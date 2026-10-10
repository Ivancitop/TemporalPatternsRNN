"""
simulacion_lazo.py
==================
Simulacion en PC del lazo completo tal como lo ejecuta el firmware:

  potes (gestos) -> ADC (12 b, ruido) -> promedio de 100 ms -> ventana de 50 pasos
        -> LSTM (una inferencia por paso nuevo, repartida en 10 ticks = 50 ms)
        -> umbral de confianza -> antirrebote -> referencia
        -> PI (cada 10 ms, anti-windup) -> puente H -> motor
        -> encoder x4 cuantizado -> filtro de velocidad -> PI

Los gestos se toman de secuencias de PRUEBA del dataset (no vistas en el
entrenamiento), interpoladas a 5 ms, con periodos quietos entre gestos.

Sirve ANTES del laboratorio para:
  1) Disenar el PI sobre un modelo de primer orden del motor (K, tau).
  2) Estimar la latencia fin del gesto -> clase activa -> velocidad.
  3) Elegir el antirrebote (DEBOUNCE_N) y ver el efecto de REF_LATCH.

El modelo del motor es SUPUESTO (K, tau abajo). Los valores reales se
obtienen con `python telemetria.py identificar` y se reemplazan aqui.

Genera resultados_sim.json.
"""

import argparse
import os

import numpy as np

import lstm_core as core
from lstm_core import LSTMNet, save_json
from cargar_pesos import leer_pesos
from run_pc import preparar, DATOS

# ---------- temporizacion del firmware (main.c) ----------
TICK = 0.005
CTRL_DIV = 2
TS = TICK * CTRL_DIV
INFER_TICKS = 10            # LSTM_T / LSTM_STEPS_PER_TICK
CONF_THRESHOLD = 0.70
DEBOUNCE_N = 3
REF_ALTA, REF_NOM = 150.0, 90.0
REF = {0: 0.0, 1: REF_ALTA, 2: REF_NOM, 3: -REF_ALTA, 4: -REF_NOM}

# ---------- modelo supuesto del motor ----------
K_MOTOR = 200.0             # rpm por unidad de duty
TAU_MOTOR = 0.08            # s
ENC_CPR = 3332.0            # cuentas por vuelta (x4), igual que main.c
RPM_LPF_ALPHA = 0.5
DT_PLANT = 0.0005
ADC_NOISE = 6.0             # LSB rms

# ---------- PI por cancelacion de polo ----------
T_CL = 0.10
KP = TAU_MOTOR / (K_MOTOR * T_CL)
KI = KP / TAU_MOTOR


class PI:
    """Copia literal de firmware/pi_ctrl.c"""

    def __init__(self, kp, ki, ts, umax=1.0):
        self.kp, self.ki, self.ts, self.umax = kp, ki, ts, umax
        self.reset()

    def reset(self):
        self.integ = 0.0

    def step(self, ref, meas):
        e = ref - meas
        u = self.kp * e + self.integ
        us = min(max(u, -self.umax), self.umax)
        if us == u or (u > self.umax and e < 0) or (u < -self.umax and e > 0):
            self.integ = min(max(self.integ + self.ki * self.ts * e, -self.umax), self.umax)
        return us


class Motor:
    def __init__(self):
        self.w = 0.0; self.pos = 0.0

    def advance(self, u, T):
        for _ in range(int(round(T / DT_PLANT))):
            self.w += DT_PLANT / TAU_MOTOR * (K_MOTOR * u - self.w)
            self.pos += self.w / 60.0 * DT_PLANT

    def counts(self):
        return int(np.floor(self.pos * ENC_CPR))


# ---------------------------------------------------------------------
def construir_escenario(d, orden=(1, 4, 3, 2), quieto=4.0, rampa=0.3, seed=0):
    """Trayectoria de niveles a 5 ms y eventos (t_inicio, t_fin, clase) de cada gesto."""
    r = np.random.default_rng(seed)
    tramos, eventos = [], []
    nivel = np.array([0.3, 0.6, 0.4])
    tramos.append(np.repeat(nivel[None], int(6.0 / TICK), 0))         # llena la ventana: default
    t = 6.0
    for c in orden:
        idx = r.choice(np.where(d["y_test"] == c)[0])
        g = d["C_test"][idx] / core.CODE_SCALE                         # (50, 3) a 100 ms
        tt = np.arange(0, core.T * core.STEP_MS / 1000, TICK)
        gesto = np.stack([np.interp(tt, np.arange(core.T) * 0.1, g[:, k]) for k in range(3)], 1)
        n_r = int(rampa / TICK)
        tramos.append(np.linspace(nivel, gesto[0], n_r)); t += rampa
        tramos.append(gesto); eventos.append((t, t + len(gesto) * TICK, c)); t += len(gesto) * TICK
        nivel = gesto[-1]
        tramos.append(np.repeat(nivel[None], int(quieto / TICK), 0)); t += quieto
    return np.vstack(tramos), eventos


def simular(net, niveles=None, pi_gains=(KP, KI), fixed_ref=None, T_total=None,
            debounce=DEBOUNCE_N, latch=False, seed=5):
    r = np.random.default_rng(seed)
    n_ticks = len(niveles) if niveles is not None else int(T_total / TICK)
    pi = PI(*pi_gains, TS); mot = Motor()
    acc = np.zeros(3, dtype=np.int64); n_acc = 0
    win = []
    busy, t_done, snap = False, 0, None
    cls_raw = cls_cand = cls_act = 0; cand = 0; pmax = 0.0
    enc_prev = 0; rpm_f = 0.0; ref = 0.0; u = 0.0
    log = {k: [] for k in ("t", "lv1", "lv2", "lv3", "cls_raw", "cls", "ref", "rpm", "u", "pmax")}
    for k in range(1, n_ticks + 1):
        t = k * TICK
        lv = niveles[k - 1] if niveles is not None else np.zeros(3)
        raw = np.clip(np.rint(lv * core.ADC_FS + r.normal(0, ADC_NOISE, 3)), 0, 4095).astype(np.int64)
        acc += raw; n_acc += 1
        nuevo = False
        if n_acc == core.SEQ_DEC:
            win.append(core.raw_to_code(acc)); win = win[-core.T:]
            acc[:] = 0; n_acc = 0; nuevo = True

        if fixed_ref is None:
            if nuevo and not busy and len(win) == core.T:
                snap = np.array(win); busy = True; t_done = k + INFER_TICKS - 1
            if busy and k >= t_done:
                p, _ = net.forward(core.codes_to_x(snap, np.float32))
                busy = False
                c = int(np.argmax(p)); pmax = float(p[c])
                cls_raw = c if pmax >= CONF_THRESHOLD else 0
                if cls_raw == cls_cand:
                    cand += 1
                else:
                    cls_cand, cand = cls_raw, 1
                if cand >= debounce and cls_act != cls_cand and not (latch and cls_cand == 0):
                    cls_act = cls_cand
                    if cls_act == 0:
                        pi.reset()

        if k % CTRL_DIV == 0:
            enc = mot.counts()
            rpm = (enc - enc_prev) * 60.0 / (ENC_CPR * TS)
            enc_prev = enc
            rpm_f += RPM_LPF_ALPHA * (rpm - rpm_f)
            if fixed_ref is not None:
                ref = fixed_ref(t); u = pi.step(ref, rpm_f)
            else:
                ref = REF[cls_act]
                if cls_act == 0:
                    pi.reset(); u = 0.0
                else:
                    u = pi.step(ref, rpm_f)
            for key, v in zip(log, (t, *lv, cls_raw, cls_act, ref, rpm_f, u, pmax)):
                log[key].append(float(v))
        mot.advance(u, TICK)
    return log


def metricas_escalon(t, y, ref, t0, banda=0.02):
    t = np.asarray(t); y = np.asarray(y)
    m = t >= t0
    tt, yy = t[m] - t0, y[m]
    y0 = yy[0]; dy = ref - y0
    yn = (yy - y0) / dy
    mp = max(0.0, float(np.max(np.sign(dy) * (yy - ref)) / abs(dy) * 100.0))
    tr = float(tt[np.argmax(yn >= 0.9)] - tt[np.argmax(yn >= 0.1)])
    fuera = np.where(np.abs(yy - ref) > banda * abs(dy))[0]
    ts = float(tt[fuera[-1] + 1]) if len(fuera) and fuera[-1] + 1 < len(tt) else float("nan")
    return {"sobreimpulso_pct": mp, "t_subida_s": tr, "t_establecimiento_s": ts,
            "error_ss_rpm": float(np.mean(yy[-20:]) - ref)}


def latencias(log, eventos):
    """Por gesto: desde que termina hasta que la clase activa es la correcta, y desde
    ahi hasta que la velocidad entra al 10 % de la referencia. Tambien cuanto tiempo
    se mantuvo la clase correcta."""
    t = np.array(log["t"]); cls = np.array(log["cls"]); rpm = np.array(log["rpm"])
    ref = np.array(log["ref"])
    out = []
    for k, (t0, t1, c) in enumerate(eventos):
        t_lim = eventos[k + 1][0] if k + 1 < len(eventos) else t[-1]
        c_prev = eventos[k - 1][2] if k > 0 else 0     # la ventana aun contiene el gesto anterior
        m = (t >= t0) & (t < t_lim)
        idx = np.where(m & (cls == c))[0]
        if not len(idx):
            out.append({"clase": c, "detectado": False}); continue
        i0 = idx[0]
        ok = np.where((t >= t[i0]) & (t < t_lim) & (np.abs(rpm - ref[i0]) <= 0.1 * abs(ref[i0])))[0]
        out.append({"clase": c, "detectado": True, "t_fin_gesto": t1,
                    "deteccion_s": float(t[i0] - t1),
                    "accion_s": float(t[ok[0]] - t[i0]) if len(ok) else float("nan"),
                    "duracion_clase_s": float(len(idx) * TS),
                    "clases_erroneas": int(np.sum(m & (cls != c) & (cls != 0) & (cls != c_prev)))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="pesos_mcu.json")
    ap.add_argument("--datos", default=DATOS)
    args = ap.parse_args()
    fuente = args.pesos if os.path.exists(args.pesos) else "resultados_pc.json"
    net = LSTMNet(dtype=np.float32); net.set_flat_params(leer_pesos(fuente))
    d = preparar(args.datos)

    niveles, eventos = construir_escenario(d)
    log = simular(net, niveles)
    lat = latencias(log, eventos)
    log_latch = simular(net, niveles, latch=True)

    esc = simular(net, fixed_ref=lambda t: REF_ALTA if t >= 0.2 else 0.0, T_total=1.5)
    m_esc = metricas_escalon(esc["t"], esc["rpm"], REF_ALTA, 0.2)

    barrido = []
    for tcl in (0.03, 0.05, 0.08, 0.10, 0.15, 0.25):
        kp = TAU_MOTOR / (K_MOTOR * tcl); ki = kp / TAU_MOTOR
        e = simular(net, pi_gains=(kp, ki), fixed_ref=lambda t: REF_ALTA if t >= 0.2 else 0.0,
                    T_total=1.5)
        mm = metricas_escalon(e["t"], e["rpm"], REF_ALTA, 0.2)
        mm.update({"t_cl": tcl, "kp": kp, "ki": ki, "u_pico": float(np.max(np.abs(e["u"])))})
        barrido.append(mm)

    barrido_db = []
    for db in (1, 2, 3, 5):
        lg = simular(net, niveles, debounce=db, seed=9)
        la = [l for l in latencias(lg, eventos) if l["detectado"]]
        c = np.array(lg["cls"])
        barrido_db.append({"debounce": db, "conmutaciones": int(np.sum(c[1:] != c[:-1])),
                           "detectados": len(la),
                           "det_media_ms": 1e3 * float(np.mean([l["deteccion_s"] for l in la])) if la else None})

    save_json("resultados_sim.json", {
        "pesos": fuente,
        "modelo": {"K": K_MOTOR, "tau": TAU_MOTOR, "enc_cpr": ENC_CPR, "t_cl": T_CL, "kp": KP, "ki": KI},
        "escenario": log, "escenario_latch": {"cls": log_latch["cls"], "ref": log_latch["ref"],
                                              "rpm": log_latch["rpm"]},
        "eventos": [list(e) for e in eventos], "latencias": lat,
        "escalon": esc, "metricas_escalon": m_esc, "barrido_tcl": barrido,
        "barrido_debounce": barrido_db})

    print(f"Pesos: {fuente}")
    print(f"PI: Kp={KP:.5f}  Ki={KI:.5f}  (T_cl={T_CL}s, modelo K={K_MOTOR}, tau={TAU_MOTOR})")
    print("Escalon 0->150 rpm:", {k: round(v, 4) for k, v in m_esc.items()})
    for l in lat:
        n = core.CLASES[l["clase"]]
        if not l["detectado"]:
            print(f"  gesto {n:12s} NO detectado"); continue
        print(f"  gesto {n:12s} deteccion {1e3 * l['deteccion_s']:+5.0f} ms tras el fin, "
              f"accion {1e3 * l['accion_s']:4.0f} ms, clase activa {l['duracion_clase_s']:.1f} s, "
              f"muestras con clase erronea {l['clases_erroneas']}")
    for b in barrido_db:
        print(f"  antirrebote={b['debounce']}  conmutaciones={b['conmutaciones']}  "
              f"detectados={b['detectados']}/4  deteccion media={b['det_media_ms']:.0f} ms")
    for b in barrido:
        print(f"  T_cl={b['t_cl']:.2f}  Mp={b['sobreimpulso_pct']:.1f}%  "
              f"ts={b['t_establecimiento_s']:.3f}s  u_pico={b['u_pico']:.2f}")


if __name__ == "__main__":
    main()
