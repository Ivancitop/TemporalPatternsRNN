"""
analisis_mcu.py
===============
Metricas de la implementacion embebida a partir de lo medido en el S32K312:

  resultados_eval.json   clasificacion en el MCU (modo 'V' sobre el 15 % de prueba)
  resultados_mcu.json    entrenamiento en el MCU (ciclos DWT de forward y BPTT, RTT UART)
  tlm_demo_*.csv         operacion en RUN (telemetria cada 10 ms)
  salida de `arm-none-eabi-size` (text, data, bss)  -> argumentos --text --data --bss

Calcula:
  clasificacion  exactitud, precision, recall, F1, matriz de confusion (MCU)
  temporal       periodo de muestreo, actualizacion de ventana, inferencia (ciclos DWT),
                 latencia sensor -> decision, decision -> actuacion (PWM),
                 frecuencia maxima de inferencia
  recursos       Flash, RAM, carga de CPU, parametros, tamano del modelo,
                 energia por inferencia (si se da --idd-ma)
  robotico       por cada activacion: tiempo de respuesta, error de seguimiento,
                 sobreimpulso, tiempo de establecimiento, estabilidad; decisiones
                 incorrectas (comparando con el gesto que se intento, ver --intento)

Uso:
    python analisis_mcu.py tlm_demo_20261011_203105.csv --text 91600 --data 4 --bss 33924 \
        [--idd-ma 45 --vdd 5.0] [--intento "1,3,2,0,0,0,0,0,0,1,4"]
  --intento: clase que se QUISO hacer en cada tramo de movimiento que detecta
  revisar_demo.py (0 = no era un gesto valido / prueba de rechazo). Sin el, se usa
  la clase que "parece" segun la coreografia.
Salidas: analisis_mcu.json, tablas/tabla_mcu.tex, tablas/tabla_robotico.tex,
         figuras/fig_demo.pdf, figuras/fig_respuesta.pdf
"""

import argparse
import os

import numpy as np

import lstm_core as core
from lstm_core import LSTMNet, load_json, save_json
from cargar_pesos import leer_pesos
from revisar_demo import niveles_100ms, replay, activa, segmentos, diagnostico
from telemetria import leer

CORE_MHZ = 120.0
TICK_MS, CTRL_MS, STEP_MS = 5.0, 10.0, core.STEP_MS
STEPS_POR_TICK = 5                       # LSTM_STEPS_PER_TICK de main.c
FLASH_TOTAL, RAM_TOTAL = 2 * 1024 * 1024, 192 * 1024     # S32K312: 2 MB PFlash, 192 KB SRAM


def memoria_modelo():
    H, G, D, K, T, NI = core.N_H, 4 * core.N_H, core.N_D, core.N_OUT, core.T, core.N_IN
    f = 4
    return {"pesos_B": core.N_PARAMS * f,
            "cache_bptt_B": (T * (G + 3 * H) + D + K) * f,
            "gradiente_B": core.N_PARAMS * f,
            "secuencias_B": 2 * T * NI * f,
            "estado_inferencia_B": (2 * H) * f + 4}


def activaciones(t, cls, ref):
    """Intervalos [i0, i1) con clase activa constante != 0."""
    out, i = [], 0
    while i < len(cls):
        if cls[i] != 0:
            j = i
            while j < len(cls) and cls[j] == cls[i]:
                j += 1
            out.append((i, j, int(cls[i]), float(ref[i])))
            i = j
        else:
            i += 1
    return out


def metricas_respuesta(t, rpm, u, ref, i0, i1, banda=0.05):
    tt = t[i0:i1] - t[i0]; y = rpm[i0:i1]; uu = u[i0:i1]
    dur = tt[-1] if len(tt) else 0.0
    cola = slice(int(0.6 * len(y)), None)
    yf = float(np.median(y[cola])) if len(y) > 10 else float("nan")
    r = {"duracion_s": float(dur), "ref": ref, "final_rpm": yf,
         "error_seg_rpm": float(np.mean(ref - y[cola])) if len(y) > 10 else float("nan"),
         "error_seg_rms": float(np.sqrt(np.mean((ref - y[cola]) ** 2))) if len(y) > 10 else float("nan"),
         "rizado_rpm": float(np.std(y[cola])) if len(y) > 10 else float("nan"),
         "saturacion_pct": float(100 * np.mean(np.abs(uu) >= 0.99))}
    k = np.where(np.abs(np.diff(np.sign(uu))) > 0)[0]
    r["latencia_pwm_ms"] = float(1000 * (tt[np.argmax(np.abs(uu) > 0)])) if np.any(np.abs(uu) > 0) else float("nan")
    if np.isfinite(yf) and abs(yf) > 5:
        s = np.sign(yf); yn = s * y
        r["t90_s"] = float(tt[np.argmax(yn >= 0.9 * abs(yf))]) if np.any(yn >= 0.9 * abs(yf)) else float("nan")
        r["t63_s"] = float(tt[np.argmax(yn >= 0.632 * abs(yf))]) if np.any(yn >= 0.632 * abs(yf)) else float("nan")
        r["sobreimpulso_final_pct"] = float(max(0.0, (yn.max() - abs(yf)) / abs(yf) * 100))
        r["sobreimpulso_ref_pct"] = float(max(0.0, (yn.max() - abs(ref)) / abs(ref) * 100))
        fuera = np.where(np.abs(y - yf) > banda * abs(yf))[0]
        r["t_establecimiento_s"] = float(tt[fuera[-1] + 1]) if len(fuera) and fuera[-1] + 1 < len(tt) else float("nan")
        r["alcanza_ref"] = bool(abs(yf) >= 0.9 * abs(ref))
    return r


def frenado(t, rpm, i1, umbral=5.0):
    """Tiempo desde que la clase vuelve a default hasta |rpm| < umbral."""
    k = np.where(np.abs(rpm[i1:]) < umbral)[0]
    return float(t[i1 + k[0]] - t[i1]) if len(k) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--pesos", default="pesos_mcu.json")
    ap.add_argument("--text", type=int, required=True)
    ap.add_argument("--data", type=int, required=True)
    ap.add_argument("--bss", type=int, required=True)
    ap.add_argument("--vdd", type=float, default=5.0)
    ap.add_argument("--idd-ma", type=float, default=None, help="corriente medida de la tarjeta en RUN")
    ap.add_argument("--intento", default=None)
    ap.add_argument("--umbral", type=float, default=core.UMBRAL)
    ap.add_argument("--antirrebote", type=int, default=core.ANTIRREBOTE)
    a = ap.parse_args()

    out = {}
    # ---------------- clasificacion en el MCU ----------------
    ev = load_json("resultados_eval.json") if os.path.exists("resultados_eval.json") else None
    if ev:
        out["clasificacion"] = {"metricas": ev["prueba"]["metricas_mcu"], "cm": ev["prueba"]["cm_mcu"],
                                "acuerdo_pc_mcu": ev["prueba"]["acuerdo_pc_mcu"],
                                "max_dif_prob": ev["prueba"]["max_dif_prob"]}

    # ---------------- entrenamiento en el MCU ----------------
    mcu = load_json("resultados_mcu.json") if os.path.exists("resultados_mcu.json") else None
    if mcu:
        out["entrenamiento_mcu"] = {k: mcu.get(k) for k in ("fwd_mean_us", "bwd_mean_us", "cyc_fwd_mean", "cyc_bwd_mean",
                                                          "cyc_fwd_max", "cyc_bwd_max", "rtt_mean_ms", "wall_time_s")}

    # ---------------- RUN: telemetria ----------------
    d = leer(a.demo)
    t = d["t_ms"] / 1000.0; cls = d["cls"].astype(int); ref = d["ref"]; rpm = d["rpm"]; u = d["u"]
    cf = d["cic_fwd"][d["cic_fwd"] > 0]
    t_inf_ms = float(np.mean(cf) / CORE_MHZ / 1000)
    carga = 100 * d["carga"]
    tem = {"muestreo_adc_ms": TICK_MS, "periodo_control_ms": CTRL_MS, "actualizacion_ventana_ms": STEP_MS,
           "adquisicion_ventana_s": core.T * STEP_MS / 1000,
           "inferencia_ciclos_media": float(np.mean(cf)), "inferencia_ciclos_min": float(np.min(cf)),
           "inferencia_ciclos_max": float(np.max(cf)),
           "inferencia_ms": t_inf_ms, "inferencia_ms_max": float(np.max(cf) / CORE_MHZ / 1000),
           "inferencia_por_tick_ms": t_inf_ms * STEPS_POR_TICK / core.T,
           "inferencia_reparto_ms": core.T / STEPS_POR_TICK * TICK_MS,
           "f_inferencia_implementada_hz": 1000.0 / STEP_MS,
           "f_inferencia_max_computo_hz": 1000.0 / t_inf_ms}
    # latencia de decision por arquitectura (peor caso desde que la ventana contiene la evidencia)
    tem["latencia_arquitectura_ms"] = {"espera_paso": STEP_MS, "inferencia_repartida": tem["inferencia_reparto_ms"],
                                       "antirrebote": (a.antirrebote - 1) * STEP_MS,
                                       "total_peor_caso": STEP_MS + tem["inferencia_reparto_ms"] + (a.antirrebote - 1) * STEP_MS}

    # tramos de movimiento (gestos) y decisiones
    net = LSTMNet(dtype=np.float32); net.set_flat_params(leer_pesos(a.pesos))
    ts, codes = niveles_100ms(d)
    lv = codes / core.CODE_SCALE
    P, quieta = replay(net, codes)
    act_pc = activa(P, quieta, a.umbral, a.antirrebote)
    segs = segmentos(ts, lv)
    intentos = [int(v) for v in a.intento.split(",")] if a.intento else \
        [diagnostico(ts, lv, s0, s1, ini, fin)[0] if not diagnostico(ts, lv, s0, s1, ini, fin)[2] else 0
         for s0, s1, ini, fin in segs]
    acts = activaciones(t, cls, ref)
    filas_seg, usadas = [], set()
    for k, (s0, s1, ini, fin) in enumerate(segs):
        t_ini, t_fin = ts[s0], ts[s1]
        t_lim = ts[segs[k + 1][0]] if k + 1 < len(segs) else t[-1]
        mias = [j for j, (i0, i1, c, r_) in enumerate(acts) if t_ini <= t[i0] < t_lim]
        usadas.update(mias)
        clases = [acts[j][2] for j in mias]
        it = intentos[k] if k < len(intentos) else 0
        primera = next((j for j in mias if acts[j][2] == it), None)
        fila = {"tramo": k + 1, "t_inicio": float(t_ini), "t_fin": float(t_fin), "intento": it,
                "activadas": clases, "detectado": (it == 0 and not clases) or (primera is not None),
                "incorrectas": int(sum(1 for c in clases if c != it))}
        if primera is not None:
            fila["lat_desde_inicio_s"] = float(t[acts[primera][0]] - t_ini)
            fila["lat_desde_fin_s"] = float(t[acts[primera][0]] - t_fin)
        m = (t >= t_ini) & (t < t_lim)
        fila["coincide_pc_mcu"] = bool(set(np.unique(cls[m]).tolist()) ==
                                       set(np.unique(act_pc[(ts >= t_ini) & (ts < t_lim)]).tolist()))
        filas_seg.append(fila)
    sin_tramo = [j for j in range(len(acts)) if j not in usadas]
    lat_ini = [f["lat_desde_inicio_s"] for f in filas_seg if "lat_desde_inicio_s" in f]
    lat_fin = [f["lat_desde_fin_s"] for f in filas_seg if "lat_desde_fin_s" in f]
    validos = [f for f in filas_seg if f["intento"] != 0]
    tem["latencia_sensor_decision"] = {
        "desde_inicio_gesto_s_media": float(np.mean(lat_ini)) if lat_ini else None,
        "desde_fin_gesto_s_media": float(np.mean(lat_fin)) if lat_fin else None,
        "desde_fin_gesto_s": lat_fin}

    # robotico por activacion
    rob = []
    for i0, i1, c, r_ in acts:
        if r_ == 0:
            continue
        m = metricas_respuesta(t, rpm, u, r_, i0, i1)
        m.update({"t": float(t[i0]), "clase": c, "frenado_s": frenado(t, rpm, i1)})
        rob.append(m)
    lat_pwm = [m["latencia_pwm_ms"] for m in rob if np.isfinite(m["latencia_pwm_ms"])]
    tem["latencia_decision_pwm_ms"] = float(np.mean(lat_pwm)) if lat_pwm else None
    tem["latencia_decision_pwm_ms_max"] = float(np.max(lat_pwm)) if lat_pwm else None
    out["temporal"] = tem

    # recursos
    mem = memoria_modelo()
    flash = a.text + a.data; ram = a.data + a.bss
    rec = {"flash_B": flash, "flash_pct": 100 * flash / FLASH_TOTAL, "ram_B": ram, "ram_pct": 100 * ram / RAM_TOTAL,
           "text_B": a.text, "data_B": a.data, "bss_B": a.bss,
           "cpu_media_pct": float(np.mean(carga)), "cpu_max_pct": float(np.max(carga)),
           "n_params": core.N_PARAMS, "modelo": mem,
           "rpm_max_motor": float(np.max(np.abs(rpm))), "vdd": a.vdd, "idd_ma": a.idd_ma}
    if a.idd_ma:
        p_mw = a.vdd * a.idd_ma
        rec["potencia_mw"] = p_mw
        rec["energia_inferencia_mj"] = p_mw * t_inf_ms / 1000
        if mcu and mcu.get("fwd_mean_us"):
            rec["energia_muestra_entrenamiento_mj"] = p_mw * (mcu["fwd_mean_us"] + mcu["bwd_mean_us"]) / 1e6
    out["recursos"] = rec

    # resumen robotico
    ok = [m for m in rob if np.isfinite(m.get("t90_s", np.nan))]
    out["robotico"] = {
        "activaciones": rob, "tramos": filas_seg,
        "gestos_validos": len(validos), "gestos_validos_detectados": int(sum(f["detectado"] for f in validos)),
        "rechazos_correctos": int(sum(1 for f in filas_seg if f["intento"] == 0 and not f["activadas"])),
        "tramos_invalidos": int(sum(1 for f in filas_seg if f["intento"] == 0)),
        "decisiones_totales": len(acts), "decisiones_incorrectas": int(sum(f["incorrectas"] for f in filas_seg)) + len(sin_tramo),
        "activaciones_sin_movimiento": len(sin_tramo),
        "coincidencia_pc_mcu_tramos": int(sum(f["coincide_pc_mcu"] for f in filas_seg)),
        "t90_media_s": float(np.mean([m["t90_s"] for m in ok])) if ok else None,
        "t63_media_s": float(np.nanmean([m["t63_s"] for m in ok])) if ok else None,
        "t_establecimiento_media_s": float(np.nanmean([m["t_establecimiento_s"] for m in ok])) if ok else None,
        "sobreimpulso_final_media_pct": float(np.mean([m["sobreimpulso_final_pct"] for m in ok])) if ok else None,
        "rizado_media_rpm": float(np.nanmean([m["rizado_rpm"] for m in rob])) if rob else None,
        "frenado_media_s": float(np.nanmean([m["frenado_s"] for m in rob])) if rob else None,
    }
    for grupo, cl in (("alta", (1, 3)), ("nominal", (2, 4))):
        g = [m for m in rob if m["clase"] in cl and np.isfinite(m["error_seg_rpm"])]
        if g:
            out["robotico"][f"error_seg_{grupo}_rpm"] = float(np.mean(np.abs([m["error_seg_rpm"] for m in g])))
            out["robotico"][f"final_{grupo}_rpm"] = float(np.mean(np.abs([m["final_rpm"] for m in g])))
            out["robotico"][f"saturacion_{grupo}_pct"] = float(np.mean([m["saturacion_pct"] for m in g]))
    save_json("analisis_mcu.json", out)
    imprimir(out)
    tablas(out, a)
    figuras(t, cls, ref, rpm, u, lv, ts, acts, rob)


def imprimir(o):
    tm, rc, rb = o["temporal"], o["recursos"], o["robotico"]
    if "clasificacion" in o:
        m = o["clasificacion"]["metricas"]
        print(f"Clasificacion MCU: exactitud {100 * m['accuracy']:.1f} %, F1 macro {m['macro_f1']:.3f}, "
              f"acuerdo PC-MCU {100 * o['clasificacion']['acuerdo_pc_mcu']:.0f} %")
    print(f"Inferencia: {tm['inferencia_ciclos_media']:.0f} ciclos = {tm['inferencia_ms']:.2f} ms "
          f"(max {tm['inferencia_ms_max']:.2f} ms), {tm['inferencia_por_tick_ms']:.3f} ms por tick, "
          f"repartida en {tm['inferencia_reparto_ms']:.0f} ms")
    print(f"Frecuencia: implementada {tm['f_inferencia_implementada_hz']:.0f} Hz, maxima por computo "
          f"{tm['f_inferencia_max_computo_hz']:.0f} Hz")
    ls = tm["latencia_sensor_decision"]
    if ls["desde_fin_gesto_s_media"] is not None:
        print(f"Latencia sensor->decision: {ls['desde_inicio_gesto_s_media']:.2f} s desde el inicio del gesto, "
              f"{ls['desde_fin_gesto_s_media']:+.2f} s respecto al fin; arquitectura (peor caso) "
              f"{tm['latencia_arquitectura_ms']['total_peor_caso']:.0f} ms")
    print(f"Latencia decision->PWM: {tm['latencia_decision_pwm_ms']} ms (max {tm['latencia_decision_pwm_ms_max']})")
    print(f"Flash {rc['flash_B']} B ({rc['flash_pct']:.2f} %), RAM {rc['ram_B']} B ({rc['ram_pct']:.1f} %), "
          f"CPU {rc['cpu_media_pct']:.2f} % (max {rc['cpu_max_pct']:.1f} %), modelo {rc['modelo']['pesos_B']} B")
    if rc.get("energia_inferencia_mj"):
        print(f"Energia: {rc['potencia_mw']:.0f} mW -> {rc['energia_inferencia_mj']:.3f} mJ por inferencia")
    print(f"Robotico: {rb['gestos_validos_detectados']}/{rb['gestos_validos']} gestos validos detectados, "
          f"{rb['rechazos_correctos']}/{rb['tramos_invalidos']} movimientos invalidos rechazados, "
          f"{rb['decisiones_incorrectas']}/{rb['decisiones_totales']} decisiones incorrectas, "
          f"PC=MCU en {rb['coincidencia_pc_mcu_tramos']}/{len(rb['tramos'])} tramos")
    print(f"  t63 {rb['t63_media_s']}, t90 {rb['t90_media_s']} s, establecimiento {rb['t_establecimiento_media_s']} s, "
          f"sobreimpulso {rb['sobreimpulso_final_media_pct']} %, rizado {rb['rizado_media_rpm']} rpm, "
          f"frenado {rb['frenado_media_s']} s")
    for g in ("alta", "nominal"):
        if f"error_seg_{g}_rpm" in rb:
            print(f"  {g}: velocidad final {rb[f'final_{g}_rpm']:.1f} rpm, error {rb[f'error_seg_{g}_rpm']:.1f} rpm, "
                  f"saturacion {rb[f'saturacion_{g}_pct']:.0f} %")
    for f in rb["tramos"]:
        print(f"  tramo {f['tramo']:2d} t={f['t_inicio']:.1f}-{f['t_fin']:.1f} intento {core.CLASES[f['intento']]:12s} "
              f"activadas {[core.CLASES[c] for c in f['activadas']]}  incorrectas {f['incorrectas']}"
              + (f"  lat {f['lat_desde_inicio_s']:.2f}/{f['lat_desde_fin_s']:+.2f} s" if 'lat_desde_fin_s' in f else ""))


def tablas(o, a):
    os.makedirs("tablas", exist_ok=True)
    tm, rc, rb = o["temporal"], o["recursos"], o["robotico"]
    e = o.get("entrenamiento_mcu") or {}
    ls = tm["latencia_sensor_decision"]
    pend = r"\pend{medir}"
    fil = [
        (r"\multicolumn{2}{l}{\textit{Temporal}}", None),
        ("Muestreo ADC (3 canales)", f"{tm['muestreo_adc_ms']:.0f} ms"),
        ("Actualización de ventana (1 paso)", f"{tm['actualizacion_ventana_ms']:.0f} ms"),
        ("Adquisición de una ventana completa", f"{tm['adquisicion_ventana_s']:.0f} s"),
        ("Inferencia (DWT, media / máx.)", f"{tm['inferencia_ms']:.2f} / {tm['inferencia_ms_max']:.2f} ms"),
        ("Inferencia por tick (5 pasos)", f"{tm['inferencia_por_tick_ms']:.2f} ms"),
        ("Forward + BPTT en entrenamiento", f"{e.get('fwd_mean_us', 0) / 1000:.2f} + {e.get('bwd_mean_us', 0) / 1000:.2f} ms"),
        ("Latencia arquitectura (peor caso)", f"{tm['latencia_arquitectura_ms']['total_peor_caso']:.0f} ms"),
        ("Sensor $\\rightarrow$ decisión (inicio del gesto)",
         f"{ls['desde_inicio_gesto_s_media']:.2f} s" if ls["desde_inicio_gesto_s_media"] is not None else pend),
        ("Sensor $\\rightarrow$ decisión (fin del gesto)",
         f"{ls['desde_fin_gesto_s_media']:+.2f} s" if ls["desde_fin_gesto_s_media"] is not None else pend),
        ("Decisión $\\rightarrow$ actuación (PWM)", f"$<$ {CTRL_MS:.0f} ms (mismo ciclo de control)"
         if tm["latencia_decision_pwm_ms_max"] is not None and tm["latencia_decision_pwm_ms_max"] < CTRL_MS
         else (f"{tm['latencia_decision_pwm_ms_max']:.0f} ms" if tm["latencia_decision_pwm_ms_max"] is not None else pend)),
        ("Frecuencia de inferencia (impl. / máx.)",
         f"{tm['f_inferencia_implementada_hz']:.0f} / {tm['f_inferencia_max_computo_hz']:.0f} Hz"),
        (r"\multicolumn{2}{l}{\textit{Recursos}}", None),
        ("Flash (text + data)", f"{rc['flash_B'] / 1024:.1f} KB ({rc['flash_pct']:.1f}\\,\\%)"),
        ("RAM (data + bss)", f"{rc['ram_B'] / 1024:.1f} KB ({rc['ram_pct']:.1f}\\,\\%)"),
        ("Carga de CPU en RUN (media / máx.)", f"{rc['cpu_media_pct']:.1f} / {rc['cpu_max_pct']:.1f}\\,\\%"),
        ("Parámetros / tamaño (float32)", f"{rc['n_params']} / {rc['modelo']['pesos_B'] / 1024:.2f} KB"),
        ("Caché BPTT + gradiente", f"{(rc['modelo']['cache_bptt_B'] + rc['modelo']['gradiente_B']) / 1024:.1f} KB"),
        ("Energía por inferencia", f"{rc['energia_inferencia_mj']:.3f} mJ" if rc.get("energia_inferencia_mj")
         else r"$V I t_{inf}$ " + pend),
    ]
    lin = [r"\begin{table}[t]", r"\centering", r"\caption{Desempeño temporal y recursos en el S32K312 (120 MHz).}",
           r"\label{tab:mcu}", r"\scriptsize", r"\begin{tabular}{lr}", r"\toprule"]
    for k, v in fil:
        lin.append(k + r" \\" if v is None else f"{k} & {v} " + r"\\")
    lin += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("tablas/tabla_mcu.tex", "w", encoding="utf-8").write("\n".join(lin) + "\n")

    lin = [r"\begin{table}[t]", r"\centering", r"\caption{Desempeño robótico en la demostración (RUN).}",
           r"\label{tab:robotico}", r"\scriptsize", r"\begin{tabular}{lr}", r"\toprule"]
    f2 = lambda k, fmt="{:.2f}": (fmt.format(rb[k]) if rb.get(k) is not None else "--")
    filas = [("Gestos válidos detectados", f"{rb['gestos_validos_detectados']} / {rb['gestos_validos']}"),
             ("Movimientos no válidos rechazados", f"{rb['rechazos_correctos']} / {rb['tramos_invalidos']}"),
             ("Decisiones incorrectas", f"{rb['decisiones_incorrectas']} / {rb['decisiones_totales']}"),
             ("Tiempo de respuesta $t_{63}$ / $t_{90}$", f"{f2('t63_media_s')} / {f2('t90_media_s')} s"),
             ("Tiempo de establecimiento (5\\,\\%)", f"{f2('t_establecimiento_media_s')} s"),
             ("Sobreimpulso (sobre valor final)", f"{f2('sobreimpulso_final_media_pct', '{:.1f}')}\\,\\%"),
             ("Rizado en estado estable (d.e.)", f"{f2('rizado_media_rpm', '{:.1f}')} rpm"),
             ("Tiempo de frenado a default", f"{f2('frenado_media_s')} s")]
    for g, n in (("alta", "Alta"), ("nominal", "Nominal")):
        if f"error_seg_{g}_rpm" in rb:
            filas.append((f"{n}: vel. final / error de seguimiento",
                          f"{rb[f'final_{g}_rpm']:.1f} / {rb[f'error_seg_{g}_rpm']:.1f} rpm"))
            filas.append((f"{n}: tiempo con PWM saturado", f"{rb[f'saturacion_{g}_pct']:.0f}\\,\\%"))
    for k, v in filas:
        lin.append(f"{k} & {v} " + r"\\")
    lin += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("tablas/tabla_robotico.tex", "w", encoding="utf-8").write("\n".join(lin) + "\n")
    print("Tablas: tablas/tabla_mcu.tex, tablas/tabla_robotico.tex")


def figuras(t, cls, ref, rpm, u, lv, ts, acts, rob):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from graficas import aplicar_estilo, guardar, AZUL, POTES, ETQ_CORTA, COLORES
    aplicar_estilo()
    t0 = t[0]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.0, 3.6), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 0.7, 1.2]})
    for k in range(3):
        a1.plot(ts - t0, lv[:, k], c=POTES[k], label=f"P{k + 1}")
    a1.set_ylabel("Nivel"); a1.legend(loc="upper left", ncol=3, fontsize=6); a1.set_ylim(-0.05, 1.15)
    a2.step(t - t0, cls, where="post", c="k")
    a2.set_yticks(range(5)); a2.set_yticklabels(ETQ_CORTA, fontsize=6); a2.set_ylabel("Clase")
    a3.plot(t - t0, ref, "--", c="k", lw=0.9, label="referencia")
    a3.plot(t - t0, rpm, c=AZUL, label="velocidad")
    a3.set_ylabel("rpm"); a3.set_xlabel("Tiempo [s]"); a3.legend(loc="lower left", ncol=2, fontsize=6)
    for i0, i1, c, r_ in acts:
        for ax in (a1, a2, a3):
            ax.axvspan(t[i0] - t0, t[min(i1, len(t) - 1)] - t0, color=COLORES[c], alpha=0.12, lw=0)
    fig.tight_layout(h_pad=0.2)
    guardar(fig, "fig_demo")

    sel = [m for m in rob if np.isfinite(m.get("t90_s", np.nan))]
    if sel:
        fig, ax = plt.subplots(figsize=(3.4, 1.7))
        for m in sel:
            i0 = int(np.argmin(np.abs(t - m["t"])))
            n = min(int(min(2.5, m['duracion_s']) / 0.01), len(t) - i0)   # solo mientras dura la activacion
            ax.plot(t[i0:i0 + n] - t[i0], np.sign(m["ref"]) * rpm[i0:i0 + n], c=COLORES[m["clase"]], lw=0.9)
        for r_, ls in ((150, "--"), (90, ":")):
            ax.axhline(r_, c="k", ls=ls, lw=0.7)
        ax.set_xlabel("Tiempo desde la decisión [s]"); ax.set_ylabel("|rpm|")
        guardar(fig, "fig_respuesta")


if __name__ == "__main__":
    main()
