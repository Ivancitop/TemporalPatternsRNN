"""
revisar_demo.py
===============
Revisa un registro de `telemetria.py demo` (tlm_demo_*.csv) para entender por
que un gesto no se clasifico.

  1) Reconstruye cada 100 ms la ventana de 5 s que vio la red, a partir de las
     lecturas crudas (raw1..raw3) y la misma calibracion del firmware.
  2) Corre la LSTM con los pesos (pesos_mcu.json) + umbral + compuerta, igual
     que el firmware, y grafica las 5 probabilidades.
  3) Encuentra cada movimiento de los potes y lo compara con la coreografia del
     dataset: posicion inicial, orden de los potes, nivel final y duracion.
  4) Con --barrido, repite la decision del firmware (umbral + antirrebote) con
     varios valores sobre TU registro, para elegir los del comando H.

Uso:
    python revisar_demo.py tlm_demo_20261010_234948.csv [--pesos pesos_mcu.json]
    python revisar_demo.py tlm_demo_...csv --umbral 0.85 --antirrebote 5 --barrido
Salida:
    <archivo>_revision.png   y un resumen por gesto en la consola
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lstm_core as core
from lstm_core import LSTMNet
from cargar_pesos import leer_pesos
from telemetria import leer

COLORES = ["#8a8985", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
POTES = ["#e87ba4", "#008300", "#6250d6"]

COREOGRAFIA = {
    1: ("los 3 en 0", "pot1 -> pot2 -> pot3", "los 3 al maximo"),
    2: ("los 3 en 0", "pot1 -> pot2 -> pot3", "los 3 a la mitad"),
    3: ("los 3 al maximo", "pot3 -> pot2 -> pot1", "los 3 en 0"),
    4: ("los 3 al maximo", "pot3 -> pot2 -> pot1", "los 3 a la mitad"),
}


def niveles_100ms(d):
    """Telemetria cada 10 ms -> codigos cada 100 ms (aprox. del promedio de 20 x 5 ms)."""
    raw = np.vstack([d["raw1"], d["raw2"], d["raw3"]]).T.astype(np.int64)
    n = (len(raw) // 10) * 10
    suma = raw[:n].reshape(-1, 10, 3).sum(1) * 2        # 10 lecturas a 10 ms ~ 20 a 5 ms
    t = d["t_ms"][:n].reshape(-1, 10)[:, -1] / 1000.0
    return t, core.raw_to_code(suma)


def replay(net, codes):
    """Probabilidades de la red y compuerta de movimiento, una ventana cada 100 ms."""
    P = np.full((len(codes), core.N_OUT), np.nan)
    quieta = np.ones(len(codes), bool)
    for k in range(core.T - 1, len(codes)):
        w = codes[k - core.T + 1:k + 1]
        P[k] = net.predict_proba(core.codes_to_x(w[None], np.float64))[0]
        quieta[k] = core.ventana_quieta(w[None])[0]
    return P, quieta


def activa(P, quieta, umbral, n):
    """Misma logica que Classify() de main.c: umbral + compuerta + antirrebote."""
    act = np.zeros(len(P), int)
    cand, cnt, a = 0, 0, 0
    for k in range(len(P)):
        if np.isnan(P[k, 0]):
            act[k] = a; continue
        c = int(np.argmax(P[k]))
        raw = c if (P[k, c] >= umbral and not quieta[k]) else 0
        if raw == cand:
            cnt += 1
        else:
            cand, cnt = raw, 1
        if cnt >= n and a != cand:
            a = cand
        act[k] = a
    return act


def resumen_tramo(t, act, a, b, extra=5.0):
    m = (t >= t[a]) & (t <= t[b] + extra)
    return [int(k) for k in np.unique(act[m]) if k != 0]


def segmentos(t, lv, umbral=0.01, hueco=1.0, minimo=0.15):
    """Tramos con movimiento (algun pote cambia > umbral por paso), unidos si el hueco < 1 s."""
    mov = np.concatenate([[False], np.any(np.abs(np.diff(lv, axis=0)) > umbral, axis=1)])
    idx = np.where(mov)[0]
    if not len(idx):
        return []
    segs, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[b] > hueco:
            segs.append((a, b)); a = i
        b = i
    segs.append((a, b))
    out = []
    for a, b in segs:
        ini = lv[max(a - 3, 0)]; fin = lv[min(b + 3, len(lv) - 1)]
        if np.max(np.abs(fin - ini)) >= minimo:
            out.append((a, b, ini, fin))
    return out


def describir(nivel):
    return "0" if nivel < 0.15 else ("mitad" if 0.35 <= nivel <= 0.7 else ("max" if nivel > 0.85 else f"{nivel:.2f}"))


def diagnostico(t, lv, a, b, ini, fin):
    cambio = fin - ini
    movidos = [k for k in range(3) if abs(cambio[k]) >= 0.2]
    orden = []
    for k in movidos:
        f = (lv[a - 3:b + 4, k] - ini[k]) / cambio[k]
        orden.append((t[a - 3 + int(np.argmax(f >= 0.5))], k))
    orden = [k for _, k in sorted(orden)]
    sube = np.mean(cambio[movidos]) > 0 if movidos else True
    alto = np.mean(fin[movidos] if sube else ini[movidos]) > 0.8 if movidos else False
    intento = (1 if alto else 2) if sube else (3 if np.mean(fin[movidos]) < 0.25 else 4) if movidos else 0
    avisos = []
    esperado_ini = np.zeros(3) if intento in (1, 2) else np.full(3, 1.02)
    esperado_fin = {1: np.full(3, 1.02), 2: np.full(3, 0.5), 3: np.zeros(3), 4: np.full(3, 0.5)}.get(intento)
    if intento:
        if len(movidos) < 3:
            avisos.append(f"solo se movieron {len(movidos)} potes (pot{', pot'.join(str(k + 1) for k in movidos)}); el gesto mueve los 3")
        if np.max(np.abs(ini - esperado_ini)) > 0.2:
            avisos.append(f"posicion inicial ({', '.join(describir(v) for v in ini)}): debe ser {COREOGRAFIA[intento][0]}")
        if np.max(np.abs(fin - esperado_fin)) > 0.2:
            avisos.append(f"posicion final ({', '.join(describir(v) for v in fin)}): debe ser {COREOGRAFIA[intento][2]}")
        esperado_orden = [0, 1, 2] if intento in (1, 2) else [2, 1, 0]
        if len(orden) == 3 and orden != esperado_orden:
            avisos.append(f"orden pot{'->pot'.join(str(k + 1) for k in orden)}: debe ser {COREOGRAFIA[intento][1]}")
        dur = t[b] - t[a]
        if dur < 2.0:
            avisos.append(f"duro {dur:.1f} s: el dataset tarda ~4 s (cada pote ~1 s, uno tras otro)")
        if dur > 5.5:
            avisos.append(f"duro {dur:.1f} s: mas que la ventana de 5 s, la red no ve el gesto completo")
    return intento, orden, avisos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivo")
    ap.add_argument("--pesos", default="pesos_mcu.json")
    ap.add_argument("--umbral", type=float, default=core.UMBRAL)
    ap.add_argument("--antirrebote", type=int, default=core.ANTIRREBOTE)
    ap.add_argument("--barrido", action="store_true")
    args = ap.parse_args()

    d = leer(args.archivo)
    net = LSTMNet(dtype=np.float32); net.set_flat_params(leer_pesos(args.pesos))
    t, codes = niveles_100ms(d)
    lv = codes / core.CODE_SCALE
    P, quieta = replay(net, codes)
    cls_rep = activa(P, quieta, args.umbral, args.antirrebote)

    print(f"{args.archivo}: {t[-1] - t[0]:.0f} s, pesos {args.pesos}")
    print(f"Rango crudo observado: " + ", ".join(
        f"pot{k + 1} {int(d[f'raw{k + 1}'].min())}-{int(d[f'raw{k + 1}'].max())}" for k in range(3))
        + f"   (calibracion: {core.ADC_MIN}-{core.ADC_MAX})")
    t_mcu = d["t_ms"] / 1000.0
    segs = segmentos(t, lv)
    for i, (a, b, ini, fin) in enumerate(segs):
        intento, orden, avisos = diagnostico(t, lv, a, b, ini, fin)
        m = (t_mcu >= t[a]) & (t_mcu <= t[b] + 5.0)
        vistas = [int(k) for k in np.unique(d["cls"][m]) if k != 0]
        pmax = np.nanmax(P[a:min(b + 50, len(P))], axis=0)
        nombre = core.CLASES[intento] if intento else "no reconocible"
        print(f"\n[{i + 1}] t={t[a]:.1f}-{t[b]:.1f} s  parece {nombre}  "
              f"({', '.join(describir(v) for v in ini)} -> {', '.join(describir(v) for v in fin)}, "
              f"orden pot{'->pot'.join(str(k + 1) for k in orden) if orden else '-'})")
        rep = resumen_tramo(t, cls_rep, a, b)
        print(f"     el MCU activo: {', '.join(core.CLASES[k] for k in vistas) or 'solo Default'}"
              f"   | PC con umbral {args.umbral} y antirrebote {args.antirrebote}: "
              f"{', '.join(core.CLASES[k] for k in rep) or 'solo Default'}")
        print("     p max de la red: " + "  ".join(f"{core.CLASES[k]} {pmax[k]:.2f}" for k in range(1, 5)))
        for a_ in avisos:
            print(f"     ! {a_}")

    if args.barrido and segs:
        print("\nBarrido (clases activas por tramo; * = coincide con lo que parece el gesto):")
        print("  umbral  antirr.  " + "  ".join(f"[{i + 1}]".ljust(9) for i in range(len(segs))))
        intentos = [diagnostico(t, lv, a, b, ini, fin)[0] for a, b, ini, fin in segs]
        for th in (0.7, 0.8, 0.85, 0.9):
            for n in (3, 5, 7):
                act = activa(P, quieta, th, n)
                celdas = []
                for (a, b, _, _), it in zip(segs, intentos):
                    r = resumen_tramo(t, act, a, b)
                    txt = "+".join(["Def", "A+", "N+", "A-", "N-"][k] for k in r) or "Def"
                    celdas.append((txt + ("*" if it in r and len(r) == 1 else "")).ljust(9))
                print(f"  {th:5.2f}   {n:5d}    " + "  ".join(celdas))
        print("  (los tramos que en realidad son acomodos de los potes deberian quedar en Def)")

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1, 0.7]})
    for k in range(3):
        a1.plot(t, lv[:, k], c=POTES[k], label=f"pot {k + 1}")
    a1.set_ylabel("Nivel calibrado"); a1.legend(loc="upper left", ncol=3); a1.set_ylim(-0.05, 1.15)
    for k in range(1, 5):
        a2.plot(t, P[:, k], c=COLORES[k], ls="-" if k in (1, 3) else "--", label=core.CLASES[k])
    a2.axhline(args.umbral, c="k", ls=":", lw=0.8)
    a2.set_ylabel("p (red)"); a2.legend(loc="upper left", ncol=4, fontsize=8)
    a3.step(t_mcu, d["cls"], where="post", c="k", label="clase activa (MCU)")
    a3.step(t, cls_rep, where="post", c="0.6", lw=0.8,
            label=f"reconstruida en PC (umbral {args.umbral}, antirrebote {args.antirrebote})")
    a3.set_yticks(range(5)); a3.set_yticklabels(["Def", "A+", "N+", "A−", "N−"])
    a3.set_xlabel("Tiempo [s]"); a3.legend(loc="upper left", ncol=2, fontsize=8)
    for ax in (a1, a2, a3):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.archivo.replace(".csv", "_revision.png")
    fig.savefig(out, dpi=130)
    print(f"\nGrafica: {out}")


if __name__ == "__main__":
    main()
