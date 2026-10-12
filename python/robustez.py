"""
robustez.py
===========
Prueba de robustez del modelo DESPLEGADO (pesos_mcu.json, entrenado en el S32K312)
sobre el 15 % de prueba, con perturbaciones controladas aplicadas a la secuencia
de 50 pasos que ve la red. Se compara contra el modelo estatico (MLP de
modelo_estatico.py) bajo las mismas perturbaciones.

Perturbaciones (nivel 0 = sin perturbar):
  ruido      ruido gaussiano de medicion, sigma en % de la escala completa
  amplitud   ganancia sobre todos los potes (error de calibracion / recorrido incompleto)
  velocidad  pendiente de cada transicion: cada pote se mueve v veces mas rapido o lento
             alrededor de su punto medio, sin cambiar CUANDO se mueve
  duracion   escala de tiempo de todo el patron: el gesto dura s x 5 s (anclado al inicio;
             si dura mas de 5 s la ventana lo corta, si dura menos los potes quedan quietos)
  perdida    probabilidad de perder un paso completo (3 potes); se mantiene el ultimo valor
             (sample-and-hold, como haria el firmware si una conversion falla)
  perdida0   igual, pero el paso perdido llega como 0 (caso pesimista, sin validacion)

Ademas, prueba de ORDEN TEMPORAL: la misma ventana invertida en el tiempo y con los
pasos barajados. Las caracteristicas del MLP son identicas en los tres casos; la
LSTM si cambia de decision.

Decision del firmware: umbral de confianza (UMBRAL) + compuerta de movimiento.
Cada nivel aleatorio se repite REPS veces (media +- d.e.).

Uso:
    python robustez.py [--pesos pesos_mcu.json] [--reps 10]
Salidas: robustez.json, figuras/fig_robustez.pdf, figuras/fig_orden.pdf,
         tablas/tabla_robustez.tex, tablas/tabla_orden.tex
"""

import argparse
import os

import numpy as np

import lstm_core as core
from lstm_core import LSTMNet, save_json, matriz_confusion, metricas_clase, con_umbral
from cargar_pesos import leer_pesos
from modelo_estatico import ModeloEstatico
from run_pc import preparar, DATOS

T = core.T
NIVELES = {
    "ruido":     [0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
    "amplitud":  [0.5, 0.7, 0.85, 1.0, 1.1, 1.2],
    "velocidad": [0.25, 0.5, 1.0, 2.0, 4.0],
    "duracion":  [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "perdida":   [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "perdida0":  [0.0, 0.02, 0.05, 0.1, 0.2],
}
NEUTRO = {"ruido": 0.0, "amplitud": 1.0, "velocidad": 1.0, "duracion": 1.0, "perdida": 0.0, "perdida0": 0.0}
ALEATORIA = {"ruido", "perdida", "perdida0"}
TITULO = {"ruido": "Ruido $\\sigma$ [% FS]", "amplitud": "Ganancia", "velocidad": "Velocidad de transición",
          "duracion": "Duración del patrón", "perdida": "Pérdida (retención) [%]",
          "perdida0": "Pérdida (valor 0) [%]"}
ETIQ = {"ruido": lambda v: f"{100 * v:g}", "amplitud": lambda v: f"{v:g}", "velocidad": lambda v: f"{v:g}x",
        "duracion": lambda v: f"{v:g}x", "perdida": lambda v: f"{100 * v:g}",
        "perdida0": lambda v: f"{100 * v:g}"}


# ---------------------------------------------------------------------
# Perturbaciones sobre codigos (N, T, 3) -> codigos enteros
# ---------------------------------------------------------------------
def _cerrar(C):
    return np.clip(np.rint(C), 0, core.CODE_MAX).astype(np.int64)


def ruido(C, sigma, r):
    return _cerrar(C + r.normal(0, sigma * core.CODE_SCALE, C.shape))


def amplitud(C, g, r=None):
    return _cerrar(C * g)


def velocidad(C, v, r=None):
    """Cada canal con transicion (|fin - inicio| > 0.1) se re-muestrea alrededor de su punto medio."""
    out = np.array(C, float)
    t = np.arange(T, dtype=float)
    for n in range(len(C)):
        for k in range(C.shape[2]):
            x = C[n, :, k].astype(float)
            dx = x[-1] - x[0]
            if abs(dx) < 100:
                continue
            f = (x - x[0]) / dx
            tm = float(np.interp(0.5, np.maximum.accumulate(f), t)) if f.max() >= 0.5 else T / 2
            out[n, :, k] = np.interp(tm + (t - tm) * v, t, x)
    return _cerrar(out)


def duracion(C, s, r=None):
    t = np.arange(T, dtype=float)
    return _cerrar(np.stack([[np.interp(t / s, t, C[n, :, k]) for k in range(C.shape[2])]
                             for n in range(len(C))], 0).transpose(0, 2, 1))


def perdida(C, p, r):
    out = np.array(C)
    for n in range(len(C)):
        perdido = r.random(T) < p
        perdido[0] = False
        for t in range(1, T):
            if perdido[t]:
                out[n, t] = out[n, t - 1]
    return out


def perdida_cero(C, p, r):
    """Paso perdido que llega como 0 (p. ej. trama corrupta sin validar): caso pesimista."""
    out = np.array(C)
    out[r.random(C.shape[:2]) < p] = 0
    return out


PERTURB = {"ruido": ruido, "amplitud": amplitud, "velocidad": velocidad, "duracion": duracion,
           "perdida": perdida, "perdida0": perdida_cero}


# ---------------------------------------------------------------------
def evaluar(net, mlp, C, y):
    P = net.predict_proba(core.codes_to_x(C, np.float32))
    yp = con_umbral(P, codes=C)
    M = matriz_confusion(y, yp)
    ym = mlp.predecir(C)
    Mm = matriz_confusion(y, ym)
    return {"acc": 100 * float(np.mean(yp == y)), "f1": metricas_clase(M)["macro_f1"],
            "acc_argmax": 100 * float(np.mean(P.argmax(1) == y)),
            "acc_mlp": 100 * float(np.mean(ym == y)), "f1_mlp": metricas_clase(Mm)["macro_f1"],
            "pred": yp, "pred_mlp": ym}


def barrido(net, mlp, d, reps, seed=0):
    out = {}
    for nombre, niveles in NIVELES.items():
        filas = []
        for v in niveles:
            n_rep = reps if (nombre in ALEATORIA and v != NEUTRO[nombre]) else 1
            rs = [evaluar(net, mlp, PERTURB[nombre](d["C_test"], v, np.random.default_rng(seed + i)), d["y_test"])
                  for i in range(n_rep)]
            fila = {"nivel": v, "cm": matriz_confusion(d["y_test"], rs[0]["pred"]).tolist(),
                    "cm_mlp": matriz_confusion(d["y_test"], rs[0]["pred_mlp"]).tolist()}
            for k in ("acc", "f1", "acc_argmax", "acc_mlp", "f1_mlp"):
                a = np.array([x[k] for x in rs]); fila[k] = float(a.mean()); fila[k + "_sd"] = float(a.std())
            filas.append(fila)
            print(f"  {nombre:10s} {ETIQ[nombre](v):>6s}  LSTM {fila['acc']:5.1f} +- {fila['acc_sd']:4.1f} %"
                  f"  (F1 {fila['f1']:.3f})   MLP {fila['acc_mlp']:5.1f} +- {fila['acc_mlp_sd']:4.1f} %")
        out[nombre] = filas
    return out


def orden_temporal(net, mlp, d, reps, seed=0):
    C, y = d["C_test"], d["y_test"]
    base = evaluar(net, mlp, C, y)
    inv = evaluar(net, mlp, C[:, ::-1].copy(), y)
    r = np.random.default_rng(seed)
    bar = [evaluar(net, mlp, C[:, r.permutation(T)].copy(), y) for _ in range(reps)]
    res = {}
    for nombre, rs in (("original", [base]), ("invertida", [inv]), ("barajada", bar)):
        res[nombre] = {
            "acc": float(np.mean([x["acc"] for x in rs])), "acc_mlp": float(np.mean([x["acc_mlp"] for x in rs])),
            "misma_decision_lstm": 100 * float(np.mean([np.mean(x["pred"] == base["pred"]) for x in rs])),
            "misma_decision_mlp": 100 * float(np.mean([np.mean(x["pred_mlp"] == base["pred_mlp"]) for x in rs]))}
    res["cm_invertida_lstm"] = matriz_confusion(y, inv["pred"]).tolist()
    res["cm_invertida_mlp"] = matriz_confusion(y, inv["pred_mlp"]).tolist()
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k:10s} LSTM {v['acc']:5.1f} % (misma decision {v['misma_decision_lstm']:5.1f} %)   "
                  f"MLP {v['acc_mlp']:5.1f} % (misma decision {v['misma_decision_mlp']:5.1f} %)")
    print("  LSTM, ventana invertida (filas: clase real, columnas: decision):")
    print(np.array(res["cm_invertida_lstm"]))
    return res


# ---------------------------------------------------------------------
def figuras(res, orden):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from graficas import aplicar_estilo, guardar, AZUL, NARANJA, GRIS
    aplicar_estilo()
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.1), sharey=True)
    axs = axs.ravel()
    for ax, (nombre, filas) in zip(axs, res.items()):
        x = np.arange(len(filas))
        for k, c, m, lab in (("acc", AZUL, "o-", "LSTM (MCU)"), ("acc_mlp", NARANJA, "s--", "MLP estático")):
            ax.errorbar(x, [f[k] for f in filas], [f[k + "_sd"] for f in filas], fmt=m, c=c, ms=3, capsize=2,
                        mfc="none" if k == "acc_mlp" else c, label=lab)
        i0 = [f["nivel"] for f in filas].index(NEUTRO[nombre])
        ax.axvline(i0, c=GRIS, lw=0.6, ls=":")
        ax.set_xticks(x); ax.set_xticklabels([ETIQ[nombre](f["nivel"]) for f in filas], fontsize=6)
        ax.set_title(TITULO[nombre], fontsize=7); ax.set_ylim(0, 103)
    axs[0].set_ylabel("Exactitud [%]"); axs[0].legend(loc="lower left", fontsize=5.5)
    fig.tight_layout(w_pad=0.3)
    guardar(fig, "fig_robustez")

    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    nombres = ["original", "invertida", "barajada"]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, [orden[n]["acc"] for n in nombres], w, color=AZUL, label="LSTM (MCU)")
    ax.bar(x + w / 2, [orden[n]["acc_mlp"] for n in nombres], w, color=NARANJA, label="MLP estático")
    for i, n in enumerate(nombres):
        ax.text(i - w / 2, orden[n]["acc"] + 2, f"{orden[n]['acc']:.0f}", ha="center", fontsize=6)
        ax.text(i + w / 2, orden[n]["acc_mlp"] + 2, f"{orden[n]['acc_mlp']:.0f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(["Ventana original", "Invertida en t", "Pasos barajados"], fontsize=7)
    ax.set_ylabel("Exactitud [%]"); ax.set_ylim(0, 115); ax.legend(loc="upper right", fontsize=6, ncol=2)
    guardar(fig, "fig_orden")


def tablas(res, orden):
    os.makedirs("tablas", exist_ok=True)
    lin = [r"\begin{table}[t]", r"\centering",
           r"\caption{Exactitud en prueba bajo perturbaciones (decisión del firmware: umbral + compuerta). "
           r"Media $\pm$ d.e. en perturbaciones aleatorias.}", r"\label{tab:robustez}", r"\scriptsize",
           r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccc}", r"\toprule",
           r"Perturbación & Nivel & LSTM [\%] & F1 LSTM & MLP [\%] \\", r"\midrule"]
    for nombre, filas in res.items():
        for j, f in enumerate(filas):
            pm = lambda k: f"{f[k]:.1f}" + (rf" $\pm$ {f[k + '_sd']:.1f}" if f[k + "_sd"] > 0 else "")
            neutro = f["nivel"] == NEUTRO[nombre]
            lin.append(" & ".join([TITULO[nombre].replace("%", r"\%") if j == 0 else "",
                                   (r"\textit{" if neutro else "") + ETIQ[nombre](f["nivel"]) + ("}" if neutro else ""),
                                   pm("acc"), f"{f['f1']:.3f}", pm("acc_mlp")]) + r" \\")
        lin.append(r"\midrule" if nombre != list(res)[-1] else r"\bottomrule")
    lin += [r"\end{tabular}", r"\end{table}"]
    open("tablas/tabla_robustez.tex", "w", encoding="utf-8").write("\n".join(lin) + "\n")

    lin = [r"\begin{table}[t]", r"\centering", r"\caption{Prueba de orden temporal sobre el conjunto de prueba.}",
           r"\label{tab:orden}", r"\scriptsize", r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{Exactitud [\%]} & \multicolumn{2}{c}{Misma decisión [\%]} \\",
           r"Ventana & LSTM & MLP & LSTM & MLP \\", r"\midrule"]
    for n, t in (("original", "Original"), ("invertida", "Invertida en el tiempo"), ("barajada", "Pasos barajados")):
        o = orden[n]
        lin.append(f"{t} & {o['acc']:.1f} & {o['acc_mlp']:.1f} & {o['misma_decision_lstm']:.1f} & "
                   f"{o['misma_decision_mlp']:.1f} " + r"\\")
    lin += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("tablas/tabla_orden.tex", "w", encoding="utf-8").write("\n".join(lin) + "\n")
    print("Tablas: tablas/tabla_robustez.tex, tablas/tabla_orden.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="pesos_mcu.json")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    d = preparar(DATOS)
    net = LSTMNet(dtype=np.float32); net.set_flat_params(leer_pesos(a.pesos))
    print(f"LSTM: {a.pesos}   decision: umbral {core.UMBRAL} + compuerta {core.MOVE_GATE_MIN}")
    mlp = ModeloEstatico(d)
    print("Perturbaciones:")
    res = barrido(net, mlp, d, a.reps)
    print("Orden temporal:")
    orden = orden_temporal(net, mlp, d, a.reps)
    limpio = next(f for f in res["ruido"] if f["nivel"] == 0.0)
    save_json("robustez.json", {"pesos": a.pesos, "umbral": core.UMBRAL, "reps": a.reps,
                                "limpio": limpio, "perturbaciones": res, "orden": orden})
    figuras(res, orden)
    tablas(res, orden)
    print("-> robustez.json")


if __name__ == "__main__":
    main()
