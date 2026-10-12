"""
estudio_hiperparametros.py
==========================
Efecto de los hiperparametros sobre la LSTM, entrenando en Python con el MISMO
procedimiento que run_pc.py (split 70/15/15 estratificado, aumento por epoca,
SGD por secuencia con BPTT, SSE, recorte de gradiente) y variando UN factor a la
vez respecto a la configuracion base (la desplegada en el S32K312):

    base: 8 unidades, ventana 5 s (50 pasos de 100 ms), 3 sensores, lr 0.1, 60 epocas

  n_h        unidades recurrentes            4, 8, 16, 32
  ventana_s  longitud de la ventana [s]      2, 3, 4, 5    (ultimos T pasos de la secuencia)
  sensores   potenciometros usados           P1 | P1+P2 | P1+P2+P3
  lr         tasa de aprendizaje             0.01, 0.03, 0.1, 0.3
  epocas     numero de epocas                5..100 (puntos de control de una sola corrida)

Cada configuracion se entrena con varias semillas de pesos iniciales (por defecto 3)
y se reporta media y desviacion estandar de:
  exactitud y F1 macro en prueba, perdida de validacion final, exactitud con la
  ventana desfasada (+-0.5 s y +1 s, como ocurre en RUN), tiempo de entrenamiento,
  numero de parametros y MAC por inferencia.

Uso:
    python estudio_hiperparametros.py                # todo (~30-60 min con 2 nucleos)
    python estudio_hiperparametros.py --solo n_h lr  # solo algunos factores
    python estudio_hiperparametros.py --semillas 1   # rapido
Las corridas ya hechas se guardan en estudio_hiperparametros.json y no se repiten.
Salidas: estudio_hiperparametros.json, figuras/fig_hiper.pdf, figuras/fig_epocas.pdf,
         tablas/tabla_hiper.tex
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import lstm_core as core
from lstm_core import LSTMNet, desplazar, matriz_confusion, metricas_clase
from run_pc import preparar, DATOS, LR, CLIP, SEED_ORDER, EPOCHS

SALIDA = "estudio_hiperparametros.json"
BASE = {"n_h": core.N_H, "ventana_s": 5.0, "sensores": [0, 1, 2], "lr": LR, "epocas": EPOCHS}
FACTORES = {
    "n_h": [4, 8, 16, 32],
    "ventana_s": [2.0, 3.0, 4.0, 5.0],
    "sensores": [[0], [0, 1], [0, 1, 2]],
    "lr": [0.01, 0.03, 0.1, 0.3],
}
EPOCAS_MAX = 100
PUNTOS_EPOCAS = [5, 10, 20, 40, 60, 80, 100]
NOMBRE = {"n_h": "Unidades recurrentes", "ventana_s": "Ventana [s]", "sensores": "Sensores",
          "lr": "Tasa de aprendizaje", "epocas": "Épocas"}
_D = None


def _datos():
    global _D
    if _D is None:
        _D = preparar(DATOS)
    return _D


def recortar(C, cfg):
    """Selecciona sensores y los ultimos T pasos (ventana mas corta = misma resolucion)."""
    T = int(round(cfg["ventana_s"] * 1000 / core.STEP_MS))
    return np.asarray(C)[..., -T:, :][..., cfg["sensores"]]


def macs(n_h, n_in, T):
    """Multiplicaciones-acumulaciones por inferencia (sin no linealidades)."""
    return T * 4 * n_h * (n_in + n_h) + core.N_D * n_h + core.N_OUT * core.N_D


def entrenar(args):
    cfg, semilla, epocas = args
    d = _datos()
    n_in = len(cfg["sensores"])
    net = LSTMNet(seed=semilla, n_h=cfg["n_h"], n_in=n_in)
    X = {k: core.codes_to_x(recortar(d[f"C_{k}"], cfg), np.float64) for k in ("val", "test")}
    orders = core.epoch_orders(len(d["C_train"]), epocas, SEED_ORDER)
    hist = {"loss_train": [], "loss_val": [], "acc_val": [], "acc_test": []}
    t_sgd = 0.0
    for ep, order in enumerate(orders):
        Ce, Ye = core.epoca_aumentada(d, ep)
        Xe = core.codes_to_x(recortar(Ce, cfg), np.float64)
        t0 = time.perf_counter(); acc = 0.0
        for i in order:
            acc += net.train_sample(Xe[i], Ye[i], cfg["lr"], CLIP)[1]
        t_sgd += time.perf_counter() - t0
        lv, av, _ = net.evaluate(X["val"], d["y_val"])
        hist["loss_train"].append(acc / len(order)); hist["loss_val"].append(lv)
        hist["acc_val"].append(av); hist["acc_test"].append(net.evaluate(X["test"], d["y_test"])[1])
    _, acc_te, P = net.evaluate(X["test"], d["y_test"])
    M = matriz_confusion(d["y_test"], P.argmax(1))
    desf = {}
    for s in (-5, 5, 10):
        C = np.array([desplazar(c, s) for c in d["C_test"]])
        desf[str(s)] = net.evaluate(core.codes_to_x(recortar(C, cfg), np.float64), d["y_test"])[1]
    return {"cfg": cfg, "semilla": semilla, "epocas": epocas, "acc_test": acc_te,
            "f1_macro": metricas_clase(M)["macro_f1"], "loss_val": hist["loss_val"][-1],
            "acc_desfase": desf, "t_sgd_s": t_sgd, "n_params": net.n_params(),
            "macs": macs(cfg["n_h"], n_in, int(round(cfg["ventana_s"] * 10))), "hist": hist}


def clave(cfg, semilla, epocas):
    return json.dumps([cfg, semilla, epocas], sort_keys=True)


def correr(trabajos, procesos):
    hechos = json.load(open(SALIDA)) if os.path.exists(SALIDA) else {}
    pend = [t for t in trabajos if clave(*t) not in hechos]
    print(f"{len(trabajos)} corridas, {len(trabajos) - len(pend)} ya hechas, {len(pend)} pendientes")
    t0 = time.time()
    with Pool(procesos) as pool:
        for k, r in enumerate(pool.imap_unordered(entrenar, pend), 1):
            hechos[clave(r["cfg"], r["semilla"], r["epocas"])] = r
            with open(SALIDA, "w") as f:
                json.dump(hechos, f)
            print(f"  [{k}/{len(pend)}] {r['cfg']} s={r['semilla']}  acc={r['acc_test']:.1f}%  "
                  f"F1={r['f1_macro']:.3f}  {r['t_sgd_s']:.0f}s  ({time.time() - t0:.0f}s)")
    return hechos


def resumen(hechos, factor, valor, semillas):
    cfg = dict(BASE); cfg[factor] = valor
    rs = [hechos.get(clave(cfg, s, EPOCHS)) for s in semillas]
    rs = [r for r in rs if r]
    if not rs:
        return None
    f = lambda k: np.array([r[k] for r in rs], float)
    des = np.array([np.mean([r["acc_desfase"][k] for k in ("-5", "5", "10")]) for r in rs])
    return {"valor": valor, "n": len(rs), "acc": f("acc_test"), "f1": f("f1_macro"), "loss_val": f("loss_val"),
            "desfase": des, "t": f("t_sgd_s"), "n_params": rs[0]["n_params"], "macs": rs[0]["macs"]}


def etiqueta(factor, v):
    if factor == "sensores":
        return "+".join(f"P{k + 1}" for k in v)
    if factor == "ventana_s":
        return f"{v:g} s"
    return f"{v:g}"


def figuras_y_tabla(hechos, semillas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from graficas import aplicar_estilo, guardar, AZUL, NARANJA, VERDE
    aplicar_estilo()
    os.makedirs("figuras", exist_ok=True); os.makedirs("tablas", exist_ok=True)

    filas = []
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for ax, factor in zip(axs, FACTORES):
        res = [resumen(hechos, factor, v, semillas) for v in FACTORES[factor]]
        res = [r for r in res if r]
        if not res:
            ax.set_visible(False); continue
        x = np.arange(len(res))
        ax.errorbar(x, [r["acc"].mean() for r in res], [r["acc"].std() for r in res], fmt="o-", c=AZUL,
                    ms=3, capsize=2, label="prueba")
        ax.errorbar(x, [r["desfase"].mean() for r in res], [r["desfase"].std() for r in res],
                    fmt="s--", c=NARANJA, ms=3, mfc="none", capsize=2, label="ventana desfasada")
        ax.set_xticks(x); ax.set_xticklabels([etiqueta(factor, r["valor"]) for r in res], fontsize=6)
        ax.set_title(NOMBRE[factor], fontsize=7); ax.set_ylim(70, 102)
        for r in res:
            filas.append((NOMBRE[factor], etiqueta(factor, r["valor"]), r, r["valor"] == BASE[factor]))
    axs[0].set_ylabel("Exactitud [%]")
    axs[0].legend(loc="lower right", fontsize=5.5)
    fig.tight_layout(w_pad=0.4)
    guardar(fig, "fig_hiper")

    # Numero de epocas: una corrida de EPOCAS_MAX por semilla, puntos de control
    curvas = [hechos.get(clave(BASE, s, EPOCAS_MAX)) for s in semillas]
    curvas = [c for c in curvas if c]
    if curvas:
        A = np.array([c["hist"]["acc_test"] for c in curvas]); V = np.array([c["hist"]["acc_val"] for c in curvas])
        L = np.array([c["hist"]["loss_val"] for c in curvas])
        ep = np.arange(1, A.shape[1] + 1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.6))
        for M_, c_, n_ in ((V, AZUL, "validación"), (A, NARANJA, "prueba")):
            a1.plot(ep, M_.mean(0), c=c_, label=n_)
            a1.fill_between(ep, M_.mean(0) - M_.std(0), M_.mean(0) + M_.std(0), color=c_, alpha=0.15, lw=0)
        a1.set_xlabel("Época"); a1.set_ylabel("Exactitud [%]"); a1.legend(loc="lower right", fontsize=5.5)
        a2.semilogy(ep, L.mean(0), c=VERDE); a2.set_xlabel("Época"); a2.set_ylabel("SSE validación")
        for a in (a1, a2):
            a.axvline(EPOCHS, c="k", ls=":", lw=0.7)
        fig.tight_layout()
        guardar(fig, "fig_epocas")
        tsgd = np.mean([c["t_sgd_s"] for c in curvas]) / EPOCAS_MAX
        for e in PUNTOS_EPOCAS:
            if e <= A.shape[1]:
                r = {"acc": A[:, e - 1], "f1": None, "loss_val": L[:, e - 1], "desfase": None,
                     "t": np.full(len(curvas), tsgd * e), "n_params": curvas[0]["n_params"], "macs": curvas[0]["macs"]}
                filas.append((NOMBRE["epocas"], str(e), r, e == EPOCHS))

    def pm(a, fmt="{:.1f}"):
        if a is None:
            return "--"
        a = np.asarray(a, float)
        return (fmt.format(a.mean()) + (r" $\pm$ " + fmt.format(a.std()) if len(a) > 1 else ""))

    lin = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Efecto de los hiperparámetros (media $\pm$ d.e., " + f"{len(semillas)}" +
           r" semillas). Base en negritas.}", r"\label{tab:hiper}", r"\scriptsize",
           r"\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{llccccrr}", r"\toprule",
           r"Factor & Valor & Exact. [\%] & F1 macro & Desf. [\%] & SSE val. & Parám. & $t$ [s] \\",
           r"\midrule"]
    previo = None
    for fac, val, r, es_base in filas:
        if previo and fac != previo:
            lin.append(r"\midrule")
        b = (lambda s: r"\textbf{" + s + "}") if es_base else (lambda s: s)
        lin.append(" & ".join([fac if fac != previo else "", b(val), pm(r["acc"]), pm(r["f1"], "{:.3f}"),
                               pm(r["desfase"]), pm(r["loss_val"], "{:.3f}"), str(r["n_params"]),
                               pm(r["t"], "{:.0f}")]) + r" \\")
        previo = fac
    lin += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join("tablas", "tabla_hiper.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lin) + "\n")
    print("Tabla: tablas/tabla_hiper.tex")
    for fac, val, r, es_base in filas:
        print(f"  {fac:22s} {val:10s} acc {pm(r['acc'])}  F1 {pm(r['f1'], '{:.3f}')}  "
              f"desf {pm(r['desfase'])}  t {pm(r['t'], '{:.0f}')} s  params {r['n_params']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", nargs="*", choices=list(FACTORES) + ["epocas"])
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--solo-figuras", action="store_true")
    a = ap.parse_args()
    semillas = list(range(1, a.semillas + 1))
    factores = a.solo or (list(FACTORES) + ["epocas"])

    trabajos = []
    if "epocas" in factores:
        trabajos += [(dict(BASE), s, EPOCAS_MAX) for s in semillas]
    for fac in factores:
        if fac == "epocas":
            continue
        for v in FACTORES[fac]:
            cfg = dict(BASE); cfg[fac] = v
            if v == BASE[fac] and fac != "n_h":
                continue                                  # la base se corre una vez (bajo n_h)
            trabajos += [(cfg, s, EPOCHS) for s in semillas]
    if "n_h" not in factores:
        trabajos += [(dict(BASE), s, EPOCHS) for s in semillas]
    trabajos = list({clave(*t): t for t in trabajos}.values())
    trabajos.sort(key=lambda t: -t[0]["n_h"] * t[2])          # las mas largas primero

    hechos = json.load(open(SALIDA)) if (a.solo_figuras and os.path.exists(SALIDA)) else correr(trabajos, a.procesos)
    figuras_y_tabla(hechos, semillas)


if __name__ == "__main__":
    main()
