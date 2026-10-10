"""
graficas.py
===========
Figuras del informe (PDF vectorial a ancho de columna IEEE). Usa los JSON que
existan; las figuras que dependen de hardware se generan solo si ya estan
los resultados correspondientes.

  resultados_pc.json, resultados_pc_f32.json   (run_pc.py)
  resultados_host.json   gemelo C en PC        (run_mcu.py --host, renombrado)
  resultados_mcu.json    entrenamiento en MCU  (run_mcu.py)
  resultados_eval.json   PC vs MCU, prueba/real (evaluar.py)
  resultados_sim.json    lazo simulado          (simulacion_lazo.py)
  tlm_escalon_*.csv      escalon real           (telemetria.py escalon)

Colores: paleta categorica validada (deuteranopia/protanopia) para las 4
clases de gesto, gris neutro para default; ademas, velocidad alta en linea
continua y nominal en discontinua para que la clase no dependa solo del color.

Uso:  python graficas.py  [--salida ../informe/figuras]
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lstm_core as core
from lstm_core import load_json

COL = 3.4
OUT = "figuras"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "lines.linewidth": 1.1, "legend.framealpha": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
})
# default, alta CW, nominal CW, alta CCW, nominal CCW
COLORES = ["#8a8985", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ESTILO = ["-", "-", "--", "-", "--"]
POTES = ["#e87ba4", "#008300", "#6250d6"]          # pot 1, 2, 3
ETQ = [f"{core.CLASE_ENUNCIADO[c]}·{core.CLASES[c]}" for c in range(core.N_OUT)]
ETQ_CORTA = ["5·Def", "1·A+", "2·N+", "3·A−", "4·N−"]


def cargar(p):
    return load_json(p) if os.path.exists(p) else None


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=200)
    plt.close(fig); print(f"  {path}.pdf")


def nombre_origen(r, defecto):
    o = (r or {}).get("config", {}).get("origen", "")
    return {"host_c": "C (gemelo en PC)", "hardware": "S32K312", "simulado": "NumPy f32"}.get(o, defecto)


# ---------------------------------------------------------------------
def fig_dataset(pc):
    s = pc["split"]
    C = np.array(s["C_train"] + s["C_val"] + s["C_test"]) / core.CODE_SCALE
    y = np.array(s["y_train"] + s["y_val"] + s["y_test"])
    t = np.arange(core.T) * core.STEP_MS / 1000
    orden = [1, 2, 3, 4, 0]
    fig, axs = plt.subplots(1, 5, figsize=(2 * COL, 1.45), sharey=True)
    for ax, c in zip(axs, orden):
        m, sd = C[y == c].mean(0), C[y == c].std(0)
        for k in range(3):
            ax.plot(t, m[:, k], c=POTES[k], label=f"pot {k + 1}")
            ax.fill_between(t, m[:, k] - sd[:, k], m[:, k] + sd[:, k], color=POTES[k], alpha=0.12, lw=0)
        ax.set_title(ETQ[c], fontsize=7); ax.set_xlabel("t [s]")
    axs[0].set_ylabel("Nivel (media ± σ)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(w_pad=0.3)
    save(fig, "fig_dataset")


def fig_convergencia(pc, pc32, host, mcu):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * COL, 1.9), gridspec_kw={"width_ratios": [1, 1.3]})
    ep = np.arange(1, len(pc["loss_train"]) + 1)
    a1.semilogy(ep, pc["loss_train"], "-", c="#2a78d6", label="PC float64 · entren.")
    a1.semilogy(ep, pc["loss_val"], "--", c="#2a78d6", label="PC float64 · valid.")
    for r, c, mk, nom in ((host, "#eb6834", "o", "C (gemelo)"), (mcu, "#1baf7a", "s", None)):
        if r:
            e = np.arange(1, len(r["loss_train"]) + 1)
            a1.semilogy(e, r["loss_train"], mk, c=c, ms=2.5, mfc="none",
                        label=f"{nom or nombre_origen(r, 'MCU')} · entren.")
    a1.set_xlabel("Época"); a1.set_ylabel("SSE media por secuencia")
    a1.legend(loc="upper right")

    p64 = np.array(pc["prob_log"]); p32 = np.array(pc32["prob_log"])
    k = np.arange(len(p64))
    a2.semilogy(k, np.maximum(np.abs(p64 - p32).max(1), 1e-12), ".", ms=0.6, c="#2a78d6",
                alpha=0.4, label=r"|PC$_{64}$ − PC$_{32}$|")
    for r, c, nom in ((host, "#eb6834", "C (gemelo)"), (mcu, "#1baf7a", None)):
        if r:
            pm = np.array(r["prob_log"]); n = min(len(pm), len(p32))
            a2.semilogy(k[:n], np.maximum(np.abs(p32[:n] - pm[:n]).max(1), 1e-12), ".", ms=0.6, c=c,
                        alpha=0.4, label=rf"|PC$_{{32}}$ − {nom or nombre_origen(r, 'MCU')}|")
    a2.axhline(np.finfo(np.float32).eps, c="k", ls=":", lw=0.8)
    a2.text(len(k) * 0.99, np.finfo(np.float32).eps * 1.6, r"$\epsilon_{32}$", ha="right", fontsize=6.5)
    a2.set_xlabel("Paso de SGD (secuencias acumuladas)"); a2.set_ylabel(r"máx$_k$ |Δ$p_k$|")
    a2.legend(loc="lower right", markerscale=12)
    fig.tight_layout()
    save(fig, "fig_convergencia")


def _cm(ax, M, titulo):
    M = np.array(M); Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(core.N_OUT):
        for j in range(core.N_OUT):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=6,
                    color="w" if Mn[i, j] > 0.6 else "k")
    ax.set_xticks(range(core.N_OUT)); ax.set_xticklabels(ETQ_CORTA, fontsize=5.5, rotation=45)
    ax.set_yticks(range(core.N_OUT)); ax.set_yticklabels(ETQ_CORTA, fontsize=5.5)
    ax.set_title(titulo, fontsize=7); ax.grid(False)


def fig_confusion(pc, host, mcu, ev):
    paneles = [(pc["cm_test"], "PC float64")]
    for r, nom in ((host, "C (gemelo)"), (mcu, None)):
        if r:
            paneles.append((r["cm_test"], nom or nombre_origen(r, "MCU")))
    if ev and "real" in ev:
        k = "cm_mcu" if "cm_mcu" in ev["real"] else "cm_pc"
        paneles.append((ev["real"][k], "Datos reales"))
    fig, axs = plt.subplots(1, len(paneles), figsize=(COL * len(paneles) / 2.0, 1.9))
    for ax, (M, t) in zip(np.atleast_1d(axs), paneles):
        _cm(ax, M, t)
    np.atleast_1d(axs)[0].set_ylabel("Real")
    for ax in np.atleast_1d(axs):
        ax.set_xlabel("Predicha", fontsize=7)
    fig.tight_layout(w_pad=0.3)
    save(fig, "fig_confusion")


def fig_desplazamiento(pc):
    r = pc["desplazamiento"]
    ms = np.array([x["ms"] for x in r]) / 1000
    fig, ax = plt.subplots(figsize=(COL, 1.5))
    ax.plot(ms, [100 * x["acc"] for x in r], "o-", c="#2a78d6", ms=3, label="todas las clases")
    ax.plot(ms, [100 * x["acc_gestos"] for x in r], "s--", c="#eb6834", ms=3, mfc="none",
            label="solo gestos (1–4)")
    ax.axvline(0, c="k", lw=0.6, ls=":")
    ax.set_xlabel("Desfase de la ventana respecto al gesto [s]  (− antes, + después)")
    ax.set_ylabel("Exactitud prueba [%]")
    ax.set_ylim(min(80, ax.get_ylim()[0]), 101)
    ax.legend(loc="lower center", ncol=2)
    save(fig, "fig_desplazamiento")


def fig_lazo(sim):
    s = sim["escenario"]; t = np.array(s["t"])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(2 * COL, 3.0), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 0.8, 1.3]})
    for i in range(3):
        a1.plot(t, s[f"lv{i + 1}"], c=POTES[i], label=f"pot {i + 1}")
    for t0, t1, c in sim["eventos"]:
        for ax in (a1, a2, a3):
            ax.axvspan(t0, t1, color=COLORES[int(c)], alpha=0.10, lw=0)
        a1.text((t0 + t1) / 2, 1.12, ETQ[int(c)], ha="center", fontsize=6)
    a1.set_ylabel("Nivel"); a1.set_ylim(-0.05, 1.25)
    a1.legend(loc="center left", ncol=3, bbox_to_anchor=(0.0, 0.62), fontsize=6)
    a2.step(t, s["cls_raw"], where="post", c="0.6", lw=0.8, label="salida de la red")
    a2.step(t, s["cls"], where="post", c="k", label="clase activa")
    a2.set_yticks(range(core.N_OUT)); a2.set_yticklabels(ETQ_CORTA, fontsize=6)
    a2.legend(loc="upper left", ncol=2, fontsize=6)
    a3.plot(t, s["ref"], "--", c="k", lw=0.9, label="referencia")
    a3.plot(t, s["rpm"], c="#2a78d6", label="velocidad medida")
    a3.set_ylabel("rpm"); a3.set_xlabel("Tiempo [s]")
    a3.legend(loc="lower left", ncol=2)
    fig.tight_layout(h_pad=0.2)
    save(fig, "fig_lazo")


def fig_control(sim, tlm_escalon=None):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(2 * COL, 1.7),
                                     gridspec_kw={"width_ratios": [1.2, 1, 1]})
    e = sim["escalon"]; t = np.array(e["t"])
    a1.plot(t - 0.2, e["ref"], "--", c="k", lw=0.9, label="referencia")
    a1.plot(t - 0.2, e["rpm"], c="#2a78d6", label="simulado")
    if tlm_escalon is not None:
        import csv
        with open(tlm_escalon) as f:
            rows = list(csv.DictReader(f))
        tr = np.array([float(r["t_ms"]) for r in rows]) / 1e3
        a1.plot(tr - tr[0], [float(r["rpm"]) for r in rows], c="#eb6834", label="S32K312")
    a1.set_xlim(-0.1, 1.0); a1.set_xlabel("Tiempo desde el escalón [s]"); a1.set_ylabel("rpm")
    a1.legend(loc="lower right")

    b = sim["barrido_tcl"]; tcl = [x["t_cl"] for x in b]
    a2.plot(tcl, [x["t_establecimiento_s"] for x in b], "o-", c="#2a78d6", ms=3, label=r"$t_s$ (2 %)")
    a2.plot(tcl, [x["t_subida_s"] for x in b], "s--", c="#eb6834", ms=3, mfc="none", label=r"$t_r$ (10–90 %)")
    a2.set_xlabel(r"$T_{cl}$ de diseño [s]"); a2.set_ylabel("Tiempo [s]")
    a2.legend(loc="upper left")
    a3.plot(tcl, [x["u_pico"] for x in b], "^-", c="#1baf7a", ms=3)
    a3.set_ylim(0, 1.1); a3.set_xlabel(r"$T_{cl}$ de diseño [s]"); a3.set_ylabel(r"$|u|_{máx}$ (duty)")
    fig.tight_layout()
    save(fig, "fig_control")


# ---------------------------------------------------------------------
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="figuras")
    OUT = ap.parse_args().salida
    os.makedirs(OUT, exist_ok=True)

    pc = load_json("resultados_pc.json"); pc32 = load_json("resultados_pc_f32.json")
    host, mcu = cargar("resultados_host.json"), cargar("resultados_mcu.json")
    if host and mcu and host.get("config") == mcu.get("config"):
        host = None                                   # mismo archivo copiado: no duplicar
    ev, sim = cargar("resultados_eval.json"), cargar("resultados_sim.json")
    tlm = sorted(glob.glob("tlm_escalon_*.csv"))

    print(f"Generando figuras en {OUT}/  (MCU: {nombre_origen(mcu, 'no') if mcu else 'no'}, "
          f"real: {'sí' if ev and 'real' in ev else 'no'})")
    fig_dataset(pc)
    fig_convergencia(pc, pc32, host, mcu)
    fig_confusion(pc, host, mcu, ev)
    fig_desplazamiento(pc)
    if sim:
        fig_lazo(sim)
        fig_control(sim, tlm[-1] if tlm else None)


if __name__ == "__main__":
    main()
