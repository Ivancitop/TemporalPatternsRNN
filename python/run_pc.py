"""
run_pc.py
=========
Entrena el modelo de referencia (LSTM) en la PC y deja todo listo para
comparar contra el microcontrolador.

Genera (en la carpeta actual):
  dataset_secuencias.csv      dataset diezmado a 50 pasos (codigos) con su particion
  resultados_pc.json          corrida principal float64 (referencia)
  resultados_pc_f32.json      misma corrida en float32 (efecto de la precision)
  ../firmware/pesos_iniciales.h

Uso:
    python run_pc.py [--datos dataset_balanceado_500.csv]
"""

import argparse
import csv
import os
import time

import numpy as np

import lstm_core as core
from lstm_core import (LSTMNet, load_dataset, split_dataset, epoch_orders, save_json, desplazar,
                       matriz_confusion, metricas_clase, con_umbral, gradient_check)

# ---------------- configuracion del experimento ----------------
# run_mcu.py importa estas constantes: PC y MCU usan exactamente lo mismo
EPOCHS = 60
AUMENTO = True          # desfase / velocidad / amplitud por epoca (ver lstm_core.aumentar)
LR = 0.10
CLIP = 1.0
SEED_WEIGHTS = 1
SEED_DATA = 0
SEED_ORDER = 7
DATOS = "dataset/dataset_balanceado_500.csv"
FIRMWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware")


def preparar(datos=DATOS):
    codes, y, ids = load_dataset(datos)
    return split_dataset(codes, y, ids, SEED_DATA)


def train(net, d, orders, lr, clip=CLIP, log=True, eval_test=False):
    """SGD por secuencia + BPTT. Registra por epoca:
      loss_train      SSE media de la epoca (cada muestra evaluada ANTES de actualizar)
      acc_train_linea exactitud en linea de esa misma pasada (igual a la del MCU)
      acc_train       exactitud de una pasada completa sobre el train (aumentado) al final de la epoca
      loss_val/acc_val, y acc_test por epoca si eval_test (estudio del numero de epocas)
      t_epoca_s       tiempo de SGD de la epoca (sin evaluaciones)
    wall_time_s = tiempo total con evaluaciones; t_entrenamiento_s = solo SGD."""
    hist = {"loss_train": [], "loss_val": [], "acc_train": [], "acc_val": [],
            "acc_train_linea": [], "acc_test": [], "t_epoca_s": [],
            "prob_log": [], "loss_log": []}
    Xtr, Ytr = d["X_train"], d["y_train"]
    t0 = time.perf_counter()
    for ep, order in enumerate(orders):
        acc = 0.0
        aciertos = 0
        if AUMENTO:
            Ce, Ytr = core.epoca_aumentada(d, ep)
            Xtr = core.codes_to_x(Ce, np.float64)
        t_ep = time.perf_counter()
        for idx in order:
            p, loss = net.train_sample(Xtr[idx], Ytr[idx], lr, clip)
            acc += loss
            aciertos += int(np.argmax(p) == Ytr[idx])
            if log:
                hist["prob_log"].append([float(v) for v in p])
                hist["loss_log"].append(loss)
        hist["t_epoca_s"].append(time.perf_counter() - t_ep)
        hist["loss_train"].append(acc / len(order))
        hist["acc_train_linea"].append(100.0 * aciertos / len(order))
        lv, av, _ = net.evaluate(d["X_val"], d["y_val"])
        _, at, _ = net.evaluate(Xtr, Ytr)
        hist["loss_val"].append(lv); hist["acc_val"].append(av); hist["acc_train"].append(at)
        if eval_test:
            hist["acc_test"].append(net.evaluate(d["X_test"], d["y_test"])[1])
        if log:
            print(f"  epoca {ep:3d}  loss_tr={hist['loss_train'][-1]:.5f}  acc_tr={at:5.1f}%  "
                  f"loss_val={lv:.5f}  acc_val={av:5.1f}%")
    hist["wall_time_s"] = time.perf_counter() - t0
    hist["t_entrenamiento_s"] = float(np.sum(hist["t_epoca_s"]))
    return hist


def robustez_desplazamiento(net, d, rango=range(-10, 21)):
    """Exactitud en prueba (con umbral 0.7) cuando la ventana de RUN no esta
    alineada con el gesto. Dice cuantas clasificaciones seguidas (antirrebote)
    puede esperar el firmware."""
    out = []
    for s in rango:
        C = np.array([desplazar(c, s) for c in d["C_test"]])
        P = net.predict_proba(core.codes_to_x(C, np.float64))
        yp = con_umbral(P, codes=C)
        out.append({"desplazamiento": s, "ms": s * core.STEP_MS,
                    "acc": float(np.mean(yp == d["y_test"])),
                    "acc_gestos": float(np.mean(yp[d["y_test"] != 0] == d["y_test"][d["y_test"] != 0])),
                    "acc_clase": [float(np.mean(yp[d["y_test"] == c] == c)) for c in range(core.N_OUT)]})
    return out


def quietos_default(net, d):
    """Potes quietos 5 s en el estado inicial o final de un gesto deben dar default."""
    ok = tot = 0
    for g in d["C_test"][d["y_test"] != 0]:
        for row in (g[0], g[-1]):
            P = net.predict_proba(core.codes_to_x(np.repeat(row[None], core.T, 0)[None], np.float64))
            ok += int(con_umbral(P)[0] == 0); tot += 1          # sin compuerta: lo decide la red
    return f"{ok}/{tot}"


def time_forward(net, X, reps=3):
    t0 = time.perf_counter()
    for _ in range(reps):
        for x in X:
            net.forward(x)
    return 1e6 * (time.perf_counter() - t0) / (reps * len(X))


def run(dtype, tag, d, orders):
    net = LSTMNet(seed=SEED_WEIGHTS, dtype=dtype)
    w0 = net.get_weights()
    print(f"\n[{tag}] dtype={np.dtype(dtype).name}  LSTM({core.N_IN}->{core.N_H})"
          f"-Dense({core.N_D})-Dense({core.N_OUT})  {net.n_params()} parametros")
    hist = train(net, d, orders, LR, eval_test=True)
    _, acc_te, Pte = net.evaluate(d["X_test"], d["y_test"])
    M = matriz_confusion(d["y_test"], Pte.argmax(1))
    Mu = matriz_confusion(d["y_test"], con_umbral(Pte, codes=d["C_test"]))
    hist.update({
        "acc_test": acc_te, "cm_test": M.tolist(), "metricas_test": metricas_clase(M),
        "cm_test_umbral": Mu.tolist(), "prob_test": Pte.tolist(),
        "time_fwd_us": time_forward(net, d["X_test"][:10]),
        "weights_init": w0, "weights_final": net.get_weights(),
        "config": {"epochs": EPOCHS, "lr": LR, "clip": CLIP, "dtype": np.dtype(dtype).name,
                   "seed_weights": SEED_WEIGHTS, "seed_data": SEED_DATA,
                   "seed_order": SEED_ORDER, "n_params": net.n_params(),
                   "T": core.T, "step_ms": core.STEP_MS, "n_h": core.N_H, "n_d": core.N_D,
                   "aumento": AUMENTO},
        "desplazamiento": robustez_desplazamiento(net, d),
        "quietos_default": quietos_default(net, d),
    })
    print(f"  prueba: {acc_te:.1f} %   (umbral {core.UMBRAL} + compuerta: {100 * metricas_clase(Mu)['accuracy']:.1f} %)")
    print(f"  parametros: {net.n_params()}   tiempo de entrenamiento: {hist['t_entrenamiento_s']:.1f} s "
          f"(SGD) / {hist['wall_time_s']:.1f} s (con evaluaciones)")
    print(f"  final: loss_tr {hist['loss_train'][-1]:.4f}  loss_val {hist['loss_val'][-1]:.4f}  "
          f"acc_tr {hist['acc_train'][-1]:.1f} %  acc_val {hist['acc_val'][-1]:.1f} %")
    return net, hist


def guardar_csv(d, path="dataset_secuencias.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "particion", "clase", "clase_enunciado", "nombre_clase", "paso",
                    "t_ms", "code1", "code2", "code3"])
        for part in ("train", "val", "test"):
            for sid, y, C in zip(d[f"id_{part}"], d[f"y_{part}"], d[f"C_{part}"]):
                for k, (a, b, c) in enumerate(C):
                    w.writerow([sid, part, int(y), core.CLASE_ENUNCIADO[int(y)],
                                core.CLASES[int(y)], k, k * core.STEP_MS, a, b, c])
    print(f"Dataset diezmado: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=DATOS)
    args = ap.parse_args()

    print("Gradient check (BPTT analitico vs diferencias centradas)")
    gc = gradient_check()
    print(f"  error relativo maximo = {gc:.3e}")

    d = preparar(args.datos)
    print(f"Secuencias: {len(d['y_train'])} entrenamiento + {len(d['y_val'])} validacion "
          f"(85 %) / {len(d['y_test'])} prueba (15 %)")
    guardar_csv(d)
    orders = epoch_orders(len(d["X_train"]), EPOCHS, SEED_ORDER)

    net64, h64 = run(np.float64, "PC float64", d, orders)
    h64["gradient_check"] = gc
    h64["split"] = {k: d[k] for k in d}
    h64["orders"] = orders
    save_json("resultados_pc.json", h64)

    _, h32 = run(np.float32, "PC float32", d, orders)
    save_json("resultados_pc_f32.json", h32)

    print("\nRobustez al desfase de la ventana (float64, umbral 0.7 + compuerta), exactitud por clase:")
    print("   desfase   " + "  ".join(f"{n[:11]:>11s}" for n in core.CLASES))
    for r in h64["desplazamiento"][::5]:
        print(f"  {r['ms']:+6d} ms  " + "  ".join(f"{100 * a:10.0f}%" for a in r["acc_clase"]))
    print("  Potes quietos al inicio/fin de los gestos de prueba -> default: "
          f"{h64['quietos_default']}")

    os.makedirs(FIRMWARE_DIR, exist_ok=True)
    path = os.path.join(FIRMWARE_DIR, "pesos_iniciales.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write(LSTMNet(seed=SEED_WEIGHTS).export_c("INIT"))
    print(f"\nPesos iniciales para el firmware: {path}")


if __name__ == "__main__":
    main()
