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
from lstm_core import (LSTMNet, load_dataset, split_dataset, epoch_orders, save_json,
                       matriz_confusion, metricas_clase, con_umbral, gradient_check)

# ---------------- configuracion del experimento ----------------
# run_mcu.py importa estas constantes: PC y MCU usan exactamente lo mismo
EPOCHS = 20
LR = 0.10
CLIP = 1.0
SEED_WEIGHTS = 1
SEED_DATA = 0
SEED_ORDER = 7
DATOS = "dataset\\dataset_balanceado_500.csv"
FIRMWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware")


def preparar(datos=DATOS):
    codes, y, ids = load_dataset(datos)
    return split_dataset(codes, y, ids, SEED_DATA)


def train(net, d, orders, lr, clip=CLIP, log=True):
    hist = {"loss_train": [], "loss_val": [], "acc_train": [], "acc_val": [],
            "prob_log": [], "loss_log": []}
    Xtr, Ytr = d["X_train"], d["y_train"]
    t0 = time.perf_counter()
    for ep, order in enumerate(orders):
        acc = 0.0
        for idx in order:
            p, loss = net.train_sample(Xtr[idx], Ytr[idx], lr, clip)
            acc += loss
            if log:
                hist["prob_log"].append([float(v) for v in p])
                hist["loss_log"].append(loss)
        hist["loss_train"].append(acc / len(order))
        lv, av, _ = net.evaluate(d["X_val"], d["y_val"])
        _, at, _ = net.evaluate(Xtr, Ytr)
        hist["loss_val"].append(lv); hist["acc_val"].append(av); hist["acc_train"].append(at)
        if log:
            print(f"  epoca {ep:3d}  loss_tr={hist['loss_train'][-1]:.5f}  "
                  f"loss_val={lv:.5f}  acc_val={av:5.1f}%")
    hist["wall_time_s"] = time.perf_counter() - t0
    return hist


def desplazar(codes, s):
    """Ventana deslizante: s > 0 -> la ventana termina s pasos DESPUES del gesto
    (los potes quedan quietos al final); s < 0 -> termina antes (quietos al inicio)."""
    if s == 0:
        return codes
    if s > 0:
        return np.concatenate([codes[s:], np.repeat(codes[-1:], s, axis=0)])
    s = -s
    return np.concatenate([np.repeat(codes[:1], s, axis=0), codes[:-s]])


def robustez_desplazamiento(net, d, rango=range(-10, 11)):
    """Exactitud en prueba (con umbral 0.7) cuando la ventana de RUN no esta
    alineada con el gesto. Dice cuantas clasificaciones seguidas (antirrebote)
    puede esperar el firmware."""
    out = []
    for s in rango:
        C = np.array([desplazar(c, s) for c in d["C_test"]])
        P = net.predict_proba(core.codes_to_x(C, np.float64))
        yp = con_umbral(P)
        out.append({"desplazamiento": s, "ms": s * core.STEP_MS,
                    "acc": float(np.mean(yp == d["y_test"])),
                    "acc_gestos": float(np.mean(yp[d["y_test"] != 0] == d["y_test"][d["y_test"] != 0]))})
    return out


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
    hist = train(net, d, orders, LR)
    _, acc_te, Pte = net.evaluate(d["X_test"], d["y_test"])
    M = matriz_confusion(d["y_test"], Pte.argmax(1))
    Mu = matriz_confusion(d["y_test"], con_umbral(Pte))
    hist.update({
        "acc_test": acc_te, "cm_test": M.tolist(), "metricas_test": metricas_clase(M),
        "cm_test_umbral": Mu.tolist(), "prob_test": Pte.tolist(),
        "time_fwd_us": time_forward(net, d["X_test"][:10]),
        "weights_init": w0, "weights_final": net.get_weights(),
        "config": {"epochs": EPOCHS, "lr": LR, "clip": CLIP, "dtype": np.dtype(dtype).name,
                   "seed_weights": SEED_WEIGHTS, "seed_data": SEED_DATA,
                   "seed_order": SEED_ORDER, "n_params": net.n_params(),
                   "T": core.T, "step_ms": core.STEP_MS, "n_h": core.N_H, "n_d": core.N_D},
        "desplazamiento": robustez_desplazamiento(net, d),
    })
    print(f"  prueba: {acc_te:.1f} %   (con umbral 0.7: {100 * metricas_clase(Mu)['accuracy']:.1f} %)")
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

    print("\nRobustez al desplazamiento de la ventana (float64, umbral 0.7):")
    print("  " + "  ".join(f"{r['desplazamiento']:+d}:{100 * r['acc']:.0f}%"
                           for r in h64["desplazamiento"]))

    os.makedirs(FIRMWARE_DIR, exist_ok=True)
    path = os.path.join(FIRMWARE_DIR, "pesos_iniciales.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write(LSTMNet(seed=SEED_WEIGHTS).export_c("INIT"))
    print(f"\nPesos iniciales para el firmware: {path}")


if __name__ == "__main__":
    main()
