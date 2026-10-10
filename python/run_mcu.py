"""
run_mcu.py
==========
Entrena la LSTM EN EL MICROCONTROLADOR con exactamente el mismo dataset, el
mismo orden de secuencias y los mismos pesos iniciales que run_pc.py.

Por cada secuencia de entrenamiento:
    Q,0,<25 pasos>  -> K
    Q,25,<25 pasos> -> K        (los niveles viajan como enteros en 3 hex)
    T,<clase>       -> <cls>,<p0..p4>,<loss>,<cicF>,<cicB>   (floats en hex IEEE-754)
Asi cualquier diferencia PC-MCU es de la aritmetica, no del formato.

Al terminar:
  - evalua validacion y prueba en modo 'V' (forward sin actualizar),
  - descarga los pesos entrenados ('W') y los guarda en
        pesos_mcu.json                  (para cargar_pesos.py tras un reinicio)
        ../firmware/pesos_entrenados.h  (para compilar con USE_PRETRAINED)
  - guarda resultados_mcu.json.

Uso:
    python run_mcu.py --puerto COM6          # hardware real
    python run_mcu.py --host                 # gemelo en C (tests/host_mcu)
    python run_mcu.py --simular              # emulacion float32 en NumPy
Opciones: --epocas N, --val-cada N (validar cada N epocas), --datos archivo.csv
"""

import argparse
import os
import sys
import time

import numpy as np

import lstm_core as core
import enlace
from lstm_core import (LSTMNet, epoch_orders, save_json, f32_to_hex, matriz_confusion,
                       metricas_clase, con_umbral)
from run_pc import EPOCHS, LR, CLIP, SEED_WEIGHTS, SEED_ORDER, DATOS, FIRMWARE_DIR, preparar

CORE_MHZ = 120.0


def evaluar(link, C, Y):
    P, loss, cf = [], 0.0, []
    for c, y in zip(C, Y):
        _, p, l, c_f, _, _ = enlace.muestra(link, "V", c, y)
        P.append(p); loss += l; cf.append(c_f)
    P = np.array(P)
    return loss / len(C), 100.0 * float(np.mean(P.argmax(1) == Y)), P, cf


def guardar_pesos(flat, origen, extra=None):
    """pesos_mcu.json (hex exacto + float) y firmware/pesos_entrenados.h"""
    flat = np.asarray(flat, dtype=np.float32)
    info = {"origen": origen, "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_params": int(len(flat)), "topologia": [core.N_IN, core.N_H, core.N_D, core.N_OUT],
            "T": core.T, "step_ms": core.STEP_MS,
            "hex": [f32_to_hex(v) for v in flat], "flat": flat.tolist()}
    if extra:
        info.update(extra)
    save_json("pesos_mcu.json", info)
    net = LSTMNet(seed=SEED_WEIGHTS, dtype=np.float32)
    net.set_flat_params(flat)
    os.makedirs(FIRMWARE_DIR, exist_ok=True)
    with open(os.path.join(FIRMWARE_DIR, "pesos_entrenados.h"), "w", encoding="utf-8") as f:
        f.write(net.export_c("TRAINED"))
    return net


def main():
    ap = argparse.ArgumentParser()
    enlace.agregar_argumentos(ap)
    ap.add_argument("--epocas", type=int, default=EPOCHS)
    ap.add_argument("--val-cada", type=int, default=1, help="validar cada N epocas (0 = solo al final)")
    ap.add_argument("--datos", default=DATOS)
    args = ap.parse_args()

    d = preparar(args.datos)
    orders = epoch_orders(len(d["C_train"]), EPOCHS, SEED_ORDER)[:args.epocas]
    Ctr, Ytr = d["C_train"], d["y_train"]

    n_tx = 2 * (len("Q,25,") + 9 * core.SEQ_CHUNK + 1) + 4
    n_rx = 2 * 2 + (2 + 6 * 9 + 16)
    t_link = (n_tx + n_rx) * 10.0 / args.baud
    n_val = len(d["C_val"]) * (args.epocas if args.val_cada == 1 else
                               (args.epocas // max(args.val_cada, 1) + 1))
    print(f"Dataset: {len(Ctr)} entren. + {len(d['C_val'])} valid. (85 %) / "
          f"{len(d['C_test'])} prueba (15 %)")
    print(f"Enlace estimado: {1e3 * t_link:.0f} ms por secuencia -> "
          f"~{t_link * (args.epocas * len(Ctr) + n_val) / 60:.1f} min (sin contar el computo)\n")

    link, origen = enlace.abrir(args)
    hist = {"loss_train": [], "loss_val": [], "acc_val": [], "epoca_val": [], "prob_log": [],
            "loss_log": [], "rtt_log": [], "cyc_fwd": [], "cyc_bwd": []}
    try:
        assert link.line("R") == "OK", "el MCU no respondio a R"
        enlace.fijar_lr(link, LR, CLIP)
        t0 = time.perf_counter()
        for ep, order in enumerate(orders):
            acc = 0.0
            for idx in order:
                _, p, loss, cf, cb, rtt = enlace.muestra(link, "T", Ctr[idx], Ytr[idx])
                acc += loss
                hist["prob_log"].append(p); hist["loss_log"].append(loss)
                hist["rtt_log"].append(rtt); hist["cyc_fwd"].append(cf); hist["cyc_bwd"].append(cb)
            hist["loss_train"].append(acc / len(order))
            msg = f"  epoca {ep:3d}  loss_tr={hist['loss_train'][-1]:.5f}"
            ultima = ep == len(orders) - 1
            if (args.val_cada > 0 and (ep + 1) % args.val_cada == 0) or ultima:
                lv, av, _, _ = evaluar(link, d["C_val"], d["y_val"])
                hist["loss_val"].append(lv); hist["acc_val"].append(av); hist["epoca_val"].append(ep)
                msg += f"  loss_val={lv:.5f}  acc_val={av:5.1f}%"
            print(msg + f"  ({time.perf_counter() - t0:.0f} s)")
        hist["wall_time_s"] = time.perf_counter() - t0

        _, acc_te, Pte, _ = evaluar(link, d["C_test"], d["y_test"])
        M = matriz_confusion(d["y_test"], Pte.argmax(1))
        Mu = matriz_confusion(d["y_test"], con_umbral(Pte))
        hist.update({"acc_test": acc_te, "cm_test": M.tolist(), "metricas_test": metricas_clase(M),
                     "cm_test_umbral": Mu.tolist(), "prob_test": Pte.tolist()})
        print(f"\nExactitud en prueba: {acc_te:.1f} %  (con umbral 0.7: "
              f"{100 * metricas_clase(Mu)['accuracy']:.1f} %)\n{M}")

        # Retroalimentacion de pesos: el MCU los vuelca y la PC los guarda
        flat = enlace.volcar_pesos(link)
        net = guardar_pesos(flat, origen, {"acc_test": acc_te})
        hist["weights_final"] = net.get_weights()
        print("Pesos entrenados -> pesos_mcu.json y firmware/pesos_entrenados.h")
    finally:
        link.close()

    rtt = np.array(hist["rtt_log"])
    hist["rtt_mean_ms"] = float(rtt.mean() * 1e3)
    cf = np.array(hist["cyc_fwd"]); cb = np.array(hist["cyc_bwd"])
    if cf.any():
        hist["cyc_fwd_mean"] = float(cf.mean()); hist["cyc_bwd_mean"] = float(cb.mean())
        hist["cyc_fwd_max"] = int(cf.max()); hist["cyc_bwd_max"] = int(cb.max())
        hist["fwd_mean_us"] = hist["cyc_fwd_mean"] / CORE_MHZ
        hist["bwd_mean_us"] = hist["cyc_bwd_mean"] / CORE_MHZ
        print(f"Forward: {hist['fwd_mean_us']:.0f} us   BPTT+actualizacion: {hist['bwd_mean_us']:.0f} us "
              f"(media, {CORE_MHZ:.0f} MHz{'; medido en la PC' if origen == 'host_c' else ''})")
    hist["config"] = {"epochs": len(orders), "lr": LR, "clip": CLIP, "baud": args.baud,
                      "origen": origen, "core_mhz": CORE_MHZ}
    save_json("resultados_mcu.json", hist)
    print(f"Ida y vuelta: {hist['rtt_mean_ms']:.2f} ms/secuencia.  -> resultados_mcu.json")


if __name__ == "__main__":
    sys.exit(main())
