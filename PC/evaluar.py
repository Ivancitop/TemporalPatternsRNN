"""
evaluar.py
==========
Clasificacion PC vs MCU con los mismos pesos y las mismas secuencias.

  - PC : LSTM en float64 con los pesos que entreno el MCU (pesos_mcu.json)
  - MCU: la misma secuencia (codigos enteros exactos) enviada en modo 'V'

Como ambos usan los mismos pesos y las mismas entradas bit a bit, cualquier
discrepancia de clase se debe solo a la aritmetica float32 del MCU.

Conjuntos evaluados:
  prueba : el 15 % de prueba del dataset (mismo split que run_pc / run_mcu)
  real   : dataset_real.csv capturado con capturar.py, si existe
           (brecha entre el dataset de entrenamiento y los potes reales)

Uso:
    python evaluar.py --puerto COM6        # el MCU debe estar encendido (los pesos se cargan solos)
    python evaluar.py --host | --solo-pc
Salida:
    resultados_eval.json
"""

import argparse
import os

import numpy as np

import lstm_core as core
import enlace
from lstm_core import (LSTMNet, load_dataset, save_json, matriz_confusion, metricas_clase,
                       con_umbral)
from cargar_pesos import leer_pesos
from run_pc import DATOS, preparar


def evaluar_conjunto(nombre, C, y, net, link):
    P_pc = net.predict_proba(core.codes_to_x(C, np.float64))
    yp_pc = con_umbral(P_pc)                        # mismo umbral que el firmware
    out = {"n": int(len(y)), "cm_pc": matriz_confusion(y, yp_pc).tolist(),
           "metricas_pc": metricas_clase(matriz_confusion(y, yp_pc)), "prob_pc": P_pc.tolist()}
    print(f"[{nombre}] PC  : exactitud {100 * out['metricas_pc']['accuracy']:.1f} %  ({len(y)} secuencias)")
    if link is not None:
        P_mcu = np.array([enlace.muestra(link, "V", c, yy)[1] for c, yy in zip(C, y)])
        yp_mcu = con_umbral(P_mcu)
        M = matriz_confusion(y, yp_mcu)
        out.update({"cm_mcu": M.tolist(), "metricas_mcu": metricas_clase(M), "prob_mcu": P_mcu.tolist(),
                    "acuerdo_pc_mcu": float(np.mean(yp_mcu == yp_pc)),
                    "max_dif_prob": float(np.max(np.abs(P_mcu - P_pc)))})
        print(f"[{nombre}] MCU : exactitud {100 * out['metricas_mcu']['accuracy']:.1f} %   "
              f"acuerdo PC-MCU {100 * out['acuerdo_pc_mcu']:.2f} %   max|dp| {out['max_dif_prob']:.2e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    enlace.agregar_argumentos(ap)
    ap.add_argument("--solo-pc", action="store_true")
    ap.add_argument("--pesos", default="pesos_mcu.json")
    ap.add_argument("--datos", default=DATOS)
    ap.add_argument("--real", default="dataset_real.csv")
    args = ap.parse_args()

    fuente = args.pesos if os.path.exists(args.pesos) else "resultados_pc.json"
    flat = leer_pesos(fuente)
    net = LSTMNet(dtype=np.float64); net.set_flat_params(flat)
    print(f"Pesos: {fuente}")

    link = None
    if not args.solo_pc:
        link, _ = enlace.abrir(args)
        enlace.escribir_pesos(link, flat)           # garantiza que el MCU usa estos pesos

    d = preparar(args.datos)
    out = {"pesos": fuente,
           "prueba": evaluar_conjunto("prueba", d["C_test"], d["y_test"], net, link)}
    if os.path.exists(args.real):
        Cr, yr, _ = load_dataset(args.real)
        out["real"] = evaluar_conjunto("real", Cr, yr, net, link)
    else:
        print(f"(sin {args.real}: captura datos reales con capturar.py)")
    if link is not None:
        link.close()
    save_json("resultados_eval.json", out)
    print("-> resultados_eval.json")


if __name__ == "__main__":
    main()
