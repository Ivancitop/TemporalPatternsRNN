"""
capturar.py
===========
Registro y anotacion de un dataset REAL de gestos con los potenciometros.

El firmware arma la secuencia con el mismo preprocesado que usa en RUN
(promedio de 100 ms, ventana de 50 pasos = 5 s) y la devuelve con el comando
'C' como codigos enteros. Por cada repeticion el script:
  1) pide colocar los potes en la posicion inicial del gesto,
  2) da la senal de inicio y marca el paso del tiempo durante 5 s,
  3) lee la ventana justo al terminar (contiene el gesto completo).

Se guarda en el MISMO formato largo que el dataset original
(sample_id, timestamp_ms, p1, p2, p3, label) pero a 100 ms, asi que
evaluar.py / lstm_core.load_dataset lo leen sin cambios.

Uso:
    python capturar.py --puerto COM6 --n 10
Salida:
    dataset_real.csv
"""

import argparse
import csv
import sys
import time

import lstm_core as core

# (clase, posicion inicial, gesto)
SESIONES = [
    (1, "los tres potes en 0", "sube pot1 al maximo, luego pot2, luego pot3 (todo en ~5 s)"),
    (2, "los tres potes en 0", "sube pot1 a la MITAD, luego pot2, luego pot3 (todo en ~5 s)"),
    (3, "los tres potes al maximo", "baja pot3 a 0, luego pot2, luego pot1 (todo en ~5 s)"),
    (4, "los tres potes al maximo", "baja pot3 a la MITAD, luego pot2, luego pot1 (todo en ~5 s)"),
    (0, "cualquier posicion (cambiala en cada repeticion)", "NO muevas los potes"),
]


def leer_ventana(ser):
    ser.reset_input_buffer()
    ser.write(b"C\n")
    t_end = time.time() + 2.0
    while time.time() < t_end:
        r = ser.readline().decode("ascii", errors="replace").strip()
        if r.startswith("C,"):
            p = r.split(",")
            return [int(v) for v in p[1:4]], core.hex_to_codes(p[4])
        if r == "N":
            return None, None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=10, help="repeticiones por clase")
    ap.add_argument("--salida", default="dataset_real.csv")
    args = ap.parse_args()

    import serial
    ser = serial.Serial(args.puerto, args.baud, timeout=0.5)
    time.sleep(0.5); ser.write(b"X\n"); time.sleep(0.1); ser.reset_input_buffer()
    dur = core.T * core.STEP_MS / 1000.0

    sid = 0
    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "timestamp_ms", "p1", "p2", "p3", "label"])
        for clase, inicio, gesto in SESIONES:
            nombre = f"clase {core.CLASE_ENUNCIADO[clase]} ({core.CLASES[clase]})"
            print(f"\n=== {nombre}: {args.n} repeticiones ===")
            k = 0
            while k < args.n:
                input(f"[{k + 1}/{args.n}] Coloca {inicio}. Enter para empezar...")
                print(f"  ¡YA! {gesto}")
                t0 = time.time()
                for s in range(int(dur)):
                    time.sleep(max(0.0, t0 + s + 1 - time.time()))
                    print(f"  {s + 1} s", end="\r", flush=True)
                time.sleep(max(0.0, t0 + dur + 0.1 - time.time()))
                raw, codes = leer_ventana(ser)
                if codes is None:
                    print("  ventana incompleta, repite"); continue
                sid += 1
                for t, (a, b, c) in enumerate(codes):
                    w.writerow([sid, t * core.STEP_MS, f"{a / 1000:.3f}", f"{b / 1000:.3f}",
                                f"{c / 1000:.3f}", clase])
                f.flush()
                print(f"  ok  crudo final={raw}  nivel final={(codes[-1] / 1000).round(2).tolist()}")
                k += 1
    ser.close()
    print(f"\nDataset real guardado en {args.salida} ({sid} secuencias)")


if __name__ == "__main__":
    sys.exit(main())
