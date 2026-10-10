"""
cargar_pesos.py
===============
Restaura en el MCU los pesos que devolvio run_mcu.py, para no reentrenar
tras un reinicio (la RAM se pierde, el archivo pesos_mcu.json no).

  1) X  -> IDLE
  2) L,<idx>,<8 hex>  x 183 lineas (1461 parametros, ~1.5 s a 115200)
  3) W  -> verifica que el MCU tiene los mismos bits
  4) V  sobre algunas secuencias de prueba -> compara con el modelo en la PC
  5) --run: entra a RUN (I) para usar el clasificador con los potes

Uso:
    python cargar_pesos.py --puerto COM6 [--pesos pesos_mcu.json] [--run]
    python cargar_pesos.py --host          # contra el gemelo en C
"""

import argparse
import sys

import numpy as np

import lstm_core as core
import enlace
from lstm_core import LSTMNet, load_json, hex_to_f32
from run_pc import DATOS, preparar


def leer_pesos(path):
    w = load_json(path)
    if "hex" in w:                                   # pesos_mcu.json
        return np.array([hex_to_f32(h) for h in w["hex"]], dtype=np.float32)
    net = LSTMNet(dtype=np.float32)                  # resultados_*.json
    net.set_weights(w["weights_final"])
    return net.flat_params().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    enlace.agregar_argumentos(ap)
    ap.add_argument("--pesos", default="pesos_mcu.json")
    ap.add_argument("--datos", default=DATOS)
    ap.add_argument("--n-check", type=int, default=10)
    ap.add_argument("--run", action="store_true", help="entrar a RUN al terminar")
    args = ap.parse_args()

    flat = leer_pesos(args.pesos)
    print(f"{len(flat)} parametros leidos de {args.pesos}")
    link, origen = enlace.abrir(args)
    try:
        assert link.line("X") == "OK", "el MCU no paso a IDLE"
        enlace.escribir_pesos(link, flat)
        back = enlace.volcar_pesos(link)
        iguales = np.array_equal(back.view(np.uint32), flat.view(np.uint32))
        print(f"Readback W: {'identico bit a bit' if iguales else 'DIFERENTE'}")
        if not iguales:
            return 1

        d = preparar(args.datos)
        net = LSTMNet(dtype=np.float32); net.set_flat_params(flat)
        n = min(args.n_check, len(d["C_test"]))
        dif, acuerdo = 0.0, 0
        for C, y in zip(d["C_test"][:n], d["y_test"][:n]):
            k, p, _, _, _, _ = enlace.muestra(link, "V", C, y)
            p_pc, _ = net.forward(core.codes_to_x(C, np.float32))
            dif = max(dif, float(np.max(np.abs(np.array(p) - p_pc))))
            acuerdo += int(k == int(np.argmax(p_pc)))
        print(f"Chequeo V en {n} secuencias de prueba: acuerdo {acuerdo}/{n}, max|dp| = {dif:.2e}")
        if args.run:
            print("RUN:", link.line("I"))
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
