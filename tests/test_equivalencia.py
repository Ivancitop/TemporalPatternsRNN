"""
test_equivalencia.py
====================
Pruebas en PC del codigo C del firmware (seq.c, lstm.c, app_cmd.c) contra el
modelo de referencia en Python, usando el gemelo tests/host_mcu.

  1) Secuencia: series crudas identicas (comando A) -> los codigos de la
     ventana (comando C) deben ser IGUALES a lstm_core.raw_series_to_codes.
  2) Inferencia incremental (comando U, la que corre en RUN repartida en
     ticks) contra el forward de Python sobre la misma ventana.
  3) Entrenamiento: N pasos de SGD+BPTT por el protocolo (Q/T) contra
     LSTMNet(float32), muestra a muestra, y pesos finales (W).
  4) Carga de pesos: L seguido de W debe devolver los mismos bits.

Uso (desde tests/):
    gcc -O2 -std=c99 -ffp-contract=off -I../firmware host_mcu.c ../firmware/lstm.c \
        ../firmware/seq.c ../firmware/app_cmd.c ../firmware/pi_ctrl.c -lm -o host_mcu
    python test_equivalencia.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import lstm_core as core  # noqa: E402
import enlace  # noqa: E402
from run_pc import preparar, SEED_WEIGHTS, LR, CLIP  # noqa: E402

DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python",
                     "dataset_balanceado_500.csv")


def test_secuencia(n=30, seed=11):
    r = np.random.default_rng(seed)
    net = core.LSTMNet(seed=SEED_WEIGHTS, dtype=np.float32)
    err_code, err_p = 0, 0.0
    for _ in range(n):
        link = enlace.HostLink()             # estado de seq limpio en cada serie
        ticks = core.SEQ_DEC * (core.T + int(r.integers(0, 8))) + int(r.integers(0, core.SEQ_DEC))
        lv = r.random(3) * 1.1
        tr = np.linspace(0, 1, ticks)[:, None] * (r.random(3) - 0.5)       # rampa
        raw = np.clip(np.rint((lv + tr) * core.ADC_FS + r.normal(0, 25, (ticks, 3))), 0, 4095).astype(int)
        for a, b, c in raw:
            link.send(f"A,{a},{b},{c}")
        resp = link.line("C").split(",")
        codes_c = core.hex_to_codes(resp[4])
        codes_py = core.raw_series_to_codes(raw)[-core.T:]
        err_code = max(err_code, int(np.max(np.abs(codes_c - codes_py))))
        u = link.line("U").split(",")
        p_c = np.array([core.hex_to_f32(h) for h in u[1:]])
        p_py, _ = net.forward(core.codes_to_x(codes_py, np.float32))
        err_p = max(err_p, float(np.max(np.abs(p_c - p_py))))
        link.close()
    return err_code, err_p


def test_entrenamiento(n=200):
    d = preparar(DATOS)
    link = enlace.HostLink()
    assert link.line("R") == "OK"
    enlace.fijar_lr(link, LR, CLIP)
    net = core.LSTMNet(seed=SEED_WEIGHTS, dtype=np.float32)
    err_p, err_l = 0.0, 0.0
    for C, y in zip(d["C_train"][:n], d["y_train"][:n]):
        _, p_c, l_c, _, _, _ = enlace.muestra(link, "T", C, y)
        p_py, l_py = net.train_sample(core.codes_to_x(C, np.float32), y, LR, CLIP)
        err_p = max(err_p, float(np.max(np.abs(np.array(p_c) - p_py))))
        err_l = max(err_l, abs(l_c - l_py))
    w_c = enlace.volcar_pesos(link)
    err_w = float(np.max(np.abs(w_c - net.flat_params())))
    link.close()
    return err_p, err_l, err_w


def test_carga_pesos(seed=5):
    r = np.random.default_rng(seed)
    flat = r.normal(0, 1, core.N_PARAMS).astype(np.float32)
    link = enlace.HostLink()
    enlace.escribir_pesos(link, flat)
    back = enlace.volcar_pesos(link)
    link.close()
    return bool(np.array_equal(back.view(np.uint32), flat.view(np.uint32)))


if __name__ == "__main__":
    ec, ep = test_secuencia()
    print(f"Secuencia  C vs Python            : error max codigo = {ec} (debe ser 0)")
    print(f"Inferencia incremental vs forward : error max en p = {ep:.2e}")
    e_p, e_l, e_w = test_entrenamiento()
    print(f"Entrenamiento (200 pasos SGD+BPTT): error max p = {e_p:.2e}, perdida = {e_l:.2e}, "
          f"pesos = {e_w:.2e}")
    ok_l = test_carga_pesos()
    print(f"Carga L + volcado W bit a bit     : {'iguales' if ok_l else 'DISTINTOS'}")
    ok = (ec == 0) and ep < 1e-5 and e_p < 1e-4 and e_w < 1e-4 and ok_l
    print("OK" if ok else "FALLO")
    sys.exit(0 if ok else 1)
