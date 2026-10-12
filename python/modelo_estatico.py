"""
modelo_estatico.py
==================
Modelo de referencia SIN memoria temporal: el MLP 4 -> 2 -> 3 -> 5 de MLP.py
(media, RMS, maximo y varianza de la ventana, sigmoides parametricas, softmax +
entropia cruzada), entrenado con el MISMO split 70/15/15 que la LSTM para que la
comparacion sea directa.

Diferencia con MLP.run(): las caracteristicas se calculan sobre la secuencia de
50 pasos de 100 ms (la misma que ve la LSTM), no sobre las 250 muestras de 20 ms,
para poder aplicar exactamente las mismas perturbaciones en robustez.py.

Propiedad clave: media, RMS, maximo y varianza de v = [p1(t), p2(t), p3(t)] no
cambian si se permutan las muestras en el tiempo (ni entre canales). El modelo no
puede distinguir un gesto de su version invertida en el tiempo.

Uso:
    python modelo_estatico.py           -> resultados_mlp.json (+ resumen en consola)
"""

import time

import numpy as np

import lstm_core as core
from lstm_core import save_json, matriz_confusion, metricas_clase
from MLP import MLP
from run_pc import preparar, DATOS

H1, H2, EPOCAS, LR_MLP, BATCH, C_SIG, SEMILLA = 2, 3, 200, 0.5, 32, 1.0, 0


def caracteristicas(C):
    """(N, T, 3) codigos -> (N, 4): media, RMS, maximo, varianza de los 3 potes juntos."""
    V = np.asarray(C, float).reshape(len(C), -1) / core.CODE_SCALE
    return np.stack([V.mean(1), np.sqrt((V ** 2).mean(1)), V.max(1), V.var(1)], 1)


class ModeloEstatico:
    def __init__(self, d, epocas=EPOCAS, semilla=SEMILLA, verbose=False):
        Xtr = caracteristicas(d["C_train"])
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        self.net = MLP(H1, H2, {"a": 1.0, "b": 1.0, "c": C_SIG, "d": 0.0}, semilla)
        t0 = time.perf_counter()
        self.hist = self.net.fit(self.norm(Xtr), d["y_train"], self.norm(caracteristicas(d["C_val"])),
                                 d["y_val"], epocas, LR_MLP, BATCH, semilla, verbose)
        self.t_entrenamiento_s = time.perf_counter() - t0
        self.n_params = int(sum(w.size for w in self.net.W) + sum(b.size for b in self.net.b))

    def norm(self, X):
        return (X - self.mu) / self.sd

    def proba(self, C):
        return self.net.forward(self.norm(caracteristicas(C)))[0]

    def predecir(self, C):
        return self.proba(C).argmax(1)


def main():
    d = preparar(DATOS)
    m = ModeloEstatico(d, verbose=True)
    yp = m.predecir(d["C_test"])
    M = matriz_confusion(d["y_test"], yp)
    met = metricas_clase(M)
    out = {"modelo": f"MLP 4-{H1}-{H2}-5 (media, RMS, max, var)", "n_params": m.n_params,
           "t_entrenamiento_s": m.t_entrenamiento_s, "epocas": EPOCAS, "lr": LR_MLP, "batch": BATCH,
           "acc_test": 100 * met["accuracy"], "cm_test": M.tolist(), "metricas_test": met,
           "hist": {k: [float(v) for v in vs] for k, vs in m.hist.items()}}
    # Invariancia temporal: la ventana invertida en el tiempo da EXACTAMENTE las mismas entradas
    Cr = d["C_test"][:, ::-1]
    out["dif_max_caracteristicas_invertida"] = float(np.abs(caracteristicas(Cr) - caracteristicas(d["C_test"])).max())
    save_json("resultados_mlp.json", out)
    print(f"\nMLP estatico: {m.n_params} parametros, prueba {out['acc_test']:.1f} %, "
          f"F1 macro {met['macro_f1']:.3f}, {m.t_entrenamiento_s:.1f} s\n{M}")
    print(f"Diferencia maxima de caracteristicas con la ventana invertida en el tiempo: "
          f"{out['dif_max_caracteristicas_invertida']:.2e}  -> resultados_mlp.json")


if __name__ == "__main__":
    main()
