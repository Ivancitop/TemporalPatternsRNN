"""
MLP 4 -> H1 -> H2 -> 5  (numpy puro)
Entradas : average, RMS, máximo, varianza (por ventana)
Ocultas  : sigmoide paramétrica f(x) = b / (a + e^(-c x)) + d   (a=1, b=1, d=0 -> salida en (0,1))
Salida   : softmax (5 clases), pérdida entropía cruzada
Clases   : 0 default | 1 alta CW | 2 baja CW | 3 alta CCW | 4 baja CCW
CSV      : id, timestamp_ms, val1, val2, val3, label   (id = identificador de ventana)
"""
import json
import argparse
import numpy as np
import pandas as pd

N_CLASSES = 5


# --------------------------------------------------------------------------
# Activaciones
# --------------------------------------------------------------------------
def sigmoid(params, x):
    x = np.clip(x, -500, 500)
    return params["b"] / (params["a"] + np.exp(-params["c"] * x)) + params["d"]


def sigmoidDiff(params, x):
    # d/dx [ b/(a+e^{-cx}) + d ] = b*c*e^{-cx} / (a+e^{-cx})^2
    x = np.clip(x, -500, 500)
    e = np.exp(-params["c"] * x)
    return params["b"] * params["c"] * e / (params["a"] + e) ** 2


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def extract_features(csv_path):
    """CSV -> DataFrame -> una fila por ventana con [average, rms, max, var] y label."""
    df = pd.read_csv(csv_path)
    rows = []
    for wid, g in df.groupby("sample_id"):
        v = g[["p1", "p2", "p3"]].to_numpy().ravel()   # señal de la ventana
        rows.append({
            "average": v.mean(),
            "rms": np.sqrt(np.mean(v ** 2)),
            "max": v.max(),
            "var": v.var(),
            "label": int(g["label"].iloc[0]),
        })
    feats = pd.DataFrame(rows)
    X = feats[["average", "rms", "max", "var"]].to_numpy(float)
    y = feats["label"].to_numpy(int)
    return X, y


def split_data(X, y, seed=0, frac=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_tr = int(frac[0] * len(X))
    n_va = int(frac[1] * len(X))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def one_hot(y, k=N_CLASSES):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def confusion_matrix(y_true, y_pred, k=N_CLASSES):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------
class MLP:
    def __init__(self, h1=2, h2=3, act_params=None, seed=0):
        self.act = act_params or {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}
        sizes = [4, h1, h2, N_CLASSES]
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / (sizes[i] + sizes[i + 1])),
                             (sizes[i], sizes[i + 1])) for i in range(3)]
        self.b = [np.zeros((1, sizes[i + 1])) for i in range(3)]

    # ---- forward ----------------------------------------------------------
    def forward(self, X):
        cache = {"A0": X}
        A = X
        for l in range(2):                      # dos capas ocultas (sigmoide paramétrica)
            Z = A @ self.W[l] + self.b[l]
            A = sigmoid(self.act, Z)
            cache[f"Z{l+1}"], cache[f"A{l+1}"] = Z, A
        Z3 = A @ self.W[2] + self.b[2]          # salida (softmax)
        Y = softmax(Z3)
        cache["Z3"], cache["A3"] = Z3, Y
        return Y, cache

    # ---- backward ---------------------------------------------------------
    def backward(self, cache, T):
        m = T.shape[0]
        dZ3 = (cache["A3"] - T) / m             # softmax + entropía cruzada
        dW3 = cache["A2"].T @ dZ3
        db3 = dZ3.sum(axis=0, keepdims=True)

        dA2 = dZ3 @ self.W[2].T
        dZ2 = dA2 * sigmoidDiff(self.act, cache["Z2"])
        dW2 = cache["A1"].T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)

        dA1 = dZ2 @ self.W[1].T
        dZ1 = dA1 * sigmoidDiff(self.act, cache["Z1"])
        dW1 = cache["A0"].T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)
        return [dW1, dW2, dW3], [db1, db2, db3]

    # ---- utilidades -------------------------------------------------------
    @staticmethod
    def loss(Y, T):
        return float(-np.mean(np.sum(T * np.log(Y + 1e-12), axis=1)))

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)

    def evaluate(self, X, y):
        Y, _ = self.forward(X)
        return self.loss(Y, one_hot(y)), float(np.mean(Y.argmax(axis=1) == y))

    # ---- entrenamiento ----------------------------------------------------
    def fit(self, Xtr, ytr, Xva, yva, epochs=200, lr=0.5, batch=32, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        Ttr = one_hot(ytr)
        hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        for ep in range(1, epochs + 1):
            order = rng.permutation(len(Xtr))
            for s in range(0, len(Xtr), batch):
                b = order[s:s + batch]
                _, cache = self.forward(Xtr[b])
                dW, db = self.backward(cache, Ttr[b])
                for l in range(3):
                    self.W[l] -= lr * dW[l]
                    self.b[l] -= lr * db[l]
            l_tr, a_tr = self.evaluate(Xtr, ytr)
            l_va, a_va = self.evaluate(Xva, yva)
            for k, v in zip(hist, (l_tr, a_tr, l_va, a_va)):
                hist[k].append(v)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
                print(f"ép {ep:4d} | loss {l_tr:.4f} acc {a_tr:.3f} | "
                      f"val_loss {l_va:.4f} val_acc {a_va:.3f}")
        return hist


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def run(csv_path, epochs=200, lr=0.5, batch=32, h1=2, h2=3, c=1.0, seed=0, out_prefix="mlp"):
    X, y = extract_features(csv_path)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_data(X, y, seed)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8      # estandarizar con stats de train
    Xtr, Xva, Xte = [(A - mu) / sd for A in (Xtr, Xva, Xte)]

    net = MLP(h1, h2, {"a": 1.0, "b": 1.0, "c": c, "d": 0.0}, seed)
    hist = net.fit(Xtr, ytr, Xva, yva, epochs, lr, batch, seed)

    te_loss, te_acc = net.evaluate(Xte, yte)
    cm = confusion_matrix(yte, net.predict(Xte))
    print(f"\nTEST  loss {te_loss:.4f}  acc {te_acc:.3f}\n{cm}")

    pd.DataFrame(hist).rename_axis("epoch").to_csv(f"{out_prefix}_history.csv")
    pd.DataFrame(cm).to_csv(f"{out_prefix}_confusion.csv")
    with open(f"{out_prefix}_metrics.json", "w") as f:
        json.dump({"test_loss": te_loss, "test_acc": te_acc, "confusion_matrix": cm.tolist(),
                   "split": [len(ytr), len(yva), len(yte)]}, f, indent=2)
    return net, hist, {"loss": te_loss, "acc": te_acc, "cm": cm}


def plot_history(hist):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("época"); ax[0].legend()
    ax[1].plot(hist["acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("época"); ax[1].legend()
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, default="dataset/dataset_balanceado_500.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--h1", type=int, default=2)
    ap.add_argument("--h2", type=int, default=3)
    ap.add_argument("--c", type=float, default=1.0)
    a = ap.parse_args()
    _, h, _ = run(a.csv, a.epochs, a.lr, a.batch, a.h1, a.h2, a.c)
    plot_history(h)