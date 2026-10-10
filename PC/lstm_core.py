"""
lstm_core.py
============
Modelo de referencia (NumPy) y utilidades compartidas por todos los scripts.
Es el gemelo de firmware/lstm.c y firmware/seq.c: mismas constantes, mismo
orden de parametros, mismas formulas. Cualquier cambio aqui se replica en C.

Red:   x_t (3 potes) -> LSTM(16) -> Dense(8, tanh) -> Dense(5) + softmax
Perdida: suma de errores al cuadrado  E = 1/2 * sum_k (p_k - t_k)^2
Entrenamiento: SGD por secuencia con BPTT completo (50 pasos) y recorte de
la norma global del gradiente.

Clases (indice interno = etiqueta del CSV):
    0 -> clase 5 del enunciado: default, referencia 0 rpm
    1 -> clase 1: velocidad alta, sentido CW
    2 -> clase 2: velocidad nominal, sentido CW
    3 -> clase 3: velocidad alta, sentido CCW
    4 -> clase 4: velocidad nominal, sentido CCW
Se conserva 0 = default para que coincida con el firmware (CLS_DEFAULT = 0).

Entrada de la red (formato de linea y de memoria del MCU):
    codigo = round(1000 * nivel)  en [0, CODE_MAX]   (entero, 3 hex por canal)
    x      = codigo * 0.002f - 1.0f                    (float32, en [-1, 1.2])
El CSV trae 250 muestras a 20 ms; se promedian bloques de DEC_DATA = 5 para
pasar a 50 pasos de 100 ms (la LSTM ve 5 s con 50 pasos).
"""

import json
import struct

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Constantes (identicas a lstm.h / seq.h)
# ---------------------------------------------------------------------
N_IN = 3
N_H = 16
N_D = 8
N_OUT = 5
T = 50                     # pasos de la secuencia
STEP_MS = 100              # periodo de un paso de la LSTM
CODE_SCALE = 1000          # codigo = nivel * 1000
CODE_MAX = 1100            # recorte del codigo (nivel 1.1)
X_GAIN = np.float32(0.002)
X_OFF = np.float32(1.0)
SEQ_CHUNK = 25             # pasos por trama 'Q' (25 * 9 = 225 caracteres)

# Adquisicion en tiempo real (seq.h)
TICK_MS = 5
SEQ_DEC = STEP_MS // TICK_MS     # 20 lecturas de ADC por paso
ADC_FS = 2700                    # cuentas del ADC para nivel 1.0

CLASES = ["Default", "Alta CW", "Nominal CW", "Alta CCW", "Nominal CCW"]
CLASE_ENUNCIADO = [5, 1, 2, 3, 4]          # numeracion del enunciado
REF_SIGNO = [0, +1, +1, -1, -1]
REF_ES_ALTA = [False, True, False, True, False]

N_PARAMS = (4 * N_H * N_IN + 4 * N_H * N_H + 4 * N_H
            + N_D * N_H + N_D + N_OUT * N_D + N_OUT)       # = 1461


# ---------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------
def load_dataset(csv_path, step_ms=STEP_MS):
    """CSV largo (sample_id, timestamp_ms, p1, p2, p3, label) -> codigos.

    Usa el timestamp para ordenar, verificar que el muestreo es uniforme y
    calcular el factor de diezmado al paso de la LSTM. Devuelve
    codes (N, T, 3) int, y (N,) int, ids (N,).
    """
    df = pd.read_csv(csv_path).sort_values(["sample_id", "timestamp_ms"])
    codes, ys, ids = [], [], []
    for sid, g in df.groupby("sample_id", sort=True):
        ts = g["timestamp_ms"].to_numpy()
        dt = np.diff(ts)
        if len(dt) == 0 or np.any(dt != dt[0]):
            raise ValueError(f"sample_id {sid}: timestamp no uniforme")
        dec = int(round(step_ms / dt[0]))
        if dec * dt[0] != step_ms:
            raise ValueError(f"periodo {dt[0]} ms no divide a {step_ms} ms")
        c = np.rint(g[["p1", "p2", "p3"]].to_numpy(float) * CODE_SCALE).astype(np.int64)
        c = np.clip(c, 0, CODE_MAX)
        n = (len(c) // dec) * dec
        c = c[:n].reshape(-1, dec, 3).sum(1)
        c = (c + dec // 2) // dec                     # promedio redondeado (entero)
        if len(c) < T:
            raise ValueError(f"sample_id {sid}: {len(c)} pasos < {T}")
        codes.append(c[-T:])                          # ultimos T pasos
        ys.append(int(g["label"].iloc[0])); ids.append(int(sid))
    return np.array(codes), np.array(ys), np.array(ids)


def codes_to_x(codes, dtype=np.float32):
    """Misma operacion que Seq_CodeToX() en C (float32)."""
    x = np.asarray(codes).astype(np.float32) * X_GAIN - X_OFF
    return x.astype(dtype)


def split_dataset(codes, y, ids, seed=0, frac_val=0.15, frac_test=0.15):
    """Estratificado: 70 % entrenamiento + 15 % validacion (= 85 %) y 15 % prueba."""
    r = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = r.permutation(np.where(y == c)[0])
        n_te = int(round(frac_test * len(idx)))
        n_va = int(round(frac_val * len(idx)))
        te += list(idx[:n_te]); va += list(idx[n_te:n_te + n_va]); tr += list(idx[n_te + n_va:])
    tr, va, te = (np.array(sorted(s)) for s in (tr, va, te))
    d = {}
    for nombre, s in (("train", tr), ("val", va), ("test", te)):
        d[f"C_{nombre}"] = codes[s]
        d[f"X_{nombre}"] = codes_to_x(codes[s], np.float64)
        d[f"y_{nombre}"] = y[s]
        d[f"id_{nombre}"] = ids[s]
    return d


def epoch_orders(n, epochs, seed):
    r = np.random.default_rng(seed)
    return [r.permutation(n).tolist() for _ in range(epochs)]


# ---------------------------------------------------------------------
# Gemelo de seq.c (adquisicion en tiempo real)
# ---------------------------------------------------------------------
def raw_to_code(acc_sum):
    """codigo = round(1000 * suma / (SEQ_DEC * ADC_FS)), aritmetica entera."""
    den = SEQ_DEC * ADC_FS
    c = (np.asarray(acc_sum, dtype=np.int64) * CODE_SCALE + den // 2) // den
    return np.minimum(c, CODE_MAX)


def raw_series_to_codes(raw):
    """raw (n_ticks, 3) lecturas crudas -> pasos (n_pasos, 3) como Seq_PushRaw."""
    raw = np.asarray(raw, dtype=np.int64)
    n = (len(raw) // SEQ_DEC) * SEQ_DEC
    return raw_to_code(raw[:n].reshape(-1, SEQ_DEC, 3).sum(1))


# ---------------------------------------------------------------------
# Formato de linea
# ---------------------------------------------------------------------
def f32_to_hex(v):
    return struct.pack(">f", float(np.float32(v))).hex().upper()


def hex_to_f32(h):
    return struct.unpack(">f", bytes.fromhex(h))[0]


def codes_to_hex(codes):
    """(n, 3) -> 'AAABBBCCC...' : 3 hex por canal, 9 por paso."""
    return "".join(f"{int(a):03X}{int(b):03X}{int(c):03X}" for a, b, c in codes)


def hex_to_codes(s):
    v = [int(s[k:k + 3], 16) for k in range(0, len(s), 3)]
    return np.array(v, dtype=np.int64).reshape(-1, 3)


def seq_frames(codes):
    """Tramas 'Q,<k0>,<hex>' para cargar una secuencia completa en el MCU."""
    return [f"Q,{k},{codes_to_hex(codes[k:k + SEQ_CHUNK])}" for k in range(0, len(codes), SEQ_CHUNK)]


# ---------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------
def _sig(z):
    z = np.clip(z, -80.0, 80.0).astype(z.dtype)
    return (1.0 / (1.0 + np.exp(-z))).astype(z.dtype)


class LSTMNet:
    """LSTM(16) -> Dense(8, tanh) -> Dense(5) + softmax. SSE + BPTT.

    Orden de compuertas en las filas de Wx, Wh, b: i, f, g, o (como Keras).
    Orden plano de parametros (igual a LstmParams en C):
        Wx[4H,N_IN], Wh[4H,H], b[4H], Wd[D,H], bd[D], Wo[K,D], bo[K]
    """

    NAMES = ("Wx", "Wh", "b", "Wd", "bd", "Wo", "bo")

    def __init__(self, seed=1, dtype=np.float64, n_h=N_H):
        self.dt = dtype
        self.H = n_h
        H, G = n_h, 4 * n_h
        r = np.random.default_rng(seed)
        lim = lambda a, b: np.sqrt(6.0 / (a + b))
        Wx = r.uniform(-lim(N_IN, G), lim(N_IN, G), (G, N_IN))
        # Recurrente ortogonal (por compuerta), estabiliza el BPTT
        Wh = np.vstack([np.linalg.qr(r.normal(size=(H, H)))[0] for _ in range(4)])
        b = np.zeros(G); b[H:2 * H] = 1.0                 # sesgo de olvido = 1
        Wd = r.uniform(-lim(H, N_D), lim(H, N_D), (N_D, H))
        Wo = r.uniform(-lim(N_D, N_OUT), lim(N_D, N_OUT), (N_OUT, N_D))
        self.P = {"Wx": Wx, "Wh": Wh, "b": b, "Wd": Wd, "bd": np.zeros(N_D),
                  "Wo": Wo, "bo": np.zeros(N_OUT)}
        self._cast()

    def _cast(self):
        for k in self.P:
            self.P[k] = np.asarray(self.P[k], dtype=self.dt)

    # ---- parametros planos ---------------------------------------------
    def n_params(self):
        return sum(v.size for v in self.P.values())

    def flat_params(self):
        return np.concatenate([self.P[k].ravel() for k in self.NAMES])

    def set_flat_params(self, flat):
        flat = np.asarray(flat, dtype=np.float64)
        assert len(flat) == self.n_params(), f"{len(flat)} != {self.n_params()}"
        o = 0
        for k in self.NAMES:
            n = self.P[k].size
            self.P[k] = flat[o:o + n].reshape(self.P[k].shape)
            o += n
        self._cast()

    def get_weights(self):
        return {k: self.P[k].tolist() for k in self.NAMES}

    def set_weights(self, w):
        for k in self.NAMES:
            self.P[k] = np.array(w[k])
        self._cast()

    # ---- forward ---------------------------------------------------------
    def forward(self, X):
        """X (T, N_IN) -> p (K,), cache. Mismo orden de operaciones que C."""
        P, H, dt = self.P, self.H, self.dt
        X = np.asarray(X, dtype=dt)
        Tn = len(X)
        gates = np.zeros((Tn, 4 * H), dt)
        cs = np.zeros((Tn, H), dt); tcs = np.zeros((Tn, H), dt); hs = np.zeros((Tn, H), dt)
        h = np.zeros(H, dt); c = np.zeros(H, dt)
        for t in range(Tn):
            z = P["b"] + P["Wx"] @ X[t] + P["Wh"] @ h
            i = _sig(z[:H]); f = _sig(z[H:2 * H])
            g = np.tanh(z[2 * H:3 * H]); o = _sig(z[3 * H:])
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            gates[t] = np.concatenate([i, f, g, o]); cs[t] = c; tcs[t] = tc; hs[t] = h
        a = np.tanh(P["bd"] + P["Wd"] @ h)
        z3 = P["bo"] + P["Wo"] @ a
        e = np.exp(z3 - z3.max())
        p = (e / e.sum()).astype(dt)
        return p, {"X": X, "gates": gates, "c": cs, "tc": tcs, "h": hs, "a": a, "p": p}

    def predict_proba(self, Xb):
        """Forward vectorizado sobre un lote (N, T, N_IN) -> (N, K)."""
        P, H, dt = self.P, self.H, self.dt
        Xb = np.asarray(Xb, dtype=dt)
        N = len(Xb)
        h = np.zeros((N, H), dt); c = np.zeros((N, H), dt)
        for t in range(Xb.shape[1]):
            z = P["b"] + Xb[:, t] @ P["Wx"].T + h @ P["Wh"].T
            i = _sig(z[:, :H]); f = _sig(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H]); o = _sig(z[:, 3 * H:])
            c = f * c + i * g
            h = o * np.tanh(c)
        a = np.tanh(P["bd"] + h @ P["Wd"].T)
        z3 = P["bo"] + a @ P["Wo"].T
        e = np.exp(z3 - z3.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    @staticmethod
    def loss(p, y):
        t = np.zeros_like(p); t[y] = 1.0
        return float(0.5 * np.sum((p - t) ** 2))

    # ---- backward (BPTT) ------------------------------------------------
    def gradients(self, cache, y):
        P, H, dt = self.P, self.H, self.dt
        p, a = cache["p"], cache["a"]
        X, gates, cs, tcs, hs = (cache[k] for k in ("X", "gates", "c", "tc", "h"))
        t = np.zeros(N_OUT, dt); t[y] = 1.0
        gp = p - t                                     # dE/dp (SSE)
        dz3 = p * (gp - np.dot(p, gp))                 # jacobiano del softmax
        G = {"Wo": np.outer(dz3, a), "bo": dz3}
        da = P["Wo"].T @ dz3
        dzd = da * (1.0 - a * a)
        G["Wd"] = np.outer(dzd, hs[-1]); G["bd"] = dzd
        dh = P["Wd"].T @ dzd
        dc = np.zeros(H, dt)
        dWx = np.zeros_like(P["Wx"]); dWh = np.zeros_like(P["Wh"]); db = np.zeros_like(P["b"])
        for k in range(len(X) - 1, -1, -1):
            i, f, g, o = (gates[k, s * H:(s + 1) * H] for s in range(4))
            tc = tcs[k]
            cp = cs[k - 1] if k > 0 else np.zeros(H, dt)
            hp = hs[k - 1] if k > 0 else np.zeros(H, dt)
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc * tc)
            dz = np.concatenate([dc * g * i * (1.0 - i),
                                 dc * cp * f * (1.0 - f),
                                 dc * i * (1.0 - g * g),
                                 do * o * (1.0 - o)])
            dWx += np.outer(dz, X[k]); dWh += np.outer(dz, hp); db += dz
            dh = P["Wh"].T @ dz
            dc = dc * f
        G["Wx"], G["Wh"], G["b"] = dWx, dWh, db
        return G

    def train_sample(self, X, y, lr, clip=5.0):
        p, cache = self.forward(X)
        loss = self.loss(p, y)
        G = self.gradients(cache, y)
        norm = np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in G.values()))
        s = (clip / norm) if (clip > 0 and norm > clip) else 1.0
        sc = self.dt(lr * s)
        for k in self.NAMES:
            self.P[k] = (self.P[k] - sc * G[k]).astype(self.dt)
        return p, loss

    def evaluate(self, Xb, y):
        P = self.predict_proba(Xb)
        T1 = np.eye(N_OUT)[y]
        loss = float(np.mean(0.5 * np.sum((P - T1) ** 2, axis=1)))
        return loss, 100.0 * float(np.mean(P.argmax(1) == y)), P

    # ---- exportacion -----------------------------------------------------
    def export_c(self, tag):
        flat = self.flat_params().astype(np.float32)
        body = []
        for k in range(0, len(flat), 4):
            body.append("    " + ", ".join(f"{v:.9e}f" for v in flat[k:k + 4]) + ",")
        g = f"PESOS_{tag}_H"
        return (f"/* Pesos LSTM({N_IN}->{self.H}) -> Dense({N_D}) -> Dense({N_OUT}), "
                f"{len(flat)} parametros.\n"
                f" * Orden plano: Wx[{4*self.H}x{N_IN}], Wh[{4*self.H}x{self.H}], b[{4*self.H}], "
                f"Wd[{N_D}x{self.H}], bd[{N_D}], Wo[{N_OUT}x{N_D}], bo[{N_OUT}]\n"
                f" * Generado automaticamente por lstm_core.py: no editar a mano. */\n"
                f"#ifndef {g}\n#define {g}\n\n"
                f"static const float LSTM_W_{tag}[{len(flat)}] = {{\n" + "\n".join(body)
                + "\n};\n\n#endif\n")


# ---------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------
def matriz_confusion(y, yp, k=N_OUT):
    M = np.zeros((k, k), dtype=int)
    for a, b in zip(y, yp):
        M[int(a), int(b)] += 1
    return M


def metricas_clase(M):
    M = np.asarray(M, float)
    tp = np.diag(M)
    prec = np.divide(tp, M.sum(0), out=np.zeros_like(tp), where=M.sum(0) > 0)
    rec = np.divide(tp, M.sum(1), out=np.zeros_like(tp), where=M.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {"accuracy": float(tp.sum() / max(M.sum(), 1)), "precision": prec.tolist(),
            "recall": rec.tolist(), "f1": f1.tolist(), "macro_f1": float(f1.mean())}


def con_umbral(P, umbral=0.70):
    """Misma decision que el firmware: si p_max < umbral -> default (0)."""
    P = np.asarray(P)
    return np.where(P.max(1) >= umbral, P.argmax(1), 0)


def gradient_check(seed=3, n_check=60, eps=1e-6):
    """Gradiente analitico (BPTT) contra diferencias centradas, en float64."""
    r = np.random.default_rng(seed)
    net = LSTMNet(seed=seed, dtype=np.float64)
    X = r.uniform(-1, 1, (12, N_IN)); y = 2
    p, cache = net.forward(X)
    G = net.gradients(cache, y)
    worst = 0.0
    for _ in range(n_check):
        k = net.NAMES[r.integers(len(net.NAMES))]
        idx = tuple(r.integers(s) for s in net.P[k].shape)
        old = net.P[k][idx]
        net.P[k][idx] = old + eps; lp = net.loss(net.forward(X)[0], y)
        net.P[k][idx] = old - eps; lm = net.loss(net.forward(X)[0], y)
        net.P[k][idx] = old
        num = (lp - lm) / (2 * eps); ana = G[k][idx]
        worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=_Enc)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
