"""
enlace.py
=========
Transporte del protocolo de app_cmd.h hacia el S32K312 (pyserial), hacia el
gemelo en C compilado para PC (tests/host_mcu) o hacia una emulacion float32
en NumPy. Todos exponen line(cmd) -> str y lines_until(cmd, fin) -> [str].

Funciones de alto nivel (las usan run_mcu.py, cargar_pesos.py, evaluar.py):
    cargar_secuencia(link, codes)        tramas Q (2 por secuencia)
    muestra(link, 'T'|'V', codes, y)     -> cls, p[5], loss, cic_fwd, cic_bwd, rtt
    volcar_pesos(link)                   W  -> np.float32[1461]
    escribir_pesos(link, flat)           L  (8 parametros por linea)
"""

import os
import subprocess
import time

import numpy as np

import lstm_core as core
from lstm_core import LSTMNet, f32_to_hex, hex_to_f32

AQUI = os.path.dirname(os.path.abspath(__file__))
HOST_BIN = os.path.join(AQUI, "..", "tests", "host_mcu")
READ_TIMEOUT_S = 3.0


class SerialLink:
    def __init__(self, port, baud):
        import serial   # solo se exige pyserial con hardware real
        self.ser = serial.Serial(port, baud, timeout=READ_TIMEOUT_S)
        time.sleep(0.5)
        self.ser.reset_input_buffer()
        self.ser.write(b"X\n")                   # asegura modo IDLE (corta la telemetria)
        time.sleep(0.05)
        self._readline()
        self.ser.reset_input_buffer()

    def _readline(self):
        while True:
            s = self.ser.readline().decode("ascii", errors="replace").strip()
            if not s.startswith("D,"):            # ignora telemetria residual
                return s

    def line(self, cmd):
        self.ser.write((cmd + "\n").encode("ascii"))
        r = self._readline()
        if not r:
            raise TimeoutError(f"Sin respuesta del MCU a {cmd[:20]!r}")
        return r

    def lines_until(self, cmd, fin="END"):
        self.ser.write((cmd + "\n").encode("ascii"))
        out = []
        while True:
            r = self._readline()
            if not r:
                raise TimeoutError("volcado incompleto")
            if r == fin:
                return out
            out.append(r)

    def close(self):
        self.ser.close()


class HostLink(SerialLink):
    """Mismo protocolo contra el gemelo en C compilado para PC."""

    def __init__(self, exe=HOST_BIN):
        if not os.path.exists(exe) and os.path.exists(exe + ".exe"):
            exe += ".exe"                             # Windows / MinGW
        if not os.path.exists(exe):
            raise FileNotFoundError(f"Compila primero {exe} (ver tests/README.md)")
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)

    def _readline(self):
        return self.p.stdout.readline().strip()

    def send(self, cmd):
        """Linea sin respuesta (comando A del gemelo)."""
        self.p.stdin.write(cmd + "\n")

    def line(self, cmd):
        self.p.stdin.write(cmd + "\n"); self.p.stdin.flush()
        return self._readline()

    def lines_until(self, cmd, fin="END"):
        self.p.stdin.write(cmd + "\n"); self.p.stdin.flush()
        out = []
        while True:
            r = self._readline()
            if r == fin:
                return out
            out.append(r)

    def close(self):
        self.p.stdin.write("Z\n"); self.p.stdin.flush(); self.p.wait(timeout=5)


class SimLink:
    """Emulacion float32 en NumPy del protocolo (no ejercita el codigo C)."""

    def __init__(self, seed=1):
        self.seed = seed
        self.net = LSTMNet(seed=seed, dtype=np.float32)
        self.lr, self.clip = 0.1, 1.0
        self.codes = np.zeros((core.T, 3), dtype=np.int64)
        self.loaded = 0

    def line(self, cmd):
        op = cmd[0]
        if op == "R":
            self.net = LSTMNet(seed=self.seed, dtype=np.float32); return "OK"
        if op in "XIFOPS":
            return "OK"
        if op == "G":
            a, b = (int(v) for v in cmd[2:].split(","))
            self.lr, self.clip = a * 1e-6, b * 1e-3; return "OK"
        if op == "Q":
            k0, hx = cmd[2:].split(",")
            k0 = int(k0); c = core.hex_to_codes(hx)
            if k0 != (0 if k0 == 0 else self.loaded):
                return "E"
            self.codes[k0:k0 + len(c)] = c; self.loaded = k0 + len(c); return "K"
        if op == "L":
            parts = cmd[2:].split(",")
            i0 = int(parts[0]); flat = self.net.flat_params()
            for k, h in enumerate(parts[1:]):
                flat[i0 + k] = hex_to_f32(h)
            self.net.set_flat_params(flat); return "K"
        if op in "TV":
            y = int(cmd[2:])
            x = core.codes_to_x(self.codes, np.float32)
            if op == "T":
                p, loss = self.net.train_sample(x, y, self.lr, self.clip)
            else:
                p, _ = self.net.forward(x); loss = self.net.loss(p, y)
            return ",".join([str(int(np.argmax(p)))] + [f32_to_hex(v) for v in p]
                            + [f32_to_hex(loss), "0", "0"])
        return "E"

    def lines_until(self, cmd, fin="END"):
        flat = self.net.flat_params()
        return [f"W,{i}," + ",".join(f32_to_hex(v) for v in flat[i:i + 8])
                for i in range(0, len(flat), 8)]

    def close(self):
        pass


def abrir(args):
    """Crea el enlace a partir de los argumentos comunes (--simular/--host/--puerto)."""
    if getattr(args, "simular", False):
        return SimLink(), "simulado"
    if getattr(args, "host", False):
        return HostLink(), "host_c"
    return SerialLink(args.puerto, args.baud), "hardware"


def agregar_argumentos(ap):
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--simular", action="store_true", help="emulacion NumPy float32")
    g.add_argument("--host", action="store_true", help="gemelo en C (tests/host_mcu)")
    ap.add_argument("--puerto", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)


# ---------------------------------------------------------------------
def _ok(r, cmd):
    if r not in ("K", "OK"):
        raise RuntimeError(f"El MCU respondio {r!r} a {cmd[:24]!r}... ¿esta en IDLE?")


def cargar_secuencia(link, codes):
    for fr in core.seq_frames(codes):
        _ok(link.line(fr), fr)


def muestra(link, modo, codes, y):
    t0 = time.perf_counter()
    cargar_secuencia(link, codes)
    r = link.line(f"{modo},{int(y)}")
    rtt = time.perf_counter() - t0
    if r == "E" or "," not in r:
        raise RuntimeError(f"El MCU rechazo {modo} (respuesta {r!r}); ¿esta en modo IDLE?")
    f = r.split(",")
    p = [hex_to_f32(h) for h in f[1:1 + core.N_OUT]]
    k = 1 + core.N_OUT
    return int(f[0]), p, hex_to_f32(f[k]), int(f[k + 1]), int(f[k + 2]), rtt


def volcar_pesos(link):
    flat = []
    for ln in link.lines_until("W"):
        flat += [hex_to_f32(h) for h in ln.split(",")[2:]]
    flat = np.array(flat, dtype=np.float32)
    if len(flat) != core.N_PARAMS:
        raise RuntimeError(f"volcado con {len(flat)} parametros (esperados {core.N_PARAMS})")
    return flat


def escribir_pesos(link, flat):
    flat = np.asarray(flat, dtype=np.float32)
    for i in range(0, len(flat), 8):
        cmd = f"L,{i}," + ",".join(f32_to_hex(v) for v in flat[i:i + 8])
        _ok(link.line(cmd), cmd)


def fijar_lr(link, lr, clip):
    _ok(link.line(f"G,{int(round(lr * 1e6))},{int(round(clip * 1e3))}"), "G")
