"""
RECOLECTOR DE DATASET CON ETIQUETADO MANUAL (para LSTM)
=======================================================

FLUJO
-----
1. Pulsas la tecla de la clase que VAS a grabar.
2. Python toma el reposo actual y espera el "onset" (que alguna perilla se
   mueva más de UMBRAL_ONSET respecto al reposo).
3. Desde el onset (con N_PRE muestras previas) graba WINDOW_LENGTH muestras,
   imprimiendo cada fila que llega del micro.
4. Muestra un análisis y lo compara con el patrón de la clase.
   Enter/s = guardar | n = descartar.
5. Se guarda la serie completa con la etiqueta que TÚ elegiste.

TECLAS (en estado libre)
------------------------
0-4 = grabar esa clase | 9 = otros gestos (se guarda como clase 0) | q = salir
En espera de movimiento: c = cancelar

PATRONES (potenciómetros P1, P2, P3; escala 0-100 %)
----------------------------------------------------
Clase 0  STOP           -> Tecla 0: no mover nada (reposo, en distintas posiciones).
                           Tecla 9: cualquier gesto que NO sea 1-4 (una o dos
                           perillas, orden equivocado, a medias, errático...).
Clase 1  Avance Rápido  -> Subir P1 -> P2 -> P3, de 0 a 100 %.
Clase 2  Avance Lento   -> Subir P1 -> P2 -> P3, de 0 a 50 %.
Clase 3  Reversa Rápida -> Bajar P3 -> P2 -> P1, de 100 a 0 %.
Clase 4  Reversa Lenta  -> Bajar P3 -> P2 -> P1, de 100 a 50 %.

Antes de las clases 1-2 deja las perillas abajo; antes de las 3-4, arriba.
"""

import csv
import msvcrt
import os
import sys
import time
from collections import deque

import numpy as np
import serial

# ==========================================
# CONFIGURACIÓN
# ==========================================
SERIAL_PORT = 'COM10'
BAUD_RATE = 115200
NOMBRE_DATASET = 'dataset_series_temporales.csv'

MUESTREO_MS = 20          # 50 Hz
WINDOW_LENGTH = 250       # 250 muestras = 5 s (ajústalo a tu gesto más lento)
N_PRE = 5                 # muestras previas al onset que se incluyen
N_WARMUP = 40             # muestras ignoradas al inicio (el filtro EMA arranca en 0)
N_BASE = 10               # muestras para calcular el reposo
UMBRAL_ONSET = 0.04       # cuánto debe moverse una perilla para iniciar
UMBRAL_RANGO = 0.10       # rango mínimo para considerar que una perilla se movió
TOL_NIVEL = 0.12          # tolerancia sobre el nivel esperado (escala 0-1)

NOMBRES = {
    0: "STOP (0)",
    1: "Avance Rápido (+150)",
    2: "Avance Lento (+50)",
    3: "Reversa Rápida (-150)",
    4: "Reversa Lenta (-50)",
}

# ini/fin en escala 0-1 (0 = 0 %, 1.0 = 100 %)
PATRONES = {
    0: dict(orden=None,              ini=None, fin=None),
    1: dict(orden=('P1', 'P2', 'P3'), ini=0.0,  fin=1.0),   # subir 0-100
    2: dict(orden=('P1', 'P2', 'P3'), ini=0.0,  fin=0.5),   # subir 0-50
    3: dict(orden=('P3', 'P2', 'P1'), ini=1.0,  fin=0.0),   # bajar 100-0
    4: dict(orden=('P3', 'P2', 'P1'), ini=1.0,  fin=0.5),   # bajar 100-50
}

CANALES = ('P1', 'P2', 'P3')


# ==========================================
# UTILIDADES
# ==========================================
def barra(v, ancho=10):
    n = int(round(min(max(v, 0.0), 1.0) * ancho))
    return '█' * n + '·' * (ancho - n)


def fila_str(i, f):
    return (f"  {i:03d} t={i * MUESTREO_MS:4d}ms | "
            f"P1 {f[0]:.3f} {barra(f[0])} | "
            f"P2 {f[1]:.3f} {barra(f[1])} | "
            f"P3 {f[2]:.3f} {barra(f[2])}")


def leer_muestra(ser):
    """Devuelve [p1, p2, p3] o None si no llegó una línea válida."""
    linea = ser.readline().decode('utf-8', errors='ignore').strip()
    if not linea:
        return None
    partes = [p for p in linea.split(',') if p != '']
    if len(partes) < 3:
        return None
    try:
        return [float(partes[0]), float(partes[1]), float(partes[2])]
    except ValueError:
        return None


def analizar(ventana):
    """Rango, delta y tiempo al 50 % de cada perilla, más el orden de activación."""
    info = {}
    for i, n in enumerate(CANALES):
        p = ventana[:, i]
        rango = float(p.max() - p.min())
        delta = float(p[-1] - p[0])
        t50 = None
        if rango >= UMBRAL_RANGO:
            t50 = int(np.argmax(np.abs(p - p[0]) >= rango / 2.0)) * MUESTREO_MS
        info[n] = dict(rango=rango, delta=delta, t50=t50)

    movidas = [n for n in CANALES if info[n]['t50'] is not None]
    orden = tuple(sorted(movidas, key=lambda n: info[n]['t50']))
    return info, movidas, orden


def verificar(clase, ventana, movidas, orden):
    """Compara la ventana con el patrón esperado. Devuelve lista de avisos."""
    pat = PATRONES[clase]
    avisos = []

    if clase == 0:
        if movidas:
            avisos.append(f"STOP pero se movieron: {', '.join(movidas)}")
        return avisos

    if len(movidas) < 3:
        faltan = [n for n in CANALES if n not in movidas]
        avisos.append(f"No se movieron: {', '.join(faltan)}")
        return avisos

    if orden != pat['orden']:
        avisos.append(f"Orden {' -> '.join(orden)} (esperado {' -> '.join(pat['orden'])})")

    for i, n in enumerate(CANALES):
        ini, fin = ventana[0, i], ventana[-1, i]
        if abs(ini - pat['ini']) > TOL_NIVEL:
            avisos.append(f"{n} empezó en {ini * 100:.0f}% (esperado {pat['ini'] * 100:.0f}%)")
        if abs(fin - pat['fin']) > TOL_NIVEL:
            avisos.append(f"{n} terminó en {fin * 100:.0f}% (esperado {pat['fin'] * 100:.0f}%)")
    return avisos


def cargar_estado():
    """Lee el CSV existente: último sample_id y conteo de muestras por clase."""
    ultimo_id = 0
    ids = {c: set() for c in NOMBRES}
    if os.path.isfile(NOMBRE_DATASET):
        with open(NOMBRE_DATASET, newline='') as f:
            for fila in csv.DictReader(f):
                sid, lab = int(fila['sample_id']), int(fila['label'])
                ultimo_id = max(ultimo_id, sid)
                ids.setdefault(lab, set()).add(sid)
    return ultimo_id, {c: len(s) for c, s in ids.items()}


def esperar_tecla(validas):
    while True:
        if msvcrt.kbhit():
            k = msvcrt.getwch().lower()
            if k in validas:
                return k
        time.sleep(0.02)


def resumen_conteo(conteo):
    return ' | '.join(f"{c}:{conteo.get(c, 0)}" for c in NOMBRES)


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 72)
    print(" RECOLECTOR LSTM - ETIQUETADO MANUAL")
    print("=" * 72)
    for c, n in NOMBRES.items():
        pat = PATRONES[c]
        if pat['orden']:
            desc = (f"{' -> '.join(pat['orden'])}, "
                    f"{pat['ini'] * 100:.0f}% -> {pat['fin'] * 100:.0f}%")
        else:
            desc = "reposo (tecla 0) / otros gestos (tecla 9)"
        print(f"  [{c}] {n:<24} {desc}")
    print("\nTeclas: 0-4 = grabar clase | 9 = otros gestos (clase 0) | q = salir\n")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"-> Conectado a {SERIAL_PORT}\n")
    except serial.SerialException as e:
        print(f"[ERROR] No se pudo abrir {SERIAL_PORT}: {e}")
        sys.exit(1)

    ultimo_id, conteo = cargar_estado()
    nuevo = not os.path.isfile(NOMBRE_DATASET)
    f_csv = open(NOMBRE_DATASET, mode='a', newline='')
    writer = csv.writer(f_csv)
    if nuevo:
        writer.writerow(['sample_id', 'timestamp_ms', 'p1', 'p2', 'p3', 'label'])
        f_csv.flush()

    print(f"Muestras por clase -> {resumen_conteo(conteo)}")
    print(f"Calentando filtro ({N_WARMUP} muestras)...")

    recientes = deque(maxlen=max(N_BASE, N_PRE + 1))
    estado = 'IDLE'            # IDLE | ARMED | RECORDING
    clase = None
    es_otro = False
    base = None
    grabando = []
    n_rx = 0

    try:
        while True:
            m = leer_muestra(ser)
            if m is not None:
                n_rx += 1
                recientes.append(m)

            tecla = msvcrt.getwch().lower() if msvcrt.kbhit() else None

            # ---------- IDLE: telemetría en vivo y selección de clase ----------
            if estado == 'IDLE':
                if m is not None:
                    listo = "LISTO" if n_rx > N_WARMUP else "calentando"
                    print(f"\r[{listo}] P1 {m[0]:.3f} {barra(m[0])} | "
                          f"P2 {m[1]:.3f} {barra(m[1])} | "
                          f"P3 {m[2]:.3f} {barra(m[2])}   (0-4 clase, 9 otros, q salir)   ",
                          end='', flush=True)

                if tecla == 'q':
                    break

                if (tecla in ('0', '1', '2', '3', '4', '9')
                        and n_rx > N_WARMUP and len(recientes) >= N_BASE):
                    es_otro = (tecla == '9')
                    clase = 0 if es_otro else int(tecla)
                    base = np.mean(np.array(list(recientes)[-N_BASE:]), axis=0)
                    print(f"\n\n>>> Clase {clase}: {NOMBRES[clase]}"
                          f"{' (OTROS GESTOS)' if es_otro else ''}")
                    print(f"    Reposo base: P1 {base[0]:.3f} | P2 {base[1]:.3f} | "
                          f"P3 {base[2]:.3f}")
                    if tecla == '0':
                        print("    Grabando STOP (no muevas nada)...\n")
                        grabando = [list(r) for r in recientes][-N_PRE:]
                        estado = 'RECORDING'
                    else:
                        print("    Esperando movimiento... (c = cancelar)\n")
                        estado = 'ARMED'

            # ---------- ARMED: espera el onset ----------
            elif estado == 'ARMED':
                if tecla == 'c':
                    print("\n    Cancelado.\n")
                    estado = 'IDLE'
                    continue
                if m is not None:
                    desv = np.max(np.abs(np.array(m) - base))
                    print(f"\r[ESPERANDO] P1 {m[0]:.3f} | P2 {m[1]:.3f} | P3 {m[2]:.3f} "
                          f"| desvío {desv:.3f}/{UMBRAL_ONSET}   ", end='', flush=True)
                    if desv > UMBRAL_ONSET:
                        print("\n\n    ¡Movimiento detectado! Grabando...\n")
                        grabando = [list(r) for r in recientes][-(N_PRE + 1):]
                        estado = 'RECORDING'
                        for i, fila in enumerate(grabando):
                            print(fila_str(i, fila))

            # ---------- RECORDING: graba, analiza y guarda ----------
            elif estado == 'RECORDING':
                if m is not None:
                    grabando.append(m)
                    print(fila_str(len(grabando) - 1, m))

                if len(grabando) >= WINDOW_LENGTH:
                    ventana = np.array(grabando[:WINDOW_LENGTH])
                    info, movidas, orden = analizar(ventana)
                    avisos = [] if es_otro else verificar(clase, ventana, movidas, orden)

                    print("\n" + "-" * 72)
                    print(f" ANÁLISIS (clase elegida: {clase} - {NOMBRES[clase]})")
                    for n in CANALES:
                        d = info[n]
                        t = f"{d['t50']} ms" if d['t50'] is not None else "sin mover"
                        print(f"   {n}: rango {d['rango']:.3f} | delta {d['delta']:+.3f} "
                              f"| 50% en {t}")
                    print(f"   Orden detectado: {' -> '.join(orden) if orden else '-'}")
                    if es_otro:
                        print("   (Clase 0 'otros gestos': sin verificación automática)")
                    elif avisos:
                        print("   ⚠ POSIBLES PROBLEMAS:")
                        for a in avisos:
                            print(f"     - {a}")
                    else:
                        print("   ✔ Coincide con el patrón esperado")
                    print("-" * 72)
                    print(" ¿Guardar? Enter/s = guardar | n = descartar ", end='', flush=True)
                    k = esperar_tecla({'\r', '\n', 's', 'n'})

                    if k == 'n':
                        print("-> Descartada.\n")
                    else:
                        ultimo_id += 1
                        for i, fila in enumerate(ventana):
                            writer.writerow([ultimo_id, i * MUESTREO_MS,
                                             f"{fila[0]:.3f}", f"{fila[1]:.3f}",
                                             f"{fila[2]:.3f}", clase])
                        f_csv.flush()
                        conteo[clase] = conteo.get(clase, 0) + 1
                        print(f"-> Guardada como muestra #{ultimo_id}")
                        print(f"   Por clase -> {resumen_conteo(conteo)}\n")

                    ser.reset_input_buffer()
                    recientes.clear()
                    grabando = []
                    estado = 'IDLE'
                    print("Regresa las perillas a su posición de reposo y elige la siguiente clase.\n")

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n\nCaptura finalizada. Por clase -> {resumen_conteo(conteo)}")
        f_csv.close()
        ser.close()


if __name__ == '__main__':
    main()
