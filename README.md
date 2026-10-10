# TemporalPatterns — LSTM en el S32K312

Evolución de *StaticPatterns* (MLP 9-8-6-4 sobre estadísticas de una ventana) a una red recurrente que
clasifica **gestos temporales** de los 3 potenciómetros y fija la referencia del PI de velocidad.
Se conservan el hardware, el planificador por DWT, el protocolo UART, el PI, la telemetría y la
máquina de estados IDLE / RUN / FIXEDREF / OPENLOOP.

## Red

```
x_t (3 potes, 50 pasos de 100 ms = 5 s)
  -> LSTM 16 unidades (i, f, g, o)          1280 parámetros
  -> Dense 8, tanh                           136
  -> Dense 5 + softmax                        45      total 1461 (5.8 KB en float32)
Pérdida: E = ½ Σ_k (p_k − t_k)²   (suma de errores al cuadrado, a través del jacobiano del softmax)
Entrenamiento: SGD por secuencia, BPTT completo, recorte de norma global (lr 0.1, clip 1.0, 20 épocas)
```

| Índice interno (CSV) | Clase del enunciado | Gesto en el dataset | Referencia |
|---|---|---|---|
| 1 | 1 · velocidad alta | sube pot1 → pot2 → pot3 hasta ~1.0 | +150 rpm |
| 2 | 2 · velocidad nominal | sube pot1 → pot2 → pot3 hasta ~0.5 | +90 rpm |
| 3 | 3 · alta, sentido contrario | baja pot3 → pot2 → pot1 de 1.0 a 0 | −150 rpm |
| 4 | 4 · nominal, sentido contrario | baja pot3 → pot2 → pot1 de 1.0 a ~0.5 | −90 rpm |
| 0 | 5 · default | potes quietos en cualquier nivel | 0 |

Internamente default es 0 (igual que `CLS_DEFAULT` del firmware anterior); los scripts muestran la
numeración del enunciado.

**Datos.** El CSV trae 500 secuencias de 250 muestras a 20 ms. `lstm_core.load_dataset` usa el
`timestamp_ms` para ordenar, verificar que el muestreo es uniforme y diezmar a 100 ms (promedio de 5
muestras), así la LSTM ve los mismos 5 s con 50 pasos: 5× menos cómputo, memoria de BPTT y bytes por
la UART. Partición estratificada: 70 % entrenamiento + 15 % validación (= 85 %) y 15 % prueba.

**Formato de entrada.** Cada nivel viaja como entero en milésimas (`código = round(1000·nivel)`, 3 hex
por canal) y en el micro se convierte con `x = código·0.002f − 1.0f`. PC y MCU parten del mismo entero.

## Estructura

```
firmware/   main.c          máquina de estados, ADC, PI, telemetría (de StaticPatterns)
            lstm.c/.h       forward, BPTT, forward incremental para RUN   (reemplaza mlp)
            seq.c/.h        promedio 100 ms + ventana de 50 pasos         (reemplaza preproc)
            app_cmd.c/.h    comandos Q T V R W L G C, portables (los usa también el gemelo de PC)
            pi_ctrl.c/.h    sin cambios
            pesos_iniciales.h, pesos_entrenados.h (generados)
python/     lstm_core.py    modelo de referencia NumPy (gemelo de lstm.c y seq.c) + utilidades
            enlace.py       transporte serie / gemelo C / emulación y funciones de protocolo
            run_pc.py       entrenamiento de referencia en PC + pesos_iniciales.h
            run_mcu.py      entrenamiento EN EL MICRO + retroalimentación de pesos
            cargar_pesos.py restaura los pesos tras un reinicio (sin reentrenar)
            evaluar.py      PC vs MCU sobre prueba y sobre datos reales
            capturar.py     dataset real de gestos con los potes
            telemetria.py   identificación, escalón, demo en RUN, ganancias, referencias
            simulacion_lazo.py  lazo completo simulado (gestos → red → PI → motor)
            graficas.py     figuras del informe
tests/      host_mcu.c      gemelo del firmware para PC (stdin/stdout)
            test_equivalencia.py
```

## Flujo de trabajo

```bash
cd python
python run_pc.py                       # referencia en PC, genera firmware/pesos_iniciales.h
# compilar y grabar el firmware (ver abajo)
python run_mcu.py --puerto COM6        # entrena en el S32K312 -> pesos_mcu.json + pesos_entrenados.h
python cargar_pesos.py --puerto COM6 --run   # tras un reinicio: restaura pesos y entra a RUN
python evaluar.py --puerto COM6        # PC vs MCU (prueba y, si existe, dataset_real.csv)
python capturar.py --puerto COM6 --n 10       # opcional: gestos reales
python telemetria.py demo --seg 60     # RUN con los potes, latencias y carga de CPU
python simulacion_lazo.py && python graficas.py
```

Sin tarjeta: `run_mcu.py --host` (código C real compilado en PC) o `--simular` (NumPy float32).

### Persistencia de los pesos

Al terminar el entrenamiento el micro vuelca los 1461 parámetros (`W`) y la PC los guarda en
`pesos_mcu.json` (hex IEEE-754 exacto) y en `firmware/pesos_entrenados.h`. Tras un reinicio hay dos
caminos, sin reentrenar:

1. `python cargar_pesos.py` — los escribe por UART (`L`, 183 líneas, ~1.5 s), los relee con `W` y
   verifica que son idénticos bit a bit.
2. Descomentar `#define USE_PRETRAINED` en `main.c` y recompilar: el binario arranca con la red
   entrenada y en RUN.

Los `pesos_mcu.json` / `pesos_entrenados.h` incluidos se generaron con el gemelo en C; sustitúyelos
por los de tu tarjeta corriendo `run_mcu.py --puerto ...`.

## Integración en S32 Design Studio

1. Quitar `mlp.c/.h`, `preproc.c/.h` y los `pesos_*.h` anteriores; agregar `lstm`, `seq`, `app_cmd` y
   los nuevos `pesos_*.h`. El `.mex` (pines, ADC, eMIOS, LPUART6, ICU) no cambia.
2. Agregar `-ffp-contract=off` a las opciones del compilador. Sin esa bandera GCC fusiona `a*b+c` en
   FMA y el micro deja de coincidir con Python en los últimos bits (sigue funcionando, solo crecen las
   diferencias PC–MCU de ~1e-7 a ~1e-5).
3. RAM nueva en `.bss`: ~35 KB (caché de BPTT 22 KB, gradiente 5.8 KB, pesos 5.8 KB, secuencias).
   El S32K312 tiene RAM de sobra, pero revisa que el linker no los mande a una región pequeña.
4. Baud: el protocolo funciona a 115200 (~47 ms de enlace por secuencia, ~7 min para 20 épocas más el cómputo).
   Si subes LPUART6 a 460800 en el `.mex`, pasa `--baud 460800` a los scripts.

### Tiempo real en RUN

Cada tick de 5 ms muestrea el ADC; cada 20 ticks (100 ms) se cierra un paso de la secuencia. Con
cada paso nuevo se congela la ventana y la inferencia (50 pasos de la celda) se reparte a 5 pasos por
tick, así ningún tick se pasa de su presupuesto: una clasificación cada 100 ms con 50 ms de latencia
de cómputo. Umbral de confianza 0.70 y antirrebote de 3 votos (300 ms) como antes.

Por diseño la ventana dura lo mismo que un gesto (5 s): la clase del gesto se mantiene mientras el
gesto está dentro de la ventana y luego vuelve a default (referencia 0), tal como pide el enunciado.
Si prefieres que el gesto quede "enganchado" hasta el siguiente, pon `REF_LATCH 1` en `main.c`
(entonces solo `X` detiene el motor).

## Verificación hecha en PC

- Gradiente BPTT analítico vs diferencias centradas: error relativo 1.7e-6.
- `tests/test_equivalencia.py` (C real vs Python float32): códigos de la secuencia idénticos;
  inferencia incremental vs forward 4e-8; 200 pasos de SGD+BPTT: Δp ≤ 4.5e-7, Δpesos ≤ 2.4e-7;
  carga `L` + volcado `W` idénticos bit a bit.
- Prueba (75 secuencias): PC float64 97.3 % (98.7 % con umbral), gemelo C 98.7 % (100 % con umbral).
- La red tolera la ventana desfasada ±1 s respecto al gesto (≥ 88 % en el peor desfase).
- Lazo simulado: los 4 gestos de prueba detectados; el motor entra al 10 % de la referencia en 200 ms.

Lo que falta medir en la tarjeta: ciclos reales de forward/BPTT (los reporta `T` y la telemetría) y
la exactitud con gestos reales (`capturar.py` + `evaluar.py`).