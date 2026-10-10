# Gemelo del firmware en PC

`host_mcu.c` enlaza los mismos `lstm.c`, `seq.c` y `app_cmd.c` que el S32K312 y habla el mismo
protocolo por stdin/stdout. Lo usan `run_mcu.py --host`, `cargar_pesos.py --host`,
`evaluar.py --host` y `test_equivalencia.py`.

```bash
cd tests
gcc -O2 -std=c99 -ffp-contract=off -I../firmware host_mcu.c ../firmware/lstm.c \
    ../firmware/seq.c ../firmware/app_cmd.c ../firmware/pi_ctrl.c -lm -o host_mcu
python test_equivalencia.py
```

En Windows con MinGW el ejecutable `host_mcu.exe` se detecta solo.

`test_equivalencia.py` necesita `python/dataset_balanceado_500.csv` y `firmware/pesos_iniciales.h`
(los genera `python run_pc.py`).
