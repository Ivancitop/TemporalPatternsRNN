/*==================================================================================================
* host_mcu.c - gemelo del firmware para PC (stdin/stdout en lugar de LPUART6)
*
* Enlaza exactamente los mismos lstm.c, seq.c y app_cmd.c que el S32K312, asi run_mcu.py --host
* y test_equivalencia.py ejercitan el codigo C real sin hardware.
*
* Comandos propios del gemelo (ademas de los de app_cmd.h):
*   A,<r0>,<r1>,<r2>   inyecta una lectura cruda del ADC (un tick de 5 ms), sin respuesta
*   U                  inferencia incremental (Lstm_Run*, 5 pasos por llamada, como en RUN)
*                      sobre la ventana actual -> U,<p0..p4 hex>   (N si no esta llena)
*   Z                  termina
*   I F O X P S        responden OK (no hay motor)
*
* Compilar (desde tests/):
*   gcc -O2 -std=c99 -ffp-contract=off -I../firmware host_mcu.c ../firmware/lstm.c \
*       ../firmware/seq.c ../firmware/app_cmd.c ../firmware/pi_ctrl.c -lm -o host_mcu
==================================================================================================*/
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "lstm.h"
#include "seq.h"
#include "app_cmd.h"
#include "pesos_iniciales.h"

static LstmParams net;
static SeqState   seq;
static AppCtx     app;
static uint16_t   rawLast[SEQ_N_CH];

static void SendOut(const char *buf, uint32_t n)
{
    (void)fwrite(buf, 1, n, stdout);
    (void)fflush(stdout);
}

/* "Ciclos" equivalentes a 120 MHz medidos con el reloj de la PC (solo referencia) */
static uint32_t Cycles(void)
{
    struct timespec ts;
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(((uint64_t)ts.tv_sec * 120000000ULL) + ((uint64_t)ts.tv_nsec * 12ULL / 100ULL));
}

static void CmdInferIncremental(void)
{
    static LstmSeq x;
    static LstmRun r;
    float p[LSTM_N_OUT];
    char  tx[64];
    uint16_t n = 0U;
    if (Seq_Ready(&seq) == 0U) { SendOut("N\n", 2U); return; }
    Seq_SnapshotX(&seq, x);
    Lstm_RunBegin(&r);
    while (Lstm_RunSteps(&net, &r, x, 5U) == 0U) { }
    Lstm_RunHead(&net, &r, p);
    tx[n++] = 'U';
    for (uint8_t k = 0U; k < LSTM_N_OUT; k++) { tx[n++] = ','; n += AppFmt_F32Hex(p[k], &tx[n]); }
    tx[n++] = '\n';
    SendOut(tx, n);
}

int main(void)
{
    char line[1024];
    Lstm_LoadFlat(&net, LSTM_W_INIT);
    Seq_Init(&seq);
    AppCmd_Init(&app);
    app.net = &net; app.initW = LSTM_W_INIT; app.seq = &seq; app.rawLast = rawLast;
    app.send = SendOut; app.cycles = Cycles; app.idle = 1U;

    while (fgets(line, sizeof(line), stdin) != NULL)
    {
        switch (line[0])
        {
            case 'Z': return 0;
            case 'A':
            {
                int32_t v[3];
                if (AppParse_Ints(&line[2], v, 3U) == 3U)
                {
                    for (uint8_t ch = 0U; ch < 3U; ch++) { rawLast[ch] = (uint16_t)v[ch]; }
                    (void)Seq_PushRaw(&seq, rawLast);
                }
                break;
            }
            case 'U': CmdInferIncremental(); break;
            case 'I': case 'F': case 'O': case 'X': case 'P': case 'S':
                SendOut("OK\n", 3U);
                break;
            default:
                if (AppCmd_Handle(&app, line) == 0U) { SendOut("E\n", 2U); }
                break;
        }
    }
    return 0;
}
