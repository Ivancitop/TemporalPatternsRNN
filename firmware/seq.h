/*==================================================================================================
* Project : TemporalPatterns
* Platform : S32K3XX
* Modulo   : seq  (secuencia temporal de los 3 potenciometros para la LSTM)
*
* Sustituye a preproc: la red ya no recibe estadisticas de una ventana sino
* la secuencia misma.
*
* Pipeline por canal (tiempo real, un llamado a Seq_PushRaw por tick de 5 ms):
*   crudo ADC (12 b) -> suma de SEQ_DEC = 20 lecturas (100 ms, acumulador entero)
*                    -> codigo = round(1020 * (suma - 20*ADC_MIN) / (20*(ADC_MAX - ADC_MIN)))
*                       (mismo "nivel en milesimas" que el CSV del dataset: los gestos del
*                       dataset van de 0.000 a ~1.020, asi que el tope fisico del pote se
*                       mapea a 1020 y el minimo a 0)
*                    -> ventana circular de LSTM_T = 50 codigos (5 s)
*
* Filtrado: el promedio de 20 lecturas (100 ms) es el pasa bajas (media movil por bloques,
* -3 dB ~ 4.4 Hz, ruido / sqrt(20)). No se agrega otro filtro: un IIR adicional solo
* retrasaria la secuencia respecto a la del dataset.
*
* Entrada a la red:  x = codigo * 0.002f - 1.0f   (nivel 0..1 -> -1..1)
*
* Por la linea serie los codigos viajan como 3 digitos hex por canal, asi la
* PC y el MCU parten exactamente del mismo entero.
*
* Gemelo de python/lstm_core.py (raw_series_to_codes, codes_to_x).
* Codigo C puro (sin RTD) para poder compilarlo en PC y probarlo.
==================================================================================================*/
#ifndef SEQ_H
#define SEQ_H

#include <stdint.h>
#include "lstm.h"

#define SEQ_N_CH       LSTM_N_IN
#define SEQ_DEC        20U        /* 20 x 5 ms = 100 ms por paso de la LSTM */
/* Calibracion: lectura cruda con el pote en el tope inferior / superior.
 * Medir con la telemetria (columnas raw1..raw3) y ajustar. */
#define SEQ_ADC_MIN    11U
#define SEQ_ADC_MAX    2670U
#define SEQ_LVL_MAX    1020U      /* codigo asignado a SEQ_ADC_MAX (nivel 1.02 del dataset) */
#define SEQ_CODE_SCALE 1000U
#define SEQ_CODE_MAX   1100U
#define SEQ_X_GAIN     0.002f
#define SEQ_X_OFF      1.0f

typedef struct {
    uint32_t acc[SEQ_N_CH];
    uint8_t  nAcc;
    uint16_t win[LSTM_T][SEQ_N_CH];   /* circular, codigos */
    uint8_t  wIdx;                    /* siguiente posicion a escribir */
    uint8_t  wCnt;
} SeqState;

void     Seq_Init(SeqState *s);
/* Agrega una lectura cruda. Devuelve 1 cuando se completo un paso nuevo. */
uint8_t  Seq_PushRaw(SeqState *s, const uint16_t raw[SEQ_N_CH]);
uint8_t  Seq_Ready(const SeqState *s);
/* Copia la ventana en orden cronologico (del mas viejo al mas nuevo) */
void     Seq_SnapshotCodes(const SeqState *s, uint16_t out[LSTM_T][SEQ_N_CH]);
void     Seq_SnapshotX(const SeqState *s, LstmSeq x);
float    Seq_CodeToX(uint16_t code);
/* Mayor (max - min) entre los 3 canales sobre la ventana: 0 = potes totalmente quietos */
uint16_t Seq_WindowRange(const SeqState *s);

#endif
