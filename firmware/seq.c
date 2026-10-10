/*==================================================================================================
* Modulo seq: promedio por bloques de 100 ms + ventana de 50 pasos. Ver seq.h.
==================================================================================================*/
#include "seq.h"
#include <string.h>

void Seq_Init(SeqState *s)
{
    (void)memset(s, 0, sizeof(*s));
}

uint8_t Seq_PushRaw(SeqState *s, const uint16_t raw[SEQ_N_CH])
{
    for (uint8_t ch = 0U; ch < SEQ_N_CH; ch++) { s->acc[ch] += raw[ch]; }
    s->nAcc++;
    if (s->nAcc < SEQ_DEC) { return 0U; }

    /* Promedio redondeado en milesimas, todo entero: 20 * 4095 * 1000 cabe en 32 b */
    const uint32_t den = SEQ_DEC * SEQ_ADC_FS;
    for (uint8_t ch = 0U; ch < SEQ_N_CH; ch++)
    {
        uint32_t code = ((s->acc[ch] * SEQ_CODE_SCALE) + (den / 2U)) / den;
        if (code > SEQ_CODE_MAX) { code = SEQ_CODE_MAX; }
        s->win[s->wIdx][ch] = (uint16_t)code;
        s->acc[ch] = 0U;
    }
    s->nAcc = 0U;
    s->wIdx = (uint8_t)((s->wIdx + 1U) % LSTM_T);
    if (s->wCnt < LSTM_T) { s->wCnt++; }
    return 1U;
}

uint8_t Seq_Ready(const SeqState *s)
{
    return (s->wCnt >= LSTM_T) ? 1U : 0U;
}

void Seq_SnapshotCodes(const SeqState *s, uint16_t out[LSTM_T][SEQ_N_CH])
{
    /* Con la ventana llena, wIdx apunta al paso mas viejo */
    uint8_t k = (s->wCnt >= LSTM_T) ? s->wIdx : 0U;
    for (uint8_t t = 0U; t < LSTM_T; t++)
    {
        for (uint8_t ch = 0U; ch < SEQ_N_CH; ch++) { out[t][ch] = s->win[k][ch]; }
        k = (uint8_t)((k + 1U) % LSTM_T);
    }
}

float Seq_CodeToX(uint16_t code)
{
    return ((float)code * SEQ_X_GAIN) - SEQ_X_OFF;
}

void Seq_SnapshotX(const SeqState *s, LstmSeq x)
{
    uint8_t k = (s->wCnt >= LSTM_T) ? s->wIdx : 0U;
    for (uint8_t t = 0U; t < LSTM_T; t++)
    {
        for (uint8_t ch = 0U; ch < SEQ_N_CH; ch++) { x[t][ch] = Seq_CodeToX(s->win[k][ch]); }
        k = (uint8_t)((k + 1U) % LSTM_T);
    }
}
