/*==================================================================================================
* Modulo preproc: media movil + ventana + media / desv. estandar / maximo.
* Ver preproc.h para la descripcion del pipeline.
==================================================================================================*/
#include "preproc.h"
#include <math.h>
#include <string.h>

void Feat_Init(FeatState *s)
{
    (void)memset(s, 0, sizeof(*s));
}

void Feat_Push(FeatState *s, const uint16_t raw[FEAT_N_CH], float *filtOut)
{
    for (uint8_t ch = 0U; ch < FEAT_N_CH; ch++)
    {
        /* Media movil con acumulador entero: sin deriva por redondeo */
        s->maSum[ch] -= s->maBuf[ch][s->maIdx];
        s->maBuf[ch][s->maIdx] = raw[ch];
        s->maSum[ch] += raw[ch];
    }
    s->maIdx = (uint8_t)((s->maIdx + 1U) % FEAT_MA_LEN);
    if (s->maCnt < FEAT_MA_LEN) { s->maCnt++; }

    /* La ventana solo recibe muestras cuando la media movil ya es valida,
     * igual que la convolucion 'valid' del modelo en Python.              */
    if (s->maCnt < FEAT_MA_LEN)
    {
        if (filtOut != NULL)
        {
            for (uint8_t ch = 0U; ch < FEAT_N_CH; ch++) { filtOut[ch] = 0.0f; }
        }
        return;
    }

    for (uint8_t ch = 0U; ch < FEAT_N_CH; ch++)
    {
        float f = (float)s->maSum[ch] / ((float)FEAT_MA_LEN * FEAT_ADC_FS);
        s->win[ch][s->wIdx] = f;
        if (filtOut != NULL) { filtOut[ch] = f; }
    }
    s->wIdx = (uint8_t)((s->wIdx + 1U) % FEAT_WIN);
    if (s->wCnt < FEAT_WIN) { s->wCnt++; }
}

uint8_t Feat_Ready(const FeatState *s)
{
    return (s->wCnt >= FEAT_WIN) ? 1U : 0U;
}

void Feat_Compute(const FeatState *s, float out[FEAT_N])
{
    const float invN = 1.0f / (float)FEAT_WIN;

    for (uint8_t ch = 0U; ch < FEAT_N_CH; ch++)
    {
        /* Dos pasadas: primero la media, luego la suma de desviaciones al
         * cuadrado. La forma E[x^2] - media^2 pierde casi todos los digitos
         * en float32 cuando la senal es casi constante (caso tipico de un
         * pote quieto): la prueba de equivalencia daba error de 3e-2 en la
         * desviacion normalizada. Con dos pasadas baja a ~1e-6.           */
        float sum = 0.0f, mx = s->win[ch][0];
        for (uint8_t k = 0U; k < FEAT_WIN; k++)
        {
            float v = s->win[ch][k];
            sum += v;
            if (v > mx) { mx = v; }
        }
        float mean = sum * invN;
        float sq = 0.0f;
        for (uint8_t k = 0U; k < FEAT_WIN; k++)
        {
            float dv = s->win[ch][k] - mean;
            sq += dv * dv;
        }
        float var = sq * invN;
        float stdn = sqrtf(var) / FEAT_STD_REF;
        if (stdn > FEAT_STD_CLIP) { stdn = FEAT_STD_CLIP; }

        out[ch]                    = (2.0f * mean) - 1.0f;
        out[FEAT_N_CH + ch]        = stdn;
        out[(2U * FEAT_N_CH) + ch] = (2.0f * mx) - 1.0f;
    }
}
