/*==================================================================================================
* Modulo lstm: forward, BPTT y forward incremental. Ver lstm.h.
==================================================================================================*/
#include "lstm.h"
#include <math.h>
#include <string.h>

#define LSTM_Z_CLAMP  80.0f      /* expf(88) desborda en float32 */
#define H             LSTM_N_H
#define G             LSTM_N_G

static inline float Sigm(float z)
{
    if (z >  LSTM_Z_CLAMP) { z =  LSTM_Z_CLAMP; }
    if (z < -LSTM_Z_CLAMP) { z = -LSTM_Z_CLAMP; }
    return 1.0f / (1.0f + expf(-z));
}

/* Un paso de la celda. Lo comparten el forward de entrenamiento y el
 * incremental, asi ambos dan exactamente el mismo resultado.             */
static void CellStep(const LstmParams *n, const float *x, const float *hp, const float *cp,
                     float *gate, float *c, float *tc, float *h)
{
    float z[G];
    for (uint8_t r = 0U; r < G; r++)
    {
        float sx = 0.0f, sh = 0.0f;
        for (uint8_t i = 0U; i < LSTM_N_IN; i++) { sx += n->Wx[(r * LSTM_N_IN) + i] * x[i]; }
        for (uint8_t j = 0U; j < H; j++)         { sh += n->Wh[(r * H) + j] * hp[j]; }
        z[r] = (n->b[r] + sx) + sh;
    }
    for (uint8_t j = 0U; j < H; j++)
    {
        float ig = Sigm(z[j]);
        float fg = Sigm(z[H + j]);
        float gg = tanhf(z[(2U * H) + j]);
        float og = Sigm(z[(3U * H) + j]);
        float cn = (fg * cp[j]) + (ig * gg);
        float t  = tanhf(cn);
        gate[j] = ig; gate[H + j] = fg; gate[(2U * H) + j] = gg; gate[(3U * H) + j] = og;
        c[j] = cn; tc[j] = t; h[j] = og * t;
    }
}

/* Dense(8, tanh) + Dense(5) + softmax estable */
static void Head(const LstmParams *n, const float *hT, float *a, float *p)
{
    for (uint8_t d = 0U; d < LSTM_N_D; d++)
    {
        float s = 0.0f;
        for (uint8_t j = 0U; j < H; j++) { s += n->Wd[(d * H) + j] * hT[j]; }
        a[d] = tanhf(n->bd[d] + s);
    }
    float z3[LSTM_N_OUT];
    float zmax = -3.4e38f;
    for (uint8_t k = 0U; k < LSTM_N_OUT; k++)
    {
        float s = 0.0f;
        for (uint8_t d = 0U; d < LSTM_N_D; d++) { s += n->Wo[(k * LSTM_N_D) + d] * a[d]; }
        z3[k] = n->bo[k] + s;
        if (z3[k] > zmax) { zmax = z3[k]; }
    }
    float sum = 0.0f;
    for (uint8_t k = 0U; k < LSTM_N_OUT; k++) { p[k] = expf(z3[k] - zmax); sum += p[k]; }
    float inv = 1.0f / sum;
    for (uint8_t k = 0U; k < LSTM_N_OUT; k++) { p[k] *= inv; }
}

void Lstm_LoadFlat(LstmParams *n, const float *flat)
{
    (void)memcpy(n, flat, sizeof(*n));
}

void Lstm_Forward(const LstmParams *n, LstmSeq x, LstmCache *c)
{
    static const float zero[H] = {0.0f};
    for (uint8_t t = 0U; t < LSTM_T; t++)
    {
        const float *hp = (t == 0U) ? zero : c->h[t - 1U];
        const float *cp = (t == 0U) ? zero : c->c[t - 1U];
        CellStep(n, x[t], hp, cp, c->gate[t], c->c[t], c->tc[t], c->h[t]);
    }
    Head(n, c->h[LSTM_T - 1U], c->a, c->p);
}

float Lstm_Loss(const float p[LSTM_N_OUT], uint8_t label)
{
    float s = 0.0f;
    for (uint8_t k = 0U; k < LSTM_N_OUT; k++)
    {
        float e = p[k] - ((k == label) ? 1.0f : 0.0f);
        s += e * e;
    }
    return 0.5f * s;
}

uint8_t Lstm_Argmax(const float p[LSTM_N_OUT])
{
    uint8_t k = 0U;
    for (uint8_t j = 1U; j < LSTM_N_OUT; j++) { if (p[j] > p[k]) { k = j; } }
    return k;
}

void Lstm_Backward(LstmParams *n, LstmSeq x, const LstmCache *c,
                   uint8_t label, float lr, float clip, LstmParams *g)
{
    float dz3[LSTM_N_OUT], dzd[LSTM_N_D], dh[H], dc[H], dz[G], dhn[H];
    (void)memset(g, 0, sizeof(*g));

    /* 1) Salida: SSE a traves del softmax  dz = p .* (gp - <p, gp>) */
    float gp[LSTM_N_OUT], dot = 0.0f;
    for (uint8_t k = 0U; k < LSTM_N_OUT; k++)
    {
        gp[k] = c->p[k] - ((k == label) ? 1.0f : 0.0f);
        dot += c->p[k] * gp[k];
    }
    for (uint8_t k = 0U; k < LSTM_N_OUT; k++)
    {
        dz3[k] = c->p[k] * (gp[k] - dot);
        for (uint8_t d = 0U; d < LSTM_N_D; d++) { g->Wo[(k * LSTM_N_D) + d] = dz3[k] * c->a[d]; }
        g->bo[k] = dz3[k];
    }
    /* 2) Dense(8, tanh) */
    const float *hT = c->h[LSTM_T - 1U];
    for (uint8_t d = 0U; d < LSTM_N_D; d++)
    {
        float s = 0.0f;
        for (uint8_t k = 0U; k < LSTM_N_OUT; k++) { s += n->Wo[(k * LSTM_N_D) + d] * dz3[k]; }
        dzd[d] = s * (1.0f - (c->a[d] * c->a[d]));
        for (uint8_t j = 0U; j < H; j++) { g->Wd[(d * H) + j] = dzd[d] * hT[j]; }
        g->bd[d] = dzd[d];
    }
    for (uint8_t j = 0U; j < H; j++)
    {
        float s = 0.0f;
        for (uint8_t d = 0U; d < LSTM_N_D; d++) { s += n->Wd[(d * H) + j] * dzd[d]; }
        dh[j] = s;
        dc[j] = 0.0f;
    }

    /* 3) BPTT: de t = T-1 hacia atras, acumulando el gradiente de Wx, Wh, b */
    for (int16_t t = (int16_t)LSTM_T - 1; t >= 0; t--)
    {
        const float *gt = c->gate[t];
        const float *tc = c->tc[t];
        for (uint8_t j = 0U; j < H; j++)
        {
            float ig = gt[j], fg = gt[H + j], gg = gt[(2U * H) + j], og = gt[(3U * H) + j];
            float cp = (t > 0) ? c->c[t - 1][j] : 0.0f;
            float d_o = dh[j] * tc[j];
            dc[j] = dc[j] + (dh[j] * og * (1.0f - (tc[j] * tc[j])));
            dz[j]            = dc[j] * gg * ig * (1.0f - ig);
            dz[H + j]        = dc[j] * cp * fg * (1.0f - fg);
            dz[(2U * H) + j] = dc[j] * ig * (1.0f - (gg * gg));
            dz[(3U * H) + j] = d_o * og * (1.0f - og);
        }
        for (uint8_t r = 0U; r < G; r++)
        {
            for (uint8_t i = 0U; i < LSTM_N_IN; i++) { g->Wx[(r * LSTM_N_IN) + i] += dz[r] * x[t][i]; }
            if (t > 0)
            {
                const float *hp = c->h[t - 1];
                for (uint8_t j = 0U; j < H; j++) { g->Wh[(r * H) + j] += dz[r] * hp[j]; }
            }
            g->b[r] += dz[r];
        }
        for (uint8_t j = 0U; j < H; j++)
        {
            float s = 0.0f;
            for (uint8_t r = 0U; r < G; r++) { s += n->Wh[(r * H) + j] * dz[r]; }
            dhn[j] = s;
            dc[j] = dc[j] * gt[H + j];      /* dc_{t-1} = dc_t * f_t */
        }
        (void)memcpy(dh, dhn, sizeof(dh));
    }

    /* 4) Recorte de norma global y actualizacion (gradiente calculado con
     *    los pesos viejos en su totalidad)                                 */
    float *pf = (float *)n;
    const float *gf = (const float *)g;
    float scale = lr;
    if (clip > 0.0f)
    {
        float ss = 0.0f;
        for (uint16_t k = 0U; k < LSTM_N_PARAMS; k++) { ss += gf[k] * gf[k]; }
        float nrm = sqrtf(ss);
        if (nrm > clip) { scale = lr * (clip / nrm); }
    }
    for (uint16_t k = 0U; k < LSTM_N_PARAMS; k++) { pf[k] -= scale * gf[k]; }
}

void Lstm_RunBegin(LstmRun *r)
{
    (void)memset(r, 0, sizeof(*r));
}

uint8_t Lstm_RunSteps(const LstmParams *n, LstmRun *r, LstmSeq x, uint8_t nsteps)
{
    float gate[G], cn[H], tc[H], hn[H];
    while ((nsteps > 0U) && (r->t < LSTM_T))
    {
        CellStep(n, x[r->t], r->h, r->c, gate, cn, tc, hn);
        (void)memcpy(r->h, hn, sizeof(hn));
        (void)memcpy(r->c, cn, sizeof(cn));
        r->t++;
        nsteps--;
    }
    return (r->t >= LSTM_T) ? 1U : 0U;
}

void Lstm_RunHead(const LstmParams *n, const LstmRun *r, float p[LSTM_N_OUT])
{
    float a[LSTM_N_D];
    Head(n, r->h, a, p);
}

float Lstm_GetParam(const LstmParams *n, uint16_t idx)
{
    /* LstmParams es un bloque contiguo de floats (sin relleno) */
    const float *flat = (const float *)n;
    return (idx < LSTM_N_PARAMS) ? flat[idx] : 0.0f;
}

void Lstm_SetParam(LstmParams *n, uint16_t idx, float v)
{
    float *flat = (float *)n;
    if (idx < LSTM_N_PARAMS) { flat[idx] = v; }
}
