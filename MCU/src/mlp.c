/*==================================================================================================
* Modulo mlp: forward, backward y utilidades. Ver mlp.h.
==================================================================================================*/
#include "mlp.h"
#include <math.h>
#include <string.h>

#define MLP_Z_CLAMP   80.0f     /* expf(88) desborda en float32 */
#define MLP_LOSS_EPS  1e-12f

/* Sigmoide parametrica por neurona. a = 1 y d = 0 -> salida en (0,1); b desplaza la curva en z
 * (umbral en z = ln(b)/c) y c cambia la pendiente.                         */
static const Zigmoid ACT_H1[MLP_N_H1] = {
    {1.0f, 0.5f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 0.0f},
    {1.0f, 1.5f, 1.0f, 0.0f}, {1.0f, 2.0f, 1.0f, 0.0f},
    {1.0f, 0.5f, 1.5f, 0.0f}, {1.0f, 1.0f, 1.5f, 0.0f},
    {1.0f, 1.5f, 1.5f, 0.0f}, {1.0f, 2.0f, 1.5f, 0.0f} };
static const Zigmoid ACT_H2[MLP_N_H2] = {
    {1.0f, 0.5f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 0.0f},
    {1.0f, 1.5f, 1.0f, 0.0f}, {1.0f, 0.5f, 2.0f, 0.0f},
    {1.0f, 1.0f, 2.0f, 0.0f}, {1.0f, 1.5f, 2.0f, 0.0f} };

/* f(z) = a / (1 + b e^{-cz}) + d */
static inline float Neuron(const Zigmoid *Z, float z)
{
    float arg = -Z->c * z;
    if (arg >  MLP_Z_CLAMP) { arg =  MLP_Z_CLAMP; }
    if (arg < -MLP_Z_CLAMP) { arg = -MLP_Z_CLAMP; }
    return (Z->a / (1.0f + (Z->b * expf(arg)))) + Z->d;
}

/* f'(z) = a c u (1-u), u = (f-d)/a  -> sin expf() en el backward */
static inline float NeuronDerFromA(const Zigmoid *Z, float aOut)
{
    float u = (aOut - Z->d) / Z->a;
    return Z->a * Z->c * u * (1.0f - u);
}

void Mlp_Load(MlpParams *n,
              const float *W1, const float *b1,
              const float *W2, const float *b2,
              const float *W3, const float *b3)
{
    (void)memcpy(n->W1, W1, sizeof(n->W1));
    (void)memcpy(n->b1, b1, sizeof(n->b1));
    (void)memcpy(n->W2, W2, sizeof(n->W2));
    (void)memcpy(n->b2, b2, sizeof(n->b2));
    (void)memcpy(n->W3, W3, sizeof(n->W3));
    (void)memcpy(n->b3, b3, sizeof(n->b3));
}

void Mlp_Forward(const MlpParams *n, const float *x, MlpCache *c)
{
    for (uint8_t j = 0U; j < MLP_N_H1; j++)
    {
        float z = n->b1[j];
        for (uint8_t i = 0U; i < MLP_N_IN; i++) { z += n->W1[(j * MLP_N_IN) + i] * x[i]; }
        c->a1[j] = Neuron(&ACT_H1[j], z);
    }
    for (uint8_t j = 0U; j < MLP_N_H2; j++)
    {
        float z = n->b2[j];
        for (uint8_t i = 0U; i < MLP_N_H1; i++) { z += n->W2[(j * MLP_N_H1) + i] * c->a1[i]; }
        c->a2[j] = Neuron(&ACT_H2[j], z);
    }

    /* Salida lineal + softmax estable (se resta el maximo antes de expf) */
    float z3[MLP_N_OUT];
    float zmax = -3.4e38f;
    for (uint8_t j = 0U; j < MLP_N_OUT; j++)
    {
        float z = n->b3[j];
        for (uint8_t i = 0U; i < MLP_N_H2; i++) { z += n->W3[(j * MLP_N_H2) + i] * c->a2[i]; }
        z3[j] = z;
        if (z > zmax) { zmax = z; }
    }
    float sum = 0.0f;
    for (uint8_t j = 0U; j < MLP_N_OUT; j++) { c->p[j] = expf(z3[j] - zmax); sum += c->p[j]; }
    float inv = 1.0f / sum;
    for (uint8_t j = 0U; j < MLP_N_OUT; j++) { c->p[j] *= inv; }
}

float Mlp_Loss(const MlpCache *c, uint8_t label)
{
    float p = c->p[label];
    if (p < MLP_LOSS_EPS) { p = MLP_LOSS_EPS; }
    return -logf(p);
}

uint8_t Mlp_Argmax(const MlpCache *c)
{
    uint8_t k = 0U;
    for (uint8_t j = 1U; j < MLP_N_OUT; j++) { if (c->p[j] > c->p[k]) { k = j; } }
    return k;
}

void Mlp_Backward(MlpParams *n, const float *x, const MlpCache *c,
                  uint8_t label, float lr)
{
    float d3[MLP_N_OUT];
    float d2[MLP_N_H2];
    float d1[MLP_N_H1];

    /* Softmax + entropia cruzada: dL/dz3 = p - onehot (sin derivada extra) */
    for (uint8_t j = 0U; j < MLP_N_OUT; j++)
    {
        d3[j] = c->p[j] - ((j == label) ? 1.0f : 0.0f);
    }
    /* Capa 2: error propagado a traves de W3 */
    for (uint8_t i = 0U; i < MLP_N_H2; i++)
    {
        float s = 0.0f;
        for (uint8_t j = 0U; j < MLP_N_OUT; j++) { s += n->W3[(j * MLP_N_H2) + i] * d3[j]; }
        d2[i] = s * NeuronDerFromA(&ACT_H2[i], c->a2[i]);
    }
    /* Capa 1: error propagado a traves de W2 */
    for (uint8_t i = 0U; i < MLP_N_H1; i++)
    {
        float s = 0.0f;
        for (uint8_t j = 0U; j < MLP_N_H2; j++) { s += n->W2[(j * MLP_N_H1) + i] * d2[j]; }
        d1[i] = s * NeuronDerFromA(&ACT_H1[i], c->a1[i]);
    }

    /* Actualizacion (todos los delta se calcularon con los pesos viejos) */
    for (uint8_t j = 0U; j < MLP_N_OUT; j++)
    {
        for (uint8_t i = 0U; i < MLP_N_H2; i++) { n->W3[(j * MLP_N_H2) + i] -= lr * d3[j] * c->a2[i]; }
        n->b3[j] -= lr * d3[j];
    }
    for (uint8_t j = 0U; j < MLP_N_H2; j++)
    {
        for (uint8_t i = 0U; i < MLP_N_H1; i++) { n->W2[(j * MLP_N_H1) + i] -= lr * d2[j] * c->a1[i]; }
        n->b2[j] -= lr * d2[j];
    }
    for (uint8_t j = 0U; j < MLP_N_H1; j++)
    {
        for (uint8_t i = 0U; i < MLP_N_IN; i++) { n->W1[(j * MLP_N_IN) + i] -= lr * d1[j] * x[i]; }
        n->b1[j] -= lr * d1[j];
    }
}

float Mlp_GetParam(const MlpParams *n, uint16_t idx)
{
    /* MlpParams es un bloque contiguo de floats (sin relleno) */
    const float *flat = (const float *)n;
    return (idx < MLP_N_PARAMS) ? flat[idx] : 0.0f;
}
