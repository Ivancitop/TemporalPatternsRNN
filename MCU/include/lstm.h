/*==================================================================================================
* Project : TemporalPatterns
* Platform : S32K3XX
* Modulo   : lstm  (LSTM(16) -> Dense(8, tanh) -> Dense(5) + softmax, BPTT en el micro)
*
*   x_t (3 potes, t = 0..49, un paso cada 100 ms)
*     -> LSTM 16 unidades (compuertas i, f, g, o; estado inicial h = c = 0)
*     -> Dense 8 neuronas, tanh, sobre el ultimo estado h_T
*     -> Dense 5 lineal + softmax (probabilidad por clase)
*
*   Perdida: suma de errores al cuadrado  E = 1/2 * sum_k (p_k - t_k)^2
*            dE/dz = p .* (g - <p, g>),  g = p - t   (jacobiano del softmax)
*   Entrenamiento: SGD por secuencia, retropropagacion a traves del tiempo
*   (BPTT) completa sobre los 50 pasos, recorte de la norma global del
*   gradiente.
*
*   Ecuaciones por paso:
*     z   = b + Wx x_t + Wh h_{t-1}            (4H filas: i | f | g | o)
*     i = s(z_i), f = s(z_f), g = tanh(z_g), o = s(z_o)
*     c_t = f c_{t-1} + i g ,   h_t = o tanh(c_t)
*
* Evolucion del modulo mlp: mismo esquema Load / Forward / Backward / GetParam
* y mismo volcado plano de parametros. Ademas trae un forward incremental
* (Lstm_Run*) para repartir la inferencia en tiempo real entre varios ticks.
*
* Gemelo de python/lstm_core.py (clase LSTMNet). Codigo C puro para pruebas en PC.
==================================================================================================*/
#ifndef LSTM_H
#define LSTM_H

#include <stdint.h>

#define LSTM_N_IN   3U
#define LSTM_N_H    16U
#define LSTM_N_G    (4U * LSTM_N_H)   /* filas de compuertas */
#define LSTM_N_D    8U
#define LSTM_N_OUT  5U
#define LSTM_T      50U               /* pasos por secuencia */

#define LSTM_N_PARAMS ((LSTM_N_G * LSTM_N_IN) + (LSTM_N_G * LSTM_N_H) + LSTM_N_G + \
                       (LSTM_N_D * LSTM_N_H) + LSTM_N_D +                         \
                       (LSTM_N_OUT * LSTM_N_D) + LSTM_N_OUT)          /* = 1461 */

/* Parametros entrenables, en el mismo orden que el volcado 'W' y que
 * LSTMNet.flat_params(). Matrices fila por neurona: W[r * n_in + i].       */
typedef struct {
    float Wx[LSTM_N_G * LSTM_N_IN];
    float Wh[LSTM_N_G * LSTM_N_H];
    float b [LSTM_N_G];
    float Wd[LSTM_N_D * LSTM_N_H];
    float bd[LSTM_N_D];
    float Wo[LSTM_N_OUT * LSTM_N_D];
    float bo[LSTM_N_OUT];
} LstmParams;

/* Lo que el BPTT necesita de cada paso (~22 KB) */
typedef struct {
    float gate[LSTM_T][LSTM_N_G];     /* i, f, g, o ya activadas */
    float c   [LSTM_T][LSTM_N_H];
    float tc  [LSTM_T][LSTM_N_H];     /* tanh(c)                 */
    float h   [LSTM_T][LSTM_N_H];
    float a   [LSTM_N_D];             /* salida de la Dense(8)   */
    float p   [LSTM_N_OUT];           /* softmax                 */
} LstmCache;

/* Forward incremental para tiempo real: solo h y c (128 B) */
typedef struct {
    float   h[LSTM_N_H];
    float   c[LSTM_N_H];
    uint8_t t;
} LstmRun;

/* Secuencia de entrada (solo lectura en lstm.c; sin const para evitar el aviso de
 * punteros a arreglos calificados de C99) */
typedef float LstmSeq[LSTM_T][LSTM_N_IN];

void    Lstm_LoadFlat(LstmParams *n, const float *flat);
void    Lstm_Forward(const LstmParams *n, LstmSeq x, LstmCache *c);
/* Calcula el gradiente en g (mismo tipo que los parametros), recorta la
 * norma global a clip (clip <= 0 desactiva) y aplica n -= lr * g.          */
void    Lstm_Backward(LstmParams *n, LstmSeq x, const LstmCache *c,
                      uint8_t label, float lr, float clip, LstmParams *g);
float   Lstm_Loss(const float p[LSTM_N_OUT], uint8_t label);
uint8_t Lstm_Argmax(const float p[LSTM_N_OUT]);

void    Lstm_RunBegin(LstmRun *r);
/* Avanza hasta nsteps pasos; devuelve 1 cuando ya se procesaron los T. */
uint8_t Lstm_RunSteps(const LstmParams *n, LstmRun *r, LstmSeq x, uint8_t nsteps);
void    Lstm_RunHead(const LstmParams *n, const LstmRun *r, float p[LSTM_N_OUT]);

/* Acceso plano a los 1461 parametros (volcado / carga por UART) */
float   Lstm_GetParam(const LstmParams *n, uint16_t idx);
void    Lstm_SetParam(LstmParams *n, uint16_t idx, float v);

#endif
