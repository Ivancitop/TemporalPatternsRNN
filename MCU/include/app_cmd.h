/*==================================================================================================
* Project : TemporalPatterns
* Modulo   : app_cmd  (comandos de entrenamiento / pesos / captura, portables)
*
* Separa del main.c la parte del protocolo que no depende del hardware, para
* que el firmware y el gemelo en PC (tests/host_mcu.c) ejecuten exactamente
* el mismo codigo. El main.c solo maneja los comandos de modo (I F O X P S).
*
* Comandos (lineas terminadas en '\n'; floats en 8 hex IEEE-754, codigos de
* nivel en 3 hex por canal):
*   Q,<k0>,<AAABBBCCC...>    carga los pasos k0.. de la secuencia (9 hex/paso) -> K
*   T,<c>                    entrena con la secuencia cargada (c = 0..4)
*   V,<c>                    evalua sin actualizar pesos
*        respuesta: <cls>,<p0>,...,<p4>,<loss>,<cicF>,<cicB>
*   R                        recarga pesos iniciales                       -> OK
*   W                        vuelca los 1461 parametros: W,<idx>,<8 hex> / END
*   L,<idx>,<h0>[,...,<h7>]  escribe hasta 8 parametros desde idx           -> K
*   G,<lr*1e6>,<clip*1e3>    cambia tasa de aprendizaje y recorte           -> OK
*   C                        ventana actual de los potes:
*                            C,<r0>,<r1>,<r2>,<50 pasos x 9 hex>   (N si no esta llena)
*   T, V, R, L y G solo se aceptan en IDLE (E en otro modo).
==================================================================================================*/
#ifndef APP_CMD_H
#define APP_CMD_H

#include <stdint.h>
#include "lstm.h"
#include "seq.h"

#define APP_TX_SIZE   512U
#define APP_LR_DEFAULT    0.10f
#define APP_CLIP_DEFAULT  1.00f

typedef void     (*AppSendFn)(const char *buf, uint32_t n);
typedef uint32_t (*AppCyclesFn)(void);

typedef struct {
    LstmParams     *net;         /* pesos en uso                          */
    const float    *initW;       /* pesos iniciales (comando R)           */
    const SeqState *seq;         /* ventana en tiempo real (comando C)    */
    const uint16_t *rawLast;     /* ultima lectura cruda (comando C)      */
    AppSendFn       send;
    AppCyclesFn     cycles;
    uint8_t         idle;        /* 1 si se permite entrenar / cargar     */
    float           lr;
    float           clip;
    uint8_t         loaded;      /* pasos validos de la secuencia cargada */
} AppCtx;

void    AppCmd_Init(AppCtx *a);
/* Devuelve 1 si la linea era un comando de este modulo (ya respondido). */
uint8_t AppCmd_Handle(AppCtx *a, const char *s);

/* Utilidades de formato compartidas con main.c */
uint8_t AppFmt_U32(uint32_t v, char *buf);
uint8_t AppFmt_I32(int32_t v, char *buf);
uint8_t AppFmt_F32Hex(float f, char *buf);
uint8_t AppParse_Ints(const char *s, int32_t *v, uint8_t nmax);

#endif
