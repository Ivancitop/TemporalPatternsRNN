/*==================================================================================================
* Project : TemporalPatterns (sin cambios respecto a StaticPatterns)
* Modulo   : pi_ctrl  (PI de velocidad discreto con anti-windup)
*
*   e[k] = r[k] - y[k]
*   u[k] = Kp e[k] + I[k]
*   I[k+1] = I[k] + Ki Ts e[k]     (solo si no empuja mas hacia la saturacion)
*   u saturada en [-UMAX, UMAX]    (u = fraccion de duty con signo)
*
* Anti-windup por integracion condicional (clamping): si la salida esta
* saturada y el error la empujaria mas alla, el integrador se congela.
==================================================================================================*/
#ifndef PI_CTRL_H
#define PI_CTRL_H

typedef struct {
    float kp;
    float ki;
    float ts;      /* periodo de control [s] */
    float umax;    /* saturacion simetrica   */
    float integ;
    float u;       /* ultima salida (saturada) */
} PiCtrl;

void  Pi_Init(PiCtrl *pi, float kp, float ki, float ts, float umax);
void  Pi_SetGains(PiCtrl *pi, float kp, float ki);
void  Pi_Reset(PiCtrl *pi);
float Pi_Step(PiCtrl *pi, float ref, float meas);

#endif
