/*==================================================================================================
* Modulo pi_ctrl. Ver pi_ctrl.h.
==================================================================================================*/
#include "pi_ctrl.h"

void Pi_Init(PiCtrl *pi, float kp, float ki, float ts, float umax)
{
    pi->kp = kp; pi->ki = ki; pi->ts = ts; pi->umax = umax;
    Pi_Reset(pi);
}

void Pi_SetGains(PiCtrl *pi, float kp, float ki)
{
    pi->kp = kp;
    pi->ki = ki;
}

void Pi_Reset(PiCtrl *pi)
{
    pi->integ = 0.0f;
    pi->u = 0.0f;
}

float Pi_Step(PiCtrl *pi, float ref, float meas)
{
    float e = ref - meas;
    float u = (pi->kp * e) + pi->integ;

    float us = u;
    if (us >  pi->umax) { us =  pi->umax; }
    if (us < -pi->umax) { us = -pi->umax; }

    /* Integracion condicional: se integra si no hay saturacion, o si el
     * error tiene el signo que saca a la salida de la saturacion.         */
    if ((us == u) || ((u > pi->umax) && (e < 0.0f)) || ((u < -pi->umax) && (e > 0.0f)))
    {
        pi->integ += pi->ki * pi->ts * e;
        /* El integrador por si solo tampoco puede exceder la saturacion */
        if (pi->integ >  pi->umax) { pi->integ =  pi->umax; }
        if (pi->integ < -pi->umax) { pi->integ = -pi->umax; }
    }
    pi->u = us;
    return us;
}
