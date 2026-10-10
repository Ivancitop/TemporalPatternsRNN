/*==================================================================================================
* Project : TemporalPatterns  (clasificacion de patrones temporales con LSTM + control PI de
*                             velocidad de un motor DC)
* Platform : S32K312 (Cortex-M7 @ 120 MHz), RTD 7.0.0
* Author   : Ivan Delgado Ramos
* Date     : 09/10/26
*
* Basado en StaticPatterns (MLP 9-8-6-4): se conservan el hardware, el planificador por DWT, el
* protocolo UART, el PI, la telemetria y la maquina de estados. Cambia el clasificador:
*
*   - 3 potenciometros por ADC0 (12 b) -> promedio de 100 ms -> secuencia de 50 pasos (5 s)
*   - LSTM(16) -> Dense(8, tanh) -> Dense(5) + softmax, entrenada en el micro con BPTT y
*     perdida de suma de errores al cuadrado (ver lstm.h)
*   - 5 clases:  0 default (clase 5 del enunciado, ref 0)   1 alta CW   2 nominal CW
*                3 alta CCW                                 4 nominal CCW
*   - encoder en cuadratura (x4) por EIRQ 3 / EIRQ 8   -> velocidad del motor
*   - 2 PWM eMIOS1 ch9 / ch10 -> LPWM y RPWM del driver BTS7960
*
* Maquina de estados parseada por UART:
*   IDLE      : motor apagado. Entrenamiento (Q/T/V), reset (R), volcado (W) y carga (L) de pesos,
*               captura de la ventana real de los potes (C).
*   RUN       : inferencia en tiempo real sobre la ventana de los potes -> referencia del PI.
*               Envia telemetria cada 10 ms.
*   FIXEDREF  : PI con referencia fija enviada por la PC (pruebas de escalon).
*   OPENLOOP  : duty fijo sin lazo (identificacion de la planta).
*
* Determinismo: tick de 5 ms con el contador de ciclos DWT (sin interrupcion). Cada tick muestrea
* el ADC y alimenta la secuencia (un paso nuevo cada 20 ticks = 100 ms). El PI corre cada 2 ticks
* (10 ms). La inferencia de la LSTM (50 pasos, ~ms) NO cabe holgada en un tick, asi que se reparte:
* con cada paso nuevo se toma una copia de la ventana y se procesan LSTM_STEPS_PER_TICK pasos por
* tick; a los 10 ticks (50 ms) se evalua la cabeza densa y se clasifica. Una clasificacion cada
* 100 ms, sin perder ticks.
*
* Protocolo UART (115200 8N1, lineas terminadas en '\n'):
*   comandos de entrenamiento y pesos: ver app_cmd.h  (Q T V R W L G C)
*   I                     entra a RUN                         -> OK + telemetria
*   F,<rpm>               entra a FIXEDREF con esa referencia -> OK + telemetria
*   O,<permil>            entra a OPENLOOP con duty +-1000    -> OK + telemetria
*   X                     vuelve a IDLE (motor apagado)       -> OK
*   P,<kp*1e6>,<ki*1e6>   cambia ganancias del PI             -> OK
*   S,<alta>,<nominal>    cambia las referencias en rpm       -> OK
*   Telemetria: D,<t_ms>,<clsRaw>,<cls>,<ref>,<rpm*10>,<u*1000>,<r0>,<r1>,<r2>,
*               <pmax*1000>,<carga*1000>,<cicF>
==================================================================================================*/

#ifdef __cplusplus
extern "C"{
#endif
/* Descomentar cuando exista pesos_entrenados.h generado por run_mcu.py: el binario arranca con la
 * red entrenada y en modo RUN. Sin recompilar, cargar_pesos.py hace lo mismo por UART.          */
/* #define USE_PRETRAINED */

#include "Clock_Ip.h"
#include "Siul2_Port_Ip.h"
#include "Emios_Mcl_Ip.h"
#include "Emios_Pwm_Ip.h"
#include "Adc_Sar_Ip.h"
#include "IntCtrl_Ip.h"
#include "Siul2_Icu_Ip.h"
#include "Siul2_Dio_Ip.h"
#include "Lpuart_Uart_Ip.h"
#include "Lpuart_Uart_Ip_Irq.h"
#include <string.h>
#include <math.h>

#include "seq.h"              /* secuencia de los potes (ventana de 50 pasos) */
#include "lstm.h"             /* red LSTM                                     */
#include "app_cmd.h"          /* comandos de entrenamiento / pesos / captura  */
#include "pi_ctrl.h"          /* control PI                                   */
#include "pesos_iniciales.h"  /* LSTM_W_INIT                                  */
#ifdef USE_PRETRAINED
#include "pesos_entrenados.h" /* LSTM_W_TRAINED                               */
#endif

/*======================================= Configuracion ==========================================*/
/* --- Instancias y canales de perifericos --- */
#define UART_INSTANCE        6U
#define UART_TIMEOUT_US      100000U
#define BUFFER_SIZE          256U    /* una trama Q ocupa ~232 B */
#define ADC_SAR_USED_CH      0U
#define ADC_SAR_USED_CH2     1U
#define ADC_SAR_USED_CH3     2U
#define ADC_WAIT_MAX         100000U
#define SIULINS              0U
#define PWM_INSTANCE         1U
#define PWM_CH_CW            9U
#define PWM_CH_CCW           10U
#define PWM_PERIOD           65534U

/* --- DWT (contador de ciclos) --- */
#define DWT_LAR     (*(volatile uint32 *)0xE0001FB0UL)
#define DWT_CTRL    (*(volatile uint32 *)0xE0001000UL)
#define DWT_CYCCNT  (*(volatile uint32 *)0xE0001004UL)
#define CORE_DEMCR  (*(volatile uint32 *)0xE000EDFCUL)
#define CORE_MHZ    120U

/* --- Temporizacion --- */
#define TICK_MS              5U
#define TICK_CYC             (CORE_MHZ * 1000U * TICK_MS)
#define CTRL_DIV             2U      /* PI cada 10 ms                               */
#define LSTM_STEPS_PER_TICK  5U      /* 50 pasos / 5 = 10 ticks = 50 ms por inferencia */
#define LOAD_WIN_TICKS       200U    /* ventana de carga de CPU: 1 s                */
#define TS_CTRL              ((float32)(TICK_MS * CTRL_DIV) * 1e-3f)

/* --- Encoder y motor --- */
#define ENC_CPR              3332.0f /* cuentas por vuelta del eje de salida en x4  */
#define ENC_SIGN             1       /* -1 si CW da cuentas negativas               */
#define RPM_LPF_ALPHA        0.5f
#define U_DEADZONE           0.0f

/* --- Control --- */
#define PI_KP_DEFAULT        0.0040f
#define PI_KI_DEFAULT        0.0500f
#define PI_UMAX              1.0f
#define REF_RPM_ALTA         150.0f  /* clases 1 y 3 */
#define REF_RPM_NOMINAL      90.0f   /* clases 2 y 4 */
#define REF_SLEW_RPM_S       0.0f    /* rampa de referencia (0 = escalon)           */

/* --- Clasificador --- */
#define CONF_THRESHOLD       0.70f   /* p_max menor -> clase default                */
#define DEBOUNCE_N           3U      /* clasificaciones iguales (x 100 ms) para conmutar */
#define REF_LATCH            0       /* 1: un gesto fija la referencia hasta el siguiente
                                        gesto (default no detiene el motor; X si)  */
#define CLS_DEFAULT          0U
#define CLS_ALTA_CW          1U
#define CLS_NOM_CW           2U
#define CLS_ALTA_CCW         3U
#define CLS_NOM_CCW          4U

typedef enum { MODE_IDLE = 0, MODE_RUN, MODE_FIXEDREF, MODE_OPENLOOP } AppMode;

/*======================================= Globales ===============================================*/
/* ADC */
volatile boolean notif_triggered = FALSE;
volatile uint16  data;
volatile uint16  data2;
volatile uint16  data3;

/* Encoder */
volatile sint32  count = 0;
volatile uint32  duty = 0;
/* UART */
volatile uint8   u8BufferIdx = 0U;
volatile uint8   au8Buffer[BUFFER_SIZE];
volatile uint8   bRxFlag = 0U;
volatile uint8   bRxRearm = 0U;
static char      tlmBuf[128];     /* telemetria (envio asincrono)                */
volatile uint32  uartStat = 0U;
volatile uint32  uartErrCnt = 0U;
volatile Lpuart_Uart_Ip_StatusType uartErr;

/* Aplicacion */
static SeqState   seq;
static LstmParams net;
static AppCtx     app;
static LstmSeq    xRun;               /* copia de la ventana para la inferencia */
static LstmRun    run;
static uint8      inferBusy = 0U;
static PiCtrl     pi;
static AppMode    mode = MODE_IDLE;
static uint16     rawLast[SEQ_N_CH];
static uint32     tickCount = 0U;
static sint32     encPrev = 0;
static float32    rpmFilt = 0.0f;
static float32    refAlta = REF_RPM_ALTA;
static float32    refNom = REF_RPM_NOMINAL;
static float32    refTarget = 0.0f;
static float32    refApplied = 0.0f;
static float32    uOut = 0.0f;
static float32    uOpenLoop = 0.0f;
static uint8      clsRaw = CLS_DEFAULT, clsCand = CLS_DEFAULT, clsActive = CLS_DEFAULT;
static uint8      candCount = 0U;
static float32    pMaxLast = 0.0f;
static uint32     cycFwdAcc = 0U, cycFwdLast = 0U;
static uint32     busyCyc = 0U, loadPm = 0U, overruns = 0U, tlmDrops = 0U;

/*======================================= Prototipos =============================================*/
void AdcEndOfChainNotif(void);
void EncoderANotify(void);
void EncoderBNotify(void);
void Uart_Callback(const uint8 HwInstance, const Lpuart_Uart_Ip_EventType Event, const void *UserData);

static void    InitPeripherals(void);
static void    ControlTick(void);
static void    InferenceTick(uint8 newStep);
static void    Classify(const float32 p[LSTM_N_OUT]);
static void    UpdateReference(void);
static void    MotorApply(float32 u);
static void    HandleCommand(void);
static void    SendTelemetry(void);
static void    UartSendLine(const char *buf, uint32 n);
static void    UartSendLineCb(const char *buf, uint32_t n);
static uint32_t CyclesCb(void);
static void    UartRxArm(void);
static inline void   DWT_Init(void);
static inline uint32 DWT_Cycles(void) { return DWT_CYCCNT; }

/*======================================= Callbacks / ISR ========================================*/
void AdcEndOfChainNotif(void)
{
    data  = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_USED_CH);
    data2 = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_USED_CH2);
    data3 = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_USED_CH3);
    notif_triggered = TRUE;
}

/* Conteo de encoder para cuadratura x4 */
void EncoderANotify(void)
{
    if (Siul2_Dio_Ip_ReadPin(chA_PORT, chA_PIN) == Siul2_Dio_Ip_ReadPin(chB_PORT, chB_PIN)) { count--; }
    else { count++; }
}
void EncoderBNotify(void)
{
    if (Siul2_Dio_Ip_ReadPin(chA_PORT, chA_PIN) == Siul2_Dio_Ip_ReadPin(chB_PORT, chB_PIN)) { count++; }
    else { count--; }
}

/* Recepcion byte a byte hasta '\n'. Al llegar el fin de linea no se entrega
 * un nuevo buffer: la recepcion termina y main() la rearma tras procesar.  */
void Uart_Callback(const uint8 HwInstance, const Lpuart_Uart_Ip_EventType Event, const void *UserData)
{
    switch (Event)
    {
        case LPUART_UART_IP_EVENT_RX_FULL:
            if ((au8Buffer[u8BufferIdx] != '\n') && (u8BufferIdx < (BUFFER_SIZE - 2U)))
            {
                u8BufferIdx++;
                Lpuart_Uart_Ip_SetRxBuffer(UART_INSTANCE, (uint8 *)&au8Buffer[u8BufferIdx], 1U);
            }
            else
            {
                bRxFlag = 1U;
            }
            break;
        case LPUART_UART_IP_EVENT_ERROR:
        {
            uint32 rem;
            uartStat = IP_LPUART_6->STAT;   /* bit19 OR, bit18 NF, bit17 FE, bit16 PF */
            uartErrCnt++;
            uartErr = Lpuart_Uart_Ip_GetReceiveStatus(UART_INSTANCE, &rem);
            bRxRearm = 1U;
            break;
        }
        default:
            break;
    }
    (void)UserData;
    (void)HwInstance;
}

/*======================================= main ===================================================*/
int main(void)
{
    InitPeripherals();
    DWT_Init();

    Seq_Init(&seq);
    Pi_Init(&pi, PI_KP_DEFAULT, PI_KI_DEFAULT, TS_CTRL, PI_UMAX);

    AppCmd_Init(&app);
    app.net = &net;
    app.initW = LSTM_W_INIT;
    app.seq = &seq;
    app.rawLast = rawLast;
    app.send = UartSendLineCb;
    app.cycles = CyclesCb;

#ifdef USE_PRETRAINED
    /* Binario autonomo: arranca con la red ya entrenada y en modo RUN */
    Lstm_LoadFlat(&net, LSTM_W_TRAINED);
    mode = MODE_RUN;
#else
    Lstm_LoadFlat(&net, LSTM_W_INIT);
#endif
    MotorApply(0.0f);
    UartRxArm();

    uint32 tNext = DWT_Cycles() + TICK_CYC;
    for (;;)
    {
        /* Planificador: resta sin signo -> robusto al desborde de CYCCNT */
        if ((sint32)(DWT_Cycles() - tNext) >= 0)
        {
            tNext += TICK_CYC;
            if ((sint32)(DWT_Cycles() - tNext) >= 0)
            {
                overruns++;                      /* se perdio al menos un tick (p. ej. durante T) */
                tNext = DWT_Cycles() + TICK_CYC;
            }
            ControlTick();
        }

        if (bRxFlag == 1U)
        {
            bRxFlag = 0U;
            uint32 c0 = DWT_Cycles();
            HandleCommand();
            busyCyc += DWT_Cycles() - c0;
            UartRxArm();
        }
        else if (bRxRearm == 1U)
        {
            bRxRearm = 0U;
            UartRxArm();
        }
        else
        {
            /* nada */
        }
    }
    return 0;
}

/*======================================= Inicializacion =========================================*/
static void InitPeripherals(void)
{
    Clock_Ip_Init(Clock_Ip_aClockConfig);
    Clock_Ip_InitClock(Clock_Ip_aClockConfig);
    while (CLOCK_IP_PLL_LOCKED != Clock_Ip_GetPllStatus()) { __asm("nop"); }
    Clock_Ip_DistributePll();

    Siul2_Port_Ip_Init(NUM_OF_CONFIGURED_PINS_PortContainer_0_BOARD_InitPeripherals,
                       g_pin_mux_InitConfigArr_PortContainer_0_BOARD_InitPeripherals);

    /* PWM */
    Emios_Mcl_Ip_Init(PWM_INSTANCE, &Emios_Mcl_Ip_Sa_1_Config);
    Emios_Mcl_Ip_EnableChannel(PWM_INSTANCE, 23U);
    Emios_Pwm_Ip_InitChannel(PWM_INSTANCE, &Emios_Pwm_Ip_Sa_I1_Ch9);
    Emios_Pwm_Ip_InitChannel(PWM_INSTANCE, &Emios_Pwm_Ip_Sa_I1_Ch10);

    /* Interrupciones: encoder (prio 0) > ADC (prio 1) > UART (prio 2) */
    IntCtrl_Ip_Init(&IntCtrlConfig_0);
    IntCtrl_Ip_EnableIrq(SIUL_0_IRQn);
    IntCtrl_Ip_EnableIrq(SIUL_1_IRQn);
    IntCtrl_Ip_EnableIrq(LPUART6_IRQn);

    /* Encoder */
    Siul2_Icu_Ip_Init(SIULINS, &Siul2_Icu_Ip_0_Config_PB);
    Siul2_Icu_Ip_EnableInterrupt(0, 3);
    Siul2_Icu_Ip_EnableInterrupt(0, 8);
    Siul2_Icu_Ip_EnableNotification(0, 3);
    Siul2_Icu_Ip_EnableNotification(0, 8);

    /* ADC con calibracion */
    Adc_Sar_Ip_Init(ADCHWUNIT_0_INSTANCE, &AdcHwUnit_0);
    IntCtrl_Ip_EnableIrq(ADC0_IRQn);
    for (uint8 k = 0U; k <= 5U; k++) { (void)Adc_Sar_Ip_DoCalibration(ADCHWUNIT_0_INSTANCE); }
    Adc_Sar_Ip_EnableNotifications(ADCHWUNIT_0_INSTANCE, ADC_SAR_IP_NOTIF_FLAG_NORMAL_ENDCHAIN);

    /* UART */
    Lpuart_Uart_Ip_Init(UART_INSTANCE, &Lpuart_Uart_Ip_xHwConfigPB_6);
}

static inline void DWT_Init(void)
{
    CORE_DEMCR |= (1UL << 24);      /* TRCENA */
    DWT_LAR     = 0xC5ACCE55UL;
    DWT_CYCCNT  = 0U;
    DWT_CTRL   |= 1UL;              /* CYCCNTENA */
}

/*======================================= Tick de control (5 ms) =================================*/
static void ControlTick(void)
{
    uint32 c0 = DWT_Cycles();
    tickCount++;

    /* 1) Muestreo de los 3 potes */
    notif_triggered = FALSE;
    (void)Adc_Sar_Ip_StartConversion(ADCHWUNIT_0_INSTANCE, ADC_SAR_IP_CONV_CHAIN_NORMAL);
    uint32 to = ADC_WAIT_MAX;
    while ((notif_triggered != TRUE) && (to > 0U)) { to--; }
    rawLast[0] = data; rawLast[1] = data2; rawLast[2] = data3;

    /* 2) Secuencia: promedio de 100 ms y ventana de 50 pasos */
    uint8 newStep = Seq_PushRaw(&seq, rawLast);

    /* 3) Inferencia repartida en ticks (solo fuera de IDLE) */
    if (mode != MODE_IDLE) { InferenceTick(newStep); }

    /* 4) Velocidad + PI cada 10 ms */
    if ((tickCount % CTRL_DIV) == 0U)
    {
        sint32 enc = count;
        sint32 dEnc = enc - encPrev;
        encPrev = enc;
        float32 rpm = ((float32)(ENC_SIGN * dEnc) * 60.0f) / (ENC_CPR * TS_CTRL);
        rpmFilt += RPM_LPF_ALPHA * (rpm - rpmFilt);

        switch (mode)
        {
            case MODE_RUN:
                UpdateReference();
                if (clsActive == CLS_DEFAULT)
                {
                    Pi_Reset(&pi);               /* estado seguro */
                    refApplied = 0.0f;
                    uOut = 0.0f;
                }
                else
                {
                    uOut = Pi_Step(&pi, refApplied, rpmFilt);
                }
                break;
            case MODE_FIXEDREF:
                if (REF_SLEW_RPM_S > 0.0f)
                {
                    float32 step = REF_SLEW_RPM_S * TS_CTRL;
                    float32 d = refTarget - refApplied;
                    refApplied += (d > step) ? step : ((d < -step) ? -step : d);
                }
                else { refApplied = refTarget; }
                uOut = Pi_Step(&pi, refApplied, rpmFilt);
                break;
            case MODE_OPENLOOP:
                uOut = uOpenLoop;
                break;
            default:
                uOut = 0.0f;
                break;
        }
        MotorApply(uOut);

        if (mode != MODE_IDLE) { SendTelemetry(); }
    }

    /* 5) Carga de CPU */
    busyCyc += DWT_Cycles() - c0;
    if ((tickCount % LOAD_WIN_TICKS) == 0U)
    {
        loadPm = (uint32)(((uint64)busyCyc * 1000ULL) / ((uint64)LOAD_WIN_TICKS * TICK_CYC));
        busyCyc = 0U;
    }
}

/* Con cada paso nuevo (100 ms) se congela la ventana y se lanza una inferencia; cada tick avanza
 * LSTM_STEPS_PER_TICK pasos de la celda. La red no cambia en RUN (entrenar exige IDLE).        */
static void InferenceTick(uint8 newStep)
{
    if ((newStep == 1U) && (inferBusy == 0U) && (Seq_Ready(&seq) == 1U))
    {
        Seq_SnapshotX(&seq, xRun);
        Lstm_RunBegin(&run);
        cycFwdAcc = 0U;
        inferBusy = 1U;
    }
    if (inferBusy == 1U)
    {
        uint32 c0 = DWT_Cycles();
        float32 p[LSTM_N_OUT];
        uint8 done = Lstm_RunSteps(&net, &run, xRun, LSTM_STEPS_PER_TICK);
        if (done == 1U) { Lstm_RunHead(&net, &run, p); }
        cycFwdAcc += DWT_Cycles() - c0;
        if (done == 1U)
        {
            cycFwdLast = cycFwdAcc;
            inferBusy = 0U;
            Classify(p);
        }
    }
}

static void Classify(const float32 p[LSTM_N_OUT])
{
    uint8 k = Lstm_Argmax(p);
    pMaxLast = p[k];
    clsRaw = (pMaxLast >= CONF_THRESHOLD) ? k : CLS_DEFAULT;

    /* Antirrebote: la clase activa solo cambia tras DEBOUNCE_N votos iguales */
    if (clsRaw == clsCand) { if (candCount < 255U) { candCount++; } }
    else { clsCand = clsRaw; candCount = 1U; }
    if ((candCount >= DEBOUNCE_N) && (clsActive != clsCand))
    {
        if ((REF_LATCH != 0) && (clsCand == CLS_DEFAULT)) { return; }   /* se mantiene el gesto */
        clsActive = clsCand;
        if (clsActive == CLS_DEFAULT) { Pi_Reset(&pi); }
    }
}

static void UpdateReference(void)
{
    switch (clsActive)
    {
        case CLS_ALTA_CW:  refTarget =  refAlta; break;
        case CLS_NOM_CW:   refTarget =  refNom;  break;
        case CLS_ALTA_CCW: refTarget = -refAlta; break;
        case CLS_NOM_CCW:  refTarget = -refNom;  break;
        default:           refTarget =  0.0f;    break;
    }
    if (REF_SLEW_RPM_S > 0.0f)
    {
        float32 step = REF_SLEW_RPM_S * TS_CTRL;
        float32 d = refTarget - refApplied;
        refApplied += (d > step) ? step : ((d < -step) ? -step : d);
    }
    else
    {
        refApplied = refTarget;
    }
}

/* u en [-1, 1]: signo = sentido, magnitud = duty. Puente H con dos PWM:
 * CW  -> CH9 = PWM, CH10 = 0 ;  CCW -> CH9 = 0, CH10 = PWM                  */
static void MotorApply(float32 u)
{
    float32 m = fabsf(u);
    if (m > 1.0f) { m = 1.0f; }
    if ((m > 0.0f) && (U_DEADZONE > 0.0f)) { m = U_DEADZONE + ((1.0f - U_DEADZONE) * m); }
    duty = (uint32)(m * (float32)PWM_PERIOD);

    if (u >= 0.0f)
    {
        (void)Emios_Pwm_Ip_SetDutyCycle(PWM_INSTANCE, PWM_CH_CCW, 0U);
        (void)Emios_Pwm_Ip_SetDutyCycle(PWM_INSTANCE, PWM_CH_CW, duty);
    }
    else
    {
        (void)Emios_Pwm_Ip_SetDutyCycle(PWM_INSTANCE, PWM_CH_CW, 0U);
        (void)Emios_Pwm_Ip_SetDutyCycle(PWM_INSTANCE, PWM_CH_CCW, duty);
    }
}

/*======================================= Comandos UART ==========================================*/
static void ResetClassifier(void)
{
    clsRaw = CLS_DEFAULT; clsCand = CLS_DEFAULT; clsActive = CLS_DEFAULT; candCount = 0U;
    inferBusy = 0U;
    refApplied = 0.0f;
}

static void HandleCommand(void)
{
    au8Buffer[u8BufferIdx] = 0U;               /* termina la cadena en '\n' */
    const char *s = (const char *)au8Buffer;

    /* Entrenamiento, pesos y captura: modulo portable compartido con el gemelo de PC */
    app.idle = (mode == MODE_IDLE) ? 1U : 0U;
    if (AppCmd_Handle(&app, s) == 1U) { return; }

    switch (s[0])
    {
        case 'I':
            Pi_Reset(&pi);
            ResetClassifier();
            mode = MODE_RUN;
            UartSendLine("OK\n", 3U);
            break;
        case 'F':
        {
            int32_t v[1];  /* stdint: tipo de AppParse_Ints */
            if (AppParse_Ints(&s[2], v, 1U) != 1U) { UartSendLine("E\n", 2U); break; }
            Pi_Reset(&pi);
            refApplied = 0.0f;
            refTarget = (float32)v[0];
            mode = MODE_FIXEDREF;
            UartSendLine("OK\n", 3U);
            break;
        }
        case 'O':
        {
            int32_t v[1];  /* stdint: tipo de AppParse_Ints */
            if (AppParse_Ints(&s[2], v, 1U) != 1U) { UartSendLine("E\n", 2U); break; }
            if (v[0] > 1000) { v[0] = 1000; }
            if (v[0] < -1000) { v[0] = -1000; }
            uOpenLoop = (float32)v[0] * 1e-3f;
            mode = MODE_OPENLOOP;
            UartSendLine("OK\n", 3U);
            break;
        }
        case 'X':
            mode = MODE_IDLE;
            Pi_Reset(&pi);
            ResetClassifier();
            uOut = 0.0f;
            MotorApply(0.0f);
            UartSendLine("OK\n", 3U);
            break;
        case 'P':
        {
            int32_t v[2];
            if (AppParse_Ints(&s[2], v, 2U) != 2U) { UartSendLine("E\n", 2U); break; }
            Pi_SetGains(&pi, (float32)v[0] * 1e-6f, (float32)v[1] * 1e-6f);
            UartSendLine("OK\n", 3U);
            break;
        }
        case 'S':
        {
            int32_t v[2];
            if ((AppParse_Ints(&s[2], v, 2U) != 2U) || (v[0] < 0) || (v[1] < 0))
            {
                UartSendLine("E\n", 2U);
                break;
            }
            refAlta = (float32)v[0];
            refNom = (float32)v[1];
            UartSendLine("OK\n", 3U);
            break;
        }
        default:
            UartSendLine("E\n", 2U);
            break;
    }
}

/* Telemetria no bloqueante: si el envio anterior no termino, se descarta */
static void SendTelemetry(void)
{
    uint32 rem = 0U;
    if (Lpuart_Uart_Ip_GetTransmitStatus(UART_INSTANCE, &rem) == LPUART_UART_IP_STATUS_BUSY)
    {
        tlmDrops++;
        return;
    }
    uint8 n = 0U;
    tlmBuf[n++] = 'D'; tlmBuf[n++] = ',';
    n += AppFmt_U32(tickCount * TICK_MS, &tlmBuf[n]);                     tlmBuf[n++] = ',';
    tlmBuf[n++] = (char)('0' + clsRaw);                                   tlmBuf[n++] = ',';
    tlmBuf[n++] = (char)('0' + clsActive);                                tlmBuf[n++] = ',';
    n += AppFmt_I32((sint32)refApplied, &tlmBuf[n]);                      tlmBuf[n++] = ',';
    n += AppFmt_I32((sint32)lroundf(rpmFilt * 10.0f), &tlmBuf[n]);        tlmBuf[n++] = ',';
    n += AppFmt_I32((sint32)lroundf(uOut * 1000.0f), &tlmBuf[n]);         tlmBuf[n++] = ',';
    for (uint8 ch = 0U; ch < SEQ_N_CH; ch++)
    {
        n += AppFmt_U32(rawLast[ch], &tlmBuf[n]);                         tlmBuf[n++] = ',';
    }
    n += AppFmt_U32((uint32)lroundf(pMaxLast * 1000.0f), &tlmBuf[n]);     tlmBuf[n++] = ',';
    n += AppFmt_U32(loadPm, &tlmBuf[n]);                                  tlmBuf[n++] = ',';
    n += AppFmt_U32(cycFwdLast, &tlmBuf[n]);
    tlmBuf[n++] = '\n';
    (void)Lpuart_Uart_Ip_AsyncSend(UART_INSTANCE, (const uint8 *)tlmBuf, (uint32)n);
}

static void UartSendLine(const char *buf, uint32 n)
{
    /* Espera a que termine la telemetria en curso (maximo ~7 ms a 115200) */
    uint32 rem = 0U, guard = 2000000U;
    while ((Lpuart_Uart_Ip_GetTransmitStatus(UART_INSTANCE, &rem) == LPUART_UART_IP_STATUS_BUSY) && (guard > 0U)) { guard--; }
    (void)Lpuart_Uart_Ip_SyncSend(UART_INSTANCE, (const uint8 *)buf, n, UART_TIMEOUT_US);
}

static void UartSendLineCb(const char *buf, uint32_t n) { UartSendLine(buf, (uint32)n); }
static uint32_t CyclesCb(void) { return DWT_Cycles(); }

static void UartRxArm(void)
{
    u8BufferIdx = 0U;
    (void)Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, (uint8 *)au8Buffer, 1U);
}

#ifdef __cplusplus
}
#endif
