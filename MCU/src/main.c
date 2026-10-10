/*==================================================================================================
* Project : StaticPatterns  (deteccion de patrones estaticos + control PI de velocidad un motor DC)
* Platform : S32K312 (Cortex-M7 @ 120 MHz), RTD 7.0.0
* Author   : Ivan Delgado Ramos
* Date     : 25/09/26
*
* Descripción de Hardware...
*   - 3 potenciometros por ADC0 con res 12 bits (P0, P1, P2 de presición)      -> entradas del clasificador (patrones estáticos)
*   - encoder en cuadratura (x4) por EIRQ 3 / EIRQ 8    -> obtencion de velocidad del motor
*   - 2 PWM eMIOS1 ch9 / ch10  -> LPWM y RPWM de driver BT7960
*   - ANN estática con back propagation, topología 9-8-6-4 entrenada  por LPUART6 con buffer asincrónico
*
* Máquina de estados parseada por UART:
*   IDLE      : motor apagado, para entrenamiento (T/V), reset (R),
*               envío de pesos finales (W) y captura de dataset con valores reales de los pots (C).
*   RUN       : inferencia sobre referencia al PI con los potes reales (solo forward)
*               Envia telemetria cada 10 ms.
*   FIXEDREF  : PI con referencia fija enviada por la PC (pruebas de escalon).
*   OPENLOOP  : duty fijo sin lazo (identificacion de la planta).
*
* Determinismo: se tiene un tick de 5 ms definido con el contador de ciclos DWT (sin interrupción.
* Cada tick muestrea el ADC y alimenta el
* preprocesado; el PI corre cada 2 ticks (10 ms) y el clasificador cada 10
* ticks (50 ms).
*
* Protocolo UART (115200 8N1, lineas terminadas en '\n'). Los floats viajan
* como 8 digitos hex del patron IEEE-754 para que PC y MCU usen exactamente
* el mismo float32:
*   T,<h0>,...,<h8>,<c>   entrena una muestra (c = 0..3)
*   V,<h0>,...,<h8>,<c>   evalua sin actualizar pesos
*        respuesta: <cls>,<p0>,<p1>,<p2>,<p3>,<loss>,<cicF>,<cicB>
*   R                     recarga pesos iniciales            -> OK
*   W                     vuelca los 162 parametros (hex)    -> W,... / END
*   C                     captura: <r0>,<r1>,<r2>,<f0..f8 hex>   (o N si no hay ventana)
*   I                     entra a RUN                         -> OK + telemetria
*   F,<rpm>               entra a FIXEDREF con esa referencia -> OK + telemetria
*   O,<permil>            entra a OPENLOOP con duty +-1000    -> OK + telemetria
*   X                     vuelve a IDLE (motor apagado)       -> OK
*   P,<kp*1e6>,<ki*1e6>   cambia ganancias del PI             -> OK
*   S,<rpm>               cambia la magnitud de la referencia -> OK
*   Telemetria: D,<t_ms>,<clsRaw>,<cls>,<ref>,<rpm*10>,<u*1000>,<r0>,<r1>,<r2>,
*               <pmax*1000>,<carga*1000>,<cicF>
==================================================================================================*/

#ifdef __cplusplus
extern "C"{
#endif
#define USE_PRETRAINED //Preprocesor statement para emplear los pesos volcados
						//(despues de entrenar)
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

#include "preproc.h" //Header con definiciones para procesar los datos de los pots
#include "mlp.h" //Header con definiciones para construir la ANN
#include "pi_ctrl.h" //Header con las definiciones del control
#include "pesos_iniciales.h" //Pesos iniciales
#ifdef USE_PRETRAINED
#include "pesos_entrenados.h"   //Pesos generados despues de entrenar
#endif

/*======================================= Configuracion ==========================================*/
/* --- Instancias y canales de perifericos --- */
#define UART_INSTANCE        6U
#define UART_TIMEOUT_US      100000U
#define BUFFER_SIZE          256U //tamaño del buffer
#define ADC_SAR_USED_CH      0U
#define ADC_SAR_USED_CH2     1U
#define ADC_SAR_USED_CH3     2U
#define ADC_WAIT_MAX         100000U
#define SIULINS              0U
#define PWM_INSTANCE         1U
#define PWM_CH_CW            9U
#define PWM_CH_CCW           10U
#define PWM_PERIOD           65534U  //cuentas máximas del pwm

/* --- DWT (contador de ciclos) --- */
#define DWT_LAR     (*(volatile uint32 *)0xE0001FB0UL)
#define DWT_CTRL    (*(volatile uint32 *)0xE0001000UL)
#define DWT_CYCCNT  (*(volatile uint32 *)0xE0001004UL)
#define CORE_DEMCR  (*(volatile uint32 *)0xE000EDFCUL)
#define CORE_MHZ    120U

/* --- Temporizacion --- */
#define TICK_MS              5U
#define TICK_CYC             (CORE_MHZ * 1000U * TICK_MS)
#define CTRL_DIV             2U      /* PI cada 10 ms          */
#define CLS_DIV              10U     /* clasificador cada 50 ms */
#define LOAD_WIN_TICKS       200U    /* ventana de carga: 1 s  */
#define TS_CTRL              ((float32)(TICK_MS * CTRL_DIV) * 1e-3f)

/* --- Encoder y motor --- */
#define ENC_CPR              3332.0f /* cuentas por vuelta del eje de salida en x4
                                        (p. ej. 17 PPR x 4 x reduccion 49)       */
#define ENC_SIGN             1       /* -1 si CW da cuentas negativas            */
#define RPM_LPF_ALPHA        0.5f    /* filtro de primer orden de la velocidad   */
#define U_DEADZONE           0.0f    /* compensacion de zona muerta (0 = off)    */

/* --- Control --- */
#define PI_KP_DEFAULT        0.0040f /* duty / rpm   */
#define PI_KI_DEFAULT        0.0500f /* duty / (rpm s)            */
#define PI_UMAX              1.0f
#define REF_RPM_DEFAULT      120.0f /* Para prueba escalon */
#define REF_SLEW_RPM_S       0.0f    /* rampa de referencia (0 = escalon)         */

/* --- Clasificador --- */
#define LEARNING_RATE        0.05f
#define CONF_THRESHOLD       0.70f   /* valor del pot menor al thres -> clase default              */
#define DEBOUNCE_N           3U      /* clasificaciones iguales para conmutar     */
#define CLS_DEFAULT          0U
#define CLS_CW               1U
#define CLS_CCW              2U
#define CLS_PARO             3U

typedef enum { MODE_IDLE = 0, MODE_RUN, MODE_FIXEDREF, MODE_OPENLOOP } AppMode;

/*======================================= Globales ===============================================*/
/* ADC */
volatile boolean notif_triggered = FALSE;
volatile uint16  data;
volatile uint16  data2;
volatile uint16  data3;

/* Encoder (32 bits: lectura atomica en M7 y sin desborde practico) */
volatile sint32  count = 0;
volatile uint32 duty = 0;
/* UART */
volatile uint8   u8BufferIdx = 0U;
volatile uint8   au8Buffer[BUFFER_SIZE];
volatile uint8   bRxFlag = 0U;
volatile uint8   bRxRearm = 0U;
static char      txBuf[128];      /* respuestas a comandos (envio sincrono)   */
static char      tlmBuf[128];     /* telemetria (envio asincrono)             */
volatile uint32 uartStat = 0U;
volatile uint32 uartErrCnt = 0U;
volatile Lpuart_Uart_Ip_StatusType uartErr;


/* Aplicacion */
static FeatState fe;
static MlpParams net;
static PiCtrl    pi;
static AppMode   mode = MODE_IDLE; //handler de estados
static uint16    rawLast[FEAT_N_CH];
static uint32    tickCount = 0U;
static sint32    encPrev = 0;
static float32   rpmFilt = 0.0f;
static float32   refMag = REF_RPM_DEFAULT;
static float32   refTarget = 0.0f;    /* referencia deseada               */
static float32   refApplied = 0.0f;   /* referencia tras la rampa         */
static float32   uOut = 0.0f;
static float32   uOpenLoop = 0.0f;
static uint8     clsRaw = CLS_DEFAULT, clsCand = CLS_DEFAULT, clsActive = CLS_DEFAULT; //Definicion de clases
static uint8     candCount = 0U;
static float32   pMaxLast = 0.0f;
static uint32    cycFwdLast = 0U;
static uint32    busyCyc = 0U, loadPm = 0U, overruns = 0U, tlmDrops = 0U;

/*======================================= Prototipos de funciones =============================================*/
void AdcEndOfChainNotif(void);
void EncoderANotify(void);
void EncoderBNotify(void);
void Uart_Callback(const uint8 HwInstance, const Lpuart_Uart_Ip_EventType Event, const void *UserData);

static void    InitPeripherals(void);
static void    LoadInitialWeights(void);
static void    ControlTick(void);
static void    Classify(void);
static void    UpdateReference(void);
static void    MotorApply(float32 u);
static void    HandleCommand(void);
static void    SendTelemetry(void);
static void    UartSendLine(const char *buf, uint32 n);
static void    UartRxArm(void);
static uint8   ParseFeatureLine(const char *s, float32 *x, uint8 *label);
static uint8   ParseInts(const char *s, sint32 *v, uint8 nmax);
static uint8   U32ToStr(uint32 v, char *buf);
static uint8   I32ToStr(sint32 v, char *buf);
static uint8   F32ToHex(float32 f, char *buf);
static inline void   DWT_Init(void);
static inline uint32 DWT_Cycles(void) { return DWT_CYCCNT; }

/*======================================= Callbacks / ISR ========================================*/
void AdcEndOfChainNotif(void)//Final de conversión de ADC
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
        	{uint32 rem;
        	uartStat = IP_LPUART_6->STAT;   /* bit19 OR, bit18 NF, bit17 FE, bit16 PF */
        	uartErrCnt++;
            uartErr = Lpuart_Uart_Ip_GetReceiveStatus(UART_INSTANCE, &rem);
            bRxRearm = 1U;          /* overrun / framing: se rearma en main */
            break;}
        default:                    /* TX_EMPTY, END_TRANSFER: nada que hacer */
            break;
    }
    (void)UserData;
    (void)HwInstance;
}

/*======================================= main ===================================================*/
int main(void)
{
    InitPeripherals(); //Inicializar el hardware
    DWT_Init();//Contador del core

    Feat_Init(&fe);//locaclización de memoria para ubicar los datos de los pots
    Pi_Init(&pi, PI_KP_DEFAULT, PI_KI_DEFAULT, TS_CTRL, PI_UMAX);
#ifdef USE_PRETRAINED
    /* Binario autonomo: arranca con la red ya entrenada y en modo RUN */
    Mlp_Load(&net, W1_TRAINED, B1_TRAINED, W2_TRAINED, B2_TRAINED, W3_TRAINED, B3_TRAINED);
    mode = MODE_RUN;
#else
    LoadInitialWeights();
#endif
    MotorApply(0.0f);
    UartRxArm();//Armar el buffer para recepción

    uint32 tNext = DWT_Cycles() + TICK_CYC;//Stamp de tiempo con No de ciclos mas offset
    for (;;)
    {
        /* Planificador: resta sin signo -> robusto al desborde de CYCCNT */
        if ((sint32)(DWT_Cycles() - tNext) >= 0)
        {
            tNext += TICK_CYC;
            if ((sint32)(DWT_Cycles() - tNext) >= 0)
            {
                overruns++;                      /* se perdio al menos un tick */
                tNext = DWT_Cycles() + TICK_CYC;
            } //funciona decente y evita configurar timer dedicado
            ControlTick();
        }

        if (bRxFlag == 1U)
        {
            bRxFlag = 0U;
            uint32 c0 = DWT_Cycles();
            HandleCommand();//Parsing de las tramas enviadas por UART
            busyCyc += DWT_Cycles() - c0;
            UartRxArm();
        }
        else if (bRxRearm == 1U)//Rearmar el buffer de recepcion
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

static void LoadInitialWeights(void)
{
    Mlp_Load(&net, W1_INIT, B1_INIT, W2_INIT, B2_INIT, W3_INIT, B3_INIT);
}

static inline void DWT_Init(void)
{
    CORE_DEMCR |= (1UL << 24);      /* TRCENA */
    DWT_LAR     = 0xC5ACCE55UL;     /* desbloqueo de dwt */
    DWT_CYCCNT  = 0U;
    DWT_CTRL   |= 1UL;              /* CYCCNTENA */
}

/*======================================= Tick de control (5 ms) =================================*/
static void ControlTick(void)
{
    uint32 c0 = DWT_Cycles();
    tickCount++;

    /* 1) Muestreo de los 3 potes (conversion unica, ~us) */
    notif_triggered = FALSE;
    (void)Adc_Sar_Ip_StartConversion(ADCHWUNIT_0_INSTANCE, ADC_SAR_IP_CONV_CHAIN_NORMAL);
    uint32 to = ADC_WAIT_MAX;
    while ((notif_triggered != TRUE) && (to > 0U)) { to--; }
    rawLast[0] = data; rawLast[1] = data2; rawLast[2] = data3;

    /* 2) Preprocesado: media movil + ventana */
    Feat_Push(&fe, rawLast, NULL);

    /* 3) Clasificacion (solo forward) cada 50 ms */
    if (((tickCount % CLS_DIV) == 0U) && (mode != MODE_IDLE) && (Feat_Ready(&fe) == 1U))
    {
        Classify();
    }

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

static void Classify(void)
{
    float32  x[FEAT_N];
    MlpCache cache;

    Feat_Compute(&fe, x);
    uint32 c0 = DWT_Cycles();
    Mlp_Forward(&net, x, &cache);
    cycFwdLast = DWT_Cycles() - c0;

    uint8 k = Mlp_Argmax(&cache);
    pMaxLast = cache.p[k];
    clsRaw = (pMaxLast >= CONF_THRESHOLD) ? k : CLS_DEFAULT;

    /* Antirrebote: la clase activa solo cambia tras DEBOUNCE_N votos iguales */
    if (clsRaw == clsCand) { if (candCount < 255U) { candCount++; } }
    else { clsCand = clsRaw; candCount = 1U; }
    if ((candCount >= DEBOUNCE_N) && (clsActive != clsCand))
    {
        clsActive = clsCand;
        if (clsActive == CLS_DEFAULT) { Pi_Reset(&pi); }
    }
}

static void UpdateReference(void)
{
    switch (clsActive)
    {
        case CLS_CW:   refTarget =  refMag; break;
        case CLS_CCW:  refTarget = -refMag; break;
        case CLS_PARO: refTarget =  0.0f;   break;
        default:       refTarget =  0.0f;   break;
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
static void HandleCommand(void)
{
    au8Buffer[u8BufferIdx] = 0U;               /* termina la cadena en '\n' */
    const char *s = (const char *)au8Buffer;
    uint8 n = 0U;

    switch (s[0])
    {
        case 'T':
        case 'V':
        {
            float32 x[FEAT_N];
            uint8   label;
            if ((mode != MODE_IDLE) || (ParseFeatureLine(&s[2], x, &label) == 0U))
            {
                UartSendLine("E\n", 2U);
                break;
            }
            MlpCache c;
            uint32 c0 = DWT_Cycles();
            Mlp_Forward(&net, x, &c);
            uint32 c1 = DWT_Cycles();
            float32 loss = Mlp_Loss(&c, label);     /* perdida ANTES de actualizar */
            if (s[0] == 'T') { Mlp_Backward(&net, x, &c, label, LEARNING_RATE); }
            uint32 c2 = DWT_Cycles();

            txBuf[n++] = (char)('0' + Mlp_Argmax(&c));
            for (uint8 j = 0U; j < MLP_N_OUT; j++) { txBuf[n++] = ','; n += F32ToHex(c.p[j], &txBuf[n]); }
            txBuf[n++] = ','; n += F32ToHex(loss, &txBuf[n]);
            txBuf[n++] = ','; n += U32ToStr(c1 - c0, &txBuf[n]);
            txBuf[n++] = ','; n += U32ToStr((s[0] == 'T') ? (c2 - c1) : 0U, &txBuf[n]);
            txBuf[n++] = '\n';
            UartSendLine(txBuf, n);
            break;
        }
        case 'R':
            LoadInitialWeights();
            UartSendLine("OK\n", 3U);
            break;
        case 'W':
            for (uint16 i = 0U; i < MLP_N_PARAMS; i += 8U)
            {
                n = 0U;
                txBuf[n++] = 'W'; txBuf[n++] = ',';
                n += U32ToStr(i, &txBuf[n]);
                for (uint16 k = i; (k < (i + 8U)) && (k < MLP_N_PARAMS); k++)
                {
                    txBuf[n++] = ','; n += F32ToHex(Mlp_GetParam(&net, k), &txBuf[n]);
                }
                txBuf[n++] = '\n';
                UartSendLine(txBuf, n);
            }
            UartSendLine("END\n", 4U);
            break;
        case 'C':
        {
            if (Feat_Ready(&fe) == 0U) { UartSendLine("N\n", 2U); break; }
            float32 x[FEAT_N];
            Feat_Compute(&fe, x);
            for (uint8 ch = 0U; ch < FEAT_N_CH; ch++)
            {
                if (ch > 0U) { txBuf[n++] = ','; }
                n += U32ToStr(rawLast[ch], &txBuf[n]);
            }
            for (uint8 j = 0U; j < FEAT_N; j++) { txBuf[n++] = ','; n += F32ToHex(x[j], &txBuf[n]); }
            txBuf[n++] = '\n';
            UartSendLine(txBuf, n);
            break;
        }
        case 'I':
            Pi_Reset(&pi);
            clsCand = CLS_DEFAULT; clsActive = CLS_DEFAULT; candCount = 0U;
            refApplied = 0.0f;
            mode = MODE_RUN;
            UartSendLine("OK\n", 3U);
            break;
        case 'F':
        {
            sint32 v[1];
            if (ParseInts(&s[2], v, 1U) != 1U) { UartSendLine("E\n", 2U); break; }
            Pi_Reset(&pi);
            refApplied = 0.0f;
            refTarget = (float32)v[0];
            mode = MODE_FIXEDREF;
            UartSendLine("OK\n", 3U);
            break;
        }
        case 'O':
        {
            sint32 v[1];
            if (ParseInts(&s[2], v, 1U) != 1U) { UartSendLine("E\n", 2U); break; }
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
            uOut = 0.0f;
            MotorApply(0.0f);
            UartSendLine("OK\n", 3U);
            break;
        case 'P':
        {
            sint32 v[2];
            if (ParseInts(&s[2], v, 2U) != 2U) { UartSendLine("E\n", 2U); break; }
            Pi_SetGains(&pi, (float32)v[0] * 1e-6f, (float32)v[1] * 1e-6f);
            UartSendLine("OK\n", 3U);
            break;
        }
        case 'S':
        {
            sint32 v[1];
            if (ParseInts(&s[2], v, 1U) != 1U) { UartSendLine("E\n", 2U); break; }
            refMag = (float32)v[0];
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
    n += U32ToStr(tickCount * TICK_MS, &tlmBuf[n]);                    tlmBuf[n++] = ',';
    tlmBuf[n++] = (char)('0' + clsRaw);                                tlmBuf[n++] = ',';
    tlmBuf[n++] = (char)('0' + clsActive);                             tlmBuf[n++] = ',';
    n += I32ToStr((sint32)refApplied, &tlmBuf[n]);                     tlmBuf[n++] = ',';
    n += I32ToStr((sint32)lroundf(rpmFilt * 10.0f), &tlmBuf[n]);       tlmBuf[n++] = ',';
    n += I32ToStr((sint32)lroundf(uOut * 1000.0f), &tlmBuf[n]);        tlmBuf[n++] = ',';
    for (uint8 ch = 0U; ch < FEAT_N_CH; ch++)
    {
        n += U32ToStr(rawLast[ch], &tlmBuf[n]);                        tlmBuf[n++] = ',';
    }
    n += U32ToStr((uint32)lroundf(pMaxLast * 1000.0f), &tlmBuf[n]);    tlmBuf[n++] = ',';
    n += U32ToStr(loadPm, &tlmBuf[n]);                                 tlmBuf[n++] = ',';
    n += U32ToStr(cycFwdLast, &tlmBuf[n]);
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

static void UartRxArm(void)
{
    u8BufferIdx = 0U;
    (void)Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, (uint8 *)au8Buffer, 1U);
}

/*======================================= Parsing / formato ======================================*/
static sint8 HexVal(char c)
{
    if ((c >= '0') && (c <= '9')) { return (sint8)(c - '0'); }
    if ((c >= 'A') && (c <= 'F')) { return (sint8)(c - 'A' + 10); }
    if ((c >= 'a') && (c <= 'f')) { return (sint8)(c - 'a' + 10); }
    return -1;
}

/* "<h0>,...,<h8>,<c>" -> x[9], label. Devuelve 1 si es valida. */
static uint8 ParseFeatureLine(const char *s, float32 *x, uint8 *label)
{
    for (uint8 j = 0U; j < FEAT_N; j++)
    {
        uint32 u = 0U;
        for (uint8 k = 0U; k < 8U; k++)
        {
            sint8 h = HexVal(*s++);
            if (h < 0) { return 0U; }
            u = (u << 4) | (uint32)h;
        }
        if (*s++ != ',') { return 0U; }
        (void)memcpy(&x[j], &u, sizeof(float32));
    }
    if ((*s < '0') || (*s > '3')) { return 0U; }
    *label = (uint8)(*s - '0');
    return 1U;
}

/* Lee hasta nmax enteros con signo separados por ',' */
static uint8 ParseInts(const char *s, sint32 *v, uint8 nmax)
{
    uint8 cnt = 0U;
    while ((cnt < nmax) && (*s != '\0') && (*s != '\n') && (*s != '\r'))
    {
        sint32 sign = 1, acc = 0;
        uint8 digits = 0U;
        if (*s == '-') { sign = -1; s++; }
        while ((*s >= '0') && (*s <= '9')) { acc = (acc * 10) + (sint32)(*s - '0'); s++; digits++; }
        if (digits == 0U) { return cnt; }
        v[cnt++] = sign * acc;
        if (*s == ',') { s++; }
    }
    return cnt;
}

static uint8 U32ToStr(uint32 v, char *buf)
{
    char  tmp[10];
    uint8 n = 0U, pos = 0U;
    if (v == 0U) { tmp[n++] = '0'; }
    while (v > 0U) { tmp[n++] = (char)('0' + (v % 10U)); v /= 10U; }
    while (n > 0U) { buf[pos++] = tmp[--n]; }
    return pos;
}

static uint8 I32ToStr(sint32 v, char *buf)
{
    if (v < 0) { buf[0] = '-'; return (uint8)(1U + U32ToStr((uint32)(-v), &buf[1])); }
    return U32ToStr((uint32)v, buf);
}

static uint8 F32ToHex(float32 f, char *buf)
{
    static const char HEX[] = "0123456789ABCDEF";
    uint32 u;
    (void)memcpy(&u, &f, sizeof(u));
    for (sint8 k = 7; k >= 0; k--) { buf[7 - k] = HEX[(u >> (4 * k)) & 0xFU]; }
    return 8U;
}

#ifdef __cplusplus
}
#endif
