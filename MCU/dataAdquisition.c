/* Including necessary configuration files. */
#include "Clock_Ip.h"
#include "IntCtrl_Ip.h"
#include "Adc_Sar_Ip.h"
#include "Siul2_Port_Ip.h"
#include "Lpuart_Uart_Ip.h"
#include "OsIf.h"
#include <stdio.h>
#include <string.h>

volatile int exit_code = 0;

/* Configuración de UART */
#define UART_INSTANCE                   6U
#define BUFFER_SIZE                     128U

/* Canales de Potenciómetros */
#define ADC_SAR_POT1                    0U
#define ADC_SAR_POT2                    1U
#define ADC_SAR_POT3                    2U
#define ADC_RAW_MAX_REAL                2660.0f

/* Filtering Configuration */
#define ADC_FILTER_ALPHA                0.15f

/* --------------------------------------------------------------------
 * ETIQUETA DE LA CLASE ACTUAL PARA EL DATASET:
 * 0: STOP (0 RPM)
 * 1: Avance Rápido (+150 RPM)
 * 2: Avance Lento (+50 RPM)
 * 3: Reversa Rápida (-150 RPM)
 * 4: Reversa Lenta (-50 RPM)
 * -------------------------------------------------------------------- */


/* ADC Variables */
volatile boolean notif_triggered = FALSE;
volatile uint16 pot1;
volatile uint16 pot2;
volatile uint16 pot3;
uint16_t ui16AdcRawpot1 = 0u;
uint16_t ui16AdcRawpot2 = 0u;
uint16_t ui16AdcRawpot3 = 0u;

float fAdcpot1;
float fAdcpot2;
float fAdcpot3;
float fAdcpot1Filtered = 0.0f;
float fAdcpot2Filtered = 0.0f;
float fAdcpot3Filtered = 0.0f;

/* Buffers y Estado UART */
uint8 au8TxBuffer[BUFFER_SIZE];
static volatile uint8_t bTxBusy = 0U;

/* Callback de Notificación ADC */
void AdcEndOfChainNotif(void)
{
    pot1 = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_POT1);
    pot2 = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_POT2);
    pot3 = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_POT3);
    notif_triggered = TRUE;
}

/* Callback de UART */
void Uart_Callback(
    const uint8 HwInstance,
    const Lpuart_Uart_Ip_EventType Event,
    const void *UserData)
{
    (void)UserData;
    (void)HwInstance;

    if (Event == LPUART_UART_IP_EVENT_TX_EMPTY)
    {
        bTxBusy = 0U;
    }
}

/* Transmisión de datos formateados */
void Send_UART_Data(void)
{
    if (bTxBusy == 1U) return;

    /* Casteo explícito a double para eliminar las advertencias del compilador */
    int len = sprintf((char *)au8TxBuffer, "%.3f,%.3f,%.3f\r\n",
                      (double)fAdcpot1Filtered,
                      (double)fAdcpot2Filtered,
                      (double)fAdcpot3Filtered);

    bTxBusy = 1U;
    Lpuart_Uart_Ip_AsyncSend(UART_INSTANCE, au8TxBuffer, (uint32)len);
}

void delay_ms(uint32 ms)
{
	volatile uint32 i;
	volatile uint32 j;

	for (i = 0; i < ms; i++)
	{
		for (j = 0; j < 10000U; j++)
		{
			__asm("nop");
		}
	}
}


int main(void)
{
    /* 1. Initialize System Clocks */
    Clock_Ip_Init(&Clock_Ip_aClockConfig[0]);

    /* 2. Initialize OS Interface (needed for timing functions) */
    OsIf_Init(NULL_PTR);

    /* 3. Initialize GPIO Pins */
    Siul2_Port_Ip_Init(
        NUM_OF_CONFIGURED_PINS_PortContainer_0_BOARD_InitPeripherals,
        g_pin_mux_InitConfigArr_PortContainer_0_BOARD_InitPeripherals);

    /* 4. Initialize Interrupts & Enable LPUART6 + ADC0 IRQ */
    IntCtrl_Ip_Init(&IntCtrlConfig_0);
    IntCtrl_Ip_EnableIrq(ADC0_IRQn);
    IntCtrl_Ip_EnableIrq(LPUART6_IRQn);

    /* 5. Initialize UART Hardware */
    Lpuart_Uart_Ip_Init(UART_INSTANCE, &Lpuart_Uart_Ip_xHwConfigPB_6);

    /* 6. Initialize ADC Hardware Unit */
    Adc_Sar_Ip_Init(ADCHWUNIT_0_INSTANCE, &AdcHwUnit_0);

    /* 7. Perform ADC Calibration (multiple times for accuracy) */
    for (uint8 Index = 0; Index <= 5; Index++) {
        Adc_Sar_Ip_DoCalibration(ADCHWUNIT_0_INSTANCE);
    }

    /* 8. Enable ADC End of Chain notifications */
    Adc_Sar_Ip_EnableNotifications(ADCHWUNIT_0_INSTANCE,
                                   ADC_SAR_IP_NOTIF_FLAG_NORMAL_ENDCHAIN);

    while(1)
    {
        /* Iniciar conversión por cadena de canales */
        Adc_Sar_Ip_StartConversion(ADCHWUNIT_0_INSTANCE, ADC_SAR_IP_CONV_CHAIN_NORMAL);

        /* Esperar interrupción de fin de conversión */
        while (notif_triggered != TRUE);
        notif_triggered = FALSE;

        /* Leer valores crudos */
        ui16AdcRawpot1 = pot1;
        ui16AdcRawpot2 = pot2;
        ui16AdcRawpot3 = pot3;

        /* Normalizar datos entre 0.0f y 1.0f */
        fAdcpot1 = ((float) ui16AdcRawpot1) / ADC_RAW_MAX_REAL;
        fAdcpot2 = ((float) ui16AdcRawpot2) / ADC_RAW_MAX_REAL;
        fAdcpot3 = ((float) ui16AdcRawpot3) / ADC_RAW_MAX_REAL;

        /* Aplicar Filtro Pasa Bajas (Filtro Exponencial) */
        fAdcpot1Filtered = (ADC_FILTER_ALPHA * fAdcpot1) + ((1.0f - ADC_FILTER_ALPHA) * fAdcpot1Filtered);
        fAdcpot2Filtered = (ADC_FILTER_ALPHA * fAdcpot2) + ((1.0f - ADC_FILTER_ALPHA) * fAdcpot2Filtered);
        fAdcpot3Filtered = (ADC_FILTER_ALPHA * fAdcpot3) + ((1.0f - ADC_FILTER_ALPHA) * fAdcpot3Filtered);

        /* Enviar telemetría serial a Python */
        Send_UART_Data();

        /* Ritmo de muestreo constante de 20ms (50 Hz) */
        delay_ms(20U);
    }
}

