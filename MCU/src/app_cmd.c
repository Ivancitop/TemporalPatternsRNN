/*==================================================================================================
* Modulo app_cmd. Ver app_cmd.h.
==================================================================================================*/
#include "app_cmd.h"
#include <string.h>

/* Buffers grandes en .bss (no en pila) */
static LstmSeq    xSeq;       /* secuencia recibida por 'Q'          */
static LstmCache  cache;      /* activaciones para el BPTT (~22 KB)  */
static LstmParams grad;       /* gradiente acumulado (~5.8 KB)       */
static char       tx[APP_TX_SIZE];

/*---------------------------------------- formato ----------------------------------------------*/
uint8_t AppFmt_U32(uint32_t v, char *buf)
{
    char  tmp[10];
    uint8_t n = 0U, pos = 0U;
    if (v == 0U) { tmp[n++] = '0'; }
    while (v > 0U) { tmp[n++] = (char)('0' + (v % 10U)); v /= 10U; }
    while (n > 0U) { buf[pos++] = tmp[--n]; }
    return pos;
}

uint8_t AppFmt_I32(int32_t v, char *buf)
{
    if (v < 0) { buf[0] = '-'; return (uint8_t)(1U + AppFmt_U32((uint32_t)(-v), &buf[1])); }
    return AppFmt_U32((uint32_t)v, buf);
}

uint8_t AppFmt_F32Hex(float f, char *buf)
{
    static const char HEX[] = "0123456789ABCDEF";
    uint32_t u;
    (void)memcpy(&u, &f, sizeof(u));
    for (int8_t k = 7; k >= 0; k--) { buf[7 - k] = HEX[(u >> (4 * k)) & 0xFU]; }
    return 8U;
}

static uint8_t FmtHex3(uint16_t v, char *buf)
{
    static const char HEX[] = "0123456789ABCDEF";
    buf[0] = HEX[(v >> 8) & 0xFU]; buf[1] = HEX[(v >> 4) & 0xFU]; buf[2] = HEX[v & 0xFU];
    return 3U;
}

/*---------------------------------------- parsing ----------------------------------------------*/
static int8_t HexVal(char c)
{
    if ((c >= '0') && (c <= '9')) { return (int8_t)(c - '0'); }
    if ((c >= 'A') && (c <= 'F')) { return (int8_t)(c - 'A' + 10); }
    if ((c >= 'a') && (c <= 'f')) { return (int8_t)(c - 'a' + 10); }
    return -1;
}

static uint8_t IsEnd(char c) { return ((c == '\0') || (c == '\n') || (c == '\r')) ? 1U : 0U; }

/* Lee nd digitos hex; devuelve 0 si alguno no es valido */
static uint8_t ParseHexN(const char **ps, uint8_t nd, uint32_t *out)
{
    uint32_t u = 0U;
    for (uint8_t k = 0U; k < nd; k++)
    {
        int8_t h = HexVal(**ps);
        if (h < 0) { return 0U; }
        u = (u << 4) | (uint32_t)h;
        (*ps)++;
    }
    *out = u;
    return 1U;
}

uint8_t AppParse_Ints(const char *s, int32_t *v, uint8_t nmax)
{
    uint8_t cnt = 0U;
    while ((cnt < nmax) && (IsEnd(*s) == 0U))
    {
        int32_t sign = 1, acc = 0;
        uint8_t digits = 0U;
        if (*s == '-') { sign = -1; s++; }
        while ((*s >= '0') && (*s <= '9')) { acc = (acc * 10) + (int32_t)(*s - '0'); s++; digits++; }
        if (digits == 0U) { return cnt; }
        v[cnt++] = sign * acc;
        if (*s == ',') { s++; }
    }
    return cnt;
}

/* Lee un entero decimal sin signo y salta la coma siguiente */
static uint8_t ParseUField(const char **ps, uint32_t *out)
{
    uint32_t acc = 0U; uint8_t d = 0U;
    while ((**ps >= '0') && (**ps <= '9')) { acc = (acc * 10U) + (uint32_t)(**ps - '0'); (*ps)++; d++; }
    if (d == 0U) { return 0U; }
    if (**ps == ',') { (*ps)++; }
    *out = acc;
    return 1U;
}

/*---------------------------------------- comandos ---------------------------------------------*/
void AppCmd_Init(AppCtx *a)
{
    a->lr = APP_LR_DEFAULT;
    a->clip = APP_CLIP_DEFAULT;
    a->loaded = 0U;
}

static void SendStr(AppCtx *a, const char *s) { a->send(s, (uint32_t)strlen(s)); }

/* Q,<k0>,<9 hex por paso> */
static void CmdLoadChunk(AppCtx *a, const char *s)
{
    uint32_t k0;
    if ((ParseUField(&s, &k0) == 0U) || (k0 >= LSTM_T)) { SendStr(a, "E\n"); return; }
    if (k0 == 0U) { a->loaded = 0U; }
    if (k0 != a->loaded) { SendStr(a, "E\n"); return; }      /* tramas fuera de orden */
    uint32_t t = k0;
    while ((IsEnd(*s) == 0U) && (t < LSTM_T))
    {
        for (uint8_t ch = 0U; ch < LSTM_N_IN; ch++)
        {
            uint32_t code;
            if (ParseHexN(&s, 3U, &code) == 0U) { a->loaded = 0U; SendStr(a, "E\n"); return; }
            xSeq[t][ch] = Seq_CodeToX((uint16_t)code);
        }
        t++;
    }
    if (IsEnd(*s) == 0U) { a->loaded = 0U; SendStr(a, "E\n"); return; }   /* sobran datos */
    a->loaded = (uint8_t)t;
    SendStr(a, "K\n");
}

/* T,<c> / V,<c> */
static void CmdTrainEval(AppCtx *a, const char *s, uint8_t train)
{
    if ((a->idle == 0U) || (a->loaded < LSTM_T) || (*s < '0') || ((uint8_t)*s > (uint8_t)('0' + LSTM_N_OUT - 1U)))
    {
        SendStr(a, "E\n");
        return;
    }
    uint8_t label = (uint8_t)(*s - '0');
    uint32_t c0 = a->cycles();
    Lstm_Forward(a->net, xSeq, &cache);
    uint32_t c1 = a->cycles();
    float loss = Lstm_Loss(cache.p, label);               /* perdida ANTES de actualizar */
    if (train == 1U) { Lstm_Backward(a->net, xSeq, &cache, label, a->lr, a->clip, &grad); }
    uint32_t c2 = a->cycles();

    uint16_t n = 0U;
    tx[n++] = (char)('0' + Lstm_Argmax(cache.p));
    for (uint8_t k = 0U; k < LSTM_N_OUT; k++) { tx[n++] = ','; n += AppFmt_F32Hex(cache.p[k], &tx[n]); }
    tx[n++] = ','; n += AppFmt_F32Hex(loss, &tx[n]);
    tx[n++] = ','; n += AppFmt_U32(c1 - c0, &tx[n]);
    tx[n++] = ','; n += AppFmt_U32((train == 1U) ? (c2 - c1) : 0U, &tx[n]);
    tx[n++] = '\n';
    a->send(tx, n);
}

static void CmdDump(AppCtx *a)
{
    for (uint16_t i = 0U; i < LSTM_N_PARAMS; i += 8U)
    {
        uint16_t n = 0U;
        tx[n++] = 'W'; tx[n++] = ',';
        n += AppFmt_U32(i, &tx[n]);
        for (uint16_t k = i; (k < (i + 8U)) && (k < LSTM_N_PARAMS); k++)
        {
            tx[n++] = ','; n += AppFmt_F32Hex(Lstm_GetParam(a->net, k), &tx[n]);
        }
        tx[n++] = '\n';
        a->send(tx, n);
    }
    SendStr(a, "END\n");
}

/* L,<idx>,<h0>,...  (hasta 8 valores) */
static void CmdLoadParams(AppCtx *a, const char *s)
{
    uint32_t idx;
    if ((a->idle == 0U) || (ParseUField(&s, &idx) == 0U)) { SendStr(a, "E\n"); return; }
    float v[8];
    uint8_t cnt = 0U;
    while ((IsEnd(*s) == 0U) && (cnt < 8U))
    {
        uint32_t u;
        if (ParseHexN(&s, 8U, &u) == 0U) { SendStr(a, "E\n"); return; }
        (void)memcpy(&v[cnt], &u, sizeof(float));
        cnt++;
        if (*s == ',') { s++; }
    }
    if ((cnt == 0U) || ((idx + cnt) > LSTM_N_PARAMS)) { SendStr(a, "E\n"); return; }
    for (uint8_t k = 0U; k < cnt; k++) { Lstm_SetParam(a->net, (uint16_t)(idx + k), v[k]); }
    SendStr(a, "K\n");
}

static void CmdCapture(AppCtx *a)
{
    if (Seq_Ready(a->seq) == 0U) { SendStr(a, "N\n"); return; }
    static uint16_t codes[LSTM_T][SEQ_N_CH];
    Seq_SnapshotCodes(a->seq, codes);
    uint16_t n = 0U;
    tx[n++] = 'C';
    for (uint8_t ch = 0U; ch < SEQ_N_CH; ch++) { tx[n++] = ','; n += AppFmt_U32(a->rawLast[ch], &tx[n]); }
    tx[n++] = ',';
    for (uint8_t t = 0U; t < LSTM_T; t++)
    {
        for (uint8_t ch = 0U; ch < SEQ_N_CH; ch++) { n += FmtHex3(codes[t][ch], &tx[n]); }
    }
    tx[n++] = '\n';
    a->send(tx, n);
}

uint8_t AppCmd_Handle(AppCtx *a, const char *s)
{
    switch (s[0])
    {
        case 'Q': CmdLoadChunk(a, &s[2]); break;
        case 'T': CmdTrainEval(a, &s[2], 1U); break;
        case 'V': CmdTrainEval(a, &s[2], 0U); break;
        case 'W': CmdDump(a); break;
        case 'L': CmdLoadParams(a, &s[2]); break;
        case 'C': CmdCapture(a); break;
        case 'R':
            if (a->idle == 0U) { SendStr(a, "E\n"); break; }
            Lstm_LoadFlat(a->net, a->initW);
            SendStr(a, "OK\n");
            break;
        case 'G':
        {
            int32_t v[2];
            if ((a->idle == 0U) || (AppParse_Ints(&s[2], v, 2U) != 2U) || (v[0] <= 0))
            {
                SendStr(a, "E\n");
                break;
            }
            a->lr = (float)v[0] * 1e-6f;
            a->clip = (float)v[1] * 1e-3f;
            SendStr(a, "OK\n");
            break;
        }
        default:
            return 0U;
    }
    return 1U;
}
