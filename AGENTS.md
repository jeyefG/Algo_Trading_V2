# AGENTS.md — State Engine (PA-first, no Taylor/VWAP)

## Propósito
Este repositorio implementa un **State Engine** para trading discrecional/semisistemático basado en **Price Action (PA)**.
El objetivo es **clasificar el estado del mercado** y **habilitar/prohibir familias de setups** mediante reglas explícitas (`ALLOW_*`), reduciendo errores estructurales y sosteniendo estabilidad cognitiva.

Este sistema **no predice dirección**. Gobierna **cuándo existe un trade**.

---

## Decisiones de diseño (no negociables)
1) **Taylor/VWAP: excluidos del modelo**.  
   No se usan como features, señales ni contexto.  
   No se recicla lógica previa por conveniencia.
2) **ML = percepción** (clasificar estado). **Reglas = decisión** (gating).  
3) **Métrica principal**: expectancy y drawdown condicionados por estado y setup.  
   Accuracy/F1 son secundarias.
4) **Transición es crítica**: su función es **prohibir direccional**.
5) **Complejidad mínima**: máximo 10–12 features H1. Si se requieren más, la definición está mal.

---

## Scope reutilizable del repo previo
Se recicla **solo infraestructura**, no lógica:
- `MT5Connector`: descarga OHLCV por símbolo/timeframe.
- Utilidades de calendario/sesión (si existen y están limpias).
- Infra mínima de dataset (resampling, timezone, NaN handling).
- Persistencia de modelos (save/load).

Todo lo demás se considera legado y **no se integra**.

---

## Definición de estado (H1)
- **Timeframe**: H1  
- **Ventanas fijas**:  
  - `W = 24` velas H1 (contexto)
  - `N = 8` velas H1 (reciente)

Estados:
- **BALANCE**: rotación dentro de un rango; múltiples rechazos; baja direccionalidad neta.
- **TRANSICIÓN**: intento de salida + falla o aceptación incompleta; expansión + reversión; alto riesgo direccional.
- **TENDENCIA**: migración sostenida con aceptación; retrocesos ordenados; reingresos raros.

Regla maestra:
> El estado se define por comportamiento agregado en ventanas fijas, no por velas individuales.

---

## Medidas base (normalización)
Para evitar acoplamiento excesivo con volatilidad:

- `ATR_W = ATR(t-W..t)` (volatilidad de contexto)
- `ATR_N = ATR(t-N..t)` (volatilidad reciente)

Normalizaciones:
- Métricas de contexto → normalizar por `ATR_W`
- Métricas recientes / ruptura → normalizar por `ATR_N`

---

## Variables (definiciones formales)
Para cada timestamp `t`:

1) **Direccionalidad neta de contexto (no es predicción)**  
   `NetMove = |C_t - C_{t-W}| / ATR_W`

2) **Longitud de camino (chop)**  
   `Path = sum_{i=t-W+1..t} |C_i - C_{i-1}| / ATR_W`

3) **Eficiencia (Kaufman)**  
   `ER = |C_t - C_{t-W}| / sum_{i=t-W+1..t} |C_i - C_{i-1}|` ∈ [0,1]

4) **Rango relativo de contexto**  
   `Range_W = (H_W - L_W) / ATR_W`

5) **Posición del cierre en el rango (dirección-neutral)**  
   `CloseLocation = (C_t - L_W) / (H_W - L_W)` en [0,1]

6) **Ruptura de rango (magnitud, no dirección)**  
   `BreakMag = max(0, |C_t - clamp(C_t, L_W, H_W)|) / ATR_N`  
   Es 0 si el cierre está dentro del rango W.

7) **ReentryCount (definición exacta)**  
   Cuenta reingresos al rango tras una ruptura durante la ventana reciente `N`:
   - `outside(i)`: `C_i > H_W` o `C_i < L_W`
   - `inside(i)`: `L_W ≤ C_i ≤ H_W`

   Un reingreso ocurre cuando existe un par consecutivo `(i-1, i)` tal que:
   - `outside(i-1) = True` y `inside(i) = True`

   `ReentryCount = # reingresos en i=t-N+1..t`

8) **InsideBarsRatio (definición exacta)**  
   En la ventana reciente `N`:
   - Inside bar en `i` si `H_i ≤ H_{i-1}` y `L_i ≥ L_{i-1}`

   `InsideBarsRatio = (# inside bars en N) / N`

9) **SwingCount (dirección-neutral)**  
   Se usa pivote simple y solo cuenta giros, no dirección:
   - Pivot high en `i` si `H_i > H_{i-1}` y `H_i > H_{i+1}`
   - Pivot low en `i` si `L_i < L_{i-1}` y `L_i < L_{i+1}`

   `SwingCount = # pivots (high + low) en i=t-W+2..t-1`

10) **Slopes (opcionales)**  
   - `ERSlope = slope(ER_{t-W+1..t})`
   - `RangeSlope = slope(Range_W_{t-W+1..t})`

---

## Auto-etiquetado inicial (bootstrap) — no es “verdad”
El etiquetado inicial existe para arrancar. No se optimiza por accuracy.

**Regla crítica:** el modelo no puede usar exactamente el mismo set mínimo que define el bootstrap como features core (evitar colapso reglas↔ML).

### Bootstrap rules (positivas para cada clase)
Umbrales iniciales (ajustables por validación de PnL/DD):

**BALANCE** si:
- `ER ≤ 0.22`
- `NetMove ≤ 0.9`
- `ReentryCount ≥ 2`
- `InsideBarsRatio ≥ 0.20`

**TENDENCIA** si:
- `ER ≥ 0.38`
- `NetMove ≥ 1.3`
- `ReentryCount = 0`
- `BreakMag ≥ 0.25`

**TRANSICIÓN** si:
- (a) `BreakMag ≥ 0.25` y `ReentryCount ≥ 1` (ruptura + reingreso), o
- (b) `NetMove ≥ 1.0` y `ER` en [0.22, 0.38] (desplazamiento sin eficiencia), o
- (c) `RangeSlope > 0` y `InsideBarsRatio` cae fuerte vs pasado (expansión post compresión).

Si hay conflicto de reglas:
- Prioridad: **TRANSICIÓN > TENDENCIA > BALANCE**  
  (penaliza falso TENDENCIA; fuerza conservadurismo).

---

## Features del State Engine (máx. 10–12)
**PA-nativas. Sin indicadores clásicos. Sin niveles.**

**Regla:** las features usadas para entrenar **no deben ser idénticas** al set mínimo que define el bootstrap label.

Set recomendado (dirección-neutral):
1) `Path`
2) `Range_W`
3) `CloseLocation`
4) `BreakMag`
5) `ReentryCount`
6) `InsideBarsRatio`
7) `SwingCount`
8) `ATR_N / ATR_W` (cambio de volatilidad)
9) `ERSlope` (opcional)
10) `RangeSlope` (opcional)

Notas:
- `ER` y `NetMove` pueden existir como métricas de diagnóstico, pero no son core features si el bootstrap se construye principalmente con ellas.
- Si se decide incluir una de ellas, se debe retirar del bootstrap o degradar su peso (no duplicar lógica).

---

## Modelos
### StateEngine (principal)
- **Modelo**: LightGBM multiclass (BALANCE/TRANSICIÓN/TENDENCIA).
- **Salida**:
  - `state_hat` (clase predicha)
  - `margin = P(top1) - P(top2)`
  - `probas` (opcional para reporting)
- **Calibración**: si se quiere usar umbrales sobre `P(state)`, se debe aplicar calibración (Platt/Isotonic) y validar estabilidad out-of-sample.
- **Por defecto**, el gating usa `state_hat` + `margin`.

### Modelos auxiliares (máximo 1, por defecto 0)
Solo si mejora PnL/DD OOS:
- **VolatilityRegime (binario)**: expansión vs rotación (sin dirección).  
  Si no mejora, se elimina.

---

## Gating (`ALLOW_*`) — política determinista
`ALLOW` **no se entrena**. Es una capa lógica.

Regla general:
- El gating se basa en `state_hat`, `margin` y guardrails simples.
- Los valores de margin se fijan por:
   - percentiles OOS del margin, o
   - targeting explícito de tasa de "no trade" 
- No se usa `P(state) ≥ umbral` salvo calibración validada.

Ejemplos:
- `ALLOW_trend_pullback = 1` si `state_hat == TENDENCIA` y `margin ≥ 0.15`
- `ALLOW_balance_fade = 1` si `state_hat == BALANCE` y `margin ≥ 0.10`
- `ALLOW_transition_failure = 1` si:
  - `state_hat == TRANSICIÓN`
  - `margin ≥ 0.10`
  - `BreakMag ≥ 0.25`
  - `ReentryCount ≥ 1`

Prohibición explícita:
- Si `state_hat == TRANSICIÓN`, se prohíbe swing direccional por defecto.

Propósito:
- Prohibir trades inexistentes.
- Reducir fatiga decisional.
- Convertir “no operar” en decisión explícita.

---

## Setups por estado (familias)
- **TENDENCIA**: continuaciones y pullbacks estructurales.
- **BALANCE**: fades de extremos, sweep & reclaim, failed breakouts.
- **TRANSICIÓN**: solo failures/reclaims tácticos.  
  **Prohibido swing direccional.**

El repo no ejecuta entradas automáticas sin confirmación PA.

---

## Evaluación (criterios válidos)
- Confusión crítica: TRANSICIÓN → TENDENCIA (debe ser baja).
- PnL/Trade y Drawdown **condicionados por estado y `ALLOW_*`**.
- Reducción de overtrading vs baseline.
- Walk-forward mensual (out-of-sample).

Éxito mínimo:
- Menos trades con mismo PnL, o
- Igual trades con menor DD, o
- Mayor expectancy aunque baje accuracy.

---

## Anti-leakage
- Features usan solo info ≤ t.
- Labels bootstrap se calculan offline y no se usan como features.
- No guardar variables auxiliares usadas para etiquetar como features.
- No usar info futura en detección de setups.

---

## Lista de veto
- Taylor/VWAP, niveles, bandas, confluencias.
- Indicadores clásicos (RSI, MACD, etc.).
- Predicción direccional next-bar.
- RL/policy learning end-to-end.
- Optimizar por F1/accuracy.
- Inflar features sin mejora en PnL/DD.

---

## Resultado esperado
Un sistema que:
- Clasifica estado con alta robustez.
- Habilita/prohíbe familias de setups con reglas claras.
- Reduce errores estructurales.
- Aumenta estabilidad y supervivencia del trader.
- Prioriza claridad y disciplina sobre actividad.

---

## Implementación actual (resumen)
Esta sección documenta **cómo se implementó** el State Engine respetando estrictamente las definiciones de este documento.
No introduce lógica adicional ni heurísticas implícitas.

### Fuentes de datos (MT5)
- Se utiliza un conector dedicado (MT5Connector) para obtener OHLCV en **H1** desde MetaTrader 5.
- No se usan fuentes externas ni feeds alternativos.
- El rango temporal de entrenamiento se define explícitamente por symbol, start, end.
- El manejo de timezone, resampling y NaN se realiza antes de cualquier cálculo de features.
- Se filtran velas H1 **cerradas** usando la hora del servidor MT5 (no se incluye la vela en formación).

### Cálculo de variables PA-first (H1)
- Todas las variables se calculan **exclusivamente con información ≤ t**.
- Se usan ventanas fijas:
   - W = 24 H1 para contexto.
   - N = 8 H1 para comportamiento reciente.
- La normalización distingue explícitamente:
   - métricas de contexto → ATR_W
   - métricas recientes / ruptura → ATR_N
- Variables implementadas:
   - Path
   - Range_W
   - CloseLocation
      - Si H_W == L_W, se asigna CloseLocation = 0.5.
    - BreakMag
    - ReentryCount
    - InsideBarsRatio
    - SwingCount
       - Calculado solo con pivots confirmados hasta t-1 (no se usa información futura).
    - ATR_N / ATR_W
    - Pendientes (ERSlope, RangeSlope) disponibles como opcionales.

No se calculan ni almacenan indicadores clásicos ni niveles externos.

### Auto-etiquetado (bootstrap)
- El bootstrap se ejecuta **offline**, previo al entrenamiento.
- Se basa únicamente en las reglas explícitas definidas en este documento.
- Las etiquetas bootstrap no representan verdad de mercado, solo un punto de partida.
- Las variables usadas para definir el bootstrap no se reutilizan automáticamente como features core del modelo.
- En caso de conflicto entre reglas:
   - Se aplica la prioridad conservadora:
   - **TRANSICIÓN > TENDENCIA > BALANCE**.

Las etiquetas bootstrap **no se guardan como features** y no están disponibles durante inferencia.

### Modelo principal (StateEngine)
- Se entrena un **LightGBM multiclass** con tres clases:
   - BALANCE
   - TRANSICIÓN
   - TENDENCIA
- El modelo recibe únicamente el set de features definido en la sección correspondiente.
- Salidas del modelo:
   - state_hat: clase predicha (argmax).
   - margin = P(top1) − P(top2) como medida de confianza relativa.
   - Probabilidades completas (probas) solo para análisis y reporting.

Por defecto:
   - **No se asume calibración de probabilidades**.
   - El sistema no depende de umbrales directos sobre P(state).

### Gating determinista (`ALLOW_*`)
- El gating es una capa lógica separada, no entrenada.
- Las decisiones ALLOW_* se derivan de:
   - state_hat
   - margin
   - guardrails simples (ruptura, reingreso, compresión/expansión).

Ejemplos implementados:
   - ALLOW_trend_pullback
   - ALLOW_balance_fade
   - ALLOW_transition_failure

Si state_hat == TRANSICIÓN:
   - El sistema **prohíbe swing direccional por defecto**, independientemente del margin.

### Pipeline de entrenamiento
- El pipeline sigue estrictamente este orden:
   1) Descarga de datos (MT5).
   2) Limpieza básica y alineación temporal.
      - Se aplica guardrail temporal con la hora del servidor MT5 para usar solo velas cerradas.
   3) Cálculo de variables PA-first.
   4) Generación de etiquetas bootstrap (offline).
   5) Entrenamiento del modelo StateEngine.
   6) Persistencia del modelo entrenado.

No se optimiza por accuracy o F1.
La evaluación se realiza **posteriormente**, condicionando PnL y drawdown por estado y ALLOW_*.

### Inferencia
- En inferencia online:
   - Se recalculan únicamente las variables PA-first.
   - El modelo produce state_hat y margin.
   - El gating determina explícitamente qué familias de setups están habilitadas.

- El sistema puede devolver **“no operar”** como resultado explícito y válido.

### Nota final de implementación

La implementación está diseñada para:
   - ser **auditable**,
   - evitar leakage temporal,
   - mantener separación clara entre percepción (ML) y decisión (reglas),
   - y permitir ajustes de parámetros **sin reescribir la lógica del sistema**.
