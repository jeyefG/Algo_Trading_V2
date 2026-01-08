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
   Si `H_W == L_W`, fijar `CloseLocation = 0.5` (neutral) o marcar NaN y descartar.

6) **Ruptura de rango (magnitud, no dirección)**  
   `BreakMag = |C_t - clamp(C_t, L_W, H_W)| / ATR_N`  
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
   Se usa pivote simple y solo cuenta giros, no dirección.  
   Se computa **solo hasta t-1** usando pivots confirmados (evitar leakage).
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

**Separación obligatoria (elige una opción y respétala):**
- **Opción A (recomendada):** bootstrap usa (`ER`, `NetMove`) + **1** confirmación secundaria (p. ej. `ReentryCount` **o** `InsideBarsRatio` **o** `BreakMag`).  
  Features core **no** incluyen `ER`/`NetMove`, pero sí pueden incluir el resto.
- **Opción B:** bootstrap usa un set **distinto** al set de features (p. ej. solo `ER`, `NetMove`, `Range_W`).  
  Las micro-features quedan **solo** para el modelo.

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

**Subtags internos (no son clases nuevas):**
- `transition_type = FAILURE` para (a)  
- `transition_type = DRIVE` para (b)  
Uso: análisis y guardrails (no para entrenar si se mantiene simple).

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
- **Calibración**: si se quiere usar umbrales sobre `P()`, se debe aplicar calibración (Platt/Isotonic) y validar estabilidad out-of-sample.
- **Por defecto**, el gating usa `state_hat` + `margin`.

### Modelos auxiliares (máximo 1, por defecto 0)
Solo si mejora PnL/DD OOS:
- **VolatilityRegime (binario)**: expansión vs rotación (sin dirección).  
  Si no mejora, se elimina.

---

## Gating (`ALLOW_*`) — política determinista
`ALLOW` **no se entrena**. Es una capa lógica.

Regla general:
- El gating se basa en `state_hat`, `margin` y guardrails simples (volatilidad/compresión).
- No se usa `P(state) ≥ umbral` salvo calibración validada.
- Los umbrales de `margin` **no son fijos**: se calibran por percentiles OOS (p. ej. p60) o targeting de tasa de “no trade”.

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
Esta sección documenta **cómo se implementó** el State Engine respetando el texto original.

### Fuentes de datos (MT5)
- Se añadió un conector dedicado para obtener OHLCV H1 desde MetaTrader 5, evitando fuentes externas.
- El flujo de entrenamiento usa exclusivamente este conector para descargar datos del símbolo y rango de fechas solicitados.

### Features PA-first (H1, W=24)
- Las features core se calculan sobre H1 con ventana fija W=24 y normalización según ATR_W/ATR_N.
- Se incluyen `Path`, `Range_W`, `CloseLocation`, `BreakMag`, `ReentryCount`, `InsideBarsRatio`, `SwingCount`.
- Se agrega `ATR_N / ATR_W` como ratio de cambio de volatilidad.
- Las pendientes (`ERSlope`, `RangeSlope`) están disponibles como opcionales si se habilitan.

### Auto-etiquetado (bootstrap)
- Se implementaron reglas iniciales de bootstrap con prioridad conservadora para TRANSICIÓN.
- El etiquetado ocurre offline y no se usa como feature.

### Modelo principal (StateEngine)
- Se implementó un wrapper de LightGBM multiclass para entrenar y predecir `state_hat`, `margin` y probabilidades.
- Las probabilidades son opcionales para reporting; el gating usa `state_hat + margin` por defecto.

### Gating determinista (`ALLOW_*`)
- Se implementó una capa de reglas deterministas para convertir `state_hat` y `margin` en ALLOWs.
- Ejemplos soportados: `ALLOW_trend_pullback`, `ALLOW_balance_fade`, `ALLOW_transition_failure`.

### Pipeline y script de entrenamiento
- Se agregó un pipeline mínimo que construye features + labels y entrena el modelo.
- El script de entrenamiento recibe `symbol`, `start`, `end`, entrena, guarda el modelo y calcula `ALLOW_*`.
