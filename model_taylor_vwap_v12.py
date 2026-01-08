# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 10:06:59 2025

@author: jfcog

Genera los niveles de Taylor y VWAP para n_días. Luego filtra los días en que no 
hay tick_volume

Luego se identifican los eventos de toque en las zonas de taylor (4 valores), 
si se está en la zona de intersección, y si se tocan las líneas de VWAP

Se definen las etiquetas en base a si hubo un toque y luego un rebote que 
alcanzara un nivel jerárquico posterior
"""

import MetaTrader5 as mt5
import pandas as pd
import ta
from ta.volatility import AverageTrueRange
from datetime import datetime, timedelta, time
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_predict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from symbols_configs import SYMBOL_CONFIGS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import os
import joblib


class MT5Connector:
    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError(f"No se pudo conectar a MetaTrader 5: {mt5.last_error()}")

    def shutdown(self):
        mt5.shutdown()

    def obtener_d1(self, symbol, fecha_sesion, num_dias, n=10):
        """Obtiene `n` velas diarias previas a ``fecha_sesion``.

        Esta lógica replica la empleada en ``Taylor_zone_V5_Max_min_VWAP.py``:
        se piden las últimas ``n`` velas D1 y se descarta la vela del día en
        curso si está presente.
        """
        rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, num_dias, n)
        if rates_d1 is None:
            return None

        df = pd.DataFrame(rates_d1)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        df = df[df.index < fecha_sesion]
        if len(df) < 4:
            return None  # Esto previene el error más arriba
        return df

    def obtener_m5(self, symbol, fecha_sesion):
        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M5,
            fecha_sesion - timedelta(hours=4),
            fecha_sesion + timedelta(days=1)
        )
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df[df.index.normalize() == fecha_sesion]

class TaylorVWAPCalculator:
    def __init__(self, vwap_win=14):
        self.vwap_win = vwap_win

    def calcular_zonas_taylor(self, df_d1):
        df_d1['PP'] = (df_d1['high'] + df_d1['low'] + df_d1['close']) / 3
        df_d1['Rally'] = df_d1['high'] - df_d1['low'].shift(1)
        df_d1['Rally_avg'] = df_d1['Rally'].rolling(3).mean()
        RR1 = df_d1.iloc[-1]['low'] + df_d1.iloc[-1]['Rally_avg']
        df_d1['BH'] = df_d1['high'] - df_d1['high'].shift(1)
        df_d1['BH_avg'] = df_d1['BH'].rolling(3).mean()
        RR3 = df_d1.iloc[-1]['high'] + df_d1.iloc[-1]['BH_avg']
        RR2 = df_d1.iloc[-1]['high']
        PP = df_d1.iloc[-1]['PP']
        RR4 = 2 * PP - df_d1.iloc[-1]['low']
        zona_alta = (RR1 + RR2 + RR3 + RR4) / 4

        df_d1['Decline'] = df_d1['high'].shift(1) - df_d1['low']
        df_d1['Decline_avg'] = df_d1['Decline'].rolling(3).mean()
        SS1 = df_d1.iloc[-1]['high'] - df_d1.iloc[-1]['Decline_avg']
        df_d1['BL'] = df_d1['low'].shift(1) - df_d1['low']
        df_d1['BL_avg'] = df_d1['BL'].rolling(3).mean()
        SS3 = df_d1.iloc[-1]['low'] - df_d1.iloc[-1]['BL_avg']
        SS2 = df_d1.iloc[-1]['low']
        SS4 = 2 * PP - df_d1.iloc[-1]['high']
        zona_baja = (SS1 + SS2 + SS3 + SS4) / 4

        return zona_baja, zona_alta, zona_alta - zona_baja

    def calcular_vwap(self, df):
        df['typical'] = df[['high', 'low', 'close']].mean(axis=1)
        df['HH'] = df['high'].rolling(self.vwap_win, min_periods=1).max()
        df['LL'] = df['low'].rolling(self.vwap_win, min_periods=1).min()
        df['HV'] = df['tick_volume'].rolling(self.vwap_win, min_periods=1).max()
        df['pivot_price'] = (df['HH'] + df['LL'] + df['close']) / 3.0
        df['pivot_vol'] = df['HV']
        df['cum_pv'] = df['pivot_price'] * df['pivot_vol']
        df['vwap'] = df['cum_pv'].cumsum() / df['pivot_vol'].cumsum()
        df['cum_p2v'] = (df['pivot_price'] ** 2 * df['pivot_vol'])
        df['sigma'] = ((df['cum_p2v'].cumsum() / df['pivot_vol'].cumsum()) - df['vwap'] ** 2).pow(0.5)
        df['vwap_hi'] = df['vwap'] + 2 * df['sigma']
        df['vwap_lo'] = df['vwap'] - 2 * df['sigma']
        return df
    
class FeatureEnricher:
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, ema_periods=[5, 20, 50, 200]):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_periods = ema_periods
        
    def enriquecer_intradia(self, df):
        """
        Arega features que son sólo de contexto diario
        
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        
        """
        
        df = df.copy()

        # Triggers de toque en t-1
        niveles = ['buy_low', 'buy_high', 'sell_low', 'sell_high', 'vwap_lo', 'vwap_hi']
        for nivel in niveles:
            if 'high' in nivel:
                df[f'toque_{nivel}'] = df['high'].shift(1) >= df[nivel].shift(1)
            else:
                df[f'toque_{nivel}'] = df['low'].shift(1) <= df[nivel].shift(1)

        df['toque_vwap'] = df['low'].shift(1) <= df['vwap'].shift(1)
        
        # 1. Detectar si hay intersección en t-1
        df['interseccion_low'] = df[['buy_low', 'sell_low']].max(axis=1)
        df['interseccion_high'] = df[['buy_high', 'sell_high']].min(axis=1)
        df['hay_interseccion'] = df['interseccion_low'].shift(1) <= df['interseccion_high'].shift(1)
        
        # Inicializar todas las zonas como columnas binarias
        zonas = ['zona_inferior_extrema', 'zona_baja_externa', 'zona_interseccion',
                 'zona_neutra','zona_alta_externa', 'zona_superior_extrema']
        for zona in zonas:
            df[zona] = 0
        
        # Variables auxiliares con shift(1) para evitar leakage
        low_min = df[['buy_low', 'sell_low']].min(axis=1).shift(1)
        high_max = df[['buy_high', 'sell_high']].max(axis=1).shift(1)
        inter_low = df['interseccion_low'].shift(1)
        inter_high = df['interseccion_high'].shift(1)
        high_1 = df['high'].shift(1)
        low_1 = df['low'].shift(1)
        
        # --- CASO CON INTERSECCIÓN ---
        cond_inter = df['hay_interseccion']
        
        df.loc[cond_inter, 'zona_inferior_extrema'] = (high_1 < low_min)[cond_inter].astype(int)
        df.loc[cond_inter, 'zona_baja_externa'] = (
            (low_1 >= low_min) & (high_1 < inter_low))[cond_inter].astype(int)
        df.loc[cond_inter, 'zona_interseccion'] = (
            (low_1 >= inter_low) & (high_1 <= inter_high)
        )[cond_inter].astype(int)
        df.loc[cond_inter, 'zona_alta_externa'] = (
            (low_1 > inter_high) & (high_1 <= high_max)
        )[cond_inter].astype(int)
        df.loc[cond_inter, 'zona_superior_extrema'] = (low_1 > high_max)[cond_inter].astype(int)
        
        # --- CASO SIN INTERSECCIÓN ---
        cond_no_inter = ~df['hay_interseccion']
        
        df.loc[cond_no_inter, 'zona_inferior_extrema'] = (
            high_1 < df['buy_low'].shift(1)
        )[cond_no_inter].astype(int)
        df.loc[cond_no_inter, 'zona_baja_externa'] = (
            (low_1 >= df['buy_low'].shift(1)) & (high_1 <= df['buy_high'].shift(1))
        )[cond_no_inter].astype(int)
        df.loc[cond_no_inter, 'zona_neutra'] = (
            (low_1 >= df['buy_high'].shift(1)) & (high_1 <= df['sell_low'].shift(1))
        )[cond_no_inter].astype(int)
        df.loc[cond_no_inter, 'zona_alta_externa'] = (
            (low_1 >= df['sell_low'].shift(1)) & (high_1 <= df['sell_high'].shift(1))
        )[cond_no_inter].astype(int)
        df.loc[cond_no_inter, 'zona_superior_extrema'] = (
            low_1 > df['sell_high'].shift(1)
        )[cond_no_inter].astype(int)

        # Precios previos para cálculo de cruces fuera de entrenamiento
        #df['vwap_prev'] = df['vwap'].shift(1)
        #df['close_prev'] = df['close'].shift(1)

        # Distancias
        #df['above_vwap'] = df['close'] > df['vwap']
        df['dist_to_vwap'] = df['close'] - df['vwap']
        df['dist_to_vwap_hi'] = df['close'] - df['vwap_hi']
        df['dist_to_vwap_lo'] = df['close'] - df['vwap_lo']
        df['dist_to_buy_low'] = df['close'] - df['buy_low']
        df['dist_to_sell_high'] = df['close'] - df['sell_high']
        df['dist_to_buy_high'] = df['close'] - df['buy_high']
        df['dist_to_sell_low'] = df['close'] - df['sell_low']
        
        df['hora_normalizada'] = df.index.hour + df.index.minute / 60
        
        # Nuevos indicadores experimentales
        
        # 1. Velocidad del precio (cambio por unidad de tiempo)
        #df['velocity'] = df['close'].diff() / df['hora_normalizada'].diff()
        
        # 2. Pendiente del VWAP (slope VWAP en últimas 3 velas)
        df['dist_vwap_sigma'] = df['dist_to_vwap'] / df['sigma']
        df['dist_vwap_atr'] = df['dist_to_vwap'] / df['atr_14']
        #df['dist_buy_low_atr'] = df['dist_to_buy_low'] / df['atr_14']
        
        # 4. Tendencia reciente del VWAP (promedio móvil del slope)
        df['vwap_slope_ema'] = df['vwap'].diff().ewm(span=5, adjust=False).mean()
        
        # 5. Señales tipo price action
        """
        rango = df['high'] - df['low']
        cuerpo = abs(df['close'] - df['open'])
        
        df['body_ratio'] = cuerpo / rango.replace(0, 1e-9)
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / rango.replace(0, 1e-9)
        df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / rango.replace(0, 1e-9)
        
        # Gaps y breakout
        df['gap_up'] = df['open'] - df['close'].shift(1)
        df['gap_down'] = df['close'].shift(1) - df['open']
        df['close_breaks_prev_high'] = (df['close'] > df['high'].shift(1)).astype(int)
        df['close_breaks_prev_low'] = (df['close'] < df['low'].shift(1)).astype(int)
        
        # Secuencias de altos y bajos crecientes/decrecientes
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        df['candle_range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['is_pinbar'] = ((df['lower_wick'] > df['candle_range'] * 0.6) & (df['upper_wick'] < df['candle_range'] * 0.2))
        
        # Cast booleanos a int para modelos
        df['is_pinbar'] = df['is_pinbar'].astype(int)
        """
                
        # --- Vecindades respecto a niveles críticos ---
        
        ancho_zona = abs(df['zona_alta'].iloc[0] - df['zona_baja'].iloc[0])

        def _vecindad(close, nivel, sigma, ancho_zona=None, es_taylor=False):
            if es_taylor:
                umbral = 0.15 * ancho_zona
            else:
                umbral = 0.2 * sigma
        
            return ((close - nivel).abs() <= umbral).astype(int)

        niveles_vec = {
            'buy_low': True,
            'buy_high': True,
            'sell_low': True,
            'sell_high': True,
            'vwap_lo': False,
            'vwap_hi': False,
        }
               

        for nivel, es_taylor in niveles_vec.items():
            ancho = ancho_zona if es_taylor else None
            df[f'vecindad_{nivel}'] = _vecindad(df['close'], df[nivel], df['sigma'], ancho, es_taylor)
            
            df[f'vecindad_persist_{nivel}'] = (
                df[f'vecindad_{nivel}'].rolling(5, min_periods=1).sum() >= 3
            ).astype(int)
        
        df['vecindad_acumulada'] = df[[f'vecindad_{n}' for n in niveles_vec]].sum(axis=1)
        
        return df

    def enriquecer_historico(self, df):
        """Calcula indicadores que requieren la serie completa por símbolo."""
        df_list = []
        for symbol, data in df.groupby('symbol'):
            data = data.sort_index().copy()
            
            # RSI
            data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=self.rsi_period).rsi()
            data['rsi_ema'] = data['rsi'].ewm(span=28, adjust=False).mean()
            data['rsi_above_ema'] = (data['rsi'] > data['rsi_ema']).astype(int)
            data['rsi_below_ema'] = (data['rsi'] < data['rsi_ema']).astype(int)
            data['rsi_vs_ema'] = data['rsi'].fillna(0) - data['rsi_ema'].fillna(0)
            data['rsi_slope'] = data['rsi'].diff().fillna(0)
            data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
            data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
            
            # MACD
            macd = ta.trend.MACD(data['close'], window_slow=self.macd_slow, window_fast=self.macd_fast, window_sign=self.macd_signal)
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_diff'] = macd.macd_diff()
            data['macd_bullish_cross'] = (data['macd'] > data['macd_signal']).astype(int)
            data['macd_bearish_cross'] = (data['macd'] < data['macd_signal']).astype(int)
            
            # ATR
            atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14, fillna=False)
            data['atr_14'] = atr.average_true_range()
            
            # EMAS            
            for p in self.ema_periods:
                data[f'ema_{p}'] = data['close'].ewm(span=p, adjust=False).mean()
        
            def _vec(close, nivel, sigma):
                umbral = 0.2 * sigma
                return ((close - nivel).abs() <= umbral).astype(int)
        
            for p in self.ema_periods:
                col = f'ema_{p}'
                if col in data:
                    data[f'vecindad_{col}'] = _vec(data['close'], data[col], data['sigma'])
                    data[f'vecindad_persist_{col}'] = (
                        data[f'vecindad_{col}'].rolling(5, min_periods=1).sum() >= 3
                    ).astype(int)
                    
                    # Flag 1/0: ¿precio > SMA?
                    data[f'above_ema_{p}'] = (data['close'] > data[f'ema_{p}']).astype(int)
                    data[f'below_ema_{p}'] = (data['close'] < data[f'ema_{p}']).astype(int)

                    # Distancia normalizada (en ATR para hacerla comparable)
                    # Evita división por cero con .replace
                    data[f'dist_to_ema_{p}'] = (data['close'] - data[f'ema_{p}'])
                    
            # Score acumulado de tendencia: cuántas MAs tengo a favor
            data['ema_trend_score'] = data[[f'above_ema_{p}' for p in self.ema_periods]].sum(axis=1)
            data['ema_trend_score_below'] = data[[f'below_ema_{p}' for p in self.ema_periods]].sum(axis=1)
            
            data['ema5_vs_ema20'] = data['ema_5'] - data['ema_20']
            data['ema5_cross_ema20'] = (data['ema_5'] > data['ema_20']).astype(int)
            
            data['ema20_vs_ema50'] = data['ema_20'] - data['ema_50']
            data['ema20_cross_ema50'] = (data['ema_20'] > data['ema_50']).astype(int)
            
            data['ema50_vs_ema200'] = data['ema_50'] - data['ema_200']
            data['ema50_cross_ema200'] = (data['ema_50'] > data['ema_200']).astype(int)
            
            #Tendencia:
            data['trend_score_alcista'] = (
                                            data['above_ema_5'] +
                                            data['above_ema_20'] +
                                            data['above_ema_50'] +
                                            data['above_ema_200'] +
                                            data['macd_bullish_cross'] +
                                            data['rsi_above_ema'] +
                                            (data['ema5_vs_ema20'] > 0).astype(int) +
                                            (data['ema20_vs_ema50'] > 0).astype(int) +
                                            (data['ema50_vs_ema200'] > 0).astype(int)
                                            )
            data['trend_score_bajista'] = (
                                            data['below_ema_5'] +
                                            data['below_ema_20'] +
                                            data['below_ema_50'] +
                                            data['below_ema_200'] +
                                            data['macd_bearish_cross'] +
                                            data['rsi_below_ema'] +
                                            (data['ema5_vs_ema20'] < 0).astype(int) +
                                            (data['ema20_vs_ema50'] < 0).astype(int) +
                                            (data['ema50_vs_ema200'] < 0).astype(int)
                                            )

            df['vol_surge'] = data['tick_volume'] / data['tick_volume'].rolling(14).mean()
            df['momentum_3'] = data['close'].pct_change(3)
            df['momentum_5'] = data['close'].pct_change(5)
        
            vec_cols = [c for c in data.columns if c.startswith('vecindad_') and not c.startswith('vecindad_persist')]
            data['vecindad_acumulada'] = data[vec_cols].sum(axis=1)
            df_list.append(data)

        return pd.concat(df_list).sort_index()

    def aplicar_clusters(self, df, curr_symbol, features=None, n_clusters=50, model_dir="clusters", evaluate=False):
        """Aplica KMeans y, opcionalmente, evalúa la calidad de los clusters.

        Si existen modelos guardados para cada símbolo se reutilizan para
        mantener consistencia. Con ``evaluate=True`` se imprime el *silhouette
        score* de cada símbolo para verificar la cohesión interna de los
        clusters.
        """
        print(f"Appliying {n_clusters} Clusters {curr_symbol}...")
        os.makedirs(model_dir, exist_ok=True)
        
        if features is None:
            features = ['rsi_above_ema',
                        'rsi_below_ema',
                        'rsi_overbought',
                        'rsi_oversold',
                        'macd_bullish_cross',
                        'macd_bearish_cross',
                        'above_ema_5',
                        'below_ema_5',
                        'above_ema_20',
                        'below_ema_20',
                        'above_ema_50',
                        'below_ema_50',
                        'above_ema_200',
                        'below_ema_200',
                        'ema_trend_score',
                        'ema_trend_score_below',
                        'trend_score_alcista',
                        'trend_score_bajista',
                        ]

        df_list = []
        eval_scores = {}
        for symbol, data in df.groupby('symbol'):
            datos = (
                data[features]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            scaler_path = os.path.join(model_dir, f"scaler_{symbol}.pkl")
            kmeans_path = os.path.join(model_dir, f"kmeans_{symbol}.pkl")

            if os.path.exists(kmeans_path) and os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                kmeans = joblib.load(kmeans_path)
                datos_scaled = scaler.transform(datos)
                clusters = kmeans.predict(datos_scaled)
            else:
                scaler = StandardScaler()
                datos_scaled = scaler.fit_transform(datos)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(datos_scaled)
                joblib.dump(scaler, scaler_path)
                joblib.dump(kmeans, kmeans_path)

            if evaluate:
                try:
                    eval_scores[symbol] = silhouette_score(datos_scaled, clusters)
                except Exception:
                    eval_scores[symbol] = float('nan')

            data = data.copy()
            data['cluster'] = clusters
            df_list.append(data)

        if evaluate and eval_scores:
            print("Silhouette scores:")
            for sym, score in eval_scores.items():
                if score == score:  # not NaN
                    print(f"  {sym}: {score:.3f}")
                else:
                    print(f"  {sym}: N/A")

        return pd.concat(df_list).sort_index()  

class LabelGenerator:
    def __init__(
        self,
        n_ahead=24,
        tolerancia_rebote_sigma=0.2,
        tolerancia_escape_sigma=0.3,
        umbral_escape_sigma=1.5,
        tolerancia_rebote_atr=0.5,
        tolerancia_escape_atr=0.5,
        umbral_escape_atr=2.0,
    ):
        self.n_ahead = n_ahead
        self.tolerancia_rebote_sigma = tolerancia_rebote_sigma
        self.tolerancia_escape_sigma = tolerancia_escape_sigma
        self.umbral_escape_sigma = umbral_escape_sigma
        self.tolerancia_rebote_atr = tolerancia_rebote_atr
        self.tolerancia_escape_atr = tolerancia_escape_atr
        self.umbral_escape_atr = umbral_escape_atr

    def generar_etiquetas_cruce_y_rebote(self, df, niveles, min_dist=0):
        """Genera etiquetas de cruce/rebote y escape usando la vecindad
        correspondiente al nivel más cercano en cada fila.
        """
        etiquetas = pd.DataFrame(index=df.index)

        # Calcular nivel más cercano para cada fila para descartar vecindades
        distancias = pd.concat(
            [(df['close'] - df[n]).abs().rename(n) for n in niveles], axis=1
        )
        nivel_mas_cercano = distancias.idxmin(axis=1)
    
        niveles_direccion = {
            "buy_low": "support",
            "sell_high": "resistance",
            "vwap_lo": "support",
            "vwap_hi": "resistance",
            "buy_high": "resistance",
            "sell_low": "support",
        }
        
        symbol = df['symbol'].iloc[0]
        market_end_tuple = SYMBOL_CONFIGS.get(symbol, {}).get('market_end')
        if market_end_tuple is None:
            raise ValueError(f"No se encontró 'market_end' para el símbolo {symbol} en SYMBOL_CONFIGS")
        
        # Convertir (HH, MM) → hora en formato decimal
        hora_fin = market_end_tuple[0] + market_end_tuple[1] / 60
        delta_horas = (self.n_ahead * 5) / 60  # 5 minutos por vela
        hora_limite = hora_fin - delta_horas
        
        for nivel in niveles:
            rebote_label = f"etiqueta_rebote_vwap_{nivel}"
            escape_label = f"etiqueta_escape_tendencia_{nivel}"
            etiquetas[rebote_label] = 0
            etiquetas[escape_label] = 0
    
            direccion = niveles_direccion.get(nivel, None)
            if direccion is None:
                continue  # nivel no válido

            vecs = df[(df[f"vecindad_{nivel}"] == 1) & (nivel_mas_cercano == nivel)]
            #vecs = vecs[vecs['hora_normalizada'] <= hora_limite]
    
            for idx in vecs.index:
                if idx not in df.index:
                    continue
                actual_idx = df.index.get_loc(idx)
                if actual_idx + self.n_ahead >= len(df):
                    continue
    
                sub_df = df.iloc[actual_idx + 1 : actual_idx + 1 + self.n_ahead]
                close_ini = df.loc[idx, 'close']
                vwap_ini = df.loc[idx, 'vwap']
                sigma = df.loc[idx, 'sigma']
                atr = df.loc[idx, 'atr_14']
                nivel_valor = df.loc[idx, nivel]
    
                # Validación de dirección del VWAP
                if direccion == "support" and vwap_ini <= nivel_valor:
                    continue
                elif direccion == "resistance" and vwap_ini >= nivel_valor:
                    continue
    
                # Distancia mínima a la que debo estar del vwap para que tenga sentido
                # etiquetar un rebote
                distancia_vwap = abs(vwap_ini - close_ini)
                if distancia_vwap < min_dist:
                    continue
    
                tolerancia_rebote = max(
                    self.tolerancia_rebote_sigma * sigma,
                    self.tolerancia_rebote_atr * atr,
                )
                tolerancia_escape = max(
                    self.tolerancia_escape_sigma * sigma,
                    self.tolerancia_escape_atr * atr,
                )
    
                min_dist_a_vwap = (sub_df['close'] - sub_df['vwap']).abs().min()
                
                se_acerco_para_rebote = min_dist_a_vwap < tolerancia_rebote
                cruce_ascendente = (sub_df['close'] > sub_df['vwap']).any() and close_ini < vwap_ini
                cruce_descendente = (sub_df['close'] < sub_df['vwap']).any() and close_ini > vwap_ini
    
                if se_acerco_para_rebote or cruce_ascendente or cruce_descendente:
                    etiquetas.at[idx, rebote_label] = 1
                    continue
    
                if min_dist_a_vwap <= tolerancia_escape:
                    continue  # no escapó suficientemente lejos
    
                # Determinar dirección de escape
                if close_ini > vwap_ini:
                    avance = sub_df['close'].max() - close_ini
                    movimiento_alcista = True
                else:
                    avance = close_ini - sub_df['close'].min()
                    movimiento_alcista = False
    
                if 'ema_50' in df.columns and 'ema_200' in df.columns:
                    tendencia_alcista = df.loc[idx, 'ema_50'] > df.loc[idx, 'ema_200']
                else:
                    tendencia_alcista = df.loc[idx, 'vwap_slope_ema'] > 0
    
                tendencia_ok = (
                    (movimiento_alcista and tendencia_alcista)
                    or (not movimiento_alcista and not tendencia_alcista)
                )
    
                avance_minimo = max(
                    self.umbral_escape_sigma * sigma,
                    self.umbral_escape_atr * atr,
                )
    
                if avance >= avance_minimo and tendencia_ok:
                    etiquetas.at[idx, escape_label] = 1
    
        return etiquetas

calculator = TaylorVWAPCalculator(vwap_win=14)

class SesionProcessor:
    def __init__(self, symbol, fecha_sesion, apertura_mq, df_d1, rates_m5, calculator):
        self.symbol = symbol
        self.fecha_sesion = fecha_sesion
        self.apertura_mq = apertura_mq
        self.df_d1 = df_d1
        self.rates_m5 = rates_m5
        self.calculator = calculator

    def procesar(self):
        df = self.rates_m5.copy()
        if df.empty:
            return None

        zona_baja, zona_alta, _ = self.calculator.calcular_zonas_taylor(self.df_d1)
        ancho_zona = zona_alta - zona_baja
        df = self.calculator.calcular_vwap(df)
        df_premarket = df[df.index < self.apertura_mq]
        if df_premarket.empty:
            return None

        precio_min = df_premarket['low'].min()
        precio_max = df_premarket['high'].max()
        buy_low = precio_min
        buy_high = buy_low + ancho_zona
        sell_high = precio_max
        sell_low = sell_high - ancho_zona
        #interseccion_low = max(buy_low, sell_low)
        #interseccion_high = min(buy_high, sell_high)
        

        df['symbol'] = self.symbol
        df['fecha'] = self.fecha_sesion.date()
        df['buy_low'] = buy_low
        df['buy_high'] = buy_high
        df['sell_low'] = sell_low
        df['sell_high'] = sell_high
        #df['interseccion_low'] = interseccion_low
        #df['interseccion_high'] = interseccion_high

        #zona_baja, zona_alta, _ = self.calculator.calcular_zonas_taylor(self.df_d1)
        df['zona_baja'] = zona_baja
        df['zona_alta'] = zona_alta
        
        df['en_premarket'] = df.index < self.apertura_mq

        return df
    
class DatasetBuilder:
    def __init__(self, connector, calculator, enricher):
        self.connector = connector
        self.calculator = calculator
        self.enricher = enricher

    def procesar_sesion(self, symbol, fecha_sesion, apertura_mq, num_dias):
        df_d1 = self.connector.obtener_d1(symbol, fecha_sesion, num_dias + 1)
        if df_d1 is None:
            print(f"[{symbol} - {fecha_sesion}] ❌ df_d1 no disponible")
            return None

        rates_m5 = self.connector.obtener_m5(symbol, fecha_sesion)
        if rates_m5 is None:
            #print(f"[{symbol} - {fecha_sesion}] ❌ M5 no disponible")
            return None

        # Aplicar escala si está definida para el símbolo
        scale = SYMBOL_CONFIGS.get(symbol, {}).get('scale', 1)
        if scale != 1:
            for col in ['open', 'high', 'low', 'close']:
                df_d1[col] = df_d1[col] * scale
                rates_m5[col] = rates_m5[col] * scale

        procesador = SesionProcessor(
            symbol, fecha_sesion, apertura_mq,
            df_d1, rates_m5,
            self.calculator
        )

        df_base = procesador.procesar()
        if df_base is None:
            return None

        return df_base
    
class ModelTrainer:
    def __init__(self, df, etiquetas_objetivo, excluir_columnas=[], ventana=12):
        self.df = df.copy()
        self.etiquetas_objetivo = etiquetas_objetivo
        self.excluir_columnas = excluir_columnas
        self.ventana = ventana
        self.modelos = {}
        self.reportes = {}
        # Almacenan resultados específicos por símbolo
        self.modelos_por_symbol = {}
        self.reportes_por_symbol = {}
        self.cont = 1
        self.n_ahead = 24

    def _seleccionar_features(self, target):
        etiquetas = [col for col in self.df.columns if col.startswith('etiqueta_')]

        # Exclusiones base
        excluidas = ['symbol', 
                     'fecha', 
                     'time',
                     'real_volume',
                     'spread',
                     'typical', 
                     'HH', 
                     'LL', 
                     'HV', 
                     'pivot_price', 
                     'pivot_vol',
                     'cum_pv', 
                     'cum_p2v', 
                     'sigma', 
                     'interseccion_low', 
                     'interseccion_high', 
                     'zona_baja',
                     'zona_alta'] + etiquetas + self.excluir_columnas
                     
    
        # Evitar leakage por flags (aunque se calculen en t-1)
        excluidas += [col for col in self.df.columns if col.startswith('toque_')]

        # Excluir vecindad puntual para evitar leakage (mantener persistencia y acumulada)
        for col in self.df.columns:
            if col.startswith('vecindad_') and not col.startswith('vecindad_persist') and col != 'vecindad_acumulada':
                excluidas.append(col)
    
        # También puedes excluir otros flags específicos si los tuvieras
        # excluidas += [col for col in self.df.columns if col.startswith('cruce_') or 'rebote_' in col]
    
        features = [col for col in self.df.columns if col not in excluidas]
        return features

    def _split_por_dias(self, df, features, target, proporcion=0.8):
        """Divide el DataFrame en conjuntos de entrenamiento y prueba por días."""
        dias = df['fecha'].drop_duplicates().sort_values()
        if len(dias) < 2:
            return df[features], df[features], df[target], df[target]

        split_idx = int(proporcion * len(dias))
        dias_entrenamiento = dias.iloc[:split_idx]

        mask_train = df['fecha'].isin(dias_entrenamiento)
        X_train = df.loc[mask_train, features]
        X_test = df.loc[~mask_train, features]
        y_train = df.loc[mask_train, target]
        y_test = df.loc[~mask_train, target]

        return X_train, X_test, y_train, y_test

    def _split_por_ventanas(self, df, features, target, proporcion=0.8, ventana=12):
        """Divide usando ventanas móviles de ``ventana`` filas.

        Parameters
        ----------
        df : DataFrame
            Datos ya enriquecidos y ordenados cronológicamente.
        features : list
            Columnas a utilizar como variables de entrada.
        target : str
            Columna objetivo.
        proporcion : float, optional
            Porcentaje de ventanas para entrenamiento. El resto se usa para prueba.
        ventana : int, optional
            Tamaño de cada ventana en número de filas/velas.

        Returns
        -------
        X_train, X_test, y_train, y_test : DataFrames/Series
            Conjuntos de entrenamiento y prueba basados en las ventanas.
        """

        df_sorted = df.sort_values('time')
        indices = np.arange(0, len(df_sorted), ventana)

        if len(indices) < 2:
            return df[features], df[features], df[target], df[target]

        split_idx = int(proporcion * len(indices))

        train_mask = np.zeros(len(df_sorted), dtype=bool)
        for start in indices[:split_idx]:
            end = min(start + ventana, len(df_sorted))
            train_mask[start:end] = True

        X_train = df_sorted.loc[train_mask, features]
        X_test = df_sorted.loc[~train_mask, features]
        y_train = df_sorted.loc[train_mask, target]
        y_test = df_sorted.loc[~train_mask, target]

        return X_train, X_test, y_train, y_test
    
    def _balance_data(self, X, y):
        """Aplica sobremuestreo de la clase minoritaria."""
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        return X_res, y_res

    def _tune_model(self, X_train, y_train, param_grid=None):
        """Realiza una búsqueda de hiperparámetros con GridSearchCV.

        Si ``param_grid`` es ``None`` se utiliza una grilla por defecto
        ampliada para explorar más combinaciones.
        """
        print(f"Tuning Model...{self.cont}")
        self.cont += 1

        pipeline = Pipeline([
            ('oversampler', RandomOverSampler(random_state=42)),
            ('model', LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1))
        ])

        categorical_feats = ['cluster'] if 'cluster' in X_train.columns else None

        if param_grid is None:
            param_grid = {
                'model__num_leaves': [31, 63],
                'model__max_depth': [-1, 7, 15],
                'model__learning_rate': [0.1, 0.05],
                'model__n_estimators': [50, 100],
                'model__min_child_samples': [20, 40],
                'model__subsample': [0.8, 1.0],
                'model__colsample_bytree': [0.8, 1.0],
            }
        else:
            param_grid = {
                (f"model__{k}" if not k.startswith("model__") else k): v
                for k, v in param_grid.items()
            }

        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='f1', n_jobs=-1)

        fit_params = {}
        if categorical_feats:
            fit_params['model__categorical_feature'] = categorical_feats

        grid.fit(X_train, y_train, **fit_params)
        
        print(f"✅ Mejor combinación: {grid.best_params_}")

        return grid.best_estimator_
    
    def simular_rentabilidad(self, etiqueta, umbral=0.7, ganancia_unitaria=2.5, perdida_unitaria=0.6):
        if etiqueta not in self.reportes:
            print(f"⚠️ No hay reporte para la etiqueta '{etiqueta}'. ¿Ejecutaste 'entrenar_todos()'?")
            return None

        reporte = self.reportes[etiqueta]
        y_test = reporte['y_test']
        y_prob = reporte['y_prob']

        # Simular entradas solo si prob > umbral
        decisiones = y_prob > umbral
        if decisiones.sum() == 0:
            print(f"⚠️ Ninguna señal superó el umbral de {umbral}.")
            return None

        aciertos = (y_test[decisiones] == 1).sum()
        errores = (y_test[decisiones] == 0).sum()

        ganancia_total = aciertos * ganancia_unitaria - errores * perdida_unitaria
        promedio_por_trade = ganancia_total / (aciertos + errores)

        print(f"\n💰 Simulación para '{etiqueta}' con umbral {umbral}")
        print(f"Operaciones simuladas: {aciertos + errores}")
        print(f"Aciertos: {aciertos}")
        print(f"Errores: {errores}")
        print(f"Ganancia total simulada: {ganancia_total:.2f}")
        print(f"Promedio por operación: {promedio_por_trade:.2f}")

        return {
            "etiqueta": etiqueta,
            "umbral": umbral,
            "aciertos": aciertos,
            "errores": errores,
            "ganancia_total": ganancia_total,
            "promedio_por_operacion": promedio_por_trade,
            "operaciones": aciertos + errores
        }

    def evaluar_todos_por_rentabilidad(self, umbrales=[0.65, 0.7, 0.75, 0.8], ganancia_unitaria=2.5, perdida_unitaria=0.6):
        resultados = []

        for etiqueta in self.etiquetas_objetivo:
            if etiqueta not in self.reportes:
                print(f"⚠️ No hay modelo entrenado para '{etiqueta}'")
                continue

            mejor = None
            for u in umbrales:
                resultado = self.simular_rentabilidad(etiqueta, umbral=u, ganancia_unitaria=ganancia_unitaria, perdida_unitaria=perdida_unitaria)
                if resultado is not None:
                    if mejor is None or resultado['ganancia_total'] > mejor['ganancia_total']:
                        mejor = resultado

            if mejor:
                resultados.append(mejor)

        # Ordenar por ganancia total
        df_resultados = pd.DataFrame(resultados).sort_values(by="ganancia_total", ascending=False)
        print("\n📊 Ranking por ganancia total:")
        print(df_resultados[['etiqueta', 'umbral', 'ganancia_total', 'operaciones', 'promedio_por_operacion']])

        return df_resultados
    
    def entrenar_todos(self, verbose=True, plot=True, tune=False, guardar=False, directorio="modelos"):
        for etiqueta in self.etiquetas_objetivo:
            if verbose:
                print(f"\n📈 Entrenando para: {etiqueta}")
            if etiqueta not in self.df.columns:
                print(f"⚠️ Etiqueta {etiqueta} no encontrada en DataFrame.")
                continue

            features = self._seleccionar_features(etiqueta)
            X_train, X_test, y_train, y_test = self._split_por_ventanas(
                self.df, features, etiqueta, ventana=self.ventana
            )

            if tune:
                model = self._tune_model(X_train, y_train)
            else:
                X_train_bal, y_train_bal = self._balance_data(X_train, y_train)
                model = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
                categorical_feats = ['cluster'] if 'cluster' in X_train_bal.columns else None
                if categorical_feats:
                    model.fit(X_train_bal, y_train_bal, categorical_feature=categorical_feats)
                else:
                    model.fit(X_train_bal, y_train_bal)
                
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            if verbose:
                print(classification_report(y_test, y_pred))
        
            if plot:
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Matriz de Confusión: {etiqueta}")
                plt.xlabel("Predicción")
                plt.ylabel("Real")
                plt.show()

            self.modelos[etiqueta] = model
            if guardar:
                os.makedirs(directorio, exist_ok=True)
                nombre = f"{etiqueta}.pkl"
                ruta = os.path.join(directorio, nombre)
                joblib.dump(model, ruta)
            self.reportes[etiqueta] = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }

    def entrenar_por_symbol(self, verbose=True, plot=True, tune=False,
                            guardar=False, directorio="modelos"):
        """Entrena modelos individualmente para cada símbolo.

        Parameters
        ----------
        guardar : bool, optional
            Si ``True`` se guardarán los modelos entrenados en ``directorio`` al
            finalizar el proceso mediante :func:`guardar_modelos_por_symbol`.
        directorio : str, optional
            Carpeta destino para los modelos guardados.
        """
        self.modelos_por_symbol = {}
        self.reportes_por_symbol = {}
    
        for symbol in self.df['symbol'].unique():
            df_sym = self.df[self.df['symbol'] == symbol]
            self.modelos_por_symbol[symbol] = {}
            self.reportes_por_symbol[symbol] = {}
    
            for etiqueta in self.etiquetas_objetivo:
                if verbose:
                    print(f"\n📈 Entrenando {symbol} para: {etiqueta}")
                if etiqueta not in df_sym.columns:
                    print(f"⚠️ Etiqueta {etiqueta} no encontrada en DataFrame.")
                    continue
    
                features = self._seleccionar_features(etiqueta)

                if len(df_sym) < self.ventana * 2:
                    print(f"⚠️ No hay suficientes datos para {symbol} - {etiqueta}")
                    continue

                X_train, X_test, y_train, y_test = self._split_por_ventanas(
                    df_sym, features, etiqueta, ventana=self.ventana
                )
    
                if tune:
                    param_grid = SYMBOL_CONFIGS.get(symbol, {}).get('tuning_params')
                    model = self._tune_model(X_train, y_train, param_grid)
                else:
                    X_train_bal, y_train_bal = self._balance_data(X_train, y_train)
                    model = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
                    categorical_feats = ['cluster'] if 'cluster' in X_train_bal.columns else None
                    if categorical_feats:
                        model.fit(X_train_bal, y_train_bal, categorical_feature=categorical_feats)
                    else:
                        model.fit(X_train_bal, y_train_bal)
                    
                #y_pred = model.predict(X_test)
                #y_prob = model.predict_proba(X_test)[:, 1]
                y_prob = model.predict_proba(X_test)[:, 1]
                umbral = 0.8
                y_pred = (y_prob >= umbral).astype(int)
    
                if verbose:
                    print(classification_report(y_test, y_pred))
    
                if plot:
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f"Matriz de Confusión: {symbol} - {etiqueta}")
                    plt.xlabel("Predicción")
                    plt.ylabel("Real")
                    plt.show()
    
                self.modelos_por_symbol[symbol][etiqueta] = model
                self.reportes_por_symbol[symbol][etiqueta] = {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                }

        if guardar:
            self.guardar_modelos_por_symbol(directorio)
    
    def obtener_modelo(self, etiqueta):
        return self.modelos.get(etiqueta, None)
    
    def obtener_reporte(self, etiqueta):
        return self.reportes.get(etiqueta, None)
    
    def obtener_importancias(self, etiqueta, symbol=None, top_n=10):
        """Devuelve las importancias de features para un modelo entrenado."""
        if symbol:
            model = self.modelos_por_symbol.get(symbol, {}).get(etiqueta)
        else:
            model = self.modelos.get(etiqueta)
    
        if model is None:
            print(f"⚠️ Modelo no encontrado para '{etiqueta}'" + (f" en {symbol}" if symbol else ""))
            return pd.DataFrame(columns=["feature", "importance"])
    
        # Si el modelo es un Pipeline (por ejemplo, tras usar GridSearchCV con
        # sobremuestreo), se extrae el estimador final que contiene las
        # importancias.
        estimator = model
        if hasattr(model, "named_steps"):
            if "model" in model.named_steps:
                estimator = model.named_steps["model"]
            else:
                for _, step in reversed(model.steps):
                    if hasattr(step, "feature_importances_"):
                        estimator = step
                        break

        if not hasattr(estimator, "feature_importances_"):
            print("⚠️ El modelo no expone 'feature_importances_'.")
            return pd.DataFrame(columns=["feature", "importance"])

        features = self._seleccionar_features(etiqueta)
        df_imp = pd.DataFrame({
            'feature': features,
            'importance': estimator.feature_importances_
        }).sort_values(by='importance', ascending=False)
        return df_imp.head(top_n)
    
    def resumen_resultados(self, symbol=None, top_n=5):
        """Genera un DataFrame resumido de métricas y top features, incluyendo métricas de distancia y avance."""
        if symbol:
            items = {symbol: self.reportes_por_symbol.get(symbol, {})}.items()
        else:
            items = self.reportes_por_symbol.items()
    
        resumen = []
        for sym, rep_dic in items:
            for etiqueta, datos in rep_dic.items():
                y_true = datos['y_test']
                y_pred = datos['y_pred']
                cm = confusion_matrix(y_true, y_pred)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
    
                accuracy = (tp + tn) / cm.sum() if cm.sum() else 0
                precision = tp / (tp + fp) if (tp + fp) else 0
                recall = tp / (tp + fn) if (tp + fn) else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
                top_feats = self.obtener_importancias(etiqueta, symbol=sym, top_n=top_n)['feature'].tolist()
    
                # --- Agregar métricas de distancia y avance ---
                dist_rebote = None
                avance_rebote = None
                avance_escape = None
                try:
                    df_sym = self.df[self.df['symbol'] == sym].copy()
                    df_sym = df_sym.loc[y_true.index]  # alineación segura
                    casos_tp = df_sym[(y_true == 1) & (y_pred == 1)]
                
                    if etiqueta.startswith("etiqueta_rebote"):
                        if not casos_tp.empty:
                            dist_rebote = (casos_tp['close'] - casos_tp['vwap']).abs().mean()
                
                            avance_real = []
                            for idx in casos_tp.index:
                                pos = df_sym.index.get_loc(idx)
                                if pos + self.n_ahead >= len(df_sym):
                                    continue
                                sub_df = df_sym.iloc[pos + 1 : pos + 1 + self.n_ahead]
                                close_ini = df_sym.loc[idx, 'close']
                
                                # Dirección según nivel de origen
                                if any(x in etiqueta for x in ['buy_low', 'vwap_lo', 'sell_low']):
                                    # Esperamos rebote alcista
                                    avance = sub_df['high'].max() - close_ini
                                else:
                                    # Rebote bajista
                                    avance = close_ini - sub_df['low'].min()
                
                                avance_real.append(avance)
                
                            if avance_real:
                                avance_rebote = np.mean(avance_real)
                
                    elif etiqueta.startswith("etiqueta_escape"):
                        avance_real = []
                        for idx in casos_tp.index:
                            pos = df_sym.index.get_loc(idx)
                            if pos + self.n_ahead >= len(df_sym):
                                continue
                            sub_df = df_sym.iloc[pos + 1 : pos + 1 + self.n_ahead]
                            close_ini = df_sym.loc[idx, 'close']
                            vwap_ini = df_sym.loc[idx, 'vwap']
                
                            if close_ini > vwap_ini:
                                avance = sub_df['close'].max() - close_ini
                            else:
                                avance = close_ini - sub_df['close'].min()
                            avance_real.append(avance)
                
                        if avance_real:
                            avance_escape = np.mean(avance_real)
                except Exception as e:
                    print(f"⚠️ Error al calcular métricas adicionales para {sym}-{etiqueta}: {e}")
    
                resumen.append({
                    'symbol': sym,
                    'etiqueta': etiqueta,
                    'accuracy': round(accuracy, 3),
                    'precision': round(precision, 3),
                    'recall': round(recall, 3),
                    'f1': round(f1, 3),
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    'top_features': ', '.join(top_feats),
                    'dist_prom_rebote': round(dist_rebote, 3) if dist_rebote is not None else None,
                    'avance_prom_rebote': round(avance_rebote, 3) if avance_rebote is not None else None,
                    'avance_prom_escape': round(avance_escape, 3) if avance_escape is not None else None,
                })
    
        return pd.DataFrame(resumen)

    def guardar_modelos_por_symbol(self, directorio="modelos"):
        """Guarda en ``directorio`` un archivo por símbolo con todos sus modelos."""
        os.makedirs(directorio, exist_ok=True)
        for symbol, modelos in self.modelos_por_symbol.items():
            nombre = f"{symbol.replace('.', '_')}.pkl"
            ruta = os.path.join(directorio, nombre)
            joblib.dump(modelos, ruta)
    
    
def revisar_coincidencias_etiquetas_flags(df, verbose=True):
    etiquetas = [col for col in df.columns if col.startswith('etiqueta_')]
    resumen = []

    for etiqueta in etiquetas:
        if 'rebote_vwap_' in etiqueta:
            nivel = etiqueta.replace('etiqueta_rebote_vwap_', '')
        elif 'escape_tendencia_' in etiqueta:
            nivel = etiqueta.replace('etiqueta_escape_tendencia_', '')
        else:
            if verbose:
                print(f"❌ Tipo de etiqueta desconocido o no relevante: {etiqueta}")
            continue

        flag_toque = f'vecindad_{nivel}'
        
        if flag_toque not in df.columns:
            if verbose:
                print(f"⚠️ Flag no encontrado para {etiqueta}: {flag_toque}")
            continue

        df_etiqueta = df[df[etiqueta] == 1]
        total = len(df_etiqueta)
        verdaderas = df_etiqueta[flag_toque].sum()
        porcentaje = verdaderas / total if total > 0 else 0

        resumen.append({
            'etiqueta': etiqueta,
            'flag': flag_toque,
            'total_etiquetas': total,
            'coincidencias': verdaderas,
            'porcentaje': porcentaje
        })

        if verbose:
            print(f"🔍 {etiqueta} → Flag: {flag_toque} | {verdaderas}/{total} coincidencias ({porcentaje:.2%})")

    if resumen:
        return pd.DataFrame(resumen).sort_values(by='porcentaje', ascending=False)
    else:
        print("⚠️ No se generaron coincidencias válidas.")
        return pd.DataFrame(columns=['etiqueta', 'flag', 'total_etiquetas', 'coincidencias', 'porcentaje'])
   
def plot_vecindad(df, niveles, symbol):
    """Grafica precios y marca las velas con vecindad en los niveles dados."""
    
    plt.ion()  # Modo interactivo
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='close', color='black')
    plt.plot(df.index, df['vwap'], label='vwap', color='blue')
    if 'vwap_hi' in df:
        plt.plot(df.index, df['vwap_hi'], '--', label='vwap_hi', color='orange')
    if 'vwap_lo' in df:
        plt.plot(df.index, df['vwap_lo'], '--', label='vwap_lo', color='orange')
    if 'ema_50' in df:
        plt.plot(df.index, df['ema_50'], label='ema_50', color='green')
    if 'ema_200' in df:
        plt.plot(df.index, df['ema_200'], label='ema_200', color='red')

    base_levels = {'vwap_hi', 'vwap_lo', 'ema_50', 'ema_200'}

    for nivel in niveles:
        if nivel in df.columns:
            if nivel not in base_levels:
                plt.plot(df.index, df[nivel], linestyle=':', label=nivel)
            mask = df.get(f'vecindad_{nivel}', pd.Series(False, index=df.index)).astype(bool)
            plt.scatter(df.index[mask], df['close'][mask], s=20, label=f'vec_{nivel}')

    
    plt.title(symbol)
    plt.legend(loc = 'center left')
    plt.tight_layout()
    plt.draw()       # Fuerza a dibujar el gráfico
    plt.pause(0.01)  # Pausa mínima para permitir actualización

def plot_etiquetas(df, niveles=None, titulo=None, usar_pred=False,
                   mostrar_reales=True, umbral=0.5):
    """Grafica el precio y marca las etiquetas reales y/o las predicciones.

    Parameters
    ----------
    df : DataFrame
        Datos a graficar. Si ``usar_pred`` es ``True`` se buscan columnas con
        el prefijo ``prediccion_``.
    niveles : list, optional
        Columnas de niveles de soporte/resistencia a mostrar.
    titulo : str, optional
        Título del gráfico.
    usar_pred : bool, optional
        Si ``True`` se graficarán las predicciones con un marcador ``x``.
    mostrar_reales : bool, optional
        Si ``True`` se muestran las etiquetas reales. Con ``False`` sólo se
        grafican las predicciones.
    umbral : float, optional
        Probabilidad mínima para mostrar una predicción.
    """
    
    if niveles is None:
        niveles = []

    etiquetas_rebote = [
        'etiqueta_rebote_vwap_buy_low',
        'etiqueta_rebote_vwap_buy_high',
        'etiqueta_rebote_vwap_sell_low',
        'etiqueta_rebote_vwap_sell_high',
        'etiqueta_rebote_vwap_vwap_lo',
        'etiqueta_rebote_vwap_vwap_hi'
    ]

    colores = ['green', 'orange', 'purple', 'brown', 'teal', 'crimson']

    df_plot = df.copy()

    plt.ion()
    plt.figure(figsize=(14, 6))
    plt.plot(df_plot.index, df_plot['close'], label='close', color='black', linewidth=1)

    if 'vwap' in df_plot.columns:
        plt.plot(df_plot.index, df_plot['vwap'], label='vwap', color='blue', linestyle='--')

    if niveles:
        for nivel in niveles:
            if nivel in df_plot.columns:
                plt.plot(df_plot.index, df_plot[nivel], linestyle=':', label=nivel)

    for etiqueta, color in zip(etiquetas_rebote, colores):
        if mostrar_reales and etiqueta in df_plot.columns:
            puntos = df_plot[df_plot[etiqueta] == 1]
            plt.scatter(
                puntos.index,
                puntos['close'],
                label=etiqueta,
                color=color,
                s=30,
                marker='o',
                zorder=5,
            )

        if usar_pred:
            col_pred = f'prediccion_{etiqueta}'
            if col_pred in df_plot.columns:
                puntos_pred = df_plot[df_plot[col_pred] >= umbral]
                plt.scatter(puntos_pred.index, puntos_pred['close'],
                            label=f'pred_{etiqueta}', color=color,
                            s=60, marker='x', zorder=6)

    plt.title(titulo or "Rebotes VWAP")
    plt.legend(loc = 'center left')
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
    
def main():
        
    from tqdm import tqdm

    # --- Configuraciones ---
    symbols = list(SYMBOL_CONFIGS.keys())
    vwap_win = 14
    n_dias = 350

    # --- Inicialización de clases ---
    connector = MT5Connector()
    calculator = TaylorVWAPCalculator(vwap_win=vwap_win)
    enricher = FeatureEnricher()
    label_generator = LabelGenerator()
    builder = DatasetBuilder(connector, calculator, enricher)

    fecha_base = (datetime.now() + timedelta(hours=6)).replace(hour=0, minute=0, second=0, microsecond=0)

   #symbols = ["SP500"]
    for symbol in symbols:
        df_total = []
        print(f"Procesando: {symbol}")
        for i in tqdm(range(n_dias), desc=f"{symbol}"):
            fecha_sesion = fecha_base - timedelta(days=i)
            apertura_mq = fecha_sesion.replace(
                hour=SYMBOL_CONFIGS[symbol]["premarket"][0],
                minute=SYMBOL_CONFIGS[symbol]["premarket"][1]
            )

            df = builder.procesar_sesion(symbol, fecha_sesion, apertura_mq, i)

            if df is not None:
                df_total.append(df)

        if df_total:
            df_concat = pd.concat(df_total)
            df_concat = df_concat[df_concat['tick_volume'] > 0]
            
            # Primero calcular indicadores históricos que dependen de la serie completa
            df_concat = enricher.enriquecer_historico(df_concat)
    
            # Luego calcular los features intradía que pueden usar los históricos
            df_concat = enricher.enriquecer_intradia(df_concat)
    
            # Convertir columnas booleanas a enteros después de añadir todas las features
            bool_cols = df_concat.select_dtypes(include='bool').columns
            df_concat[bool_cols] = df_concat[bool_cols].astype(int)
            
            df_concat.sort_values(by=['symbol', 'fecha', 'time'], ascending=[True, True, True], inplace=True)
    
            niveles = ['buy_low', 'buy_high', 'sell_low', 'sell_high', 'vwap_lo', 'vwap_hi']
            df_labeled = []
            for sym, data in df_concat.groupby('symbol'):
                etiquetas = label_generator.generar_etiquetas_cruce_y_rebote(
                    data, niveles, SYMBOL_CONFIGS[sym]['min_dist']
                )
                df_labeled.append(pd.concat([data, etiquetas], axis=1))
            
            df_concat = pd.concat(df_labeled).sort_index()
            
            # Chequeo de correlación excesiva para descartar posibles leakages
            print(f"\n🔎 Chequeando correlaciones para potencial leakage en {symbol}...")
            
            etiquetas = [col for col in df_concat.columns if col.startswith('etiqueta_')]
            
            for etiqueta in etiquetas:
                features = [col for col in df_concat.columns 
                            if col not in etiquetas 
                            and col not in ['symbol', 'fecha', 'time']
                            and not col.startswith('toque_')]
            
                # ⚠️ Verifica si la etiqueta actual quedó en features por error
                if etiqueta in features:
                    print(f"⚠️ ERROR: etiqueta {etiqueta} aún está en features.")
                    features.remove(etiqueta)  # Eliminala para evitar distorsión
            
                # Construir subset y calcular correlación
                df_corr = df_concat[features + [etiqueta]].dropna()
                if len(df_corr) < 100:
                    continue  # Demasiado pocos datos
            
                corrs = df_corr.corr()[etiqueta].abs().sort_values(ascending=False)
                sospechosas = corrs[corrs > 0.8]
                sospechosas = sospechosas[sospechosas.index != etiqueta]  # 🔧 evita autocomparación
            
                if not sospechosas.empty:
                    print(f"\n⚠️ Posibles features con leakage para {etiqueta}:")
                    print(sospechosas.head(10).to_string())
    
            # Asignar cluster por símbolo según volatilidad y momentum
            df_concat = enricher.aplicar_clusters(df_concat, symbol, evaluate=True)
            #df_concat.to_csv("v5.csv", index=True, sep=';', decimal=',')
            #print("✅ Dataset guardado como 'v5.csv'")
        else:
            print("⚠️ No se generaron datos válidos.")

    
        print(f"\n🔍 Procesando símbolo: {symbol}")
        revisar_coincidencias_etiquetas_flags(df_concat, True)
        
        etiquetas = [col for col in df_concat.columns if col.startswith('etiqueta_')]
        trainer = ModelTrainer(df_concat, etiquetas, ventana = SYMBOL_CONFIGS.get(symbol, {}).get('ventana', 6))
        
        # Entrenamiento global sin imprimir detalles
        #trainer.entrenar_todos(verbose=False, plot=False)
        #trainer.evaluar_todos_por_rentabilidad()
        
        # Entrenamiento por símbolo y resumen compacto con búsqueda de hiperparámetros
        trainer.entrenar_por_symbol(verbose=False, plot=False, tune=False, guardar = True, directorio = SYMBOL_CONFIGS.get(symbol, {}).get('dir_modelo', symbol))
        df_resumen = trainer.resumen_resultados(top_n=15)
        
        # Asignar expected gain por fila desde la configuración
        df_resumen['expected_gain'] = df_resumen['symbol'].map(
            {k: v['expected_gain'] for k, v in SYMBOL_CONFIGS.items()}
        )
        
        # Calcular ganancias y pérdidas esperadas
        df_resumen['ganancia_tp'] = df_resumen['tp'] * df_resumen['expected_gain']
        df_resumen['perdida_fp'] = -df_resumen['fp'] * df_resumen['expected_gain']
        df_resumen['resultado_estimado'] = df_resumen['ganancia_tp'] + df_resumen['perdida_fp']
        df_resumen['op_netas'] = df_resumen['tp'] - df_resumen['fp']

        # Ordenar por tp, f1 y precision de mayor a menor
        df_resumen = df_resumen.sort_values(by=['resultado_estimado'], ascending=False).reset_index(drop=True)
        
        df_resumen.to_csv(SYMBOL_CONFIGS.get(symbol, {}).get('dir_csv', symbol), index=False, sep=';', decimal=',')
           
    
        # --- Visualización para inspección de vecindades activadas ---
        
        # Seleccionamos un símbolo de interés (puedes cambiarlo manualmente si quieres otro)
        symbol_viz = symbol  # Cambia por cualquier símbolo disponible en tu dataset
        df_viz = df_concat[df_concat['symbol'] == symbol_viz].copy()
        
        # Definimos los niveles que nos interesa visualizar
        niveles_vecindad = ['buy_low', 'buy_high', 'sell_low', 'sell_high',
                            'vwap_lo', 'vwap_hi', 'ema_50', 'ema_200']
        

        fecha_base_dt = datetime.combine(fecha_base, datetime.min.time())
        df_viz['fecha'] = pd.to_datetime(df_viz['fecha'], errors='coerce')
        df_viz = df_viz[df_viz['fecha'] >= fecha_base_dt]
        
        #plot_vecindad(df_viz, niveles_vecindad, symbol)
        
        # Filtra por símbolo y últimos 200 registros
        df_viz = df_concat[df_concat['symbol'] == symbol].tail(2000)
        
        niveles=['vwap', 'vwap_lo', 'vwap_hi','buy_low', 'buy_high', 'sell_low', 'sell_high']
        
        #plot_etiquetas(df_viz,niveles,titulo=symbol)
    
        """
        # --- Predicciones históricas con validación temporal ---
        df_sym = df_concat[df_concat['symbol'] == symbol].copy()
        
        for etiqueta in etiquetas:

            features = trainer._seleccionar_features(etiqueta)

            if etiqueta not in df_sym.columns or len(df_sym) < 50:
                continue

            X = df_sym[features]
            y = df_sym[etiqueta]

            pipeline = Pipeline([
                ('oversampler', RandomOverSampler(random_state=42)),
                ('model', LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1))
            ])

            tscv = TimeSeriesSplit(n_splits=5)
            fit_params = {}
            if 'cluster' in X.columns:
                fit_params['model__categorical_feature'] = ['cluster']

            probas = np.empty(len(X))
            for train_idx, test_idx in tscv.split(X):
                pipeline.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx],
                    **fit_params
                )
                probas[test_idx] = pipeline.predict_proba(
                    X.iloc[test_idx]
                )[:, 1]

            df_sym[f'prediccion_{etiqueta}'] = probas

        # Graficamos todas las predicciones históricas

        plot_etiquetas(
            df_sym.tail(2000),
            niveles,
            titulo=symbol,
            usar_pred=True,
            mostrar_reales=False,
            umbral=0.8,
        )

        print("\n📊 Predicciones históricas (últimas filas):")
        cols_pred = [f'prediccion_{e}' for e in etiquetas if f'prediccion_{e}' in df_sym.columns]
        print(df_sym[cols_pred].tail().to_string())
        """

if __name__ == "__main__":
    main()