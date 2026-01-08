"""MetaTrader 5 connectivity for OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd


@dataclass
class MT5Connector:
    """Simple MT5 connector for OHLCV retrieval."""

    def __post_init__(self) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"No se pudo conectar a MetaTrader 5: {mt5.last_error()}")

    def shutdown(self) -> None:
        mt5.shutdown()

    def obtener_h1(
        self,
        symbol: str,
        fecha_inicio: datetime,
        fecha_fin: datetime,
    ) -> pd.DataFrame:
        """Obtener velas H1 en el rango dado."""
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, fecha_inicio, fecha_fin)
        if rates is None or len(rates) == 0:
            raise RuntimeError("No se pudieron obtener datos H1 desde MT5.")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df


__all__ = ["MT5Connector"]
