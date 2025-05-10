# File: data_handler.py

from abc import ABC, abstractmethod
from enum import Enum, auto
import pandas as pd
import numpy as np
from data_handler.indicators import ema, macd, rsi, cci, adx

class HandlerType(Enum):
    DAILY = auto()
    MINUTE = auto()

class MinuteVersion(Enum):
    SC201 = auto()
    SC202 = auto()
    SC203 = auto()


# --- Handlers ---

class DataHandlerBase(ABC):
    """
    추상 기본 클래스: DataHandler 인터페이스 정의
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)

    @abstractmethod
    def get_observation(self, step: int, **kwargs) -> np.ndarray:
        pass

class DailyDataHandler(DataHandlerBase):
    """
    하루 단위 거래 데이터 처리기
    - 잔고, 조정 종가, 보유수량
    - MACD, RSI, CCI, ADX
    """
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.df['MACD'], self.df['MACD_SIGNAL'], self.df['MACD_HIST'] = macd(self.df['Adj Close'])
        self.df['RSI'] = rsi(self.df['Adj Close'])
        self.df['CCI'] = cci(self.df['High'], self.df['Low'], self.df['Close'])
        self.df['ADX'] = adx(self.df['High'], self.df['Low'], self.df['Close'])
        # TODO csv로 데이터를 추출하는 코드에서 헤더 이름을 high, low, close로 바꾸기

    def get_observation(self, step: int, balance: float, shares_held: int) -> np.ndarray:
        row = self.df.loc[step]
        obs = [
            balance,
            row['Adj Close'],
            shares_held,
            row['MACD'],
            row['RSI'],
            row['CCI'],
            row['ADX'],
        ]
        return np.array(obs, dtype=np.float32)

class MinuteDataHandlerBase(DataHandlerBase):
    """
    분 단위 거래용 공통 처리기
    """
    def __init__(self, df: pd.DataFrame, lob_levels: int = 10, lookback: int = 9):
        super().__init__(df)
        self.lob_levels = lob_levels
        self.lookback = lookback

    @abstractmethod
    def get_additional_features(self, step: int, **kwargs) -> list:
        pass

    def get_observation(self, step: int, position: int, **kwargs) -> np.ndarray:
        features = self.get_additional_features(step, position, **kwargs)
        return np.array(features, dtype=np.float32)

class Sc201Handler(MinuteDataHandlerBase):
    """
    Sc201: 호가 10단계 + 직전 9틱 LOB 스냅샷 + 현재 포지션
    """
    def get_additional_features(self, step: int, position: int, **kwargs) -> list:
        row = self.df.loc[step]
        lob = row[:self.lob_levels*2].tolist()
        snapshots = []
        for t in range(step-self.lookback+1, step+1):
            if t < 0: continue
            snapshots.extend(self.df.loc[t, :self.lob_levels*2].tolist())
        return lob + snapshots + [position]

class Sc202Handler(Sc201Handler):
    """
    Sc202: Sc201 + 미실현 P&L
    """
    def get_additional_features(self, step: int, position: int, pnl: float = 0.0, **kwargs) -> list:
        base = super().get_additional_features(step, position)
        return base + [pnl]

class Sc203Handler(Sc202Handler):
    """
    Sc203: Sc202 + bid-ask 스프레드
    """
    def get_additional_features(self, step: int, position: int, pnl: float = 0.0, spread: float = 0.0) -> list:
        base = super().get_additional_features(step, position, pnl)
        return base + [spread]

def create_data_handler(
    handler_type: HandlerType,
    df: pd.DataFrame,
    version: MinuteVersion = MinuteVersion.SC201
) -> DataHandlerBase:
    if handler_type == HandlerType.DAILY:
        return DailyDataHandler(df)
    elif handler_type == HandlerType.MINUTE:
        if version == MinuteVersion.SC201:
            return Sc201Handler(df)
        elif version == MinuteVersion.SC202:
            return Sc202Handler(df)
        elif version == MinuteVersion.SC203:
            return Sc203Handler(df)
    raise ValueError(f"Unknown handler type {handler_type} or version {version}")
