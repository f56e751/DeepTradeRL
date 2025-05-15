# File: data_handler.py

from abc import ABC, abstractmethod
from enum import Enum, auto
import pandas as pd
import numpy as np
from src.data_handler.indicators import ema, macd, rsi, cci, adx
from typing import List, Dict

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

    def get_length(self) -> int:
        """
        df의 길이를 반환함
        """
        return len(self.df)

    @abstractmethod
    def get_observation(self, step: int) -> np.ndarray:
        pass

class DailyDataHandler(DataHandlerBase):
    """
    하루 단위 거래 데이터 처리기
      - 잔고, 조정 종가, 보유수량
      - MACD, RSI, CCI, ADX 등 기술 지표
    
    # Adj Close 와 RSI만 쓰고 싶다면
    handler = DailyDataHandler(df, feature_cols=['Adj Close','RSI'])
    """
    def __init__(self, df: pd.DataFrame, feature_cols=None):
        super().__init__(df)
        # 기술 지표 계산
        self.df['MACD'], self.df['MACD_SIGNAL'], self.df['MACD_HIST'] = macd(self.df['Adj Close'])
        self.df['RSI'] = rsi(self.df['Adj Close'])
        self.df['CCI'] = cci(self.df['High'], self.df['Low'], self.df['Close'])
        self.df['ADX'] = adx(self.df['High'], self.df['Low'], self.df['Close'])
        
        # 기본 feature 순서 정의
        default_feats = ['Adj Close', 'MACD', 'RSI', 'CCI', 'ADX']
        # 넘겨받은 리스트가 있으면 그걸, 아니면 기본 전체
        self.feature_cols = feature_cols or default_feats

    def get_observation(self, step: int) -> np.ndarray:
        row = self.df.loc[step]
        obs = [        ]
        # feature_cols 에 정의된 지표만 추가
        for col in self.feature_cols:
            obs.append(row[col])
        return np.array(obs, dtype=np.float32)



import pandas as pd
import numpy as np
from typing import Dict, List

class MultiDataHandler:
    """
    여러 종목의 DailyDataHandler를 모아,
    step 단위로 모든 종목의 관측값을 하나의 벡터로 반환.
    결측치(NaN)가 포함된 행은 모두 제거하며, 제거 전후 티커별 행 수를 로깅합니다.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: List[str] = None):
        # 1) 'Ticker' 컬럼 검증
        if 'Ticker' not in df.columns:
            raise ValueError("DataFrame에 'Ticker' 컬럼이 없습니다.")

        # 2) 원본 티커별 행 수 집계
        orig_counts = df['Ticker'].value_counts().to_dict()

        # 3) 결측치가 있는 모든 행 제거
        df_clean = df.dropna()

        # 4) 클린된 티커별 행 수 집계
        clean_counts = df_clean['Ticker'].value_counts().to_dict()

        # 5) 티커별로 제거된 행 수 및 남은 행 수 로깅
        for ticker, orig_count in orig_counts.items():
            clean_count = clean_counts.get(ticker, 0)
            removed = orig_count - clean_count
            print(f"Ticker '{ticker}': removed {removed} rows, remaining {clean_count} rows")

        # 6) 고유 티커 목록 추출
        self.tickers = sorted(df_clean['Ticker'].unique())

        # 7) 사용할 피처 컬럼 저장 (None이면 DailyDataHandler 기본값 사용)
        self.feature_cols = feature_cols

        # 8) 정리된 DataFrame 검증
        self._validate_dataframe(df_clean)

        # 9) 종목별로 DailyDataHandler 생성 및 indicator NaN 제거
        self.handlers: Dict[str, DailyDataHandler] = {}
        for t in self.tickers:
            # 각 티커별 데이터 분리
            df_t = df_clean[df_clean['Ticker'] == t].reset_index(drop=True)
            # DailyDataHandler 인스턴스 생성 (지표 계산 수행)
            handler = DailyDataHandler(df_t, feature_cols=self.feature_cols)
            # 지표 계산 후 NaN 제거: feature_cols에 따라 subset 지정
            subset_cols = handler.feature_cols if handler.feature_cols is not None else ['Adj Close']
            orig_len = len(handler.df)
            handler.df.dropna(subset=subset_cols, inplace=True)
            # 인덱스를 리셋하여 0 부터 시작하도록 함
            handler.df.reset_index(drop=True, inplace=True)
            new_len = len(handler.df)
            removed = orig_len - new_len
            print(f"After indicator NaN removal, ticker '{t}': removed {removed} rows, remaining {new_len} rows")
            # 핸들러 딕셔너리에 저장
            self.handlers[t] = handler

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame):
        """
        - df에 'Date', 'Ticker' 컬럼이 있는지 확인
        - 각 Date에 모든 티커 데이터가 존재하는지 확인
        """
        required_cols = {'Date', 'Ticker'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"DataFrame에 '{missing}' 컬럼이 없습니다.")

        # 날짜별 티커 집합 확인
        df_dates = df[['Date', 'Ticker']].drop_duplicates()
        grouped = df_dates.groupby('Date')['Ticker'].apply(set)
        all_tickers = set(df['Ticker'].unique())
        missing_info = {
            date: all_tickers - present
            for date, present in grouped.items()
            if all_tickers - present
        }
        if missing_info:
            msgs = []
            for date, miss in missing_info.items():
                miss_list = ", ".join(sorted(miss))
                msgs.append(f"  {date.date()}: missing {miss_list}")
            detail = "\n".join(msgs)
            raise ValueError(
                "다음 날짜들에 일부 티커 데이터가 누락되었습니다:\n" + detail
            )

    def get_length(self) -> int:
        """가장 짧은 종목 데이터 길이(최대 step 수)를 반환합니다."""
        return min(h.get_length() for h in self.handlers.values())

    def get_num_tickers(self) -> int:
        """포함된 고유 티커 개수를 반환합니다."""
        return len(self.tickers)

    def get_observation(self, step: int) -> np.ndarray:
        """
        주어진 step에서 모든 종목의 observation을 하나의 벡터로 합쳐 반환합니다.
        """
        obs_parts = []
        for t in self.tickers:
            obs_t = self.handlers[t].get_observation(step)
            obs_parts.append(obs_t)
        return np.concatenate(obs_parts, axis=0)

# ====================================================
# MinuteDataHandlerBase 입력 df 설명

# 예: lob_levels=3, lookback=2 라고 가정
# 컬럼 0~2: 매수 레벨 1~3 잔량
# 컬럼 3~5: 매도 레벨 1~3 잔량

# 컬럼 순서: [bid_1, bid_2, bid_3, ask_1, ask_2, ask_3]  
# – bid_1 은 최우선 매수호가(best bid)
# – ask_1 은 최우선 매도호가(best ask)


# 예시 데이터 (잔량@가격):
#   bid_1 = 500@100원   ← 최우선 매수호가: 100원에 500주 매수 대기
#   bid_2 = 300@ 99원   ← 2차 매수호가:   99원에 300주 매수 대기
#   bid_3 = 200@ 98원   ← 3차 매수호가:   98원에 200주 매수 대기
#   ask_1 = 600@101원   ← 최우선 매도호가: 101원에 600주 매도 대기
#   ask_2 = 400@102원   ← 2차 매도호가:   102원에 400주 매도 대기
#   ask_3 = 250@103원   ← 3차 매도호가:   103원에 250주 매도 대기
#
# 따라서 spread = ask_1 가격 − bid_1 가격 = 101원 − 100원 = 1원



# 행: n step에서의 잔량
# 열: 매수, 매도 호가

# df 모습
#    0   1   2   3   4   5
#0  23  45  12  78  34  56
#1  10  67  89  23  11   9
#2  54  33  21  45  66  72
#3  19  28  37  46  55  64
#4   5  15  25  35  45  55


# 이 df를 Sc201Handler에 넘기면,
# step=2일 때 get_observation(2, position=1) 은
#   - 컬럼 0~5 의 현재 잔량 (6개)
#   - 과거 2틱(lookback=2)이면 step=1,2의 잔량 (2×6=12개)
#   - position (1개)
# 총 6 + 12 + 1 = 19 차원 벡터를 반환합니다.
class TickDataHandlerBase(DataHandlerBase):
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
    
class Sc201Handler(TickDataHandlerBase):
    """
    Sc201: 호가 10단계 + 직전 lookback 틱 LOB 스냅샷(부족분 0 패딩) + 현재 포지션
    
    현재 포지션
    +1 → 롱(Long) 포지션
    0 → 중립(Flat)
    -1 → 숏(Short)
    """
    def __init__(self, df, lob_levels=10, lookback=9, price_cols=20):
        super().__init__(df, lob_levels, lookback)
        # price_cols = 2*lob_levels  (px 컬럼 개수)
        self.price_cols = price_cols

    def get_additional_features(self, step, position, **kwargs):
        row = self.df.iloc[step]

        # 1) 현재 잔량: price_cols 위치 다음부터 2*lob_levels개
        start = self.price_cols
        end = start + self.lob_levels*2
        lob = row.iloc[start:end].tolist()

        # 2) 과거 lookback 스냅샷
        snapshots = []
        for t in range(step - self.lookback, step):
            if t < 0:
                snapshots.extend([0.0] * (self.lob_levels*2))
            else:
                snapshots.extend(self.df.iloc[t, start:end].tolist())

        # 3) 현재 포지션
        return snapshots + lob + [position]


class Sc202Handler(Sc201Handler):
    """
    Sc202: Sc201 + 미실현 P&L
    """
    def get_additional_features(self, step: int, position: int, pnl: float = 0.0, **kwargs) -> list:
        base = super().get_additional_features(step, position)
        return base + [pnl]

class Sc203Handler(Sc202Handler):
    """
    Sc203: Sc202 + bid-ask 스프레드(최우선 매도호가와 최우선 매수호가의 차이)
    컬럼 순서: [bid_1, bid_2, ..., bid_n, ask_1, ask_2, ..., ask_n]
    """
    def calculate_spread(self, step: int) -> float:
        """
        주어진 step에서 최우선 매도호가 - 최우선 매수호가를 계산하여 반환.
        예: lob_levels=3 이면
            best_bid = row.iloc[0]    # bid_1
            best_ask = row.iloc[3]    # ask_1
        """
        bid = self.df.loc[step, 'bid_px_00']
        ask = self.df.loc[step, 'ask_px_00']
        spread = ask - bid
        return spread
    
    def get_additional_features(self, step: int, position: int, pnl: float = 0.0, **kwargs) -> list:
        # Sc202까지의 features (lob, snapshots, position, pnl)
        base = super().get_additional_features(step, position, pnl)
        # spread를 직접 계산
        spread = self.calculate_spread(step)
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
