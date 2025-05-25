from collections import deque
from enum import Enum, auto
import numpy as np
from typing import Tuple

class InputType(Enum):
    MLP = auto()
    LSTM = auto()

class Observation:
    """
    Observation 클래스

    - Flatten된 feature 벡터를 구조화하여 다음을 분리·저장:
      1) orderbook 스냅샷: 과거 lookback 틱 수만큼의 bid/ask 레벨 + 현 스텝
      2) position (현재 포지션)
      3) (선택) pnl (미실현 손익)
      4) (선택) spread (bid-ask 스프레드)
      5) (선택) OHLCV (open, high, low, close, volume)

    용어 설명:
    - lookback:
      하나의 관측(feature 벡터) 내에 포함할 과거 LOB 스냅샷의 깊이입니다.
      예: lookback=9 이면, 현 스텝 직전 9틱의 bid/ask 데이터를 벡터에 포함합니다.
      MLP 입력시 사용

    - window_size:
      모델(예: LSTM)에 입력할 관측 시퀀스의 길이(타임스텝 수)를 정합니다.
      예: window_size=5 이면, 과거 5개의 관측(feature 벡터)을 순서대로 보관하고 반환합니다.
      LSTM 입력시 사용

    - lookback은 "벡터 내부"의 시차 깊이를, window_size는 "시퀀스 길이"를 조절합니다.

    - lookback과 window_size를 같은 크기로 하는 것을 권장!!!!!!!!
    """
    def __init__(
        self,
        lob_levels: int,
        lookback: int,
        include_pnl: bool      = False,
        include_spread: bool   = False,
        include_ohlcv: bool    = False,
        window_size: int       = 1
    ):
        self.lob_levels     = lob_levels
        self.lookback       = lookback
        self.include_pnl    = include_pnl
        self.include_spread = include_spread
        self.include_ohlcv  = include_ohlcv
        self.window_size    = window_size

        # 최대 window_size 만큼 과거 관측을 보관
        self.history = deque(maxlen=window_size)

        # feature 차원
        self.dim_snapshots = 2 * lob_levels * lookback
        self.dim_current   = 2 * lob_levels
        self.dim_position  = 1
        self.dim_pnl       = 1 if include_pnl else 0
        self.dim_spread    = 1 if include_spread else 0
        self.dim_ohlcv     = 5 if include_ohlcv else 0

        self.dim_total = (
            self.dim_snapshots
            + self.dim_current
            + self.dim_position
            + self.dim_pnl
            + self.dim_spread
            + self.dim_ohlcv
        )

    def reset(self):
        """히스토리(과거 관측)를 비웁니다."""
        self.history.clear()

    def append(self, features: list):
        """
        Flatten된 features 리스트를 받아
        각 파트별로 분리·저장합니다.
        """
        if len(features) != self.dim_total:
            print(features)
            raise ValueError(f"feature 길이 불일치: 기대 {self.dim_total}, 입력 {len(features)}")
        idx = 0

        # 1) 과거 lookback 스냅샷
        snap = features[idx : idx + self.dim_snapshots]
        idx += self.dim_snapshots

        # 2) 현재 orderbook
        curr = features[idx : idx + self.dim_current]
        idx += self.dim_current

        # 3) position
        position = features[idx]
        idx += 1

        # 4) pnl (optional)
        pnl = None
        if self.include_pnl:
            pnl = features[idx]
            idx += 1

        # 5) spread (optional)
        spread = None
        if self.include_spread:
            spread = features[idx]
            idx += 1

        # 6) OHLCV (optional)
        ohlcv = None
        if self.include_ohlcv:
            ohlcv = features[idx : idx + 5]
            idx += 5

        self.history.append({
            'snapshots': np.array(snap, dtype=float),
            'current':   np.array(curr, dtype=float),
            'position':  float(position),
            **({'pnl': float(pnl)}       if self.include_pnl    else {}),
            **({'spread': float(spread)} if self.include_spread else {}),
            **({'ohlcv': np.array(ohlcv, dtype=float)} if self.include_ohlcv else {}),
        })

    def get_mlp_input(self) -> np.ndarray:
        """
        가장 최신 스텝 하나를 위한 MLP 입력 벡터(shape=(dim_total,))를 반환합니다.
        """
        if not self.history:
            raise ValueError("히스토리가 비어 있습니다. append()로 먼저 관측을 추가하세요.")
        h = self.history[-1]
        parts = [
            h['snapshots'],
            h['current'],
            np.array([h['position']], dtype=float)
        ]
        if self.include_pnl:
            parts.append(np.array([h['pnl']], dtype=float))
        if self.include_spread:
            parts.append(np.array([h['spread']], dtype=float))
        if self.include_ohlcv:
            parts.append(h['ohlcv'])
        return np.concatenate(parts, axis=0)

    # def get_lstm_input(self) -> np.ndarray:
    #     """
    #     히스토리에 저장된 전체 시퀀스를 위한 LSTM 입력(shape=(T, dim_total))을 반환합니다.
    #     T = 현재 히스토리 길이 (<= window_size)
    #     """
    #     if not self.history:
    #         raise ValueError("히스토리가 비어 있습니다. append()로 먼저 관측을 추가하세요.")
    #     seq = []
    #     for h in self.history:
    #         parts = [
    #             h['snapshots'],
    #             h['current'],
    #             np.array([h['position']], dtype=float)
    #         ]
    #         if self.include_pnl:
    #             parts.append(np.array([h['pnl']], dtype=float))
    #         if self.include_spread:
    #             parts.append(np.array([h['spread']], dtype=float))
    #         if self.include_ohlcv:
    #             parts.append(h['ohlcv'])
    #         seq.append(np.concatenate(parts, axis=0))
    #     return np.vstack(seq)

    def get_lstm_input(self) -> dict:
        if not self.history:
            raise ValueError("히스토리가 비어 있습니다. append()로 먼저 관측을 추가하세요.")
        snaps_list, others_list = [], []
        for h in self.history:
            part1 = np.concatenate([h['snapshots'], h['current']], axis=0)
            rest = [np.array([h['position']], dtype=float)]
            if self.include_pnl:    rest.append(np.array([h['pnl']], dtype=float))
            if self.include_spread: rest.append(np.array([h['spread']], dtype=float))
            if self.include_ohlcv:  rest.append(h['ohlcv'])
            part2 = np.concatenate(rest, axis=0)
            snaps_list.append(part1)
            others_list.append(part2)
        return {
            'snapshots': np.vstack(snaps_list),
            'others':    np.vstack(others_list)
        }

    def get_dimension(self, input_type: InputType):
        """
        모델 입력 차원을 반환합니다.
        - InputType.MLP: 단일 스텝 feature 길이 (dim_total)
        - InputType.LSTM: (window_size, dim_total) 튜플
        """
        if input_type == InputType.MLP:
            return self.dim_total
        elif input_type == InputType.LSTM:
            return (self.window_size, self.dim_total)
        else:
            raise ValueError(f"알 수 없는 InputType: {input_type}")
