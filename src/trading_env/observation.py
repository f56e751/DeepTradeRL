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
    - lookback은 mlp에 입력으로 들어오는 step 개수
    - window size는 lstm에 필요
    """
    def __init__(
        self,
        lob_levels: int,
        lookback: int,
        include_pnl: bool      = False,
        include_spread: bool   = False,
        include_ohlcv: bool    = False,
        include_tech:bool      = False,
        window_size: int       = 1,
        tech_dim: int          = 0,
    ):
        self.lob_levels     = lob_levels
        self.lookback       = lookback
        self.include_pnl    = include_pnl
        self.include_spread = include_spread
        self.include_ohlcv  = include_ohlcv
        self.window_size    = window_size
        self.include_tech   = include_tech

        # 최대 window_size 만큼 과거 관측을 보관
        self.history = deque(maxlen=window_size)

        # feature 차원
        self.dim_snapshots = 2 * lob_levels * lookback
        self.dim_current   = 2 * lob_levels
        self.dim_position  = 1
        self.dim_pnl       = 1 if include_pnl else 0
        self.dim_spread    = 1 if include_spread else 0
        self.dim_ohlcv     = 5 if include_ohlcv else 0
        self.dim_tech      = tech_dim if include_tech else 0

        self.dim_total = (
            self.dim_snapshots
            + self.dim_current
            + self.dim_position
            + self.dim_pnl
            + self.dim_spread
            + self.dim_ohlcv
            + self.dim_tech
        )

    def reset(self):
        """히스토리(과거 관측)를 비웁니다."""
        self.history.clear()

    def fill_window_size(self, init_feats):
        for _ in range(self.window_size - 1):
            self.append(init_feats)

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

        tech = None
        if self.include_tech:
            tech = features[idx:idx+5]; idx += 5

        self.history.append({
            'snapshots': np.array(snap, dtype=float),
            'current':   np.array(curr, dtype=float),
            'position':  float(position),
            **({'pnl': float(pnl)}       if self.include_pnl    else {}),
            **({'spread': float(spread)} if self.include_spread else {}),
            **({'ohlcv': np.array(ohlcv, dtype=float)} if self.include_ohlcv else {}),
            **({'tech': np.array(tech)}   if self.include_tech  else {})
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
        if self.include_tech:
            parts.append(h['tech'])
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
            if self.include_tech:   rest.append(h['tech'])
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


import numpy as np
from collections import deque

class ObservationBuffer:
    """
    ObservationBuffer는 시계열 데이터와 순간 데이터를 관리하는 클래스입니다.

    매개변수
    ----------
    lookback : int
        보관할 과거 타임스텝 수입니다.
    ts_keys : list of str
        시계열로 버퍼링할 키 목록입니다 (예: 'ohlcv', 'orderbook').
    inst_keys : list of str
        순간적으로 반환할 키 목록입니다 (예: 'position', 'pnl').
    """
    def __init__(self, lookback, ts_keys=None, inst_keys=None):
        self.lookback = lookback
        self.ts_keys = ts_keys or []          # 시계열 키
        self.inst_keys = inst_keys or []      # 순간 키

        # lookback 크기의 deque를 사용해 시계열 버퍼 초기화
        self.buffers = {
            key: deque(maxlen=lookback)
            for key in self.ts_keys
        }
        # 순간 데이터는 최신 값만 저장
        self.current = {key: None for key in self.inst_keys}

    def update(self, data: dict):
        """
        새로운 데이터를 받아 버퍼와 현재 값에 업데이트합니다.

        매개변수
        ----------
        data : dict
            ts_keys와 inst_keys에 정의된 모든 키를 포함해야 합니다.
            - 시계열 키: 리스트나 배열 형식의 값
            - 순간 키: 스칼라나 배열 형식의 값
        """
        # 시계열 데이터 업데이트
        for key in self.ts_keys:
            if key not in data:
                raise KeyError(f"시계열 키 누락: {key}")
            # 입력 데이터를 복사하여 deque에 추가
            self.buffers[key].append(np.array(data[key], dtype=float))

        # 순간 데이터 업데이트
        for key in self.inst_keys:
            if key not in data:
                raise KeyError(f"순간 키 누락: {key}")
            self.current[key] = np.array(data[key], dtype=float)

    def get_observation(self):
        """
        현재 관찰(observation)을 반환합니다.

        반환값
        -------
        obs : dict
            - ts_keys: (lookback, ...) 형태의 numpy 배열
            - inst_keys: 배열 또는 스칼라 값
        """
        obs = {}
        for key in self.ts_keys:
            buf = list(self.buffers[key])
            # 버퍼가 완전히 채워지지 않았으면 앞부분을 0으로 패딩
            if len(buf) < self.lookback:
                feat_shape = buf[0].shape if buf else (1,)
                padding = [np.zeros(feat_shape, dtype=float)] * (self.lookback - len(buf))
                buf = padding + buf
            # 타임스텝 축 방향으로 쌓기
            obs[key] = np.stack(buf, axis=0)

        for key in self.inst_keys:
            val = self.current.get(key)
            if val is None:
                raise ValueError(f"순간 키 '{key}'가 설정되지 않았습니다.")
            obs[key] = val

        return obs

    def reset(self):
        """
        모든 버퍼와 순간 데이터를 초기화합니다.
        """
        for key in self.ts_keys:
            self.buffers[key].clear()
        for key in self.inst_keys:
            self.current[key] = None

# 사용 예시
if __name__ == "__main__":
    lookback = 5
    buf = ObservationBuffer(
        lookback=lookback,
        ts_keys=['ohlcv', 'orderbook'],
        inst_keys=['position', 'pnl']
    )

    # 예시 데이터 업데이트 및 관찰 출력
    for t in range(7):
        data = {
            'ohlcv': [100 + t, 101 + t, 99 + t, 100 + t, 500 + 10*t],  # [open, high, low, close, volume]
            'orderbook': [1.0*t, 2.0*t, 3.0*t],                         # [bid, ask, spread]
            'position': t % 2,                                           # 현재 포지션
            'pnl': (t * 10.0)                                            # PnL
        }
        buf.update(data)
        obs = buf.get_observation()
        print(f"시간 {t}, ohlcv 버퍼 형태: {obs['ohlcv'].shape}, position: {obs['position']}")
