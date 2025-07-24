import gym
from gym import spaces
import numpy as np
import pandas as pd

import sys
import os


from .inventory import Inventory
from ..data_handler import OHLCVPositionHandler, OHLCVPositionPnlHandler
from .observation import Observation
# # Add the src directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from src.env.inventory import Inventory
# from src.data_handler.data_handler import OHLCVPositionHandler, OHLCVPositionPnlHandler
# from src.env.observation import Observation


class MinutelyOHLCVEnv(gym.Env):
    """
    분 단위 OHLCV 전용 거래 환경 (MLP 입력 전용, Observation 사용)

    - OHLCV 데이터 + position (+ optional P&L) 정보 사용
    - handler_cls 인자를 통해 OHLCVPositionHandler 또는 OHLCVPositionPnlHandler 선택 가능
    - Observation을 통해 피처 윈도우 관리 및 MLP 입력 반환
    - 연속형 액션: Box(low=-1, high=1, shape=(1,), dtype=np.float32)
      • action ∈ [-1,1]
      • real_act = round(action * h_max)
      • |real_act| ≤ h_max
    - 거래 가격은 해당 스텝의 close 가격 사용
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        handler_cls,
        initial_cash: float = 100000.0,
        lookback: int = 9,
        transaction_fee: float = 0.0023,
        h_max: int = 1,
        hold_threshold: float = 0.2,
        window_size: int = 1
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.transaction_fee = transaction_fee
        self.h_max = h_max
        self.hold_threshold = hold_threshold

        # handler 선택: OHLCV only (+position) 또는 OHLCV+Pnl
        self.handler = handler_cls(self.df)
        include_pnl = issubclass(handler_cls, OHLCVPositionPnlHandler)

        # Observation 생성
        self.observation = Observation(
            lob_levels=0,
            lookback=lookback,
            include_pnl=include_pnl,
            include_spread=False,
            include_ohlcv=True,
            include_tech=False,
            window_size=window_size
        )

        # Inventory 및 스텝 초기화
        self.inventory = Inventory(initial_cash)
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        # Action 및 Observation 스페이스 설정
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,), dtype=np.float32
        )
        obs_dim = self.observation.dim_total
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

    def get_price(self):
        return float(self.df.loc[self.current_step, 'close'])

    def reset(self):
        """
        환경 초기화 및 첫 관측 반환
        """
        self.current_step = 0
        self.inventory.reset()

        self.observation.reset()
        # 초기 position, pnl
        init_feats = self.handler.get_observation(
            step=0, position=0, pnl=0.0
        )
        self.observation.append(init_feats)
        return self.observation.get_mlp_input()

    def _get_obs(self):
        pos = int(np.sign(self.inventory.get_position('TICKER')))
        # mid price 활용하여 unrealized PnL 계산
        price = float(self.df.loc[self.current_step, 'close'])
        pnl = self.inventory.get_unrealized_pnl({'TICKER': price})

        feats = self.handler.get_observation(
            step=self.current_step,
            position=pos,
            pnl=pnl
        )
        self.observation.append(feats)
        return self.observation.get_mlp_input()

    def step(self, action):
        """
        연속 액션 실행, 보상 계산 및 다음 관측 반환
        """
        done = False
        invalid = False

        # 1) 액션 스케일링 및 real_act 계산
        act = float(np.clip(action, -1.0, 1.0))
        if abs(act) < self.hold_threshold:
            real_act = 0
        else:
            real_act = int(np.rint(act * self.h_max))
        reward = 0.0

        # 2) 거래 수행: close 가격 사용
        price = float(self.df.loc[self.current_step, 'close'])
        # 매도
        if real_act < 0:
            qty = abs(real_act)
            if self.inventory.can_sell('TICKER', qty):
                tx = self.inventory.sell('TICKER', qty, price)
                fee = qty * price * self.transaction_fee
                reward = tx.realized_pnl - fee
            else:
                invalid = True
        # 매수
        elif real_act > 0:
            qty = real_act
            fee = qty * price * self.transaction_fee
            if self.inventory.can_buy('TICKER', qty, price, fee):
                tx = self.inventory.buy('TICKER', qty, price)
                self.inventory.cash -= fee
            else:
                invalid = True
        # real_act == 0: 홀드

        # 3) 스텝 증가 및 종료 판정
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_obs() if not done else None
        info = {
            'invalid': invalid,
            'cash': self.inventory.get_cash(),
            'position': self.inventory.get_position('TICKER')
        }
        return obs, float(reward), done, info

    def render(self, mode='human'):
        price = float(self.df.loc[self.current_step, 'close'])
        pos = self.inventory.get_position('TICKER')
        cash = self.inventory.get_cash()
        print(
            f"Step:{self.current_step} | Close:{price:.2f} | "
            f"Pos:{pos} | Cash:{cash:.2f}"
        )

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
