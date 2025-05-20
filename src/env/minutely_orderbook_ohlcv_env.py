import gym
from gym import spaces
import numpy as np
import pandas as pd

from src.env.inventory import Inventory
from src.env.transaction_info import TransactionInfo
from src.data_handler.csv_processor import merge_lob_and_ohlcv
from src.data_handler.data_handler import Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler
from src.env.observation import Observation, InputType

class MinutelyOrderbookOHLCVEnv(gym.Env):
    """
    분 단위 Orderbook + OHLCV 통합 거래 환경

    - LOB 10단계 + 분 단위 OHLCV 병합 데이터 사용
    - handler_cls에 따라 자동으로 pnl/spread 포함 여부 결정
    - lookback틱 과거 LOB 스냅샷, 현 스텝 LOB, 포지션, (선택) pnl/spread/OHLCV 포함
    - MLP 또는 LSTM 입력 타입 지원
    - 액션 연속형: Box(low=-1, high=1, shape=(1,), dtype=np.float32)
      • action ∈ [-1,1]
      • |action| < hold_threshold → 홀드
      • real_act = round(action * h_max)
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        handler_cls,
        initial_cash: float = 100000.0,
        lob_levels: int = 10,
        lookback: int = 9,
        window_size: int = 9,
        input_type: InputType = InputType.MLP,
        transaction_fee: float = 0.0023,
        h_max: int = 1,
        hold_threshold: float = 0.2,
    ):
        super().__init__()
        # 데이터 설정
        self.df = df.reset_index(drop=True)
        self.transaction_fee = transaction_fee
        self.h_max = h_max
        self.hold_threshold = hold_threshold

        # handler 생성 및 include 옵션 자동 결정
          # lstm 출력용이면 lookback을 1로 함 (데이터 중복 방지)
        if input_type == InputType.LSTM:
            lookback = 1
        self.handler = handler_cls(df=self.df, lob_levels=lob_levels, lookback=lookback)
        include_pnl = isinstance(self.handler, (Sc202OHLCVHandler, Sc203OHLCVHandler))
        include_spread = isinstance(self.handler, Sc203OHLCVHandler)
        include_ohlcv = True

        # Observation 생성


        self.observation = Observation(
            lob_levels=lob_levels,
            lookback=lookback,
            include_pnl=include_pnl,
            include_spread=include_spread,
            include_ohlcv=include_ohlcv,
            window_size=window_size
        )
        self.input_type = input_type

        # Inventory 및 스텝
        self.inventory = Inventory(initial_cash)
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        sample = self._init_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample.shape, dtype=np.float32)

    def _init_observation(self):
        self.observation.reset()
        pos, pnl = 0, 0.0
        feats = self.handler.get_observation(step=0, position=pos, pnl=pnl)
        self.observation.append(feats)
        return self._get_obs()

    def reset(self):
        """
        환경 초기화 후 초기 관측 생성 및 히스토리 채우기
        - window_size-1 만큼 hold action(변경 없음) 으로 초기 피처를 반복 추가
        - 마지막으로 _get_obs() 호출하여 history 가득 채운 상태로 반환
        """
        self.current_step = 0
        self.inventory.reset()
        # Observation history 초기화
        self.observation.reset()
        # 초기 피처 생성 (현 스텝 = 0)
        pos, pnl = 0, 0.0
        init_feats = self.handler.get_observation(step=0, position=pos, pnl=pnl)
        # window_size-1 만큼 hold (동일 피처)로 채우기
        for _ in range(self.observation.window_size - 1):
            self.observation.append(init_feats)
        # 마지막으로 현재 스텝 obs 추가 및 반환
        return self._get_obs()

    def _get_best_bid(self):
        return float(self.df.loc[self.current_step, 'bid_px_00'])

    def _get_best_ask(self):
        return float(self.df.loc[self.current_step, 'ask_px_00'])

    def _get_mid_price(self):
        return (self._get_best_bid() + self._get_best_ask()) / 2

    def _get_obs(self):
        pos = np.sign(self.inventory.get_position('TICKER'))
        mid = self._get_mid_price()
        pnl = self.inventory.get_unrealized_pnl({'TICKER': mid})
        feats = self.handler.get_observation(
            step=self.current_step, position=int(pos), pnl=float(pnl)
        )
        self.observation.append(feats)
        if self.input_type == InputType.MLP:
            return self.observation.get_mlp_input()
        return self.observation.get_lstm_input()

    def step(self, action):
        """연속 액션 실행 및 보상 계산"""
        done, invalid = False, False
        # 1) 스칼라 값으로 변환
        act = float(np.clip(action, -1.0, 1.0))
        # 2) 홀드 영역 처리
        if abs(act) < self.hold_threshold:
            real_act = 0
        else:
            real_act = int(np.rint(act * self.h_max))
        reward = 0.0

        # 3) 매도
        if real_act < 0:
            qty = abs(real_act)
            price = self._get_best_bid()
            if self.inventory.can_sell('TICKER', qty):
                tx = self.inventory.sell('TICKER', qty, price)
                fee = qty * price * self.transaction_fee
                reward = tx.realized_pnl - fee
            else:
                invalid = True
        # 4) 매수
        elif real_act > 0:
            qty = real_act
            price = self._get_best_ask()
            fee = qty * price * self.transaction_fee
            if self.inventory.can_buy('TICKER', qty, price, fee):
                tx = self.inventory.buy('TICKER', qty, price)
                self.inventory.cash -= fee
            else:
                invalid = True
        # real_act == 0: 홀드(no-op)

        # 5) 다음 스텝
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = {
            'invalid': invalid,
            'cash': self.inventory.get_cash(),
            'position': self.inventory.get_position('TICKER')
        }
        return obs, float(reward), done, info

    def render(self, mode='human'):
        bid, ask = self._get_best_bid(), self._get_best_ask()
        pos = self.inventory.get_position('TICKER')
        cash = self.inventory.get_cash()
        print(f"Step:{self.current_step} | Bid:{bid:.2f} | Ask:{ask:.2f} | Pos:{pos} | Cash:{cash:.2f}")

    def get_obs_dim(self):
        """관측(observation) 벡터의 차원(shape)을 반환합니다."""
        return self.observation_space.shape

    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        np.random.seed(seed)
        return [seed]