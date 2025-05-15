import gym
from gym import spaces
import numpy as np
import pandas as pd
from src.env.inventory import Inventory
from src.data_handler.data_handler import MultiDataHandler

class DailyStockTradingEnv(gym.Env):
    """
    연속형 다중 종목 거래 환경 (MultiDataHandler + Inventory 사용):
      - df로부터 MultiDataHandler로 가격 및 기술 지표 데이터를 관리
      - action: ndarray(shape=(D,)), 각 요소 ∈ [-1,1]
      - real_act[i] = round(action[i] * h_max) ∈ [-h_max, +h_max]
         • h_max: 한 번에 매수·매도할 수 있는 최대 주식 수를 지정하는 파라미터
      - transaction_fee: 거래 시 부과되는 비율 (예: 0.001 = 0.1%)
      - Inventory 클래스가 현금 및 포지션(보유 수량) 로직을 처리
      - 관측(observation): [cash, positions…, data_handler 관측 벡터…]
      - 보상(reward): 이전 포트폴리오 가치 대비 변화량
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 df: pd.DataFrame,
                 initial_balance: float = 1e6,
                 h_max: int = 10,
                 transaction_fee: float = 0.001,
                 wanted_features: list = ['Adj Close', 'MACD', 'RSI', 'CCI', 'ADX'] ):
        super().__init__()

        # 1) MultiDataHandler 생성 시 feature_cols 인자로 전달
        # 가격 정보만 원하면 wanted_features = ['Adj Close'] 이렇게 하면 됨
        self.data_handler    = MultiDataHandler(df, feature_cols=wanted_features)
        self.tickers         = self.data_handler.tickers
        self.n_stock         = self.data_handler.get_num_tickers()

        # 2) 가격 시퀀스 추출 (각 티커의 Adj Close)
        self.price_array = np.stack([
            self.data_handler.handlers[t].df["Adj Close"].values
            for t in self.tickers
        ], axis=1)  # shape (T, D)

        # 3) 파라미터
        self.initial_balance = initial_balance
        self.h_max           = h_max
        self.transaction_fee = transaction_fee

        # 4) Inventory 인스턴스 생성
        self.inventory = Inventory(initial_cash=initial_balance)

        # 5) Action & Observation space 정의
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_stock,),
            dtype=np.float32
        )
        sample_feat = self.data_handler.get_observation(0)
        obs_dim     = 1 + self.n_stock + sample_feat.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        # Inventory 초기화 및 step, prev_value 초기화
        self.inventory.reset()
        self.current_step = 0
        # 초기 포트폴리오 가치는 초기 잔고와 동일
        self.prev_portfolio_value = self.initial_balance
        return self._get_observation()

    def step(self, action):
        # 1) 연속 action → 정수 매매량 real_act ∈ [-h_max, +h_max]
        action   = np.clip(action, -1.0, 1.0)
        real_act = np.rint(action * self.h_max).astype(int)

        # 2) 현 시점 가격 벡터
        prices = self.price_array[self.current_step]

        # 3) 사전 유효성 검사
        sell_qty = np.clip(-real_act, 0, None)
        buy_qty  = np.clip(real_act,  0, None)
        # 매도 가능 여부 체크
        for i, ticker in enumerate(self.tickers):
            if sell_qty[i] > self.inventory.get_position(ticker):
                obs = self._get_observation()
                return obs, 0.0, False, {"invalid": True}
        # 매수 비용 초과 체크 (수수료 포함)
        total_cost = np.dot(buy_qty, prices) * (1 + self.transaction_fee)
        if total_cost > self.inventory.get_cash():
            obs = self._get_observation()
            return obs, 0.0, False, {"invalid": True}

        # 4) 거래 실행 (수수료 반영)
        executed_sells = np.zeros(self.n_stock, dtype=int)
        executed_buys  = np.zeros(self.n_stock, dtype=int)
        for i, ticker in enumerate(self.tickers):
            qty   = real_act[i]
            price = prices[i]
            if qty < 0:
                sold = self.inventory.sell(ticker, abs(qty), price)
                fee  = sold * price * self.transaction_fee
                self.inventory.cash -= fee
                executed_sells[i] = sold
            elif qty > 0:
                bought = self.inventory.buy(ticker, qty, price)
                fee    = bought * price * self.transaction_fee
                self.inventory.cash -= fee
                executed_buys[i] = bought

        # 5) 스텝 진행
        self.current_step += 1
        done = (self.current_step >= len(self.price_array) - 1)

        # 6) 보상 계산: 이전 대비 포트폴리오 가치 변화
        price_map      = {t: prices[i] for i, t in enumerate(self.tickers)}
        new_value      = self.inventory.get_portfolio_value(price_map)
        reward         = new_value - self.prev_portfolio_value
        self.prev_portfolio_value = new_value

        # 7) 관측 & info 반환
        obs = self._get_observation()
        info = {
            "executed_sells": executed_sells,
            "executed_buys":  executed_buys,
            "cash":           self.inventory.get_cash(),
            "positions":      self.inventory.get_positions()
        }
        return obs, reward, done, info

    def get_obs_dim(self) -> int:
        """관측(observation) 벡터의 차원(길이)을 반환합니다."""
        return self.observation_space.shape[0]

    def _get_observation(self):
        cash     = self.inventory.get_cash()
        shares   = np.array(
            [self.inventory.get_position(t) for t in self.tickers],
            dtype=float
        )
        data_feat = self.data_handler.get_observation(self.current_step)
        return np.concatenate(([cash], shares, data_feat)).astype(np.float32)

    def render(self, mode="human"):
        cash   = self.inventory.get_cash()
        shares = {t: self.inventory.get_position(t) for t in self.tickers}
        price  = self.price_array[self.current_step]
        print(f"Step {self.current_step}: cash={cash:.0f}")
        for t, p in zip(self.tickers, price):
            print(f"  {t}: shares={shares[t]}, price={p:.2f}")

    def close(self):
        pass
