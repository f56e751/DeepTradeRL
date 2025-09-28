import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Type, Optional

from .rewards import RewardStrategy
from .actions import ActionStrategy
from .inventory import Inventory
from ..data_handler import DataHandlerBase
from .observation import Observation, ObservationBuffer


class UnifiedTradingEnv(gym.Env):

    def __init__(self, 
                df: pd.DataFrame,
                reward_strategy: Type[RewardStrategy],
                action_strategy: ActionStrategy,
                # handler_cls: Type[DataHandlerBase],
                initial_cash: float = 100000.0,
                transaction_fee: float = 0.0023,
                lookback: int = 9,
                lob_levels: int = 0,
                h_max: int = 1,
                hold_threshold: float = 0.2,
                include_ohlcv: Optional[bool] = True,
                include_tech: Optional[bool] = False,
                include_pnl: Optional[bool] = True,
                include_spread: Optional[bool] = False,
                include_position: Optional[bool] = True,
                include_orderbook: Optional[bool] = True,
                include_portfolio_value: Optional[bool] = True,
                include_cash: bool = False,
                tech_dim: Optional[int] = 0,
                price_hint: bool = False,
                price_hint_accuracy: float = 1.0,
                ):
        
        super().__init__()
        self.df = df.drop(columns=['timestamp', 'tic']).reset_index(drop=True)
        print(self.df)
        self.h_max = h_max
        self.hold_threshold = hold_threshold

        # 스텝 초기화
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        # 포함 변수 flag 초기화
        # self.include_ohlcv = include_ohlcv
        self.include_tech = include_tech
        self.include_pnl = include_pnl
        self.include_spread = include_spread
        self.include_position = include_position
        self.include_portfolio_value = include_portfolio_value
        self.include_cash = include_cash
        # self.include_orderbook = include_orderbook
        self.price_hint = price_hint
        self.price_hint_accuracy = float(np.clip(price_hint_accuracy, 0.0, 1.0))



        # Inventory 초기화
        self.inventory = Inventory(initial_cash)

        # reward, action strategy 초기화
        self.transaction_fee = transaction_fee
        # TODO 이 부분 다양한 종목에도 적용되게 하기
        price = self.get_price()
        price_map = {'TICKER': price}
        self.reward_strategy = reward_strategy(self.inventory, self.transaction_fee, price_map)
            # TODO ActionStrategy 도 비슷하게 인자 정의에 맞춰 초기화
        self.action_strategy = action_strategy()
        


        # handler 생성
        inst_keys=['pnl','position']
        if self.include_cash:
            inst_keys.append('cash')
        if self.include_portfolio_value:
            inst_keys.append('portfolio_value')
        if self.price_hint:
            inst_keys.append('price_hint')

        # Observation 생성
        self.observation_buffer = ObservationBuffer(
            lookback=lookback,
            ts_keys=['df'],
            inst_keys=inst_keys
        )

        # Action 스페이스 설정
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,), dtype=np.float32
        )

        # Observation 스페이스 설정
        # 1) df에서 남은 컬럼 개수 계산
        num_df_feats = self.df.shape[1]  # df.drop(['timestamp','tic']) 후 컬럼 수

        # 2) 포함 옵션에 따라 추가 차원 계산
        obs_dim = num_df_feats * lookback
        print("--- 관측 공간(Observation Space) 차원 계산 ---")
        print(f"  - 기본 (df 피처 수 * lookback): {num_df_feats} * {lookback} = {num_df_feats * lookback}")

        # if include_ohlcv:
        #     obs_dim += 5 * lookback           # [open, high, low, close, volume]
        if include_pnl:
            obs_dim += 1           # PnL
            print("  - PnL: +1")
        if include_spread:
            obs_dim += 1           # bid-ask spread
            print("  - Spread: +1")
        if include_position:
            obs_dim += 1           # 포지션 정보
            print("  - Position: +1")
        if self.include_cash:
            obs_dim += 1           # 현금 정보
            print("  - Cash: +1")
        if self.include_portfolio_value:
            obs_dim += 1           # 포트폴리오 가치
            print("  - Portfolio Value: +1")
        if self.price_hint:
            obs_dim += 1           # 현재 가격 힌트
            print("  - Price Hint: +1")
        
        print("-------------------------------------------")
        print(f"  >>> 총 관측 차원(obs_dim): {obs_dim}")
        print("-------------------------------------------")
        # 3) action_space, observation_space 설정
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # 필요시 멤버로도 저장
        self.obs_dim = obs_dim
        self.lookback = lookback


    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)
        
        # 스텝과 인벤토리 초기화
        self.current_step = 0
        self.inventory.reset()
        # 히스토리 버퍼 초기화
        self.observation_buffer.reset()
        # reward_strategy 초기화
        # TODO 이 부분 다양한 종목에도 적용되게 하기
        price = self.get_price()
        price_map = {'TICKER': price}
        self.reward_strategy.reset(price_map)

        
        # 첫 관측 생성 및 info 딕셔너리 반환
        obs = self._get_obs()
        info = {
            'cash': self.inventory.get_cash(),
            'position': self.inventory.get_positions()
        }
        return obs, info

    def _get_obs(self):
        # 1) 현재 행 가져오기
        row = self.df.iloc[self.current_step]
        next_row = self.df.iloc[self.current_step + 1] if self.current_step + 1 < self.max_steps else row

        # 2) 피처 리스트 조립
        # feats = []
        data = {}
        data['df'] = row.tolist()



        # # — 기술지표 추가 (데이터프레임의 마지막 tech_dim 컬럼에서 가져온다고 가정)
        # if self.observation.include_tech and getattr(self.observation, "tech_dim", 0) > 0:
        #     start = len(row) - self.observation.tech_dim
        #     feats.extend(row.iloc[start : ].tolist())

        # TODO 기술적 지표 포함하기

        # — 현재 포지션
        if self.include_position:
            pos = int(np.sign(self.inventory.get_position("TICKER")))
            data['position'] = [pos]
        
        # — 보유 현금
        if self.include_cash:
            data['cash'] = [self.inventory.get_cash()]

        # — 미실현 PnL
        if self.include_pnl:
            price = float(row["close"])
            pnl = self.inventory.get_unrealized_pnl({"TICKER": price})
            data['pnl'] = pnl
            # feats.append(pnl)

        # — 포트폴리오 가치
        if self.include_portfolio_value:
            price = float(row["close"])
            pf_val = self.inventory.get_portfolio_value({"TICKER": price})
            data['portfolio_value'] = [pf_val]

        if self.price_hint:
            current_price = float(row["close"])
            next_price = float(next_row["close"])
            price_diff_ratio = (next_price - current_price) / max(1e-12, current_price)
            threshold = 0.02  # 2%

            # 실제 정답 힌트(상승=+1, 하락=-1, 변화없음=0)
            if price_diff_ratio > threshold:
                true_hint = 1
            elif price_diff_ratio < -threshold:
                true_hint = -1
            else:
                true_hint = 0

            # 난수 소스 선택
            rng = getattr(self, "_np_random", None)
            if rng is None:
                rng = np.random

            if true_hint == 0:
                price_hint = 0
            else:
                # p 확률로 정답, (1-p) 확률로 반대 신호
                p = self.price_hint_accuracy
                flip = rng.random() >= p
                price_hint = -true_hint if flip else true_hint

            data['price_hint'] = [price_hint]


        self.observation_buffer.update(data)
        obs = self.observation_buffer.get_observation_vector()
        return obs


    
    def get_price(self):
        return float(self.df.loc[self.current_step, 'close'])


    def step(self, action):
        """
        연속 액션 실행, 보상 계산 및 다음 관측 반환
        """
        # 1) 행동 전략 위임
        act = float(np.clip(action, -1.0, 1.0))
        price = self.get_price()
        result = self.action_strategy.compute(
            raw_action=act,
            inventory=self.inventory,
            price=price,
            h_max=self.h_max,
            hold_thr=self.hold_threshold,
            transaction_fee=self.transaction_fee
        )
        real_act = result.quantity
        invalid = result.invalid

        # 2) 거래 수행 (invalid=False일 때만)
        tx = None
        if not invalid:
            # 매도
            if real_act < 0:
                qty = abs(real_act)
                tx = self.inventory.sell('TICKER', qty, price)
                fee = qty * price * self.transaction_fee
                self.inventory.cash -= fee       # ← 매도 수수료 차감
                # 수수료는 RewardStrategy에서 처리하도록 해도 되고,
                # 여기서 직접 차감해도 됩니다.
            # 매수
            elif real_act > 0:
                qty = real_act
                tx = self.inventory.buy('TICKER', qty, price)
                # self.inventory.cash -= qty * price * self.transaction_fee
            # real_act == 0: 홀드 (tx remains None)

        # 3) 다음 스텝 및 종료 판정
        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        # 4) 보상 계산
        price = self.get_price() if not done else price
        pf_val = self.inventory.get_portfolio_value({'TICKER': price})
        reward = 0.0 if invalid else self.reward_strategy.compute(tx, pf_val)

        # 5) 관측 & info 반환
        obs = None if done else self._get_obs()
        info = {
            'invalid': invalid,
            'cash': self.inventory.get_cash(),
            'position': self.inventory.get_position('TICKER')
        }
        return obs, float(reward), done, False, info



    def render(self, mode='human'):
        price = float(self.df.loc[self.current_step, 'close'])
        pos = self.inventory.get_position('TICKER')
        cash = self.inventory.get_cash()
        print(
            f"Step:{self.current_step} | Close:{price:.2f} | "
            f"Pos:{pos} | Cash:{cash:.2f}"
        )

    def seed(self, seed=None):
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]