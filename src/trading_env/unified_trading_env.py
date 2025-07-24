import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Type, Optional

from .rewards import RewardStrategy
from .actions import ActionStrategy
from .inventory import Inventory
from ..data_handler import DataHandlerBase
from .observation import Observation


class UnifiedTradingEnv(gym.Env):

    def __init__(self, 
                df: pd.DataFrame,
                reward_strategy: Type[RewardStrategy],
                action_strategy: ActionStrategy,
                handler_cls: Type[DataHandlerBase],
                initial_cash: float = 100000.0,
                transaction_fee: float = 0.0023,
                lookback: int = 9,
                lob_levels: int = 0,
                h_max: int = 1,
                hold_threshold: float = 0.2,
                include_ohlcv: Optional[bool] = False,
                include_tech: Optional[bool] = False,
                include_pnl: Optional[bool] = False,
                include_spread: Optional[bool] = False,
                tech_dim: Optional[int] = 0
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.h_max = h_max
        self.hold_threshold = hold_threshold

        # 스텝 초기화
        self.current_step = 0
        self.max_steps = len(self.df) - 1



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

        # Observation 생성
        self.observation = Observation(
            lob_levels=lob_levels,
            lookback=lookback,
            include_pnl=include_pnl,
            include_spread=include_spread,
            include_ohlcv=include_ohlcv,
            include_tech=include_tech,
            window_size=1,
            tech_dim = tech_dim
        )

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


    def reset(self):
        # 스텝과 인벤토리 초기화
        self.current_step = 0
        self.inventory.reset()
        # 히스토리 버퍼 초기화
        self.observation.reset()
        # reward_strategy 초기화
        # TODO 이 부분 다양한 종목에도 적용되게 하기
        price = self.get_price()
        price_map = {'TICKER': price}
        self.reward_strategy.reset(price_map)

        
        # 첫 관측 생성
        return self._get_obs()

    def _get_obs(self):
        # 1) 현재 행 가져오기
        row = self.df.iloc[self.current_step]

        # 2) 피처 리스트 조립
        feats = []

        # — OHLCV 추가
        if self.observation.include_ohlcv:
            feats.extend([
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            ])

        # — 기술지표 추가 (데이터프레임의 마지막 tech_dim 컬럼에서 가져온다고 가정)
        if self.observation.include_tech and getattr(self.observation, "tech_dim", 0) > 0:
            start = len(row) - self.observation.tech_dim
            feats.extend(row.iloc[start : ].tolist())

        # — 현재 포지션
        pos = int(np.sign(self.inventory.get_position("TICKER")))
        feats.append(pos)

        # — 미실현 PnL
        if self.observation.include_pnl:
            price = float(row["close"])
            pnl = self.inventory.get_unrealized_pnl({"TICKER": price})
            feats.append(pnl)

        # — 스프레드 (Orderbook이 없으므로 0으로 대체)
        if self.observation.include_spread:
            feats.append(0.0)

        # 3) Observation 버퍼에 추가 후 MLP 입력 반환
        self.observation.append(feats)
        return self.observation.get_mlp_input()
    
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
                self.inventory.cash -= qty * price * self.transaction_fee
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
        return obs, float(reward), done, info


    # def step(self, action):
    #     """
    #     연속 액션 실행, 보상 계산 및 다음 관측 반환
    #     """
    #     done = False
    #     invalid = False

    #     # TODO 이 부분 action 종류 갈아낄 수 있게

    #     # 1) 액션 스케일링 및 real_act 계산
    #     act = float(np.clip(action, -1.0, 1.0))
    #     if abs(act) < self.hold_threshold:
    #         real_act = 0
    #     else:
    #         real_act = int(np.rint(act * self.h_max))
    #     # reward = 0.0

    #     # 2) 거래 수행: close 가격 사용
    #     price = self.get_price()
    #     tx = None
    #     # 매도
    #     if real_act < 0:
    #         qty = abs(real_act)
    #         if self.inventory.can_sell('TICKER', qty):
    #             tx = self.inventory.sell('TICKER', qty, price)
    #             fee = qty * price * self.transaction_fee
    #             # reward = tx.realized_pnl - fee
    #         else:
    #             invalid = True
    #     # 매수
    #     elif real_act > 0:
    #         qty = real_act
    #         fee = qty * price * self.transaction_fee
    #         if self.inventory.can_buy('TICKER', qty, price, fee):
    #             tx = self.inventory.buy('TICKER', qty, price)
    #             self.inventory.cash -= fee
    #         else:
    #             invalid = True
    #     # real_act == 0: 홀드

    #     # 3) 스텝 증가 및 종료 판정
    #     self.current_step += 1
    #     if self.current_step >= self.max_steps:
    #         done = True


    #     # reward 계산
    #     # TODO 이 부분 다양한 종목에도 적용되게 하기
    #     price = self.get_price()
    #     price_map = {'TICKER': price}
    #     pf_val = self.inventory.get_portfolio_value(price_map)
    #     reward = self.reward_strategy.compute(tx, pf_val)

    #     obs = self._get_obs() if not done else None
    #     info = {
    #         'invalid': invalid,
    #         'cash': self.inventory.get_cash(),
    #         'position': self.inventory.get_position('TICKER')
    #     }
    #     return obs, float(reward), done, info

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