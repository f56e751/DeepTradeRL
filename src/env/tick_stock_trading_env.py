import gym
from gym import spaces
import numpy as np
import pandas as pd
from src.env.inventory import Inventory
from src.data_handler.data_handler import Sc201Handler, Sc202Handler, Sc203Handler, TickDataHandlerBase
from src.env.transaction_info import TransactionInfo

# Inventory, TickDataHandlerBase, Sc201Handler, Sc202Handler, Sc203Handler 클래스가 같은 스코프에 있어야 합니다.
class TickStockTradingEnv(gym.Env):
    """
    틱 단위 종목 거래 환경 (시장가 주문, 희소 보상)

    주요 특징:
    - 매수: 최우선 매도호가(best ask)로 1주 매수
    - 매도: 최우선 매수호가(best bid)로 1주 매도
    - 일일 손절: 보유 전량을 최우선 매수호가로 청산
    - transaction_fee: 거래 시 부과되는 비율 (예: 0.001 = 0.1%)

    TODO 수량에 따라 일일손절을 단일 가격이 아니라 여러 가격으로 나눠서 할 수 있게 수정

    보상 분배:
    - 논문에 따른 희소 보상(sparse reward)
      포지션 청산(매도 또는 일일 손절) 시 inventory.sell이 반환하는 realized PnL 사용
    - 그 외 스텝에서는 보상 0

    관측값(obs) 구성 (handler_cls에 따라):
    - 현재 호가 잔량(매수/매도 레벨)
    - 과거 lookback 틱의 LOB 스냅샷
    - 현재 포지션 상태(-1, 0, +1)
    - (Sc202 이상) 미실현 손익(pnl)
    - (Sc203) 스프레드(best ask - best bid)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        handler_cls,
        initial_cash: float = 100000.0,
        lob_levels: int = 10,
        lookback: int = 9,
        ticker: str = "TICKER",
        transaction_fee: float = 0.0023,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.ticker = ticker
        self.handler = handler_cls(df, lob_levels=lob_levels, lookback=lookback)
        self.inventory = Inventory(initial_cash)
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        self.action_space = spaces.Discrete(4)
        sample_obs = self.handler.get_observation(step=0, position=0, pnl=0.0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=sample_obs.shape, dtype=np.float32,
        )
        self.transaction_fee = transaction_fee

    def reset(self):
        """환경 초기화 후 초기 obs 반환"""
        self.current_step = 0
        self.inventory.reset()
        return self._get_obs()

    def _get_best_bid(self) -> float:
        """
        최우선 매수호가 가격(best bid price) 반환
        df 컬럼명: 'bid_px_00'
        """
        return float(self.df.loc[self.current_step, 'bid_px_00'])

    def _get_best_ask(self) -> float:
        """
        최우선 매도호가 가격(best ask price) 반환
        df 컬럼명: 'ask_px_00'
        """
        return float(self.df.loc[self.current_step, 'ask_px_00'])

    def _get_mid_price(self) -> float:
        """
        mid price 계산: (best bid + best ask) / 2
        """
        bid = self._get_best_bid()
        ask = self._get_best_ask()
        return (bid + ask) / 2

    def _get_obs(self) -> np.ndarray:
        pos = np.sign(self.inventory.get_position(self.ticker))
        mid_price = self._get_mid_price()
        pnl = self.inventory.get_unrealized_pnl({self.ticker: mid_price})
        return self.handler.get_observation(
            step=self.current_step, position=int(pos), pnl=float(pnl)
        )

    def step(self, action: int):
        """
        액션 실행 및 보상 계산
        - 0=매도, 2=매수, 3=일일 손절, 1=대기
        - 보상: 포지션 청산 시 realized PnL 반환
        - 현금/주식 부족 시 해당 액션은 no-op 처리, invalid flag 반환
        """
        done = False
        reward = 0.0
        invalid = False

        # 0=매도
        if action == 0:
            if self.inventory.can_sell(self.ticker, 1):
                price = self._get_best_bid()
                transactionInfo: TransactionInfo = self.inventory.sell(self.ticker, qty=1, price=price)
                pnl = transactionInfo.realized_pnl
                fee = price * 1 * self.transaction_fee
                reward = pnl - fee
            else:
                # 보유 주식 없으면 invalid
                invalid = True

        # 2=매수
        elif action == 2:
            price = self._get_best_ask()
            qty = 1
            fee = price * qty * self.transaction_fee
            if self.inventory.can_buy(self.ticker, 1, price, fee):
                transactionInfo: TransactionInfo = self.inventory.buy(self.ticker, qty=1, price=price)
                # 매수 시에도 수수료 부과 (현금에서 추가 차감)
                
                self.inventory.cash -= fee
            else:
                # 현금 부족 시 invalid
                invalid = True

        # 3=일일 손절
        elif action == 3:
            qty = self.inventory.get_position(self.ticker)
            if qty > 0:
                price = self._get_best_bid()
                transactionInfo: TransactionInfo = self.inventory.sell(self.ticker, qty=qty, price=price)
                pnl = transactionInfo.realized_pnl
                fee = price * qty * self.transaction_fee
                reward = pnl - fee
            else:
                # 보유 주식 없으면 invalid
                invalid = True

        # 1=대기: 항상 valid no-op

        # 스텝 진행 및 종료 여부 판단
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = {
            "invalid": invalid,
            "cash": self.inventory.get_cash(),
            "position": self.inventory.get_position(self.ticker),
        }
        return obs, float(reward), done, info


    def render(self, mode="human"):
        bid = self._get_best_bid()
        ask = self._get_best_ask()
        pos = self.inventory.get_position(self.ticker)
        cash = self.inventory.get_cash()
        print(f"Step:{self.current_step} | Bid:{bid:.2f} | Ask:{ask:.2f} | Pos:{pos} | Cash:{cash:.2f}")
