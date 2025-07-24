from abc import ABC, abstractmethod
from .inventory import Inventory
import numpy as np
from .transaction_info import TransactionInfo




class RewardStrategy(ABC):
    """
    추상 보상 전략 클래스

    - inventory: 포트폴리오 가치 계산에 사용
    - prev_portfolio_value: 직전 스텝 포트폴리오 총가치 저장
    """
    def __init__(self, inventory: Inventory, transaction_fee: float, price_map):
        self.inventory = inventory
        # 최초 prev 값 설정 (reset 시에도 초기화해 주세요)
        self.prev_portfolio_value = self.inventory.get_portfolio_value(price_map)
        self.transaction_fee = transaction_fee

    def reset(self, price_map):
        # 최초 prev 값 설정 (reset 시에도 초기화해 주세요)
        self.prev_portfolio_value = self.inventory.get_portfolio_value(price_map)

    @abstractmethod
    def compute(self, tx, current_portfolio_value: float) -> float:
        """
        보상을 계산해서 반환

        Args:
            tx: TransactionInfo 객체 또는 None (거래 시점의 실현 PnL 정보 포함)
            current_portfolio_value: 현 스텝 만료 후 포트폴리오 총가치

        Returns:
            float: 이 스텝의 reward
        """
        ...


    # 이 함수는 없어도 될듯?
    # def update(self, current_portfolio_value: float):
    #     """
    #     스텝이 끝난 뒤 prev_portfolio_value 갱신
    #     """
    #     self.prev_portfolio_value = current_portfolio_value


class RealizedPnLReward(RewardStrategy):
    """거래 시점에만 실현손익(tx.realized_pnl) - 수수료 페널티를 줍니다."""
    def compute(self, tx: TransactionInfo, current_portfolio_value: float) -> float:
        if tx is None:
            return 0.0
        fee = abs(tx.quantity) * tx.price * self.transaction_fee
        return tx.realized_pnl - fee


class LogPortfolioReturnReward(RewardStrategy):
    """포트폴리오 가치의 로그수익률: r_t = ln(p_t / p_{t-1})"""
    def compute(self, tx, current_portfolio_value: float) -> float:
        prev = self.prev_portfolio_value
        # zero-division 방지
        if prev <= 0 or current_portfolio_value <= 0:
            return 0.0
        return np.log(current_portfolio_value / prev)


class CombinedReward(RewardStrategy):
    """
    실현 PnL 보상 + 로그수익률 보상을 합산
    """
    def compute(self, tx, current_portfolio_value: float) -> float:
        r1 = RealizedPnLReward(self.inventory).compute(tx, current_portfolio_value)
        # 이 객체의 prev 값을 그대로 쓸 수 있도록 로그 전략도 init 때 받거나,
        # 아래처럼 임시 생성하여 사용해도 됩니다.
        log_strat = LogPortfolioReturnReward(self.inventory)
        log_strat.prev_portfolio_value = self.prev_portfolio_value
        r2 = log_strat.compute(tx, current_portfolio_value)
        return r1 + r2
