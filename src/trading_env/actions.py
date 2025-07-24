from abc import ABC, abstractmethod
from .inventory import Inventory
import numpy as np
from .transaction_info import TransactionInfo
from typing import NamedTuple



class ActionResult(NamedTuple):
    quantity: int       # 매수(+)/매도(–)/홀드(0) 수량
    invalid: bool = False  # 제약 위반 여부

class ActionStrategy(ABC):
    @abstractmethod
    def compute(
        self,
        raw_action: float,
        inventory: Inventory,
        price: float,
        h_max: int,
        hold_thr: float,
        transaction_fee: float
    ) -> ActionResult:
        """
        raw_action: 에이전트가 뱉은 연속값 (-1~1)
        inventory: 현재 보유 현금/포지션 정보
        price: 현 스텝 종가
        h_max: 최대 매매 수량
        hold_thr: 홀드 기준 임계값
        transaction_fee: 수수료율
        """
        ...




class TestActionStrategy(ActionStrategy):
    def compute(self, raw_action, inventory, price, h_max, hold_thr, transaction_fee):
        if abs(raw_action) < hold_thr:
            return ActionResult(0)
        # 원하는 수량
        qty = int(np.rint(raw_action * h_max))
        # 가능 여부만 체크
        if qty > 0 and not inventory.can_buy('TICKER', qty, price, qty*price*transaction_fee):
            return ActionResult(0, invalid=True)
        if qty < 0 and not inventory.can_sell('TICKER', abs(qty)):
            return ActionResult(0, invalid=True)
        return ActionResult(qty)



class ClippedActionStrategy(ActionStrategy):
    def compute(self, raw_action, inventory, price, h_max, hold_thr, transaction_fee):
        if abs(raw_action) < hold_thr:
            return ActionResult(0)

        desired = int(np.rint(raw_action * h_max))
        if desired > 0:
            # 현금 기반 최대 구매량 계산
            max_buy = int(inventory.cash // (price*(1+transaction_fee)))
            qty = min(desired, max_buy)
        else:
            max_sell = abs(inventory.get_position('TICKER'))
            qty = -min(abs(desired), max_sell)

        return ActionResult(qty)



class PercentPortfolioStrategy(ActionStrategy):
    def compute(self, raw_action, inventory, price, *_):
        # 현재 포트폴리오 가치
        pv = inventory.get_portfolio_value({'TICKER': price})
        target_value = raw_action * pv  # 음수면 공매도(숏)
        curr_pos_value = inventory.get_position('TICKER') * price
        delta = target_value - curr_pos_value
        qty = int(delta / price)
        # +/-만큼 사고/팔되, feasibility는 Clipped 전략처럼 처리
        # (여기선 단순화)
        return ActionResult(qty)

