from typing import Dict

class Inventory:
    """
    다수 종목의 보유 수량과 현금 잔고 관리
    - 매수 시 진입 가격(cost basis) 누적
    - 매도 시 평균 단가를 기준으로 cost basis 차감
    - 미실현 P&L 계산 기능 제공
    """
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        """인벤토리 초기화"""
        self.cash: float = self.initial_cash
        self.positions: Dict[str, int] = {}
        # 각 종목의 총 취득원가 누적
        self.cost_basis: Dict[str, float] = {}

    def can_buy(self, ticker: str, qty: int, price: float) -> bool:
        return qty > 0 and (qty * price) <= self.cash

    def can_sell(self, ticker: str, qty: int) -> bool:
        return qty > 0 and self.positions.get(ticker, 0) >= qty

    def buy(self, ticker: str, qty: int, price: float) -> int:
        """
        시장가 매수: 가격*수량 만큼 현금 차감, cost basis 누적
        """
        if not self.can_buy(ticker, qty, price):
            return 0
        cost = qty * price
        self.cash -= cost
        self.positions[ticker] = self.positions.get(ticker, 0) + qty
        self.cost_basis[ticker] = self.cost_basis.get(ticker, 0.0) + cost # 해당 ticker가 없으면 0 반환
        return qty

    def sell(self, ticker: str, qty: int, price: float) -> int:
        """
        시장가 매도: 평균 단가 기준 cost basis 차감 후 현금 증가
        """
        if not self.can_sell(ticker, qty):
            return 0
        # 평균 취득 단가
        total_qty = self.positions[ticker]
        avg_cost = self.cost_basis[ticker] / total_qty
        # cost basis 차감
        self.cost_basis[ticker] -= avg_cost * qty

        proceeds = qty * price
        self.cash += proceeds
        self.positions[ticker] -= qty
        if self.positions[ticker] == 0:
            del self.positions[ticker]
            del self.cost_basis[ticker]
        return qty

    def get_unrealized_pnl(self, price_map: Dict[str, float]) -> float:
        """
        각 종목별 (현재가 - 평균 취득단가)*보유수량 합산
        """
        total_pnl = 0.0
        for ticker, qty in self.positions.items():
            current_price = price_map.get(ticker)
            if current_price is None:
                continue
            avg_cost = self.cost_basis[ticker] / qty
            total_pnl += (current_price - avg_cost) * qty
        return total_pnl

    def get_portfolio_value(self, price_map: Dict[str, float]) -> float:
        """현금 + 보유 종목의 시가 총액"""
        total = self.cash
        for ticker, qty in self.positions.items():
            price = price_map.get(ticker)
            if price is not None:
                total += qty * price
        return total
    
    def get_cash(self) -> float:
        """현재 현금 잔고 반환."""
        return self.cash

    def get_position(self, ticker: str) -> int:
        """특정 종목의 보유 수량 반환."""
        return self.positions.get(ticker, 0)