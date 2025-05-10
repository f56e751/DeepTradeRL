from typing import Dict

class Inventory:
    """
    주식 인벤토리 관리 클래스
    - 다수 종목의 보유 수량과 현금 잔고를 관리
    - 매수/매도 실행, 유효성 검사, 포트폴리오 가치 계산 기능 제공
    """
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        """인벤토리를 초기화(잔고 재설정, 포지션 전량 청산)."""
        self.cash: float = self.initial_cash
        self.positions: Dict[str, int] = {}

    def can_buy(self, ticker: str, qty: int, price: float) -> bool:
        """qty 주를 price 가격에 매수할 수 있는지(현금 잔고) 검사."""
        return qty > 0 and (qty * price) <= self.cash

    def can_sell(self, ticker: str, qty: int) -> bool:
        """qty 주를 매도할 수 있는지(보유 수량) 검사."""
        return qty > 0 and self.positions.get(ticker, 0) >= qty

    def buy(self, ticker: str, qty: int, price: float) -> int:
        """
        최대 qty만큼 매수 실행.
        실행된 수량을 반환. (invalid인 경우 0 반환)
        """
        if not self.can_buy(ticker, qty, price):
            return 0
        cost = qty * price
        self.cash -= cost
        self.positions[ticker] = self.positions.get(ticker, 0) + qty
        return qty

    def sell(self, ticker: str, qty: int, price: float) -> int:
        """
        최대 qty만큼 매도 실행.
        실행된 수량을 반환. (invalid인 경우 0 반환)
        """
        if not self.can_sell(ticker, qty):
            return 0
        proceeds = qty * price
        self.cash += proceeds
        self.positions[ticker] -= qty
        if self.positions[ticker] == 0:
            del self.positions[ticker]
        return qty

    def get_cash(self) -> float:
        """현재 현금 잔고 반환."""
        return self.cash

    def get_position(self, ticker: str) -> int:
        """특정 종목의 보유 수량 반환."""
        return self.positions.get(ticker, 0)

    def get_positions(self) -> Dict[str, int]:
        """모든 종목의 보유 수량 딕셔너리 반환."""
        return dict(self.positions)

    def get_portfolio_value(self, price_map: Dict[str, float]) -> float:
        """
        현금 + (모든 종목 보유 수량 × 현재 가격) 의 합으로 포트폴리오 전체 가치를 계산.
        price_map: {ticker: current_price, ...}
        """
        total = self.cash
        for ticker, qty in self.positions.items():
            price = price_map.get(ticker)
            if price is not None:
                total += qty * price
        return total
