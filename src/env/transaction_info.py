from enum import Enum, auto

class TradeType(Enum):
    BUY = auto()
    SELL = auto()

class TransactionInfo:
    def __init__(self):
        self.trade_type: TradeType = None
        self.quantity: int = None
        self.realized_pnl: float = None
        self.price = None
        self.is_valid = None
    
    def set_new_val(self, trade_type: TradeType, quantity: int, realized_pnl: float, price: float = None, is_valid: bool = False):
        self.trade_type = trade_type
        self.quantity = quantity
        self.realized_pnl = realized_pnl
        self.price = price
        self.is_valid = is_valid