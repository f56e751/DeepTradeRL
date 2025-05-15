from enum import Enum, auto

class TradeType(Enum):
    BUY = auto()
    SELL = auto()

class TransactionInfo:
    def __init__(self):
        self.trade_type: TradeType = None
        self.quantity: int = None
        self.realized_pnl: float = None
    
    def set_new_val(self, trade_type: TradeType, quantity: int, realized_pnl: float):
        self.trade_type = trade_type
        self.quantity = quantity
        self.realized_pnl = realized_pnl