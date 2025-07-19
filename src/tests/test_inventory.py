from ..env import Inventory, TransactionInfo, TradeType

# from src.env.inventory import Inventory
# from src.env.transaction_info import TransactionInfo, TradeType

def test_inventory_pnl_with_transaction_info():
    # 1) 인벤토리 초기화
    inv = Inventory(initial_cash=1000.0)
    assert inv.get_cash() == 1000.0
    assert inv.get_position("AAPL") == 0
    assert inv.get_unrealized_pnl({"AAPL": 10.0}) == 0.0

    # 2) 2주 매수 (주당 10.0)
    tx_buy = inv.buy("AAPL", qty=2, price=10.0)
    # 반환 타입 확인
    assert isinstance(tx_buy, TransactionInfo)
    assert tx_buy.trade_type == TradeType.BUY
    assert tx_buy.quantity == 2
    assert tx_buy.realized_pnl == 0.0
    # 현금: 1000 - 2*10 = 980
    assert inv.get_cash() == 980.0
    # 포지션: 2주
    assert inv.get_position("AAPL") == 2
    # 미실현 PnL (현재가 12.0 기준): (12 - 10)*2 = 4
    unpnl = inv.get_unrealized_pnl({"AAPL": 12.0})
    assert abs(unpnl - 4.0) < 1e-6

    # 3) 1주 매도 (주당 15.0)
    tx_sell = inv.sell("AAPL", qty=1, price=15.0)
    assert isinstance(tx_sell, TransactionInfo)
    assert tx_sell.trade_type == TradeType.SELL
    assert tx_sell.quantity == 1
    # realized PnL = (15 - 10)*1 = 5
    assert abs(tx_sell.realized_pnl - 5.0) < 1e-6
    # 포지션: 1주
    assert inv.get_position("AAPL") == 1
    # 현금: 980 + 1*15 = 995
    assert abs(inv.get_cash() - 995.0) < 1e-6
    # 미실현 PnL (현재가 12.0): (12 - 10)*1 = 2
    unpnl2 = inv.get_unrealized_pnl({"AAPL": 12.0})
    assert abs(unpnl2 - 2.0) < 1e-6

    # 4) 전량 청산
    tx_sell2 = inv.sell("AAPL", qty=1, price=11.0)
    assert isinstance(tx_sell2, TransactionInfo)
    assert tx_sell2.trade_type == TradeType.SELL
    assert tx_sell2.quantity == 1
    # realized PnL = (11 - 10)*1 = 1
    assert abs(tx_sell2.realized_pnl - 1.0) < 1e-6
    # 포지션: 0주
    assert inv.get_position("AAPL") == 0
    # 현금: 995 + 11 = 1006
    assert abs(inv.get_cash() - 1006.0) < 1e-6
    # 미실현 PnL: 포지션 없으므로 0
    assert inv.get_unrealized_pnl({"AAPL": 12.0}) == 0.0

    print("Inventory TransactionInfo tests passed!")

if __name__ == "__main__":
    test_inventory_pnl_with_transaction_info()
