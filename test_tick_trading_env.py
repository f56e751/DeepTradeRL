from src.env.tick_stock_trading_env import TickStockTradingEnv
from src.data_handler.data_handler import Sc201Handler, Sc202Handler, Sc203Handler
from src.data_handler.lob_csv_processor import LOBCSVProcessor
import pandas as pd
import numpy as np
from io import StringIO

def load_df(csv_path: str, lob_levels: int = 10) -> pd.DataFrame:
    """
    주어진 CSV 파일에서 LOB 데이터를 로드하여
    TickStockTradingEnv에 사용할 형식의 DataFrame으로 가공 후 반환합니다.

    Parameters:
        csv_path: str
            LOB CSV 파일 경로
        lob_levels: int
            호가 레벨 수 (기본 10)

    Returns:
        pd.DataFrame
            [bid_px_00...ask_px_{L-1}, 0,1,...,2*L-1] 형태의 DataFrame
    """
    # LOB 가공기 생성
    processor = LOBCSVProcessor(lob_levels=lob_levels)
    # CSV 가공
    df_processed = processor.load_and_process(csv_path)
    return df_processed

def main():
    # 1) 파라미터 설정
    n_steps    = 5000
    lob_levels = 10
    lookback   = 5

    # 2) LOBCSVProcessor로 가공
    csv_path = "src/db/AAPL_orderbook_mbp-10_data_2025-05-13_1400.csv"
    df_processed = load_df(csv_path, lob_levels=lob_levels)

    # 3) 환경 생성 및 초기화
    env = TickStockTradingEnv(
        df=df_processed,
        handler_cls=Sc203Handler,
        initial_cash=1000.0,
        lob_levels=lob_levels,
        lookback=lookback,
        ticker="TEST",
        transaction_fee = 0.0023
    )
    obs = env.reset()

    # 4) 초기 obs 검증
    base_dim = 2*lob_levels + lookback*(2*lob_levels)
    
    # handler_cls에 따라 추가 피처 개수 결정
    if isinstance(env.handler, Sc203Handler):
        extra = 3
    elif isinstance(env.handler, Sc202Handler):
        extra = 2
    else:  # Sc201Handler
        extra = 1
    expected_dim = base_dim + extra

    assert isinstance(obs, np.ndarray), "obs가 numpy 배열이 아닙니다"
    assert obs.shape == (expected_dim,), f"obs.shape={obs.shape}, expected={(expected_dim,)}"
    print(f"[OK] 초기 obs shape: {obs.shape}")

    # 5) 랜덤 액션으로 몇 스텝 진행
    for step in range(1, n_steps + 1):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Attempt {step:2d} | Action={action} | Reward={reward:.2f} | Cash={info['cash']:.2f} | Pos={info['position']} | Invalid={info['invalid']} | Obs shape={obs.shape}")
        assert obs.shape == (expected_dim,)
        assert isinstance(reward, float)
        if done:
            print(f"[INFO] 에피소드 종료 (step {step})")
            break

if __name__ == "__main__":
    main()