from src.env.tick_stock_trading_env import TickStockTradingEnv
from src.data_handler.data_handler import Sc201Handler
import pandas as pd
import numpy as np


def test_tick_stock_trading_env_basic():
    # 파라미터 설정
    n_steps    = 50
    lob_levels = 10
    lookback   = 5

    # 1) 더미 LOB 데이터 생성 (랜덤 정수), 컬럼명 없이 위치 기반 iloc 사용
    data = np.random.randint(1, 100, size=(n_steps, lob_levels * 2))
    df = pd.DataFrame(data)

    # 2) 환경 생성 (Sc201Handler 사용)
    env = TickStockTradingEnv(
        df=df,
        handler_cls=Sc201Handler,
        initial_cash=1000.0,
        lob_levels=lob_levels,
        lookback=lookback,
        ticker="TEST"
    )

    # 3) reset 및 초기 관측치 확인
    obs = env.reset()
    # 기대 차원: 2*lob_levels + lookback*2*lob_levels + 1
    expected_dim = 2*lob_levels + lookback*(2*lob_levels) + 1
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (expected_dim,), f"obs.shape={obs.shape}, expected={(expected_dim,)}"

    # 4) 임의의 액션으로 몇 스텝 진행
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (expected_dim,)
        assert isinstance(reward, float)
        assert "cash" in info and "position" in info
        if done:
            break

if __name__ == "__main__":
    test_tick_stock_trading_env_basic()