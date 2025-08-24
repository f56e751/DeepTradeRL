import pandas as pd
from pathlib import Path

from gymnasium import spaces

from ..trading_env import UnifiedTradingEnv, RealizedPnLReward, LogPortfolioReturnReward, CombinedReward, ActionStrategy, TestActionStrategy, ClippedActionStrategy, PercentPortfolioStrategy, StrictActionStrategy, NormalizationWrapper
from ..data_handler import FeatureEngineer

def test_unified_env_initialization():
    # 1) CSV 로드 (timestamp 컬럼이 있다면 parse_dates 지정)
    csv_path = Path("src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv")
    raw_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    
    # 테스트용으로  tic 열이 없으면 임의로 AAPL 추가하도록 함 
    if "tic" not in raw_df.columns:
        raw_df["tic"] = "AAPL"


    # 2) FeatureEngineer로 전처리
    fe = FeatureEngineer(
        use_technical_indicator=True,
        # tech_indicator_list=["rsi_14","macd","cci"],  # 예시
        use_turbulence=False,
        user_defined_feature=False
    )
    df = fe.preprocess_data(raw_df)

    # 3) 환경 생성
    env = UnifiedTradingEnv(
        df=df,
        reward_strategy=RealizedPnLReward,       # 또는 LogPortfolioReturnReward
        action_strategy=StrictActionStrategy,   # 기본 액션 스케일링 전략
        initial_cash=100000.0,
        transaction_fee=0.0023,
        lookback=9,
        lob_levels=0,
        h_max=1,
        hold_threshold=0.2,
        include_ohlcv=True,
        include_tech=False,
        include_pnl=True,
        include_spread=False,
        tech_dim=0
    )
    env = NormalizationWrapper(env)  # 필요시 정규화 래퍼 적용

    # 4) observation / action_space 타입 검사
    obs, info = env.reset()
    assert isinstance(obs, (list, tuple)) or hasattr(obs, "shape")
    assert isinstance(info, dict)
    assert isinstance(env.action_space, spaces.Box)
    assert isinstance(env.observation_space, spaces.Box)

    # 5) 여러 스텝 실행해 보기
    n_steps = 500
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        assert isinstance(reward, float)
        assert set(info.keys()) >= {"cash","position","invalid"}
        if done:
            break

    # 6) reset 후 다시 여러 스텝 실행
    obs2, info2 = env.reset()
    print(obs2)
    assert isinstance(obs2, (list, tuple)) or hasattr(obs2, "shape")
    assert isinstance(info2, dict)
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs2, reward2, terminated2, truncated2, info2 = env.step(action)
        done2 = terminated2 or truncated2
        print(obs2)
        
        price = env.get_price()
        portfolio_val = env.inventory.get_portfolio_value({'TICKER': price})
        print(f"price is {price}")
        print(f"portfolio value is {portfolio_val}")
        assert isinstance(reward2, float)
        assert set(info2.keys()) >= {"cash","position","invalid"}
        if done2:
            break

    print("✔ UnifiedTradingEnv 초기화, 여러 스텝 실행 및 reset 후 반복 실행 성공")



if __name__ == "__main__":
    test_unified_env_initialization()
    print("hi")