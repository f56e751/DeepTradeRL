# test_stock_env.py

import pandas as pd
from src.data_handler.csv_folder_loader import CSVFolderLoader
from src.data_handler.data_handler import MultiDataHandler
from src.env.daily_stock_trading_env import DailyStockTradingEnv


def test_data_handler(folder_path: str, obs_step: int = 0):
    """
    CSV 폴더에 있는 모든 csv 파일을 합쳐서 하나의 pd 데이터 프레임으로 제작함
    만약 다른 csv 파일 조합을 시도하고 싶으면 다른 폴더를 만들어 원하는 csv 파일을 두면 됨
    MultiDataHandler의 기능(길이, 티커 수, 지정 스텝 관측) 확인을 위한 테스트 함수
    """
    print("--- Testing DataHandler ---")
    # 1) CSVFolderLoader로 지정 폴더 내 모든 CSV 읽어 합치기
    csv_loader = CSVFolderLoader(folder_path=folder_path)
    combined_df = csv_loader.load()
    print(f"Loaded CSVs, total rows: {len(combined_df)}")

    # 2) MultiDataHandler 초기화
    mdh = MultiDataHandler(df=combined_df)
    #    - get_length(): 모든 종목 중 가장 짧은 시계열 길이 반환
    length = mdh.get_length()
    print(f"DataHandler length (minimum series length): {length}")

    #    - get_num_tickers(): 고유 티커 개수 반환
    ticker_num = mdh.get_num_tickers()
    print(f"DataHandler number of tickers: {ticker_num}")

    # 3) 지정된 스텝(obs_step)의 관측 벡터 확인
    obs = mdh.get_observation(step=obs_step)
    print(f"[step {obs_step}] observation vector: {obs}")

    return combined_df


def test_env(df: pd.DataFrame, n_steps: int = 10):
    """
    StockTradingEnv의 동작 검증을 위한 테스트 함수
    - 랜덤 액션을 n_steps만큼 실행하며 환경 반응 관찰
    """
    print("\n--- Testing StockTradingEnv ---")
    # 환경 생성 (기본 h_max=10, transaction_fee=0.001)
    env = DailyStockTradingEnv(df=df, wanted_features = ['Adj Close', 'MACD', 'RSI', 'CCI', 'ADX'])

    # 환경 초기화 및 첫 관측값 확인
    obs = env.reset()
    # obs는 [cash, positions..., data_features...] 순서의 벡터
    print(f"env obs dim: {env.get_obs_dim()}")
    print(f"Initial environment observation (shape={env.get_obs_dim()}): {obs}")

    for i in range(n_steps):
        # action_space.sample()로 랜덤 액션 생성
        # - action 값은 실수 [-1,1] 범위
        # - 범위를 벗어나면 내부적으로 -1 또는 +1로 clipping
        action = env.action_space.sample()

        # step() 호출
        # - invalid action: 주문 수량 > 보유량(매도), 또는 비용 > 현금(매수)인 경우
        #    → step이 증가하지 않고 reward=0.0, info={'invalid': True} 반환
        # - 정상 action: h_max 비율만큼 주문 실행, 거래 수수료(transaction_fee) 반영
        obs, reward, done, info = env.step(action)
        print(f"Attempt {i:02d} | action: {action} | reward: {reward:.2f} | done: {done} | info: {info}")

        if done:
            print(f"Reached end of data at step {i}")
            break


if __name__ == "__main__":
    # 1) DataHandler 기능 테스트
    folder_path = "src/db/day"
    combined_df = test_data_handler(folder_path=folder_path, obs_step=26)

    # 2) Env 동작 테스트
    test_env(df=combined_df, n_steps=20)
