import numpy as np

from ..data_handler import CSVFolderLoader, merge_lob_and_ohlcv, merge_lob_and_ohlcv_extended, DataSplitter, Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler, Sc203OHLCVTechHandler
from ..env import MinutelyOrderbookOHLCVEnv, InputType


def test_merge_and_env(
    lob_csv_path: str,
    ohlcv_csv_path: str
):      
    
    df_all = merge_lob_and_ohlcv(lob_csv_path, ohlcv_csv_path)
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)
    # 환경 인스턴스화
    # TODO MinutelyOrderbookOHLCVEnv 입력으로 df만 들어오게 하기
    env =     MinutelyOrderbookOHLCVEnv(
        df=df_train,
        handler_cls=Sc203OHLCVHandler,
        initial_cash = 10000000000.0,
        lob_levels = 10,
        lookback = 9,
        window_size = 100,
        input_type = InputType.LSTM,
        transaction_fee = 0.0023,
        h_max = 100,
        hold_threshold = 0.2,
    )

    obs = env.reset()

    # lstm 출력 형식의 경우, obs를 dict로 반환함
    # snapshots가  (ask price, volume) -> 20쌍
    # 순서는?
    if isinstance(obs, dict):
        snaps, others = obs['lstm_snapshots'], obs['mlp_input']
        print(f"Initial lstm_snapshots shape: {snaps.shape}, mlp_input shape: {others.shape}")
    else:
        print(f"Initial obs shape: {obs.shape}")

    n_steps = 20
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i:02d} | action: {action[0]:+.2f} | reward: {reward:.2f} | invalid: {info['invalid']} | position: {info['position']} | cash: {info['cash']:.2f}")
        if done:
            print(f"Reached end at step {i}")
            break

def test_merge_and_env_tech(
    lob_csv_path: str,
    ohlcv_extended_csv_path: str
):      
    
    df_all = merge_lob_and_ohlcv_extended(lob_csv_path, ohlcv_extended_csv_path)
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)
    # 환경 인스턴스화
    # TODO MinutelyOrderbookOHLCVEnv 입력으로 df만 들어오게 하기
    env =     MinutelyOrderbookOHLCVEnv(
        df=df_train,
        handler_cls=Sc203OHLCVTechHandler,
        initial_cash = 10000000000.0,
        lob_levels = 10,
        lookback = 9,
        window_size = 9,
        input_type = InputType.LSTM,
        transaction_fee = 0.0023,
        h_max = 100,
        hold_threshold = 0.2,
    )

    obs = env.reset()

    # lstm 출력 형식의 경우, obs를 dict로 반환함
    # snapshots가  (ask price, volume) -> 20쌍
    # 순서는?
    if isinstance(obs, dict):
        snaps, others = obs['lstm_snapshots'], obs['mlp_input']
        print(f"Initial lstm_snapshots shape: {snaps.shape}, mlp_input shape: {others.shape}")
    else:
        print(f"Initial obs shape: {obs.shape}")

    n_steps = 20
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i:02d} | action: {action[0]:+.2f} | reward: {reward:.2f} | invalid: {info['invalid']} | position: {info['position']} | cash: {info['cash']:.2f}")
        if done:
            print(f"Reached end at step {i}")
            break

if __name__ == "__main__":
    # 테스트용 폴더 경로 지정
    lob_csv_path = "src/db/AAPL_minute_orderbook_2019_01-07_combined.csv"
    ohlcv_csv_path = "src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv"
    ohlcv_extended_csv_path = "src/db/indicator/AAPL_with_indicators_v2.csv"

    # Sc201, Sc202, Sc203 핸들러 중 하나 선택
    test_merge_and_env(
        lob_csv_path,
        ohlcv_csv_path
    )

    # test_merge_and_env_tech(
    #     lob_csv_path,
    #     ohlcv_extended_csv_path 
    # )




