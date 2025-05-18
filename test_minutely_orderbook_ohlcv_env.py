import numpy as np
from src.data_handler.csv_folder_loader import CSVFolderLoader
from src.data_handler.csv_processor import merge_lob_and_ohlcv, DataSplitter
from src.data_handler.data_handler import Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler
from src.env.minutely_orderbook_ohlcv_env import MinutelyOrderbookOHLCVEnv
from src.env.observation import InputType

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
        handler_cls=Sc201OHLCVHandler,
        initial_cash = 10000000000.0,
        lob_levels = 10,
        lookback = 9,
        window_size = 9,
        input_type = InputType.MLP,
        transaction_fee = 0.0023,
        h_max = 100,
        hold_threshold = 0.2,
    )

    obs = env.reset()
    obs_dim = env.get_obs_dim()
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
    # Sc201, Sc202, Sc203 핸들러 중 하나 선택
    test_merge_and_env(
        lob_csv_path,
        ohlcv_csv_path
    )




