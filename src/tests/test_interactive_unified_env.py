import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the Python path to allow for absolute imports
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..trading_env import UnifiedTradingEnv, RealizedPnLReward, ScaledRealizedPnLReward, ClippedActionStrategy, PortfolioScalingWrapper, NormalizationWrapper
from ..data_handler import FeatureEngineer

def run_interactive_test():
    """
    An interactive test for UnifiedTradingEnv.
    Allows the user to manually input actions at each step.
    To run this script, execute `python src/tests/test_interactive_unified_env.py` from the project root directory.
    """
    # 1. Load data
    # Using a default CSV for this test.
    # csv_path = "src/db/AAPL_minute_ohlcv_orderbook_2019_01-07_combined.csv"
    csv_path = "src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv"  # Alternative path if running from project root
    if not os.path.exists(csv_path):
        print(f"Error: Cannot find data file at '{csv_path}'")
        print("Please ensure you are running the script from the project root directory (e.g., C:\Github\DeepTradeRL).")
        return

    print(f"Loading data from {csv_path}...")
    raw_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        # 테스트용으로  tic 열이 없으면 임의로 AAPL 추가하도록 함 
    if "tic" not in raw_df.columns:
        raw_df["tic"] = "AAPL"


    # 2) FeatureEngineer로 전처리
    fe = FeatureEngineer(
        use_technical_indicator=False,
        # tech_indicator_list=["rsi_14","macd","cci"],  # 예시
        use_turbulence=False,
        user_defined_feature=False
    )
    df = fe.preprocess_data(raw_df)
    # For simplicity, we'll use a small slice of the data for the test
    test_df = df.iloc[0:1000].reset_index(drop=True)
    print(f"Using {len(test_df)} steps for the test.")

    # 2. Initialize the environment
    try:
        env = UnifiedTradingEnv(
            df=test_df,
            reward_strategy=ScaledRealizedPnLReward,
            action_strategy=ClippedActionStrategy,
            initial_cash=100000.0,
            transaction_fee=0.001,
            lookback=9,
            lob_levels=0,
            h_max=250,
            hold_threshold=0.2,
            include_ohlcv=True,
            include_tech=False,
            include_pnl=True,
            include_spread=False,
            include_position = True,
            include_orderbook = False,
            include_portfolio_value = True,
            include_cash= True,
            tech_dim=0
        )
        # env = NormalizationWrapper(env)  # 필요시 정규화 래퍼 적용
        env = PortfolioScalingWrapper(env)  # 필요시 포트폴리오 가치 스케일링 래퍼 적용
    except Exception as e:
        print(f"Error initializing environment: {e}")
        print("This might be due to changes in the environment's __init__ method.")
        return

    # 3. Run the interactive loop
    obs, info = env.reset()
    done = False
    step_count = 0

    while not done:
        # Print current status
        print("\n" + "="*50)
        print(f"STEP: {step_count}")
        print("Current Portfolio:")
        env.render() # Use the env's render method for a quick status

        # Get user input
        action_input = input(f"Enter action for step {step_count} (-1.0 to 1.0, or 'q' to quit): ")

        if action_input.lower() == 'q':
            print("Exiting interactive test.")
            break

        try:
            # Convert input to a numpy array for the action space
            action = np.array([float(action_input)], dtype=np.float32)
            if not env.action_space.contains(action):
                print(f"Warning: Action {action[0]} is outside the valid range [-1, 1]. It will be clipped by the environment.")

        except ValueError:
            print("Invalid input. Please enter a number between -1.0 and 1.0.")
            continue

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Print step results
        print(f"\n--- Step {step_count} Result ---")
        print(f"obs: {obs}  ")
        print(f"Reward: {reward:.6f}")
        print(f"Info: {info}")
        
        step_count += 1

        if done:
            print("\n" + "="*50)
            print("✅ Episode finished!")
            print("Final Portfolio Status:")
            env.render()
            final_value = env.inventory.get_portfolio_value({'TICKER': env.get_price()})
            print(f"Final Portfolio Value: ${final_value:,.2f}")


if __name__ == "__main__":
    run_interactive_test()
