# evaluate_pretrained.py

import os
import sys
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from ..data_handler import load_ohlcv_df, DataSplitter, DFProcessMode, OHLCVPositionHandler, OHLCVPositionPnlHandler
from ..env import MinutelyOHLCVEnv
from ..agent import evaluate_model

# # 프로젝트의 src 폴더를 경로에 추가
# # sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# # 필요한 모듈 임포트
# from src.data_handler.csv_processor import load_ohlcv_df, DataSplitter, DFProcessMode
# from src.data_handler.data_handler import OHLCVPositionHandler, OHLCVPositionPnlHandler
# from src.env.minutely_ohlcv_env import MinutelyOHLCVEnv
# from src.agent.util import evaluate_model  # evaluate_model 정의된 모듈 경로로 변경

def main():
    parser = argparse.ArgumentParser(description="Load trained PPO model and evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                        help="저장된 PPO 모델 경로 (예: runs/1623456789_agent.zip)")
    parser.add_argument("--ohlcv_csv", type=str, default="src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv",
                        help="평가할 OHLCV CSV 경로")
    parser.add_argument("--include_pnl", action="store_true",
                        help="PnL 포함 핸들러 사용 여부")
    parser.add_argument("--episodes", type=int, default=1,
                        help="평가할 에피소드 수" )
    parser.add_argument("--initial_cash", type=float, default=100000.0)
    parser.add_argument("--transaction_fee", type=float, default=0.0023)
    parser.add_argument("--h_max", type=int, default=250)
    parser.add_argument("--hold_threshold", type=float, default=0.2)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 랜덤 시드 고정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # OHLCV 데이터 로드 & 분할
    df = load_ohlcv_df(args.ohlcv_csv, process_mode=DFProcessMode.FillWithLast)
    df = df.apply(lambda col: (col - col.min()) / (col.max() - col.min()))
    splitter = DataSplitter(0.7, 0.2, 0.1)
    df_train, df_val, df_test = splitter.split(df)

    # 핸들러 선택
    handler_cls = OHLCVPositionPnlHandler if args.include_pnl else OHLCVPositionHandler

    # 환경 생성
    env = MinutelyOHLCVEnv(
        df=df_test,
        handler_cls=handler_cls,
        initial_cash=args.initial_cash,
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold,
        window_size=args.window_size
    )

    # 모델 로드
    model = PPO.load(args.model_path, env=env)

    # 평가 실행
    results = evaluate_model(
        model,
        env,
        num_episodes=args.episodes,
        dataset_name="Test",
        initial_cash=args.initial_cash
    )

    print("\n=== 평가 결과 요약 ===")
    print(f"평균 에피소드 보상: {results['avg_reward']:.4f}")
    print(f"보상 표준편차: {results['std_reward']:.4f}")
    print(f"포트폴리오 최종 가치: {results['financial_metrics']['portfolio_values'][-1]:.2f}")

if __name__ == "__main__":
    main()
