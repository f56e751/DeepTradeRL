# evaluate_single_env.py

import os
import sys
import argparse
import random

import numpy as np
import torch
from stable_baselines3 import PPO

# 프로젝트 루트의 src 폴더를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.infrastructure.pytorch_util import init_gpu
from src.data_handler.csv_processor import merge_lob_and_ohlcv, merge_lob_and_ohlcv_extended, DataSplitter
from src.data_handler.data_handler import Sc203OHLCVHandler, Sc203OHLCVTechHandler
from src.trading_env.minutely_orderbook_ohlcv_env import MinutelyOrderbookOHLCVEnv
from src.trading_env.observation import InputType
from src.agent.wrapper import LSTMObsWrapper, load_pretrained_lstm
from src.agent.util import evaluate_model  # 평가 함수가 정의된 위치로 변경

def make_env(lob_csv, ohlcv_csv, handler_cls, args, device):
    # 1) merge LOB+OHLCV
    df = merge_lob_and_ohlcv_extended(lob_csv, ohlcv_csv) if args.include_tech \
         else merge_lob_and_ohlcv(lob_csv, ohlcv_csv)
    # 2) normalize
    df = df.apply(lambda c: (c - c.min())/(c.max()-c.min()))
    # 3) split and take test portion
    splitter = DataSplitter(args.train_ratio, args.val_ratio, args.test_ratio)
    _, _, df_test = splitter.split(df)
    # 4) build env
    env = MinutelyOrderbookOHLCVEnv(
        df=df_test,
        handler_cls=handler_cls,
        initial_cash=args.initial_cash,
        lob_levels=args.lob_levels,
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType[args.input_type],
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )
    # 5) wrap LSTM if requested
    if InputType[args.input_type] == InputType.LSTM:
        lstm = load_pretrained_lstm()
        env = LSTMObsWrapper(env, lstm, args.window_size, device=device)
    return env

def main():
    parser = argparse.ArgumentParser("Evaluate trained PPO on a single environment")
    parser.add_argument("--model_path", required=True,
                        help="저장된 PPO 모델 경로 (예: runs/.../agent.zip)")
    parser.add_argument("--lob_csv", required=False, default="src/db/AAPL_minute_orderbook_2022_01-06_combined.csv",help="LOB CSV 파일 경로")
    parser.add_argument("--ohlcv_csv", required=False, default="src/db/AAPL_minute_ohlcv_2022_01-06_combined.csv", help="OHLCV CSV 파일 경로")
    parser.add_argument("--include_tech", action="store_true",
                        help="기술지표 포함 여부 (Sc203OHLCVTechHandler 사용)")
    parser.add_argument("--initial_cash", type=float, default=100000.0)
    parser.add_argument("--lob_levels", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=9)
    parser.add_argument("--window_size", type=int, default=250)
    parser.add_argument("--input_type", choices=["MLP","LSTM"], default="MLP")
    parser.add_argument("--transaction_fee", type=float, default=0.0023)
    parser.add_argument("--h_max", type=int, default=1)
    parser.add_argument("--hold_threshold", type=float, default=0.2)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.2)
    parser.add_argument("--test_ratio",  type=float, default=0.1)
    parser.add_argument("--episodes",    type=int,   default=1)
    parser.add_argument("--no_gpu",      action="store_true")
    parser.add_argument("--which_gpu",   type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    # 시드 고정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # GPU 초기화
    init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    # 핸들러 클래스 선택
    handler_cls = Sc203OHLCVTechHandler if args.include_tech else Sc203OHLCVHandler

    # 환경 생성
    env = make_env(
        args.lob_csv,
        args.ohlcv_csv,
        handler_cls,
        args,
        device
    )

    # 모델 로드 (env 인자는 dummy이지만, SB3 내부에서 observation_space/action_space만 사용)
    model = PPO.load(args.model_path, env=env)

    # 평가 실행
    results = evaluate_model(
        model,
        env,
        num_episodes=args.episodes,
        dataset_name="SingleEnv",
        initial_cash=args.initial_cash
    )

    # 요약 출력
    print("\n=== Evaluation Summary ===")
    print(f"Average reward: {results['avg_reward']:.4f}")
    print(f"Reward stddev : {results['std_reward']:.4f}")
    final_val = results['financial_metrics']['portfolio_values'][-1]
    print(f"Final portfolio value: {final_val:.2f}")

    env.close()

if __name__ == "__main__":
    main()
