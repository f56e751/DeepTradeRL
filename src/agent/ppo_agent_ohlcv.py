import argparse
import time
import os
import random

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

import sys

from ..infrastructure import init_gpu, TrainingStatusCallback
from ..data_handler import load_ohlcv_df, DataSplitter, DFProcessMode, OHLCVPositionHandler, OHLCVPositionPnlHandler
from ..env import MinutelyOHLCVEnv

# # Add project root to path so we can import src modules
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from src.infrastructure.pytorch_util import init_gpu
# from src.infrastructure.callback import TrainingStatusCallback
# from src.data_handler.csv_processor import load_ohlcv_df, DataSplitter, DFProcessMode
# from src.data_handler.data_handler import OHLCVPositionHandler, OHLCVPositionPnlHandler
# from src.env.minutely_ohlcv_env import MinutelyOHLCVEnv


def main(args):
    # 1) Seed 설정
    if args.seed is None:
        args.seed = int(random.random() * 1e4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 2) GPU 초기화
    init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    # 3) OHLCV 데이터 로드 및 전처리
    df_all = load_ohlcv_df(
        args.ohlcv_csv_path,
        process_mode=DFProcessMode.FillWithLast
    )
    # 정규화
    df_all = df_all.apply(lambda col: (col - col.min()) / (col.max() - col.min()))

    # 4) Train / Val / Test 분할
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)

    # 5) Handler 클래스 선택
    handler_cls = OHLCVPositionPnlHandler if args.include_pnl else OHLCVPositionHandler

    # 6) 환경 생성 (MLP 전용)
    env = MinutelyOHLCVEnv(
        df=df_train,
        handler_cls=handler_cls,
        initial_cash=args.initial_cash,
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold,
        window_size=args.window_size
    )

    # 7) 모델 학습
    save_dir = f"runs/{time.time():.0f}_{args.tag or ''}"
    os.makedirs(save_dir, exist_ok=True)
    # 파라미터 기록
    with open(os.path.join(save_dir, "parameters.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    model = PPO(
        "MlpPolicy", env,
        verbose=0,
        device=device,
        seed=args.seed,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        max_grad_norm=args.grad_clip,
        batch_size=128,
        policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64])),
        tensorboard_log=os.path.join(save_dir, "tensorboard")
    )
    # 로거 설정
    logger = configure(save_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    callback = TrainingStatusCallback(verbose=1)
    print(f"Training for {args.iters} timesteps on {device}...")
    model.learn(total_timesteps=args.iters, callback=callback)

    # 8) 저장
    model.save(os.path.join(save_dir, "agent"))
    env.close()
    print("Training complete, model saved to:", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv_csv_path", type=str,
                        default="src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv")
    parser.add_argument("--include_pnl", action="store_true",
                        help="포지션 + 미실현 PnL 핸들러 사용")
    parser.add_argument("--initial_cash", type=float, default=100000.0)
    parser.add_argument("--transaction_fee", type=float, default=0.0023)
    parser.add_argument("--h_max", type=int, default=250)
    parser.add_argument("--hold_threshold", type=float, default=0.2)
    parser.add_argument("--window_size", type=int, default=1,
                        help="Observation window size (MLP 전용이므로 보통 1)")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.002)
    parser.add_argument("--grad_clip", type=float, default=0.7)
    parser.add_argument("--iters", "-i", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)

    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    main(args)
