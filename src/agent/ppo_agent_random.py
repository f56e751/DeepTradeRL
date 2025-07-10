import argparse
import time
import os
import random

import numpy as np
import torch
import yaml
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

import gym
from gym import spaces

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.infrastructure.pytorch_util import init_gpu
from src.infrastructure.callback import TrainingStatusCallback
from src.data_handler.csv_processor import merge_lob_and_ohlcv, merge_lob_and_ohlcv_extended, DataSplitter
from src.data_handler.data_handler import (
    Sc201OHLCVHandler,
    Sc202OHLCVHandler,
    Sc203OHLCVHandler,
    Sc203OHLCVTechHandler
)
from src.env.minutely_orderbook_ohlcv_env import MinutelyOrderbookOHLCVEnv
from src.agent.wrapper import LSTMObsWrapper, load_pretrained_lstm
from src.env.observation import InputType

# 매 에피소드 시작마다 랜덤으로 상승장이나 하락장으로 환경을 초기화
class MixedMinutelyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df_bull, df_bear, handler_cls, **env_kwargs):
        self.bull_env = MinutelyOrderbookOHLCVEnv(df=df_bull, handler_cls=handler_cls, **env_kwargs)
        self.bear_env = MinutelyOrderbookOHLCVEnv(df=df_bear, handler_cls=handler_cls, **env_kwargs)
        self.active = None

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.bull_env.seed(seed)
        self.bear_env.seed(seed)
        return [seed]

    @property
    def action_space(self):
        return self.bull_env.action_space

    @property
    def observation_space(self):
        return self.bull_env.observation_space

    def reset(self):
        self.active = random.choice([self.bull_env, self.bear_env])
        return self.active.reset()

    def step(self, action):
        return self.active.step(action)

    def render(self, mode='human'):
        return self.active.render(mode)

    def close(self):
        self.bull_env.close()
        self.bear_env.close()



def main(args):
    # 1) Seed
    if args.seed is None:
        args.seed = int(random.random() * 1e4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 2) GPU
    init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    # 3) 데이터 로드
    if args.include_tech:
        merge_fn = merge_lob_and_ohlcv_extended
    else:
        merge_fn = merge_lob_and_ohlcv

    # Bull
    df_bull_all = merge_fn(args.bull_lob_csv_path, args.bull_ohlcv_csv_path)
    df_bull_all = df_bull_all.apply(lambda col: (col - col.min())/(col.max()-col.min()))
    splitter = DataSplitter(0.7, 0.2, 0.1)
    df_bull_train, _, _ = splitter.split(df_bull_all)

    # Bear
    df_bear_all = merge_fn(args.bear_lob_csv_path, args.bear_ohlcv_csv_path)
    df_bear_all = df_bear_all.apply(lambda col: (col - col.min())/(col.max()-col.min()))
    df_bear_train, _, _ = splitter.split(df_bear_all)

    # 4) handler 선택
    handler_cls = Sc203OHLCVTechHandler if args.include_tech else Sc203OHLCVHandler

    # 5) 혼합 환경 생성
    mixed_env = MixedMinutelyEnv(
        df_bull=df_bull_train,
        df_bear=df_bear_train,
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

    # LSTM wrapping
    if InputType[args.input_type] == InputType.LSTM:
        pretrained = load_pretrained_lstm()
        env = LSTMObsWrapper(mixed_env, pretrained, args.window_size, device=device)
    else:
        env = mixed_env

    # 6) 학습
    save_dir = f"runs/{int(time.time())}_{args.tag or ''}"
    os.makedirs(save_dir, exist_ok=True)
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
    logger = configure(save_dir, ["stdout","csv","tensorboard"])
    model.set_logger(logger)

    callback = TrainingStatusCallback(verbose=1)
    print(f"Training for {args.iters} timesteps on {device} ...")
    model.learn(total_timesteps=args.iters, callback=callback)

    # 7) 저장 및 종료
    model.save(os.path.join(save_dir, "agent"))
    env.close()
    print("Done. Saved to", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Bull market CSVs
    parser.add_argument("--bull_lob_csv_path", type=str, default="src/db/AAPL_minute_orderbook_2019_01-07_combined.csv",
                        required=False, help="Bull market LOB CSV")
    parser.add_argument("--bull_ohlcv_csv_path", type=str, default="src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv",
                        required=False, help="Bull market OHLCV CSV")
    # Bear market CSVs
    parser.add_argument("--bear_lob_csv_path", type=str, default="src/db/AAPL_minute_orderbook_2022_01-06_combined.csv",
                        required=False, help="Bear market LOB CSV")
    parser.add_argument("--bear_ohlcv_csv_path", type=str, default="src/db/AAPL_minute_ohlcv_2022_01-06_combined.csv",
                        required=False, help="Bear market OHLCV CSV")

    parser.add_argument("--include_tech", action="store_true",
                        help="Include technical indicators")
    parser.add_argument("--initial_cash", type=float, default=100000.0)
    parser.add_argument("--lob_levels", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=9)
    parser.add_argument("--h_max", type=int, default=250)
    parser.add_argument("--hold_threshold", type=float, default=0.2)
    parser.add_argument("--window_size", type=int, default=9)
    parser.add_argument("--transaction_fee", type=float, default=0.0023)
    parser.add_argument("--input_type", choices=["MLP","LSTM"], default="MLP")

    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)

    parser.add_argument("--iters", "-i", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.002)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None)

    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    main(args)
