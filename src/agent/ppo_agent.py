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
import os
import pandas as pd
# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.infrastructure.pytorch_util import init_gpu
from src.env.minutely_orderbook_ohlcv_env import MinutelyOrderbookOHLCVEnv
from src.data_handler.data_handler import Sc203Handler
from src.data_handler.data_handler import Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler
from src.env.observation import Observation, InputType
from src.data_handler.csv_processor import merge_lob_and_ohlcv, DataSplitter

# from eval_environment import eval_agent

def main(args):
    if args.seed is None:
        args.seed = int(random.random() * 10000)

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize GPU
    init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    save_directory = str(time.time()) + (args.tag if args.tag is not None else '')
    
    df_all = merge_lob_and_ohlcv(args.lob_csv_path, args.ohlcv_csv_path)
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)
    
    # 틱 단위 거래 환경 생성
    env = MinutelyOrderbookOHLCVEnv(
        df=df_train,
        handler_cls=Sc203OHLCVHandler,
        initial_cash=args.initial_cash, # Starting cash
        lob_levels=args.lob_levels,                     # Max shares to trade
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType.MLP,
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )

    train_agent(env, save_directory, device, args)


def train_agent(env, save_directory, device, args):
    os.makedirs('runs/' + save_directory, exist_ok=True)

    with open(os.path.join('runs/' + save_directory, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    # Define policy network architecture with proper initialization
    """
    policy_kwargs = dict(
        net_arch=dict(
            pi=args.net_arch,  # Policy network
            vf=args.net_arch   # Value network
        ),
        activation_fn=torch.nn.ReLU,
        normalize_images=True
    )
    """

    model = PPO('MlpPolicy', verbose=1, env=env,
                gamma=args.gamma, ent_coef=args.ent_coef, max_grad_norm=args.grad_clip,
                learning_rate=args.lr, # policy_kwargs=policy_kwargs,
                device=device, batch_size=128, seed=args.seed,
                tensorboard_log=os.path.join('runs/' + save_directory, 'tensorboard'))

    logger = configure(os.path.join('runs/' + save_directory, '0'), ["csv", "tensorboard"])
    model.set_logger(logger)
    model.learn(args.iters, reset_num_timesteps=True)

    env.close()
    model.save(os.path.join(os.path.join('runs/' + save_directory), 'agent'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and Environment arguments
    parser.add_argument("--lob_csv_path", type=str, default="src/db/AAPL_minute_orderbook_2019_01-07_combined.csv",
                        required=False, help='LOB CSV path')
    parser.add_argument("--ohlcv_csv_path", type=str, default="src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv",
                        required=False, help='OHLCV CSV path')
    parser.add_argument("--initial_cash", type=float, default=100000.0,
                        required=False, help='Starting cash')
    parser.add_argument("--lob_levels", type=int, default=10,
                        required=False, help='Max shares to trade')
    parser.add_argument("--lookback", type=int, default=9,
                        required=False, help='Lookback')
    parser.add_argument("--h_max", type=int, default=1,
                        required=False, help='Max action')
    parser.add_argument("--hold_threshold", type=float, default=0.2,
                        required=False, help='Hold threshold')
    parser.add_argument("--window_size", type=int, default=9,
                        required=False, help='Window size')
    parser.add_argument("--transaction_fee", type=float, default=0.0023,
                        required=False, help='Transaction fee')

    # GPU related arguments
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    
    # Training related arguments
    parser.add_argument("--iters", "-i", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=.00032)
    parser.add_argument("--grad_clip", type=float, default=.5)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--ent_coef", type=float, default=.0089)
    parser.add_argument("--net_arch", type=int, default=[64, 64])

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help='Tag to add to the save directory')

    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    main(args)