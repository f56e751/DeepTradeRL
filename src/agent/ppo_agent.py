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
from src.env.tick_stock_trading_env import TickStockTradingEnv
from src.data_handler.data_handler import Sc203Handler
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
    data_path = args.data_path
    df = pd.read_csv(data_path)
    
    # 틱 단위 거래 환경 생성
    env = TickStockTradingEnv(
        df=df,
        handler_cls=Sc203Handler,
        initial_cash=args.initial_cash, # Starting cash
        lob_levels=args.lob_levels,                     # Max shares to trade
        lookback=args.lookback,
        ticker=args.ticker,
        transaction_fee=args.transaction_fee
    )

    train_agent(env, save_directory, device)


def train_agent(env, save_directory, device):
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
    parser.add_argument("--data_path", type=str, default='./src/db/AAPL_minute_orderbook_2019_01-07_combined.csv',
                        required=False, help='Path to the training data') # 삼성전자 2014~2017년 데이터(업로드 된 것 사용)
    parser.add_argument("--initial_cash", type=float, default=100000.0,
                        required=False, help='Starting cash')
    parser.add_argument("--lob_levels", type=int, default=10,
                        required=False, help='Max shares to trade')
    parser.add_argument("--lookback", type=int, default=9,
                        required=False, help='Lookback')
    parser.add_argument("--ticker", type=str, default="TICKER",
                        required=False, help='Ticker')
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