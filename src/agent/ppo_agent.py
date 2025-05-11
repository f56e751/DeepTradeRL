import argparse
import time
import os
import random
import numpy as np

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from src.infrastructure import pytorch_util as ptu


# from data_manager import *
from src.env.stock_trading_env import StockTradingEnv
# from eval_environment import eval_agent

def main(args):
    if args.seed is None:
        args.seed = int(random.random() * 10000)

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize GPU
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    save_directory = str(time.time()) + (args.tag if args.tag is not None else '')

    env = StockTradingEnv(
        df=args.df,
        initial_balance=args.initial_balance, # Starting cash
        h_max=args.h_max,                     # Max shares to trade
        transaction_fee=args.transaction_fee, # 0.1% fee
        wanted_features=args.wanted_features  # Optional: customize features
    )

    train_agent(env, save_directory, device)


def train_agent(env, save_directory, device):
    os.makedirs('runs/' + save_directory, exist_ok=True)

    with open(os.path.join('runs/' + save_directory, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    model = PPO('MultiInputPolicy', verbose=1, env=env,
                gamma=args.gamma, ent_coef=args.ent_coef, max_grad_norm=args.grad_clip,
                learning_rate=args.lr, policy_kwargs=dict(net_arch=args.net_arch), seed=args.seed,
                device=device, batch_size=128)

    logger = configure(os.path.join('runs/' + save_directory, '0'), ["csv", "tensorboard"])
    model.set_logger(logger)
    # model.learn(args.iters, reset_num_timesteps=False) # 여러개의 환경을 적용할 때 사용
    model.learn(args.iters, reset_num_timesteps=True)

    env.close()
    model.save(os.path.join(os.path.join('runs/' + save_directory), 'agent'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and Environment arguments
    parser.add_argument("--df", type=str, default='db/samsung_data_2014_2017.csv',
                        required=False, help='Path to the training data') # 삼성전자 2014~2017년 데이터(업로드 된 것 사용)
    parser.add_argument("--initial_balance", type=float, default=1e6,
                        required=False, help='Starting cash')
    parser.add_argument("--h_max", type=int, default=10,
                        required=False, help='Max shares to trade')
    parser.add_argument("--transaction_fee", type=float, default=0.001,
                        required=False, help='Transaction fee')
    parser.add_argument("--wanted_features", type=list, default=['Adj Close', 'MACD', 'RSI', 'CCI', 'ADX'],
                        required=False, help='Features to use')

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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help='Tag to add to the save directory')

    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    main(args)