import argparse
import time
import os
import random
import numpy as np
import psutil
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
import sys
import os
import pandas as pd
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.infrastructure.pytorch_util import init_gpu
from src.env.minutely_orderbook_ohlcv_env import MinutelyOrderbookOHLCVEnv
from src.data_handler.data_handler import Sc203Handler
from src.data_handler.data_handler import Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler
from src.env.observation import Observation, InputType
from src.data_handler.csv_processor import merge_lob_and_ohlcv, DataSplitter
from src.infrastructure.callback import TrainingStatusCallback
from src.agent.wrapper import LSTMObsWrapper

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
    # data frame 스케일 조정(normalize)
    df_all = df_all.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)
    
    # 틱 단위 거래 환경 생성
    env_original = MinutelyOrderbookOHLCVEnv(
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
    
    # Pretrained LSTM 모델 로드 (모델 전체)
    pretrained_lstm = torch.load(
        "src/deeplob/best_pretrained_deeplob",
        map_location=device,
        weights_only=False    # <-- 명시적으로 False
    )

    # Wrapper 적용
    wrapped_env = LSTMObsWrapper(env_original, 
                                pretrained_lstm, 
                                train_seq_len=100, 
                                device=device)

    train_agent(wrapped_env, save_directory, device, args)


def train_agent(env, save_directory, device, args):
    os.makedirs('runs/' + save_directory, exist_ok=True)

    with open(os.path.join('runs/' + save_directory, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    # Initialize model with custom logger
    model = SAC('MlpPolicy', verbose=0, env=env,
                gamma=args.gamma, 
                tau=args.tau,
                ent_coef=args.ent_coef,
                target_update_interval=args.target_update_interval,
                target_entropy=args.target_entropy,
                use_sde=args.use_sde,
                sde_sample_freq=args.sde_sample_freq,
                use_sde_at_warmup=args.use_sde_at_warmup,
                learning_rate=args.lr,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                device=device, 
                seed=args.seed,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[64, 64])),
                tensorboard_log=os.path.join('runs/' + save_directory, 'tensorboard'))

    # Set up logging with more detailed configuration
    log_path = os.path.join('runs/' + save_directory, '0')
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["csv", "tensorboard", "stdout"])
    model.set_logger(logger)
    
    # Add training status callback with save path
    status_callback = TrainingStatusCallback(
        verbose=1
    )
    
    print("\nStarting training...")
    print(f"Total timesteps: {args.iters}")
    print(f"Device: {device}")
    print(f"Model saved to: runs/{save_directory}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Learning starts: {args.learning_starts}")
    print(f"Batch size: {args.batch_size}")
    print(f"Train frequency: {args.train_freq}")
    print(f"Target entropy: {args.target_entropy}")
    print("\nTraining Status:")
    
    # Train with progress tracking
    model.learn(args.iters, reset_num_timesteps=True, callback=status_callback)
    
    print("\n\nTraining completed!")
    print(f"Total training time: {(time.time() - status_callback.training_start)/60:.1f} minutes")
    
    env.close()
    model.save(os.path.join(os.path.join('runs/' + save_directory), 'sac_agent'))


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
    
    # SAC specific training arguments
    parser.add_argument("--iters", "-i", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument("--tau", type=float, default=0.005,
                        help='Soft update coefficient for target networks')
    parser.add_argument("--ent_coef", type=str, default='auto',
                        help='Entropy coefficient (auto or float)')
    parser.add_argument("--target_update_interval", type=int, default=1,
                        help='Target network update interval')
    parser.add_argument("--target_entropy", type=str, default='auto',
                        help='Target entropy (auto or float)')
    parser.add_argument("--use_sde", action="store_true", default=False,
                        help='Use State Dependent Exploration')
    parser.add_argument("--sde_sample_freq", type=int, default=-1,
                        help='Sample frequency for SDE')
    parser.add_argument("--use_sde_at_warmup", action="store_true", default=False,
                        help='Use SDE at warmup')
    parser.add_argument("--buffer_size", type=int, default=1000000,
                        help='Replay buffer size')
    parser.add_argument("--learning_starts", type=int, default=100,
                        help='Learning starts after this many steps')
    parser.add_argument("--batch_size", type=int, default=256,
                        help='Batch size for training')
    parser.add_argument("--train_freq", type=int, default=1,
                        help='Training frequency')
    parser.add_argument("--gradient_steps", type=int, default=1,
                        help='Gradient steps per update')

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help='Tag to add to the save directory')

    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    main(args)