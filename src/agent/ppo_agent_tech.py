import argparse
import time
import os
import random
import numpy as np
import psutil
import torch
import yaml
from stable_baselines3 import PPO
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
from src.data_handler.data_handler import Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler, Sc203OHLCVTechHandler
from src.env.observation import Observation, InputType
from src.data_handler.csv_processor import merge_lob_and_ohlcv, merge_lob_and_ohlcv_extended, DataSplitter
from src.infrastructure.callback import TrainingStatusCallback
from src.agent.wrapper import LSTMObsWrapper, load_pretrained_lstm
from src.deeplob.model import deeplob

from src.data_handler.data_handler import ResampledSc201OHLCVHandler, ResampledSc202OHLCVHandler, ResampledSc203OHLCVHandler
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
    
    if args.include_tech == False:
        df_all = merge_lob_and_ohlcv(args.lob_csv_path, args.ohlcv_csv_path)
    else:
        df_all = merge_lob_and_ohlcv_extended(args.lob_csv_path, args.ohlcv_extended_csv_path)
        
    # data frame 스케일 조정(normalize)
    df_all = df_all.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)
    
    # handler_cls 선택
    # handler = Sc203OHLCVTechHandler if args.include_tech else Sc203OHLCVHandler
    if args.resample_rule:
        if args.include_tech:
            handler = ResampledSc203OHLCVHandler
        else:
            handler = ResampledSc203OHLCVHandler
    else:
        if args.include_tech:
            handler = Sc203OHLCVTechHandler
        else:
            handler = Sc203OHLCVHandler



    # 틱 단위 거래 환경 생성
    default_env = MinutelyOrderbookOHLCVEnv(
        df=df_train,
        handler_cls=handler,
        initial_cash=args.initial_cash, # Starting cash
        lob_levels=args.lob_levels,                     # Max shares to trade
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType[args.input_type],
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )

    if InputType[args.input_type] == InputType.LSTM:
        pretrained_lstm = load_pretrained_lstm()
        env = LSTMObsWrapper(default_env, pretrained_lstm, args.window_size, device=device)
    elif InputType[args.input_type] == InputType.MLP:
        env = default_env

    train_agent(env, save_directory, device, args)


def train_agent(env, save_directory, device, args):
    os.makedirs('runs/' + save_directory, exist_ok=True)

    with open(os.path.join('runs/' + save_directory, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    # Initialize model with custom logger
    model = PPO('MlpPolicy', verbose=0, env=env,
                gamma=args.gamma, ent_coef=args.ent_coef, max_grad_norm=args.grad_clip,
                learning_rate=args.lr,
                device=device, batch_size=128, seed=args.seed,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
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
    print("\nTraining Status:")
    
    # Train with progress tracking
    model.learn(args.iters, reset_num_timesteps=True, callback=status_callback)
    
    print("\n\nTraining completed!")
    print(f"Total training time: {(time.time() - status_callback.training_start)/60:.1f} minutes")
    
    env.close()
    model.save(os.path.join(os.path.join('runs/' + save_directory), 'agent'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and Environment arguments
    parser.add_argument("--lob_csv_path", type=str, default="src/db/AAPL_minute_orderbook_2019_01-07_combined.csv",
                        required=False, help='LOB CSV path')
    parser.add_argument("--ohlcv_csv_path", type=str, default="src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv",
                        required=False, help='OHLCV CSV path')
    parser.add_argument("--ohlcv_extended_csv_path", type=str, default="src/db/indicator/AAPL_with_indicators_v2.csv",
                        required=False, help='OHLCV Extended CSV path')
    parser.add_argument("--include_tech", type=bool, default=False,
                        required=False, help='include_tech')
    parser.add_argument("--initial_cash", type=float, default=100000.0,
                        required=False, help='Starting cash')
    parser.add_argument("--lob_levels", type=int, default=10,
                        required=False, help='Max shares to trade')
    parser.add_argument("--lookback", type=int, default=9,
                        required=False, help='Lookback')
    parser.add_argument("--h_max", type=int, default=250,
                        required=False, help='Max action')
    parser.add_argument("--hold_threshold", type=float, default=0.2,
                        required=False, help='Hold threshold')
    parser.add_argument("--window_size", type=int, default=100,
                        required=False, help='Window size')
    parser.add_argument("--transaction_fee", type=float, default=0.0023,
                        required=False, help='Transaction fee')
    parser.add_argument("--input_type", type=str, choices=["MLP", "LSTM"], default="MLP",
                        required=False, help="Type of observation to use: MLP or LSTM")

    # GPU related arguments
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    
    # Training related arguments
    parser.add_argument("--iters", "-i", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=.0001)
    parser.add_argument("--grad_clip", type=float, default=.7)
    parser.add_argument("--gamma", type=float, default=.95)
    parser.add_argument("--ent_coef", type=float, default=.002)

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help='Tag to add to the save directory')
    
    # 10분봉, 15분봉으로 교체
    parser.add_argument(
        "--resample_rule",
        type=str,
        default=None,
        help="리샘플링 주기 (예: '10T', '15T')"
    )
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    main(args)