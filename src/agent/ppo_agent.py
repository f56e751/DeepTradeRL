import argparse
import time
import os
import random
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
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



class TrainingStatusCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []  # Raw rewards from each episode
        self.episode_lengths = []
        self.metrics_to_track = [
            'train/reward',  # This will be the mean reward
            'train/raw_reward',  # This will be the raw reward
            'train/ep_len_mean',
            'train/explained_variance',
            'train/learning_rate'
        ]

    def _on_step(self):
        # Track episode rewards and lengths
        if len(self.model.ep_info_buffer) > 0:
            # Get raw rewards from each episode
            raw_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            self.episode_rewards.extend(raw_rewards)
            self.episode_lengths.extend([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            # Log the raw reward from the last episode
            if raw_rewards:
                self.logger.record("train/raw_reward", raw_rewards[-1])
            
            self.model.ep_info_buffer.clear()

        # Log metrics
        self.logger.record("train/reward", np.mean(self.episode_rewards) if self.episode_rewards else 0.0)
        self.logger.record("train/ep_len_mean", np.mean(self.episode_lengths) if self.episode_lengths else 0.0)
        
        # Log progress
        if self.num_timesteps % 1000 == 0:
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            raw_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
            mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
            progress = (self.num_timesteps / self.locals['total_timesteps']) * 100
            
            print(f"\nStep {self.num_timesteps}")
            print(f"Mean Reward: {mean_reward:.2f}")
            print(f"Raw Reward: {raw_reward:.2f}")
            print(f"Mean Episode Length: {mean_length:.2f}")
            print(f"Progress: {progress:.1f}%")
            print(f"Learning Rate: {self.locals['self'].learning_rate:.6f}")

        return True

    def _on_training_end(self):
        # Create final summary
        mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        
        summary = {
            'mean_reward': mean_reward,
            'mean_episode_length': mean_length,
            'total_episodes': len(self.episode_rewards),
            'total_timesteps': self.num_timesteps
        }
        
        # Save summary to file
        summary_path = os.path.join(self.locals['self'].log_dir, 'training_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f)
        
        print("\nTraining Summary:")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.2f}")
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Total Timesteps: {self.num_timesteps}")
        print(f"\nSummary saved to: {summary_path}")

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

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help='Tag to add to the save directory')

    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    main(args)