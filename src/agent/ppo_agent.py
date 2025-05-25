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
    def __init__(self, save_path=None, save_freq=10000, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()
        self.last_time = self.start_time
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.training_start = time.time()
        self.total_episodes = 0
        self.save_path = save_path
        self.save_freq = save_freq
        
        # Initialize metrics tracking
        self.metrics_to_track = [
            'train/ep_rew_mean',
            'train/ep_len_mean',
            'train/explained_variance',
            'train/learning_rate',
            'train/reward',
            'train/realized_pnl',
            'train/unrealized_pnl'
        ]
        self.metrics_history = {metric: [] for metric in self.metrics_to_track}
        
    def _on_step(self):
        # Get episode info from logger
        if self.logger is not None:
            # Get the latest logged values
            logged_values = self.logger.name_to_value
            
            # Track metrics
            for metric in self.metrics_to_track:
                if metric in logged_values:
                    self.metrics_history[metric].append(logged_values[metric])
            
            # Track episode rewards and lengths
            if 'train/ep_rew_mean' in logged_values:
                mean_reward = logged_values['train/ep_rew_mean']
                mean_length = logged_values['train/ep_len_mean']
                explained_var = logged_values.get('train/explained_variance', 0)
                lr = logged_values.get('train/learning_rate', 0)
                reward = logged_values.get('train/reward', 0)
                realized_pnl = logged_values.get('train/realized_pnl', 0)
                unrealized_pnl = logged_values.get('train/unrealized_pnl', 0)
                
                self.total_episodes += 1
                self.episode_rewards.append(mean_reward)
                self.episode_lengths.append(mean_length)
                self.episode_times.append(time.time() - self.last_time)
                self.last_time = time.time()
                
                # Calculate progress
                progress = (self.num_timesteps / self.locals['total_timesteps']) * 100
                
                # Print status with more metrics
                print(f"\rProgress: {progress:.1f}% | "
                      f"Episodes: {self.total_episodes} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Reward: {reward:.2f} | "
                      f"Realized PnL: {realized_pnl:.2f} | "
                      f"Unrealized PnL: {unrealized_pnl:.2f} | "
                      f"Mean Length: {mean_length:.1f} | "
                      f"Explained Var: {explained_var:.2f} | "
                      f"LR: {lr:.2e} | ")
        
        # Save model checkpoint if save_path is provided
        if self.save_path and self.num_timesteps % self.save_freq == 0:
            checkpoint_path = os.path.join(self.save_path, f"model_{self.num_timesteps}")
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"\nSaved model checkpoint to {checkpoint_path}")
        
        return True
    
    def on_training_end(self):
        """Called when training ends."""
        training_time = time.time() - self.training_start
        
        # Save training summary if save_path is provided
        if self.save_path:
            # Get the final metrics
            final_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
            final_length = self.episode_lengths[-1] if self.episode_lengths else 0.0
            final_explained_var = self.metrics_history['train/explained_variance'][-1] if self.metrics_history['train/explained_variance'] else 0.0
            final_lr = self.metrics_history['train/learning_rate'][-1] if self.metrics_history['train/learning_rate'] else 0.0
            
            summary = {
                'total_timesteps': self.num_timesteps,
                'total_episodes': self.total_episodes,
                'training_time': training_time,
                'final_reward': final_reward,
                'final_length': final_length,
                'final_explained_variance': final_explained_var,
                'final_learning_rate': final_lr
            }
            
            # Save summary to file
            with open(os.path.join(self.save_path, 'training_summary.txt'), 'w') as f:
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
        
        if self.verbose > 0:
            if self.episode_rewards:
                print(f"Final mean reward: {self.episode_rewards[-1]:.2f}")
            if self.episode_lengths:
                print(f"Final mean episode length: {self.episode_lengths[-1]:.1f}")
            if self.metrics_history['train/explained_variance']:
                print(f"Final explained variance: {self.metrics_history['train/explained_variance'][-1]:.2f}")

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
        save_path=os.path.join('runs', save_directory),
        save_freq=10000,  # Save model every 10000 timesteps
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