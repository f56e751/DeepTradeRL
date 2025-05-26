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
from src.data_handler.data_handler import Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler
from src.env.observation import Observation, InputType
from src.data_handler.csv_processor import merge_lob_and_ohlcv, DataSplitter
from src.infrastructure.callback import TrainingStatusCallback
import matplotlib.pyplot as plt

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
        df=df_test,
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
    # model = PPO('MlpPolicy', verbose=0, env=env,
    #            gamma=args.gamma, ent_coef=args.ent_coef, max_grad_norm=args.grad_clip,
    #            learning_rate=args.lr,
    #            device=device, batch_size=128, seed=args.seed,
    #            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    #            tensorboard_log=os.path.join('runs/' + save_directory, 'tensorboard'))

    model = PPO.load("runs/1748256701.9330027/agent", env = env)

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
    # model.learn(args.iters, reset_num_timesteps=True, callback=status_callback)
    evaluate_model(model, env, num_episodes = 1)
    
    print("\n\nTraining completed!")
    print(f"Total training time: {(time.time() - status_callback.training_start)/60:.1f} minutes")
    
    env.close()
    model.save(os.path.join(os.path.join('runs/' + save_directory), 'agent'))

def train_agent(env, save_directory, device, args):
    os.makedirs('runs/' + save_directory, exist_ok=True)

    with open(os.path.join('runs/' + save_directory, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    # Load the pre-trained model
    model = PPO.load("runs/1748256701.9330027/agent", env=env)

    # Set up logging with more detailed configuration
    log_path = os.path.join('runs/' + save_directory, '0')
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["csv", "tensorboard", "stdout"])
    model.set_logger(logger)
    
    # Add training status callback with save path
    status_callback = TrainingStatusCallback(
        verbose=1
    )
    
    print("\nStarting evaluation...")
    print(f"Device: {device}")
    print(f"Model loaded from: runs/1748256701.9330027/agent")
    print("\nEvaluation Status:")
    
    # Comment out the training part
    # model.learn(args.iters, reset_num_timesteps=True, callback=status_callback)
    
    # Evaluate the model instead
    evaluate_model(model, env, num_episodes=1)
    
    print("\n\nEvaluation completed!")
    
    env.close()
    # Optional: save the model again if needed
    # model.save(os.path.join(os.path.join('runs/' + save_directory), 'agent'))


def evaluate_model(model, env, num_episodes=1):
    """
    Evaluate the trained model and visualize rewards
    """
    all_episode_rewards = []
    all_step_rewards = []
    all_cumulative_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_step_rewards = []
        episode_cumulative_rewards = []
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Use the model to predict the next action (evaluation mode)
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action in the environment
            obs, reward, done, info = env.step(action)
            
            # Record rewards
            episode_step_rewards.append(reward)
            episode_reward += reward
            episode_cumulative_rewards.append(episode_reward)
            
            step += 1
            
            # Optional: print step-by-step info
            if step % 100 == 0:
                print(f"  Step {step}, Reward: {reward:.4f}, Cumulative: {episode_reward:.4f}")
        
        all_episode_rewards.append(episode_reward)
        all_step_rewards.append(episode_step_rewards)
        all_cumulative_rewards.append(episode_cumulative_rewards)
        
        print(f"Episode {episode + 1} finished with total reward: {episode_reward:.4f}")
    
    # Print evaluation summary
    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.4f}")
    print(f"Standard deviation: {std_reward:.4f}")
    print(f"Min reward: {np.min(all_episode_rewards):.4f}")
    print(f"Max reward: {np.max(all_episode_rewards):.4f}")
    
    # Visualize rewards
    plot_reward_curves(all_step_rewards, all_cumulative_rewards, num_episodes)
    
    return all_episode_rewards


def plot_reward_curves(step_rewards, cumulative_rewards, num_episodes):
    """
    Plot reward curves for visualization
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for episode in range(num_episodes):
        episode_steps = range(len(step_rewards[episode]))
        
        # Plot step-wise rewards
        axes[0].plot(episode_steps, step_rewards[episode], 
                    label=f'Episode {episode + 1}', alpha=0.7)
        axes[0].set_title('Step-wise Rewards', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot cumulative rewards
        axes[1].plot(episode_steps, cumulative_rewards[episode], 
                    label=f'Episode {episode + 1}', linewidth=2)
        axes[1].set_title('Cumulative Rewards', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Cumulative Reward')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # Add moving average for step-wise rewards if there's only one episode
    if num_episodes == 1:
        rewards = step_rewards[0]
        if len(rewards) > 50:  # Only add moving average if we have enough data points
            window_size = min(50, len(rewards) // 10)
            moving_avg = pd.Series(rewards).rolling(window=window_size, center=True).mean()
            axes[0].plot(range(len(rewards)), moving_avg, 
                        color='red', linewidth=2, 
                        label=f'Moving Average (window={window_size})')
            axes[0].legend()
    
    plt.tight_layout()
    plt.savefig('reward_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nReward curves saved as 'reward_curves.png'")
    
    # Additional statistics for single episode
    if num_episodes == 1:
        rewards = step_rewards[0]
        print(f"\n=== Step-wise Reward Statistics ===")
        print(f"Total steps: {len(rewards)}")
        print(f"Average step reward: {np.mean(rewards):.4f}")
        print(f"Step reward std: {np.std(rewards):.4f}")
        print(f"Min step reward: {np.min(rewards):.4f}")
        print(f"Max step reward: {np.max(rewards):.4f}")
        print(f"Final cumulative reward: {cumulative_rewards[0][-1]:.4f}")


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
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
    parser.add_argument("--h_max", type=int, default=100,
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
    parser.add_argument("--iters", "-i", type=int, default=0)
    parser.add_argument("--lr", type=float, default=.00032)
    parser.add_argument("--grad_clip", type=float, default=.5)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--ent_coef", type=float, default=.0089)

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help='Tag to add to the save directory')

    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    # model load
    

    main(args)