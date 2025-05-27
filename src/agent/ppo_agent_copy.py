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
from src.infrastructure.callback import TrainingStatusCallback
import matplotlib.pyplot as plt


class TrainingMetricsCallback(BaseCallback):
    """
    Callback to collect training metrics for visualization
    """
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.learning_rates = []
        self.timesteps = []
        self.episodes = []
        
        # Temporary storage for current episode
        self.current_episode_reward = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Collect episode rewards from the environment info
        infos = self.locals.get('infos', [])
        if infos and len(infos) > 0:
            info = infos[0]
            if 'episode' in info:
                # Episode finished, collect the total reward
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                self.episode_rewards.append(episode_reward)
                self.episodes.append(self.episode_count)
                self.timesteps.append(self.num_timesteps)
                self.episode_count += 1
                
                if self.verbose >= 1:
                    print(f"Episode {self.episode_count}: Reward = {episode_reward:.4f}, Length = {episode_length}")
            
        return True
    
    def _on_rollout_end(self) -> None:
        # Collect training metrics from logger
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Get the latest training metrics
            log_dict = self.model.logger.name_to_value
            
            if 'train/policy_gradient_loss' in log_dict:
                self.policy_losses.append(log_dict['train/policy_gradient_loss'])
            elif 'train/policy_loss' in log_dict:
                self.policy_losses.append(log_dict['train/policy_loss'])
                
            if 'train/value_loss' in log_dict:
                self.value_losses.append(log_dict['train/value_loss'])
                
            if 'train/entropy_loss' in log_dict:
                self.entropy_losses.append(log_dict['train/entropy_loss'])
            elif 'train/entropy' in log_dict:
                self.entropy_losses.append(-log_dict['train/entropy'])  # Negative entropy as loss
                
            if 'train/learning_rate' in log_dict:
                self.learning_rates.append(log_dict['train/learning_rate'])
    
    def get_metrics(self):
        """Return collected metrics"""
        return {
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'learning_rates': self.learning_rates,
            'timesteps': self.timesteps,
            'episodes': self.episodes
        }


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
    
    # Load and preprocess data
    df_all = merge_lob_and_ohlcv(args.lob_csv_path, args.ohlcv_csv_path)
    # Normalize data
    df_all = df_all.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Split data into train, validation, and test sets
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)
    
    print(f"Data split - Train: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}")
    
    # Create training environment
    env_train = MinutelyOrderbookOHLCVEnv(
        df=df_train,
        handler_cls=Sc203OHLCVHandler,
        initial_cash=args.initial_cash,
        lob_levels=args.lob_levels,
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType.MLP,
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )

    # Train the agent
    model, training_metrics = train_agent(env_train, save_directory, device, args)
    
    # Create evaluation environments
    env_val = MinutelyOrderbookOHLCVEnv(
        df=df_val,
        handler_cls=Sc203OHLCVHandler,
        initial_cash=args.initial_cash,
        lob_levels=args.lob_levels,
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType.MLP,
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )
    
    env_test = MinutelyOrderbookOHLCVEnv(
        df=df_test,
        handler_cls=Sc203OHLCVHandler,
        initial_cash=args.initial_cash,
        lob_levels=args.lob_levels,
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType.MLP,
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )

    # Evaluate on both validation and test sets
    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)
    
    print("\n🔍 Evaluating on Validation Set...")
    val_rewards = evaluate_model(model, env_val, num_episodes=1, dataset_name="Validation")

    model = PPO.load("runs/Best_model_v1/agent", env = env_train)
    
    print("\n🔍 Evaluating on Test Set...")
    test_rewards = evaluate_model(model, env_test, num_episodes=1, dataset_name="Test")
    #test_rewards = evaluate_model(model, env_train, num_episodes=1, dataset_name="Test")
    
    # Compare validation and test performance
    compare_performance(val_rewards, test_rewards, save_directory)
    
    # Close environments
    env_train.close()
    env_val.close()
    env_test.close()


def train_agent(env, save_directory, device, args):
    """Train the PPO agent"""
    os.makedirs('runs/' + save_directory, exist_ok=True)

    # Save training parameters
    with open(os.path.join('runs/' + save_directory, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    # Initialize PPO model
    model = PPO('MlpPolicy', 
                env=env,
                verbose=1,
                gamma=args.gamma, 
                ent_coef=args.ent_coef, 
                max_grad_norm=args.grad_clip,
                learning_rate=args.lr,
                device=device, 
                batch_size=128, 
                seed=args.seed,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                tensorboard_log=os.path.join('runs/' + save_directory, 'tensorboard'))

    # Set up logging
    log_path = os.path.join('runs/' + save_directory, '0')
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["csv", "tensorboard", "stdout"])
    model.set_logger(logger)
    
    # Add training status callback and metrics callback
    status_callback = TrainingStatusCallback(verbose=1)
    metrics_callback = TrainingMetricsCallback(verbose=1)
    
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    print(f"🚀 Starting training...")
    print(f"📊 Total timesteps: {args.iters}")
    print(f"💻 Device: {device}")
    print(f"💾 Model will be saved to: runs/{save_directory}")
    
    # Train the model
    if args.iters > 0:
        model.learn(args.iters, reset_num_timesteps=True, callback=[status_callback, metrics_callback])
        
        print(f"\n✅ Training completed!")
        print(f"⏱️  Total training time: {(time.time() - status_callback.training_start)/60:.1f} minutes")
        
        # Save the trained model
        model.save(os.path.join('runs/' + save_directory, 'agent'))
        print(f"💾 Model saved successfully!")
        
        # Get training metrics and plot curves
        training_metrics = metrics_callback.get_metrics()
        print(f"\n📈 Plotting training curves...")
        plot_training_reward_curves(training_metrics, save_directory)
        
    else:
        print("⚠️  No training iterations specified (iters=0), skipping training...")
        training_metrics = {}
    
    return model, training_metrics


def plot_training_reward_curves(training_metrics, save_directory):
    """
    Plot training reward curves and loss curves
    """
    # Create subplot layout: 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Visualization', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    if training_metrics.get('episode_rewards'):
        episodes = training_metrics['episodes']
        episode_rewards = training_metrics['episode_rewards']
        
        axes[0, 0].plot(episodes, episode_rewards, color='blue', linewidth=2, alpha=0.7)
        
        # Add moving average if we have enough data points
        if len(episode_rewards) > 10:
            window_size = min(50, len(episode_rewards) // 5)
            moving_avg = pd.Series(episode_rewards).rolling(window=window_size, center=True).mean()
            axes[0, 0].plot(episodes, moving_avg, color='red', linewidth=2, 
                           label=f'Moving Average (window={window_size})')
            axes[0, 0].legend()
        
        axes[0, 0].set_title('Episode Rewards During Training', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Print statistics
        print(f"📊 Training Episode Statistics:")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average episode reward: {np.mean(episode_rewards):.4f}")
        print(f"Episode reward std: {np.std(episode_rewards):.4f}")
        print(f"Min episode reward: {np.min(episode_rewards):.4f}")
        print(f"Max episode reward: {np.max(episode_rewards):.4f}")
    else:
        axes[0, 0].text(0.5, 0.5, 'No episode reward data collected\n(Training may be too short)', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Episode Rewards During Training', fontweight='bold')
    
    # Plot 2: Policy Loss
    if training_metrics.get('policy_losses'):
        update_steps = range(len(training_metrics['policy_losses']))
        axes[0, 1].plot(update_steps, training_metrics['policy_losses'], 
                       color='red', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Policy Loss During Training', fontweight='bold')
        axes[0, 1].set_xlabel('Training Update')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No policy loss data collected', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Policy Loss During Training', fontweight='bold')
    
    # Plot 3: Value Loss
    if training_metrics.get('value_losses'):
        update_steps = range(len(training_metrics['value_losses']))
        axes[1, 0].plot(update_steps, training_metrics['value_losses'], 
                       color='orange', linewidth=2, alpha=0.7)
        axes[1, 0].set_title('Value Loss During Training', fontweight='bold')
        axes[1, 0].set_xlabel('Training Update')
        axes[1, 0].set_ylabel('Value Loss')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No value loss data collected', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Value Loss During Training', fontweight='bold')
    
    # Plot 4: Entropy Loss
    if training_metrics.get('entropy_losses'):
        update_steps = range(len(training_metrics['entropy_losses']))
        axes[1, 1].plot(update_steps, training_metrics['entropy_losses'], 
                       color='purple', linewidth=2, alpha=0.7)
        axes[1, 1].set_title('Entropy Loss During Training', fontweight='bold')
        axes[1, 1].set_xlabel('Training Update')
        axes[1, 1].set_ylabel('Entropy Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No entropy loss data collected', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Entropy Loss During Training', fontweight='bold')
    
    plt.tight_layout()
    training_curves_filename = f'runs/{save_directory}/training_curves.png'
    plt.savefig(training_curves_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Training curves saved as '{training_curves_filename}'")
    
    # 🔍 Loss 분석 및 해석
    print(f"\n🔍 Training Loss Analysis:")
    if training_metrics.get('policy_losses'):
        policy_losses = training_metrics['policy_losses']
        print(f"Policy Loss - Initial: {policy_losses[0]:.6f}, Final: {policy_losses[-1]:.6f}")
        print(f"Policy Loss - Mean: {np.mean(policy_losses):.6f}, Std: {np.std(policy_losses):.6f}")
        
        # 정책 손실의 변화 패턴 분석
        if len(policy_losses) > 10:
            first_half_mean = np.mean(policy_losses[:len(policy_losses)//2])
            second_half_mean = np.mean(policy_losses[len(policy_losses)//2:])
            if abs(first_half_mean - second_half_mean) < 0.001:
                print("⚠️  Policy loss has converged - training may need more exploration or different hyperparameters")
    
    if training_metrics.get('value_losses'):
        value_losses = training_metrics['value_losses']
        print(f"Value Loss - Initial: {value_losses[0]:.2f}, Final: {value_losses[-1]:.2f}")
        print(f"Value Loss - Mean: {np.mean(value_losses):.2f}, Std: {np.std(value_losses):.2f}")
        
        # 가치 손실이 너무 높으면 문제가 있을 수 있음
        if np.mean(value_losses) > 1000:
            print("⚠️  Value loss is very high - check reward scaling or value function architecture")
    
    print(f"\n💡 Loss Analysis Insights:")
    print(f"• PPO policy loss naturally fluctuates (not always decreasing)")
    print(f"• Value loss spikes indicate value function is learning complex patterns")
    print(f"• If losses plateau early, consider:")
    print(f"  - Increasing learning rate ({args.lr if 'args' in globals() else 'current'})")
    print(f"  - Adjusting entropy coefficient for more exploration")
    print(f"  - Checking if reward signal is informative enough")
    print(f"  - Ensuring sufficient training steps per episode")


def evaluate_model(model, env, num_episodes=1, dataset_name=""):
    """
    Evaluate the trained model and return rewards
    """
    print(f"\n📈 Evaluating on {dataset_name} Dataset")
    print("-" * 40)
    
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
        
        print(f"Episode {episode + 1}/{num_episodes} - {dataset_name}")
        
        # Use tqdm for progress bar
        pbar = tqdm(desc=f"Steps", unit="step")
        
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
            pbar.update(1)
            pbar.set_postfix({
                'Reward': f'{reward:.4f}', 
                'Cumulative': f'{episode_reward:.4f}'
            })
        
        pbar.close()
        
        all_episode_rewards.append(episode_reward)
        all_step_rewards.append(episode_step_rewards)
        all_cumulative_rewards.append(episode_cumulative_rewards)
        
        print(f"✅ Episode {episode + 1} finished - Total reward: {episode_reward:.4f}")
    
    # Print evaluation summary
    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    
    print(f"\n📊 {dataset_name} Evaluation Summary")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Standard deviation: {std_reward:.4f}")
    print(f"Min reward: {np.min(all_episode_rewards):.4f}")
    print(f"Max reward: {np.max(all_episode_rewards):.4f}")
    
    # Visualize rewards for this dataset using the original function
    plot_reward_curves(all_step_rewards, all_cumulative_rewards, num_episodes, dataset_name)
    
    return {
        'episode_rewards': all_episode_rewards,
        'step_rewards': all_step_rewards, 
        'cumulative_rewards': all_cumulative_rewards,
        'avg_reward': avg_reward,
        'std_reward': std_reward
    }


def plot_reward_curves(step_rewards, cumulative_rewards, num_episodes, dataset_name):
    """
    Plot reward curves for visualization (original function, unchanged)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for episode in range(num_episodes):
        episode_steps = range(len(step_rewards[episode]))
        
        # Plot step-wise rewards
        axes[0].plot(episode_steps, step_rewards[episode], 
                    label=f'Episode {episode + 1}', alpha=0.7)
        axes[0].set_title(f'{dataset_name} - Step-wise Rewards', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True, alpha=0.3)
        if num_episodes <= 5:  # Only show legend if not too many episodes
            axes[0].legend()
        
        # Plot cumulative rewards
        axes[1].plot(episode_steps, cumulative_rewards[episode], 
                    label=f'Episode {episode + 1}', linewidth=2)
        axes[1].set_title(f'{dataset_name} - Cumulative Rewards', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Cumulative Reward')
        axes[1].grid(True, alpha=0.3)
        if num_episodes <= 5:
            axes[1].legend()
    
    # Add moving average for step-wise rewards if there's only one episode
    if num_episodes == 1:
        rewards = step_rewards[0]
        if len(rewards) > 50:
            window_size = min(50, len(rewards) // 10)
            moving_avg = pd.Series(rewards).rolling(window=window_size, center=True).mean()
            axes[0].plot(range(len(rewards)), moving_avg, 
                        color='red', linewidth=2, 
                        label=f'Moving Average (window={window_size})')
            axes[0].legend()
    
    plt.tight_layout()
    filename = f'reward_curves_{dataset_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 {dataset_name} reward curves saved as '{filename}'")
    
    # Additional statistics for single episode
    if num_episodes == 1:
        rewards = step_rewards[0]
        print(f"\n📊 {dataset_name} Step-wise Reward Statistics")
        print(f"Total steps: {len(rewards)}")
        print(f"Average step reward: {np.mean(rewards):.4f}")
        print(f"Step reward std: {np.std(rewards):.4f}")
        print(f"Min step reward: {np.min(rewards):.4f}")
        print(f"Max step reward: {np.max(rewards):.4f}")
        print(f"Final cumulative reward: {cumulative_rewards[0][-1]:.4f}")


def compare_performance(val_results, test_results, save_directory):
    """
    Compare validation and test performance
    """
    print("\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON")
    print("="*50)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Average rewards comparison
    datasets = ['Validation', 'Test']
    avg_rewards = [val_results['avg_reward'], test_results['avg_reward']]
    std_rewards = [val_results['std_reward'], test_results['std_reward']]
    
    axes[0, 0].bar(datasets, avg_rewards, yerr=std_rewards, capsize=5, 
                   color=['skyblue', 'lightcoral'], alpha=0.7)
    axes[0, 0].set_title('Average Reward Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (avg, std) in enumerate(zip(avg_rewards, std_rewards)):
        axes[0, 0].text(i, avg + std + 0.01, f'{avg:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Cumulative rewards comparison (if single episode)
    if len(val_results['cumulative_rewards']) == 1 and len(test_results['cumulative_rewards']) == 1:
        val_cum = val_results['cumulative_rewards'][0]
        test_cum = test_results['cumulative_rewards'][0]
        
        axes[0, 1].plot(val_cum, label='Validation', linewidth=2, color='skyblue')
        axes[0, 1].plot(test_cum, label='Test', linewidth=2, color='lightcoral')
        axes[0, 1].set_title('Cumulative Rewards Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Step rewards distribution
    val_step_rewards = np.concatenate(val_results['step_rewards'])
    test_step_rewards = np.concatenate(test_results['step_rewards'])
    
    axes[1, 0].hist(val_step_rewards, bins=50, alpha=0.7, label='Validation', 
                    color='skyblue', density=True)
    axes[1, 0].hist(test_step_rewards, bins=50, alpha=0.7, label='Test', 
                    color='lightcoral', density=True)
    axes[1, 0].set_title('Step Rewards Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Step Reward')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics table
    axes[1, 1].axis('off')
    
    # Create performance summary table
    metrics_data = [
        ['Metric', 'Validation', 'Test', 'Difference'],
        ['Avg Reward', f'{val_results["avg_reward"]:.4f}', 
         f'{test_results["avg_reward"]:.4f}', 
         f'{test_results["avg_reward"] - val_results["avg_reward"]:.4f}'],
        ['Std Reward', f'{val_results["std_reward"]:.4f}', 
         f'{test_results["std_reward"]:.4f}', 
         f'{test_results["std_reward"] - val_results["std_reward"]:.4f}'],
        ['Min Reward', f'{np.min(val_results["episode_rewards"]):.4f}', 
         f'{np.min(test_results["episode_rewards"]):.4f}', 
         f'{np.min(test_results["episode_rewards"]) - np.min(val_results["episode_rewards"]):.4f}'],
        ['Max Reward', f'{np.max(val_results["episode_rewards"]):.4f}', 
         f'{np.max(test_results["episode_rewards"]):.4f}', 
         f'{np.max(test_results["episode_rewards"]) - np.max(val_results["episode_rewards"]):.4f}']
    ]
    
    table = axes[1, 1].table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                            cellLoc='center', loc='center',
                            colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(metrics_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Performance Metrics Comparison', fontweight='bold', pad=20)
    
    plt.tight_layout()
    comparison_filename = f'performance_comparison.png'
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Performance comparison saved as '{comparison_filename}'")
    
    # Print detailed comparison
    print(f"\n📈 Detailed Performance Analysis:")
    print(f"Validation avg reward: {val_results['avg_reward']:.4f} ± {val_results['std_reward']:.4f}")
    print(f"Test avg reward: {test_results['avg_reward']:.4f} ± {test_results['std_reward']:.4f}")
    
    diff = test_results['avg_reward'] - val_results['avg_reward']
    if diff > 0:
        print(f"✅ Test performance is {diff:.4f} better than validation")
    else:
        print(f"⚠️  Test performance is {abs(diff):.4f} worse than validation")
    
    # Check for overfitting
    if val_results['avg_reward'] > test_results['avg_reward']:
        print("⚠️  Potential overfitting detected (validation > test performance)")
    else:
        print("✅ No obvious overfitting (test >= validation performance)")


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
    parser.add_argument("--h_max", type=int, default=200,
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
    parser.add_argument("--iters", "-i", type=int, default=10000)  # Changed default from 0 to 10000
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