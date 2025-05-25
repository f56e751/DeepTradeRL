import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def plot_training_metrics(log_dir):
    """
    Plot training metrics from tensorboard logs
    
    Args:
        log_dir: Directory containing the tensorboard logs (e.g., 'runs/timestamp/0')
    """
    # Read the CSV file
    csv_file = os.path.join(log_dir, 'progress.csv')
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Print available columns for debugging
    print("Available columns:", df.columns.tolist())
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot rewards
    ax = axes[0, 0]
    if 'train/reward' in df.columns:
        ax.plot(df['time/total_timesteps'], df['train/reward'], label='Reward')
    if 'train/realized_pnl' in df.columns:
        ax.plot(df['time/total_timesteps'], df['train/realized_pnl'], label='Realized PnL')
    ax.set_title('Rewards and PnL')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    
    # Plot PnL
    ax = axes[0, 1]
    if 'train/unrealized_pnl' in df.columns:
        ax.plot(df['time/total_timesteps'], df['train/unrealized_pnl'], label='Unrealized PnL')
    if 'train/realized_pnl' in df.columns:
        ax.plot(df['time/total_timesteps'], df['train/realized_pnl'], label='Realized PnL')
    ax.set_title('Profit and Loss')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('PnL')
    ax.legend()
    ax.grid(True)
    
    # Plot episode length
    ax = axes[1, 0]
    if 'train/ep_len_mean' in df.columns:
        ax.plot(df['time/total_timesteps'], df['train/ep_len_mean'])
        ax.set_title('Episode Length')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Length')
        ax.grid(True)
    
    # Plot explained variance
    ax = axes[1, 1]
    if 'train/explained_variance' in df.columns:
        ax.plot(df['time/total_timesteps'], df['train/explained_variance'])
        ax.set_title('Explained Variance')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Variance')
        ax.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(log_dir), 'training_plots.png')
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")
    plt.close()

def main():
    # Find the most recent training run
    runs_dir = 'runs'
    if not os.path.exists(runs_dir):
        print(f"Error: Could not find {runs_dir} directory")
        return
    
    # Get all timestamp directories
    timestamp_dirs = glob(os.path.join(runs_dir, '*'))
    if not timestamp_dirs:
        print("Error: No training runs found")
        return
    
    # Get the most recent run
    latest_run = max(timestamp_dirs, key=os.path.getctime)
    log_dir = os.path.join(latest_run, '0')
    
    print(f"Plotting metrics from {log_dir}")
    plot_training_metrics(log_dir)

if __name__ == '__main__':
    main() 