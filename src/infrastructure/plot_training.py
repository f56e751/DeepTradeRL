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
    
    # Display all available columns
    print("\nAvailable columns in the log file:")
    print("=" * 50)
    for col in df.columns:
        print(f"- {col}")
    print("=" * 50)
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Training Rewards', fontsize=16)
    
    # Plot mean reward
    if 'train/reward' in df.columns:
        ax1.plot(df['time/total_timesteps'], df['train/reward'], label='Mean Reward', color='blue')
        ax1.set_title('Mean Reward Over Time')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Mean Reward')
        ax1.grid(True)
    
    # Plot raw reward
    if 'train/raw_reward' in df.columns:
        ax2.plot(df['time/total_timesteps'], df['train/raw_reward'], label='Raw Reward', color='red')
        ax2.set_title('Raw Reward Over Time')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Raw Reward')
        ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(log_dir), 'training_plots.png')
    plt.savefig(save_path)
    print(f"\nPlots saved to {save_path}")
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