import os
import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from tqdm import tqdm
import time



class TrainingStatusCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []  # Raw rewards from each episode
        self.episode_lengths = []
        self.training_start = time.time()  # Initialize training start time
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
        
        # Save summary to file using the logger's directory
        summary_path = os.path.join(self.logger.dir, 'training_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f)
        
        print("\nTraining Summary:")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.2f}")
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Total Timesteps: {self.num_timesteps}")
        print(f"\nSummary saved to: {summary_path}")