import os
import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from tqdm import tqdm
import time

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt



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
    

class ValidationCallback(BaseCallback):
    """
    Callback for performing validation during training
    """
    def __init__(self, val_env, eval_freq=1000, n_eval_episodes=1, save_directory=None, verbose=0):
        super(ValidationCallback, self).__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_directory = save_directory
        self.validation_rewards = []
        self.validation_timesteps = []
        self.best_mean_reward = -float('inf')
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Perform validation
            episode_rewards = []
            
            for _ in range(self.n_eval_episodes):
                obs, info = self.val_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                    
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.val_env.step(action)
                    done = terminated or truncated
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    if isinstance(done, tuple):
                        done = done[0] if len(done) > 0 else done
                    episode_reward += reward
                    
                episode_rewards.append(episode_reward)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            self.validation_rewards.append(mean_reward)
            self.validation_timesteps.append(self.n_calls)
            
            if self.verbose > 0:
                print(f"\n📊 Validation at step {self.n_calls}: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.save_directory:
                    best_model_path = os.path.join('runs', self.save_directory, 'best_model')
                    self.model.save(best_model_path)
                    if self.verbose > 0:
                        print(f"🏆 New best model saved! Reward: {mean_reward:.2f}")
            
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training to save validation plots"""
        if len(self.validation_rewards) > 0 and self.save_directory:
            try:
                # Plot validation curve
                plt.figure(figsize=(10, 6))
                plt.plot(self.validation_timesteps, self.validation_rewards, 'b-', linewidth=2, label='Validation Reward')
                plt.xlabel('Training Steps')
                plt.ylabel('Mean Episode Reward')
                plt.title('Validation Performance During Training')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot without showing
                plot_path = os.path.join('runs', self.save_directory, 'validation_curve.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()  # Important: close the figure to free memory
                
                if self.verbose > 0:
                    print(f"📈 Validation curve saved to: {plot_path}")
                    print(f"🏆 Best validation reward: {self.best_mean_reward:.2f}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️ Warning: Could not save validation plot: {e}")
                # Make sure to close any open figures
                plt.close('all')
    
    def get_validation_data(self):
        """Return validation data for analysis"""
        return {
            'timesteps': self.validation_timesteps,
            'rewards': self.validation_rewards,
            'best_reward': self.best_mean_reward
        }
