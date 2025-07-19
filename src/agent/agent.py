import argparse
import time
import os
import random
import numpy as np
import psutil
import torch
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.infrastructure.pytorch_util import init_gpu
from src.env.minutely_orderbook_ohlcv_env import MinutelyOrderbookOHLCVEnv
from src.data_handler.data_handler import Sc203Handler, Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler, Sc203OHLCVTechHandler
from src.env.observation import Observation, InputType
from src.data_handler.csv_processor import merge_lob_and_ohlcv, merge_lob_and_ohlcv_extended, DataSplitter
from src.infrastructure.callback import TrainingStatusCallback
from src.agent.wrapper import LSTMObsWrapper, load_pretrained_lstm
from src.deeplob.model import deeplob

# Import visualization and utility functions
from src.agent.util import (
    TrainingMetricsCallback,
    calculate_financial_metrics,
    plot_comprehensive_financial_analysis,
    plot_training_reward_curves,
    plot_reward_curves,
    compare_performance,
    evaluate_model
)

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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
                obs = self.val_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                    
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.val_env.step(action)
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

"""
Unified RL Agent Training Script

This script supports:
- Agent Types: PPO, SAC
- Input Types: MLP, LSTM (with pretrained DeepLOB)
- Technical Indicators: Optional inclusion
- Complete evaluation pipeline with visualization
- Flexible hyperparameter configuration

Usage examples:
1. Basic PPO with MLP: python agent.py --agent_type PPO --input_type MLP
2. SAC with LSTM: python agent.py --agent_type SAC --input_type LSTM
3. PPO with tech indicators: python agent.py --agent_type PPO --include_tech True
4. SAC with all features: python agent.py --agent_type SAC --input_type LSTM --include_tech True
"""


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
    
    # Load and preprocess data based on tech indicator option
    if args.include_tech == False:
        df_all = merge_lob_and_ohlcv(args.lob_csv_path, args.ohlcv_csv_path)
    else:
        df_all = merge_lob_and_ohlcv_extended(args.lob_csv_path, args.ohlcv_extended_csv_path)
        
    # Normalize data
    df_all = df_all.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Split data into train, validation, and test sets
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)
    
    print(f"Data split - Train: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}")
    
    # Select handler class based on tech indicator option
    handler = Sc203OHLCVTechHandler if args.include_tech else Sc203OHLCVHandler
    
    # Create training environment
    env_train_original = MinutelyOrderbookOHLCVEnv(
        df=df_train,
        handler_cls=handler,
        initial_cash=args.initial_cash,
        lob_levels=args.lob_levels,
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType[args.input_type],
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )

    # Apply wrapper based on input type
    if InputType[args.input_type] == InputType.LSTM:
        pretrained_lstm = load_pretrained_lstm()
        env_train = LSTMObsWrapper(env_train_original, pretrained_lstm, args.window_size, device=device)
    elif InputType[args.input_type] == InputType.MLP:
        env_train = env_train_original

    # Create validation environment
    env_val_original = MinutelyOrderbookOHLCVEnv(
        df=df_val,
        handler_cls=handler,
        initial_cash=args.initial_cash,
        lob_levels=args.lob_levels,
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType[args.input_type],
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )

    # Apply wrapper to validation environment if needed
    if InputType[args.input_type] == InputType.LSTM:
        env_val = LSTMObsWrapper(env_val_original, pretrained_lstm, args.window_size, device=device)
    elif InputType[args.input_type] == InputType.MLP:
        env_val = env_val_original

    # Train the agent with validation during training
    model, training_metrics, validation_data = train_agent(env_train, env_val, save_directory, device, args)
    
    # Create test environment for final evaluation
    env_test_original = MinutelyOrderbookOHLCVEnv(
        df=df_test,
        handler_cls=handler,
        initial_cash=args.initial_cash,
        lob_levels=args.lob_levels,
        lookback=args.lookback,
        window_size=args.window_size,
        input_type=InputType[args.input_type],
        transaction_fee=args.transaction_fee,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold
    )

    # Apply wrapper to test environment if needed
    if InputType[args.input_type] == InputType.LSTM:
        env_test = LSTMObsWrapper(env_test_original, pretrained_lstm, args.window_size, device=device)
    elif InputType[args.input_type] == InputType.MLP:
        env_test = env_test_original

    # Final evaluation on test set only
    print("\n" + "="*50)
    print("FINAL EVALUATION PHASE")
    print("="*50)
    
    print(f"\n� Validation Summary:")
    print(f"   • Best validation reward: {validation_data['best_reward']:.2f}")
    print(f"   • Total validation runs: {len(validation_data['rewards'])}")
    
    print("\n🔍 Final Evaluation on Test Set...")
    test_rewards = evaluate_model(model, env_test, num_episodes=1, dataset_name="Test", initial_cash=args.initial_cash, save_directory=save_directory)
    
    # Load and evaluate best model on test set
    best_model_path = os.path.join('runs', save_directory, 'best_model')
    if os.path.exists(best_model_path + '.zip'):
        print("\n🏆 Evaluating Best Model on Test Set...")
        if args.agent_type == "PPO":
            best_model = PPO.load(best_model_path, env=env_test)
        elif args.agent_type == "SAC":
            best_model = SAC.load(best_model_path, env=env_test)
        
        best_test_rewards = evaluate_model(best_model, env_test, num_episodes=1, dataset_name="Test (Best Model)", initial_cash=args.initial_cash, save_directory=save_directory)
    
    # Close environments
    env_train.close()
    env_val.close()
    env_test.close()
    
    # Also close original environments if LSTM was used
    if InputType[args.input_type] == InputType.LSTM:
        env_train_original.close()
        env_val_original.close()
        env_test_original.close()


def train_agent(env_train, env_val, save_directory, device, args):
    """Train the selected agent (PPO or SAC) with validation during training"""
    os.makedirs('runs/' + save_directory, exist_ok=True)

    # Save training parameters
    with open(os.path.join('runs/' + save_directory, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    # Initialize model based on agent type
    if args.agent_type == "PPO":
        model = PPO('MlpPolicy', 
                    env=env_train,
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
    
    elif args.agent_type == "SAC":
        # Handle string arguments for SAC
        ent_coef = args.ent_coef if args.ent_coef != 'auto' else 'auto'
        target_entropy = args.target_entropy if args.target_entropy != 'auto' else 'auto'
        
        model = SAC('MlpPolicy', 
                    env=env_train,
                    verbose=1,
                    gamma=args.gamma, 
                    tau=args.tau,
                    ent_coef=ent_coef,
                    target_update_interval=args.target_update_interval,
                    target_entropy=target_entropy,
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
    
    else:
        raise ValueError(f"Unsupported agent type: {args.agent_type}")

    # Set up logging
    log_path = os.path.join('runs/' + save_directory, '0')
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["csv", "tensorboard", "stdout"])
    model.set_logger(logger)
    
    # Add training status callback and metrics callback
    status_callback = TrainingStatusCallback(verbose=1)
    metrics_callback = TrainingMetricsCallback(verbose=1)
    
    # Add validation callback
    validation_callback = ValidationCallback(
        val_env=env_val,
        eval_freq=args.validation_freq,
        n_eval_episodes=1,
        save_directory=save_directory,
        verbose=1
    )
    
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    print(f"🚀 Starting training with {args.agent_type} agent...")
    print(f"📊 Total timesteps: {args.iters}")
    print(f"💻 Device: {device}")
    print(f"💾 Model will be saved to: runs/{save_directory}")
    print(f"✅ Validation every {args.validation_freq} steps")
    
    # Print agent-specific parameters
    if args.agent_type == "SAC":
        print(f"🎯 Buffer size: {args.buffer_size}")
        print(f"🎯 Learning starts: {args.learning_starts}")
        print(f"🎯 Batch size: {args.batch_size}")
        print(f"🎯 Train frequency: {args.train_freq}")
        print(f"🎯 Target entropy: {args.target_entropy}")
    
    # Train the model with all callbacks
    if args.iters > 0:
        model.learn(args.iters, reset_num_timesteps=True, 
                   callback=[status_callback, metrics_callback, validation_callback])
        
        print(f"\n✅ Training completed!")
        print(f"⏱️  Total training time: {(time.time() - status_callback.training_start)/60:.1f} minutes")
        
        # Save the final trained model
        model.save(os.path.join('runs/' + save_directory, 'final_agent'))
        print(f"💾 Final model saved successfully!")
        
        # Get training metrics and plot curves
        training_metrics = metrics_callback.get_metrics()
        validation_data = validation_callback.get_validation_data()
        
        print(f"\n📈 Plotting training curves...")
        plot_training_reward_curves(training_metrics, save_directory, args)
        
    else:
        print("⚠️  No training iterations specified (iters=0), skipping training...")
        training_metrics = {}
        validation_data = {'timesteps': [], 'rewards': [], 'best_reward': -float('inf')}
    
    return model, training_metrics, validation_data


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    parser = argparse.ArgumentParser()
    
    # Data and Environment arguments
    parser.add_argument("--lob_csv_path", type=str, default="src/db/AAPL_minute_orderbook_2019_01-07_combined.csv",
                        required=False, help='LOB CSV path')
    parser.add_argument("--ohlcv_csv_path", type=str, default="src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv",
                        required=False, help='OHLCV CSV path')
    parser.add_argument("--ohlcv_extended_csv_path", type=str, default="src/db/indicator/AAPL_with_indicators_v2.csv",
                        required=False, help='OHLCV Extended CSV path')
    parser.add_argument("--include_tech", type=bool, default=False,
                        required=False, help='Include technical indicators')
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

    # Agent selection
    parser.add_argument("--agent_type", type=str, choices=["PPO", "SAC"], default="PPO",
                        required=False, help="Type of RL agent to use: PPO or SAC")

    # GPU related arguments
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    
    # Training related arguments
    parser.add_argument("--iters", "-i", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=.00032)
    parser.add_argument("--validation_freq", type=int, default=10000,
                        help="Frequency of validation during training (in timesteps)")
    
    # PPO specific arguments
    parser.add_argument("--grad_clip", type=float, default=.5,
                        help="PPO: Gradient clipping")
    parser.add_argument("--gamma", type=float, default=.99,
                        help="Discount factor (both PPO and SAC)")
    parser.add_argument("--ent_coef", type=str, default=".0089",
                        help="PPO: Entropy coefficient (float) / SAC: 'auto' or float")
    
    # SAC specific arguments
    parser.add_argument("--tau", type=float, default=0.005,
                        help="SAC: Soft update coefficient for target networks")
    parser.add_argument("--target_update_interval", type=int, default=1,
                        help="SAC: Target network update interval")
    parser.add_argument("--target_entropy", type=str, default='auto',
                        help="SAC: Target entropy ('auto' or float)")
    parser.add_argument("--use_sde", action="store_true", default=False,
                        help="SAC: Use State Dependent Exploration")
    parser.add_argument("--sde_sample_freq", type=int, default=-1,
                        help="SAC: Sample frequency for SDE")
    parser.add_argument("--use_sde_at_warmup", action="store_true", default=False,
                        help="SAC: Use SDE at warmup")
    parser.add_argument("--buffer_size", type=int, default=1000000,
                        help="SAC: Replay buffer size")
    parser.add_argument("--learning_starts", type=int, default=100,
                        help="SAC: Learning starts after this many steps")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="SAC: Batch size for training")
    parser.add_argument("--train_freq", type=int, default=1,
                        help="SAC: Training frequency")
    parser.add_argument("--gradient_steps", type=int, default=1,
                        help="SAC: Gradient steps per update")

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help='Tag to add to the save directory')

    args = parser.parse_args()
    
    # Adjust default values based on agent type
    if args.agent_type == "PPO":
        # Convert ent_coef to float for PPO
        if args.ent_coef != 'auto':
            args.ent_coef = float(args.ent_coef)
    elif args.agent_type == "SAC":
        # Keep ent_coef as string for SAC ('auto' or convert to float)
        if args.ent_coef != 'auto' and args.ent_coef != '.0089':
            try:
                args.ent_coef = float(args.ent_coef)
            except ValueError:
                args.ent_coef = 'auto'
        elif args.ent_coef == '.0089':  # PPO default, change to SAC default
            args.ent_coef = 'auto'
    
    torch.use_deterministic_algorithms(True)

    main(args)