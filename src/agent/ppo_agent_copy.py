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
import warnings
warnings.filterwarnings('ignore')

from ..infrastructure import init_gpu, TrainingStatusCallback
from ..trading_env import MinutelyOrderbookOHLCVEnv, observation
from ..data_handler import Sc203Handler, Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler, merge_lob_and_ohlcv, DataSplitter


# # Add the src directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from src.infrastructure.pytorch_util import init_gpu
# from src.env.minutely_orderbook_ohlcv_env import MinutelyOrderbookOHLCVEnv
# from src.data_handler.data_handler import Sc203Handler
# from src.data_handler.data_handler import Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler
# from src.env.observation import Observation, InputType
# from src.data_handler.csv_processor import merge_lob_and_ohlcv, DataSplitter
# from src.infrastructure.callback import TrainingStatusCallback

# Import visualization and utility functions
from .util import (
    TrainingMetricsCallback,
    calculate_financial_metrics,
    plot_comprehensive_financial_analysis,
    plot_training_reward_curves,
    plot_reward_curves,
    compare_performance,
    evaluate_model
)


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
    val_rewards = evaluate_model(model, env_val, num_episodes=1, dataset_name="Validation", initial_cash=args.initial_cash)

    #model = PPO.load("runs/Best_model_v1/agent", env = env_train)
    
    print("\n🔍 Evaluating on Test Set...")
    test_rewards = evaluate_model(model, env_test, num_episodes=1, dataset_name="Test", initial_cash=args.initial_cash)
    #test_rewards = evaluate_model(model, env_train, num_episodes=1, dataset_name="Test")
    
    # Compare validation and test performance
    #compare_performance(val_rewards, test_rewards, save_directory)
    
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
        plot_training_reward_curves(training_metrics, save_directory, args)
        
    else:
        print("⚠️  No training iterations specified (iters=0), skipping training...")
        training_metrics = {}
    
    return model, training_metrics


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
    parser.add_argument("--h_max", type=int, default=250,
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