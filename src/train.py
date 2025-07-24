# train.py
import os
import argparse

import gym
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.logger import configure

from .trading_env import UnifiedTradingEnv
from .trading_env import RealizedPnLReward, LogPortfolioReturnReward, CombinedReward
from .trading_env import ClippedActionStrategy, PercentPortfolioStrategy
from .infrastructure import TrainingStatusCallback # , TrainingMetricsCallback, ValidationCallback
from .agent import TrainingMetricsCallback
from .data_handler import FeatureEngineer  # 혹은 DataHandlerBase 구현체

# 선택 가능한 알고리즘 매핑
ALGOS = {
    'ppo': PPO,
    'sac': SAC,
    'a2c': A2C,
    'ddpg': DDPG,
    'td3': TD3,
}

# 기본 파라미터
DEFAULT_TOTAL_TIMESTEPS = 1_000_000
RUNS_DIR = "runs"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=ALGOS.keys(), default='ppo',
                   help="RL algorithm")
    p.add_argument('--policy', type=str, default='MlpPolicy',
                   help="Policy network architecture")
    p.add_argument('--env-csv', type=str, required=True,
                   help="OHLCV+tech indicators CSV 파일 경로")
    p.add_argument('--total-timesteps', type=int,
                   default=DEFAULT_TOTAL_TIMESTEPS)
    p.add_argument('--tensorboard-log', type=str,
                   default=os.path.join(RUNS_DIR, "tb"))
    p.add_argument('--save-dir', type=str,
                   default=os.path.join(RUNS_DIR, "{algo}_{time}"))
    p.add_argument('--eval-freq', type=int, default=10_000,
                   help="몇 스텝마다 Validation 실행할지")
    p.add_argument('--n-eval-episodes', type=int, default=3)
    return p.parse_args()


def make_env(args):
    # 1) 데이터 로드 및 전처리
    raw_df = pd.read_csv(args.env_csv, parse_dates=['timestamp'])
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=None,
        use_turbulence=False,
        user_defined_feature=False
    )
    df = fe.preprocess_data(raw_df)

    # 2) Env 생성
    env = UnifiedTradingEnv(
        df=df,
        reward_strategy=RealizedPnLReward,           # 원하는 RewardStrategy
        action_strategy=ClippedActionStrategy,      # 또는 ClippedActionStrategy 등
        initial_cash=100_000,
        transaction_fee=0.0023,
        lookback=9,
        lob_levels=0,
        h_max=200,
        hold_threshold=0.1,
        include_ohlcv=True,
        include_tech=True,
        include_pnl=True,
        include_spread=False,
        tech_dim=len(fe.tech_indicator_list or [])
    )
    return env


def main():
    args = parse_args()
    AlgoClass = ALGOS[args.algo]

    # 1) 학습 및 평가 환경
    train_env = make_env(args)
    # Validation을 위해 같은 파라미터의 별도 env를 하나 더 생성
    eval_env = make_env(args)

    # 2) 로그 디렉토리 설정
    time_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir.format(algo=args.algo, time=time_str)
    os.makedirs(save_dir, exist_ok=True)

    # SB3 내부 로거를 텐서보드까지 쓰도록 구성
    new_logger = configure(save_dir, ["stdout", "csv", "tensorboard"])

    # 3) 모델 초기화
    model = AlgoClass(
        policy=args.policy,
        env=train_env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        # 필요 시 추가 하이퍼파라미터 예: learning_rate=3e-4
    )
    model.set_logger(new_logger)

    # 4) 콜백 준비
    cb_list = CallbackList([
        TrainingStatusCallback(verbose=1),
        TrainingMetricsCallback(verbose=0),
        # EvalCallback 대신 직접 만든 ValidationCallback 사용
        ValidationCallback(
            val_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_directory=save_dir,
            verbose=1
        )
    ])

    # 5) 학습 실행
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=cb_list
    )

    # 6) 최종 모델 저장
    final_path = os.path.join(save_dir, f"{args.algo}_final")
    model.save(final_path)
    print(f"▶️ Training finished. Model saved to {final_path}")


if __name__ == '__main__':
    main()
