# train.py
import os
import argparse

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml

from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.logger import configure

from .trading_env import UnifiedTradingEnv
from .trading_env import RealizedPnLReward, LogPortfolioReturnReward, CombinedReward, ScaledRealizedPnLReward
from .trading_env import ClippedActionStrategy, PercentPortfolioStrategy, StrictActionStrategy, FloatClippedActionStrategy
from .trading_env import NormalizationWrapper, PortfolioScalingWrapper
from .infrastructure import TrainingStatusCallback, ValidationCallback # , TrainingMetricsCallback, ValidationCallback
from .agent import TrainingMetricsCallback, plot_training_reward_curves, evaluate_model
from .data_handler import FeatureEngineer  # 혹은 DataHandlerBase 구현체
from .model_factory import ModelFactory

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

# parser로 전달하는 reward, action 방식 매핑
reward_map = {
    "RealizedPnLReward": RealizedPnLReward,
    "LogPortfolioReturnReward": LogPortfolioReturnReward,
    "CombinedReward": CombinedReward,
    "ScaledRealizedPnLReward": ScaledRealizedPnLReward,
}
action_map = {
    "ClippedActionStrategy": ClippedActionStrategy,
    "PercentPortfolioStrategy": PercentPortfolioStrategy,
    "StrictActionStrategy": StrictActionStrategy,
    "FloatClippedActionStrategy": FloatClippedActionStrategy,
}

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="RL 기반 코인/주식 트레이딩 에이전트 학습 스크립트")

    # ======================
    # 1) 데이터 & 환경 설정
    # ======================
    parser.add_argument(
        "--csv_path", type=str,
        # default="src/db/AAPL_minute_ohlcv_orderbook_2019_01-07_combined.csv",
        default="src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv",
        help="CSV 파일 경로"
    )
    parser.add_argument(
        "--include_tech", action="store_true",
        help="기술 지표 포함 여부"
    )
    parser.add_argument(
        "--initial_cash", type=float, default=100000.0,
        help="초기 자본"
    )
    parser.add_argument(
        "--transaction_fee", type=float, default=0.0023,
        help="거래 수수료 비율"
    )
    parser.add_argument(
        "--lookback", type=int, default=9,
        help="관찰 히스토리 길이 (lookback window)"
    )
    parser.add_argument(
        "--lob_levels", type=int, default=10,
        help="Order Book 레벨 수"
    )
    parser.add_argument(
        "--h_max", type=int, default=250,
        help="한 번에 최대 거래 수량"
    )
    parser.add_argument(
        "--hold_threshold", type=float, default=0.2,
        help="보유 임계치 비율"
    )
    parser.add_argument(
        "--include_portfolio_value", action="store_true",
        help="관측값에 포트폴리오 가치 포함 여부"
    )
    parser.add_argument(
        "--include_cash", action="store_true",
        help="관측값에 보유 현금 포함 여부"
    )

    parser.add_argument(
        "--wrapper", type=str, choices=["normalization", "portfolio_scaling"], default="normalization",
        help="사용할 환경 Wrapper 종류"
    )

    # ======================
    # 2) 데이터 분할 비율
    # ======================
    parser.add_argument(
        "--train_ratio", type=float, default=0.7,
        help="학습용 데이터 비율"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="검증용 데이터 비율"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="테스트용 데이터 비율"
    )

    # ======================
    # 3) 에이전트 & 정책 설정
    # ======================
    parser.add_argument(
        "--agent", choices=["ppo","sac","a2c","ddpg","td3"],
        default="ppo", help="사용할 RL 알고리즘"
    )
    parser.add_argument(
        "--policy", type=str, default="MlpPolicy",
        help="정책 네트워크 클래스 이름 (e.g. MlpPolicy, CnnPolicy)"
    )

    # ======================
    # 4) 공통 학습 하이퍼파라미터
    # ======================
    parser.add_argument(
        "--iters", type=int, default=1000000,
        help="총 학습 스텝 수"
    )
    parser.add_argument(
        "--validation_freq", type=int, default=100000,
        help="검증(Validation) 주기 (스텝 단위)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드"
    )

    # ======================
    # 5) 디바이스 설정
    # ======================
    parser.add_argument(
        "--no_gpu", action="store_true",
        help="GPU 사용하지 않음"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="사용할 GPU ID"
    )
    parser.add_argument(
        "--which_gpu", type=int, default=0,
        help="사용할 GPU ID (cuda:INDEX)"
    )

    # =================================
    # 6) PPO 전용 하이퍼파라미터
    # =================================
    parser.add_argument("--ppo_gamma",           type=float, default=0.95,    help="PPO 할인율")
    parser.add_argument("--ppo_lr",              type=float, default=0.0001,    help="PPO 학습률")
    parser.add_argument("--ppo_ent_coef",        type=float, default=0.002,     help="PPO 엔트로피 계수")
    parser.add_argument("--ppo_vf_coef",         type=float, default=0.5,     help="PPO 가치 함수 계수")
    parser.add_argument("--ppo_max_grad_norm",   type=float, default=0.7,     help="PPO 최대 그래디언트 노름")
    parser.add_argument("--ppo_net_arch_pi",     nargs="+", type=int, default=[64,64], help="PPO 정책 네트워크 구조")
    parser.add_argument("--ppo_net_arch_vf",     nargs="+", type=int, default=[64,64], help="PPO 가치 네트워크 구조")

    # =================================
    # 7) SAC 전용 하이퍼파라미터
    # =================================
    parser.add_argument("--sac_gamma",               type=float, default=0.99,      help="SAC 할인율")
    parser.add_argument("--sac_tau",                 type=float, default=0.005,     help="타겟 네트워크 부드럽게 갱신 비율")
    parser.add_argument("--sac_ent_coef",            type=str,   default="auto",    help="SAC 엔트로피 계수 ('auto' 또는 float)")
    parser.add_argument("--sac_target_update_interval", type=int, default=1,        help="타겟 네트워크 업데이트 빈도")
    parser.add_argument("--sac_target_entropy",      type=str,   default="auto",    help="목표 엔트로피 ('auto' 또는 float)")
    parser.add_argument("--sac_use_sde",             action="store_true",            help="SAC에 SDE 사용 여부")
    parser.add_argument("--sac_sde_sample_freq",     type=int,   default=-1,        help="SDE 샘플링 빈도")
    parser.add_argument("--sac_use_sde_at_warmup",   action="store_true",            help="웜업 시기에 SDE 사용 여부")
    parser.add_argument("--sac_lr",                  type=float, default=3e-4,      help="SAC 학습률")
    parser.add_argument("--sac_buffer_size",         type=int,   default=1_000_000, help="리플레이 버퍼 크기")
    parser.add_argument("--sac_learning_starts",     type=int,   default=100,       help="학습 시작 스텝")
    parser.add_argument("--sac_batch_size",          type=int,   default=256,       help="배치 크기")
    parser.add_argument("--sac_train_freq",          type=int,   default=1,         help="학습 빈도")
    parser.add_argument("--sac_gradient_steps",      type=int,   default=1,         help="그레이디언트 스텝 수")
    parser.add_argument("--sac_net_arch_pi",         nargs="+", type=int, default=[64,64], help="SAC 정책 네트워크 구조")
    parser.add_argument("--sac_net_arch_qf",         nargs="+", type=int, default=[64,64], help="SAC Q 함수 네트워크 구조")

    # =================================
    # 8) A2C 전용 하이퍼파라미터
    # =================================
    parser.add_argument("--a2c_gamma",       type=float, default=0.99,    help="A2C 할인율")
    parser.add_argument("--a2c_lr",          type=float, default=7e-4,    help="A2C 학습률")
    parser.add_argument("--a2c_ent_coef",    type=float, default=0.0,     help="A2C 엔트로피 계수")
    parser.add_argument("--a2c_vf_coef",     type=float, default=0.5,     help="A2C 가치 함수 계수")
    parser.add_argument("--a2c_n_steps",     type=int,   default=5,       help="A2C n_steps")
    parser.add_argument("--a2c_net_arch",    nargs="+", type=int, default=[64,64], help="A2C 네트워크 구조")

    # =================================
    # 9) DDPG 전용 하이퍼파라미터
    # =================================
    parser.add_argument("--ddpg_gamma",         type=float, default=0.99,      help="DDPG 할인율")
    parser.add_argument("--ddpg_lr",            type=float, default=1e-3,      help="DDPG 학습률")
    parser.add_argument("--ddpg_tau",           type=float, default=0.005,     help="타겟 네트워크 부드럽게 갱신 비율")
    parser.add_argument("--ddpg_batch_size",    type=int,   default=256,       help="배치 크기")
    parser.add_argument("--ddpg_buffer_size",   type=int,   default=1_000_000, help="리플레이 버퍼 크기")
    parser.add_argument("--ddpg_learning_starts", type=int, default=100,       help="학습 시작 스텝")
    parser.add_argument("--ddpg_train_freq",    type=int,   default=1,         help="학습 빈도")
    parser.add_argument("--ddpg_gradient_steps", type=int,  default=1,         help="그레이디언트 스텝 수")
    parser.add_argument("--ddpg_net_arch",      nargs="+", type=int, default=[400,300], help="DDPG 네트워크 구조")

    # =================================
    # 10) TD3 전용 하이퍼파라미터
    # =================================
    parser.add_argument("--td3_gamma",         type=float, default=0.99,      help="TD3 할인율")
    parser.add_argument("--td3_lr",            type=float, default=1e-3,      help="TD3 학습률")
    parser.add_argument("--td3_tau",           type=float, default=0.005,     help="타겟 네트워크 부드럽게 갱신 비율")
    parser.add_argument("--td3_batch_size",    type=int,   default=256,       help="배치 크기")
    parser.add_argument("--td3_buffer_size",   type=int,   default=1_000_000, help="리플레이 버퍼 크기")
    parser.add_argument("--td3_learning_starts", type=int, default=100,       help="학습 시작 스텝")
    parser.add_argument("--td3_train_freq",    type=int,   default=1,         help="학습 빈도")
    parser.add_argument("--td3_gradient_steps", type=int,  default=1,         help="그레이디언트 스텝 수")
    parser.add_argument("--td3_net_arch_pi",   nargs="+", type=int, default=[400,300], help="TD3 정책 네트워크 구조")
    parser.add_argument("--td3_net_arch_qf",   nargs="+", type=int, default=[400,300], help="TD3 Q 함수 네트워크 구조")

    # =================================
    # 11) Action, Reward 방식
    # =================================
    parser.add_argument(
        "--reward_strategy",
        choices=["RealizedPnLReward", "LogPortfolioReturnReward", "CombinedReward", "ScaledRealizedPnLReward"],
        default="RealizedPnLReward",
        help="사용할 리워드 전략 클래스 이름"
    )
    parser.add_argument(
        "--action_strategy",
        choices=["ClippedActionStrategy", "PercentPortfolioStrategy", "StrictActionStrategy", "FloatClippedActionStrategy"],
        default="ClippedActionStrategy",
        help="사용할 액션 전략 클래스 이름"
    )

    # ======================
    # 12) price hint
    # ======================
    parser.add_argument(
        "--price_hint", action="store_true",
        help="관측값에 다음 스텝의 가격 힌트 포함 여부 (디버깅용)"
    )

    return parser.parse_args()

def make_env_from_df(df: pd.DataFrame, args) -> UnifiedTradingEnv:
    "Create a UnifiedTradingEnv from a pre-split DataFrame."
    print("==========================")
    print(df)
    return UnifiedTradingEnv(
        df=df,
        reward_strategy=reward_map[args.reward_strategy],
        action_strategy=action_map[args.action_strategy],
        initial_cash=args.initial_cash,
        transaction_fee=args.transaction_fee,
        lookback=args.lookback,
        lob_levels=args.lob_levels,
        h_max=args.h_max,
        hold_threshold=args.hold_threshold,
        include_ohlcv=True,
        include_tech=args.include_tech,
        include_pnl=True,
        include_spread=False,
        include_position = True,
        include_orderbook = False,
        include_portfolio_value=args.include_portfolio_value,
        include_cash=args.include_cash,
        tech_dim=len(df.columns) - 5,  # or however you compute tech_dim
        price_hint=args.price_hint
    )

# def make_env(args):
#     # TODO 이 부분에서 바로 csv를 읽어올지 아니면 여러 csv를 합칠지 결정하기
#     df_ohlcv = pd.read_csv(args.ohlcv_csv_path, parse_dates=['timestamp'])
#     fe = FeatureEngineer(
#         use_technical_indicator=args.include_tech,
#         tech_indicator_list=None,
#         use_turbulence=False,
#         user_defined_feature=False
#     )
#     df = fe.preprocess_data(df_ohlcv)


#     env = UnifiedTradingEnv(
#         df=df,
#         reward_strategy=RealizedPnLReward,
#         action_strategy=ClippedActionStrategy,
#         initial_cash=args.initial_cash,
#         transaction_fee=args.transaction_fee,
#         lookback=args.lookback,
#         lob_levels=args.lob_levels,
#         h_max=args.h_max,
#         hold_threshold=args.hold_threshold,
#         include_ohlcv=True,
#         include_tech=args.include_tech,
#         include_pnl=True,
#         include_spread=False,
#         tech_dim=len(fe.tech_indicator_list or [])
#     )
#     return env


def test_df(args):
    raw_df = pd.read_csv(args.csv_path, parse_dates=['timestamp'])
    fe     = FeatureEngineer(use_technical_indicator=args.include_tech,
                             tech_indicator_list=None,
                             use_turbulence=False,
                             user_defined_feature=False)
    
    # TODO 이 부분 수정하기
    if "tic" not in raw_df.columns:
        raw_df["tic"] = "AAPL"
    df_all = fe.preprocess_data(raw_df)

    print(df_all)


def main(args):
    # 0) sanity‐check split ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    assert abs(total - 1.0) < 1e-6, "train+val+test ratios must sum to 1.0"

    # GPU 세팅
    device = 'cpu' if args.no_gpu else f'cuda:{args.which_gpu}'
    args.device = device
    torch.use_deterministic_algorithms(True)

    # # agent_type 소문자로 바꿔서 ModelFactory에 넘김
    args.algo = args.agent

    # 1) 학습/검증 환경
    # TODO 이 부분에서 바로 csv를 읽어올지 아니면 여러 csv를 합칠지 결정하기
    raw_df = pd.read_csv(args.csv_path, parse_dates=['timestamp'])
    fe     = FeatureEngineer(use_technical_indicator=args.include_tech,
                             tech_indicator_list=None,
                             use_turbulence=False,
                             user_defined_feature=False)
    
    # TODO 이 부분 수정하기
    if "tic" not in raw_df.columns:
        raw_df["tic"] = "AAPL"
    df_all = fe.preprocess_data(raw_df)

    n = len(df_all)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    df_train = df_all.iloc[:n_train]
    df_val   = df_all.iloc[n_train : n_train + n_val]
    df_test  = df_all.iloc[n_train + n_val :]

    # now instantiate three envs from those splits
    if args.wrapper == "portfolio_scaling":
        if not args.include_portfolio_value:
            print("경고: PortfolioScalingWrapper는 --include_portfolio_value 플래그가 필요합니다. 해당 플래그를 활성화합니다.")
            args.include_portfolio_value = True
        WrapperClass = PortfolioScalingWrapper
    else:
        WrapperClass = NormalizationWrapper

    train_env = WrapperClass(make_env_from_df(df_train, args))
    eval_env  = WrapperClass(make_env_from_df(df_val,   args))
    test_env  = WrapperClass(make_env_from_df(df_test,  args))


    # 2) 로그 디렉토리
    time_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    exp_name   = f"{args.algo}_{time_str}"
    save_dir   = os.path.join(RUNS_DIR, exp_name)
    # save_dir = os.path.join(RUNS_DIR, f"{args.algo}_{time_str}")
    os.makedirs(save_dir, exist_ok=True)

    # —— 학습 파라미터 YAML로 저장
    with open(os.path.join(save_dir, 'parameters.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    tb_log = os.path.join(save_dir, "tensorboard")
    logger = configure(save_dir, ["stdout","csv","tensorboard"])

    # 3) ModelFactory로 모델 생성
    model = ModelFactory.create(
        args=args,
        env=train_env,
        tensorboard_log=tb_log,
        logger=logger
    )

    # 4) 콜백
    status_cb  = TrainingStatusCallback(verbose=1)
    metrics_cb = TrainingMetricsCallback(verbose=0)
    val_cb     = ValidationCallback(
        val_env=eval_env,
        eval_freq=args.validation_freq,
        n_eval_episodes=1,
        save_directory=exp_name,
        verbose=1
    )
    callbacks = CallbackList([status_cb, metrics_cb, val_cb])

    # 5) 학습
    model.learn(
        total_timesteps=args.iters,
        callback=callbacks
    )

    # 6) 저장
    final_path = os.path.join(save_dir, f"{args.algo}_final")
    model.save(final_path)
    print(f"▶️ Done. Model saved to {final_path}")

    # 7) 학습 후 로그 저장 및 리워드 곡선 플롯
    # 에피소드 단위 메트릭 플롯
    episode_metrics = metrics_cb.get_episode_metrics()
    plot_training_reward_curves(episode_metrics, exp_name, args)

    # 스텝 단위 상세 로그 저장
    step_metrics = metrics_cb.get_step_metrics()
    step_log_df = pd.DataFrame(step_metrics)
    step_log_path = os.path.join(RUNS_DIR, exp_name, 'training_step_log.csv')
    step_log_df.to_csv(step_log_path, index=False)
    print(f"💾 Detailed step-wise training log saved to '{step_log_path}'")


    # 8) 학습 후 test 환경에서 모델 평가 및 결과 로깅
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    results = evaluate_model(model, test_env, save_directory=exp_name)

    # Clean and log the results to a YAML file
    if results.get('financial_metrics') and results['financial_metrics'] is not None:
        results['financial_metrics'].pop('portfolio_values', None)
        results['financial_metrics'].pop('returns', None)
        results['financial_metrics'].pop('drawdown', None)
        results['financial_metrics'].pop('step_rewards', None)
    if results.get('action_stats'):
        results['action_stats'].pop('raw_actions', None)
        results['action_stats'].pop('action_types', None)
    results.pop('step_rewards', None)
    results.pop('cumulative_rewards', None)
    results.pop('portfolio_values', None)
    
    results_log_path = os.path.join(RUNS_DIR, exp_name, 'test_evaluation_results.yaml')
    with open(results_log_path, 'w') as file:
        yaml.dump(results, file)
    print(f"💾 Test evaluation results saved to '{results_log_path}'")


if __name__ == '__main__':
    # Set environment variable for deterministic CuBLAS operations, as required by PyTorch >= 1.8 and CUDA >= 10.2
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    args = parse_args()
    torch.use_deterministic_algorithms(True)
    main(args)
    # test_df(args)
