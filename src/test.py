#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import pandas as pd
import torch

from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3
# train.py에 정의된 make_env_from_df 함수 사용
from .train import make_env_from_df
# 에이전트 평가를 위한 함수 import
from .agent import evaluate_model
from .data_handler import FeatureEngineer

# 알고리즘 이름과 클래스 매핑
ALGOS = {
    'ppo': PPO,
    'sac': SAC,
    'a2c': A2C,
    'ddpg': DDPG,
    'td3': TD3,
}

def main():
    # 명령행 인자 파서 설정
    parser = argparse.ArgumentParser(description="학습된 RL 트레이딩 에이전트를 테스트합니다.")
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='모델 파일 및 parameters.yaml이 저장된 디렉터리 경로'
    )
    args = parser.parse_args()

    # parameters.yaml 로드
    params_path = os.path.join(args.model_dir, 'parameters.yaml')
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    # 딕셔너리를 네임스페이스 객체로 변환하여 속성 접근 편의 제공
    from argparse import Namespace
    params_ns = Namespace(**params)

    # 디바이스 설정 (GPU/CPU)
    params_ns.device = 'cpu' if params_ns.no_gpu else f"cuda:{params_ns.which_gpu}"

    # 원본 CSV 읽기 및 전처리
    raw_df = pd.read_csv(params_ns.csv_path, parse_dates=['timestamp'])

    # FeatureEngineer 인스턴스 생성
    fe = FeatureEngineer(
        use_technical_indicator=params_ns.include_tech,
        tech_indicator_list=None,
        use_turbulence=False,
        user_defined_feature=False
    )
    # tic 컬럼이 없으면 기본값 설정
    if 'tic' not in raw_df.columns:
        raw_df['tic'] = 'AAPL'
    # 전처리 수행
    df_all = fe.preprocess_data(raw_df)

    # 학습/검증/테스트 분할
    n = len(df_all)
    n_train = int(n * params_ns.train_ratio)
    n_val   = int(n * params_ns.val_ratio)
    df_test = df_all.iloc[n_train + n_val:]

    # 테스트 환경 생성
    test_env = make_env_from_df(df_test, params_ns)

    # 학습된 모델 불러오기
    algo = params_ns.agent
    ModelClass = ALGOS[algo]
    # 모델 파일 경로: <algo>_final.zip
    model_path = os.path.join(args.model_dir, f"{algo}_final")
    model = ModelClass.load(model_path, env=test_env, device=params_ns.device)

    # 결과 저장용 폴더 생성 (model_dir 안에 'test_results' 디렉터리)
    result_dir = os.path.join(args.model_dir, 'test_results')
    os.makedirs(result_dir, exist_ok=True)

    # evaluate_model 호출 시, 내부에서 'runs/'가 붙으므로
    # save_directory에는 'runs/' 이후 부분(모델명/test_results)만 전달
    rel_save_dir = os.path.join(os.path.basename(args.model_dir), 'test_results')

    # 에이전트 평가 수행 (dataset_name 지정)
    evaluate_model(
        model,
        test_env,
        dataset_name="test",
        save_directory=rel_save_dir
    )

if __name__ == '__main__':
    main()
