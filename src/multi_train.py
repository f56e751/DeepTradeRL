# multi_train.py
import subprocess
import itertools
import os

# 1) 실험에 사용할 파라미터 리스트 정의
ALGOS = ['ppo', 'sac']
REWARDS = ['LogPortfolioReturnReward']
# REWARDS = ['RealizedPnLReward', 'LogPortfolioReturnReward']
ACTIONS = ['ClippedActionStrategy', 'StrictActionStrategy']
CSV_PATHS = [
    'src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv',
    'src/db/src/db/AAPL_minute_ohlcv_orderbook_2019_01-07_combined.csv',
    # 필요에 따라 추가
]

# (필요하다면) 공통 고정 파라미터
COMMON_ARGS = [
    '--lookback', '9',
    '--lob_levels', '10',
    '--initial_cash', '100000',
    '--iters', '10000',
    # ...
]


def main():
    for csv_path, algo, reward, action in itertools.product(CSV_PATHS, ALGOS, REWARDS, ACTIONS):
        cmd = [
            'python', '-m', 'src.train',
            '--csv_path', csv_path,
            '--agent', algo,
            '--reward_strategy', reward,
            '--action_strategy', action,
            *COMMON_ARGS
        ]
        print('Running:', ' '.join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f'(!) Error on {csv_path}, {algo}, {reward}, {action}')
        else:
            print(f'[OK] {csv_path}, {algo}, {reward}, {action}')

if __name__ == '__main__':
    main()