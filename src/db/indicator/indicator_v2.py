# -*- coding: utf-8 -*-
"""indicator.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Tw0fdKpreYo5XToqfA19KxOq-42ATx4h
"""

import pandas as pd

# 기술 지표 계산 함수들 정의
def calculate_price_dispersion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    ma = df["close"].rolling(window=window).mean()
    dispersion = (df["close"] - ma) / ma * 100
    return dispersion

def add_price_dispersion(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df[f"dispersion{window}"] = calculate_price_dispersion(df, window)
    return df

def calculate_stochastic_slow(df: pd.DataFrame, k_window: int = 20, k_smooth: int = 5, d_smooth: int = 3):
    low_min = df["low"].rolling(window=k_window).min()
    high_max = df["high"].rolling(window=k_window).max()
    fast_k = (df["close"] - low_min) / (high_max - low_min) * 100
    slow_k = fast_k.rolling(window=k_smooth).mean()
    slow_d = slow_k.rolling(window=d_smooth).mean()
    return slow_k, slow_d

def add_stochastic_slow(df: pd.DataFrame, k_window: int = 20, k_smooth: int = 5, d_smooth: int = 3) -> pd.DataFrame:
    df["slowK"], df["slowD"] = calculate_stochastic_slow(df, k_window, k_smooth, d_smooth)
    return df

def calculate_moving_average(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].rolling(window=window).mean()

def add_moving_averages(df: pd.DataFrame, windows=(5, 20)) -> pd.DataFrame:
    for w in windows:
        df[f"MA{w}"] = calculate_moving_average(df, w)
    return df

# 데이터 로드
url = "https://raw.githubusercontent.com/yangmin-seok/YAI-CON_RL-HFT/refs/heads/main/src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv"
df = pd.read_csv(url)

# DateTime 형식 처리
df['timestamp'] = pd.to_datetime(df['timestamp'])

for index, row in df.iterrows():
    if pd.isna(row['open']) and index > 0:
        prev_close = df.at[index - 1, 'close']
        next_open = df.at[index + 1, 'open']

        df.at[index, 'open'] = prev_close

        if pd.isna(next_open):
          df.at[index, 'high'] = prev_close
          df.at[index, 'low'] = prev_close
          df.at[index, 'close'] = prev_close
        else:
          df.at[index, 'high'] = max(prev_close, next_open)
          df.at[index, 'low'] = min(prev_close, next_open)
          df.at[index, 'close'] = next_open

df

# 지표 추가
df = add_price_dispersion(df, window=20)
df = add_stochastic_slow(df, k_window=20, k_smooth=5, d_smooth=3)
df = add_moving_averages(df, windows=(5, 20))

# 결과 저장
output_path = "AAPL_with_indicators_v2.csv"
df.to_csv(output_path, index=False)
print(f"저장 완료: {output_path}")

