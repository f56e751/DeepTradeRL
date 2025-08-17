import pandas as pd

# 파일 경로
ohlcv_path = 'src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv'    # timestamp,open,high,low,close,volume
lob_path   = 'src/db/AAPL_minute_orderbook_2019_01-07_combined.csv'  # timestamp,timestamp,bid_px_00,...

# 1) OHLCV 데이터 읽기
df_ohlcv = pd.read_csv(ohlcv_path, parse_dates=['timestamp'])

# 2) Orderbook 데이터 읽기
df_lob = pd.read_csv(lob_path)

# 중복된 timestamp 컬럼 처리: 첫 번째만 'timestamp'로, 두 번째는 삭제
first_col, second_col = df_lob.columns[:2]
df_lob = df_lob.rename(columns={first_col: 'timestamp'}).drop(columns=[second_col])
df_lob['timestamp'] = pd.to_datetime(df_lob['timestamp'])

# 3) bid_ct, ask_ct 컬럼 제외
to_drop = [col for col in df_lob.columns if 'bid_ct' in col or 'ask_ct' in col]
df_lob = df_lob.drop(columns=to_drop)

# 4) 병합 (inner join: 양쪽에 모두 있는 타임스탬프만)
df_merged = pd.merge(df_ohlcv, df_lob, on='timestamp', how='inner')

# 5) 결과 저장
file_path = 'src/db/AAPL_minute_ohlcv_orderbook_2019_01-07_combined.csv'
df_merged.to_csv(file_path, index=False)

print(f"병합 완료: {len(df_merged)}개 행 → {file_path}")
