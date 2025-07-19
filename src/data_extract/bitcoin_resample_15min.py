import pandas as pd

# --- 설정 ---
input_file_path = 'src/db/BTC_USDT_1m_all_history_data.csv' # 입력 1분봉 CSV 파일 경로
output_file_name = 'src/db/BTC_USDT_15m_all_history_data.csv'     # 출력 15분봉 CSV 파일 이름
# --------------

print(f"'{input_file_path}' 파일에서 1분봉 데이터를 로드합니다.")

try:
    # CSV 파일 로드
    # 'timestamp' 컬럼을 datetime 형식으로 파싱하고 인덱스로 사용합니다.
    df = pd.read_csv(input_file_path, parse_dates=['timestamp'], index_col='timestamp')
    
    # 데이터가 비어있는지 확인
    if df.empty:
        print("오류: 로드된 데이터가 비어있습니다. 파일 경로를 확인하거나 데이터가 유효한지 확인하세요.")
        exit()

    print(f"총 {len(df)}개의 1분봉 데이터가 로드되었습니다. 기간: {df.index.min()} ~ {df.index.max()}")

    # 15분봉으로 리샘플링
    # 'OHLC' (Open, High, Low, Close) 데이터와 'Volume'을 재계산합니다.
    # .first() : 15분 봉의 시작 가격
    # .max()   : 15분 동안의 최고 가격
    # .min()   : 15분 동안의 최저 가격
    # .last()  : 15분 봉의 마감 가격
    # .sum()   : 15분 동안의 총 거래량
    
    df_15min = df['open'].resample('15min').first().to_frame()
    df_15min['high'] = df['high'].resample('15min').max()
    df_15min['low'] = df['low'].resample('15min').min()
    df_15min['close'] = df['close'].resample('15min').last()
    df_15min['volume'] = df['volume'].resample('15min').sum()

    # NaN 값 처리 (거래가 없는 15분 봉)
    # 예를 들어, 거래량이 0인 경우 해당 봉은 NaN이 될 수 있습니다.
    # 여기서는 이전 유효한 값으로 채우거나 (ffill), 아예 제거하는 방법을 고려할 수 있습니다.
    # 일반적으로 거래량이 없는 봉은 그냥 놔두거나 0으로 채웁니다.
    df_15min.fillna(method='ffill', inplace=True) # 이전 값으로 채우기
    df_15min.fillna(0, inplace=True) # 남은 NaN은 0으로 채우기 (예: 맨 앞부분)

    # 리샘플링으로 인해 생성된 불필요한 행(예: 데이터 시작 전) 제거
    df_15min.dropna(inplace=True)

    # 결과 확인
    print(f"15분봉 데이터 생성 완료. 총 {len(df_15min)}개의 15분봉 데이터가 생성되었습니다.")
    print("15분봉 데이터의 일부:")
    print(df_15min.head())
    print(df_15min.tail())

    # CSV 파일로 저장
    output_file_path = f"{output_file_name}" # 출력 경로 설정
    df_15min.to_csv(output_file_path, encoding='utf-8-sig')

    print(f"✅ 모든 작업 완료! 15분봉 파일이 '{output_file_path}'으로 저장되었습니다.")

except FileNotFoundError:
    print(f"오류: 파일 '{input_file_path}'을(를) 찾을 수 없습니다. 경로를 다시 확인해주세요.")
except Exception as e:
    print(f"데이터 처리 중 오류 발생: {e}")