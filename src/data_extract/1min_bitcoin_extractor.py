import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# --- 설정 ---
symbol = 'BTC/USDT'      # 원하는 암호화폐 심볼
timeframe = '1m'         # 1분봉
exchange_name = 'binance'# 거래소 이름
# --------------

print("스크립트를 시작합니다. 1년치 데이터를 다운로드합니다.")

# 1. 거래소 객체 생성
try:
    exchange = getattr(ccxt, exchange_name)()
except AttributeError:
    print(f"오류: '{exchange_name}' 거래소를 찾을 수 없습니다.")
    exit()

# 2. 1년 전 날짜 계산 (타임스탬프 ms 단위로)
since = exchange.parse8601((datetime.now() - timedelta(days=365)).isoformat())

# 3. 모든 데이터를 저장할 리스트 생성
all_ohlcvs = []

# 4. 1년치 데이터를 모두 가져올 때까지 반복
while since < exchange.milliseconds():
    try:
        print(f"{pd.to_datetime(since, unit='ms')} 부터 데이터 가져오는 중...")
        
        # fetch_ohlcv로 데이터 가져오기
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        
        if not ohlcvs:
            break
            
        all_ohlcvs.extend(ohlcvs)
        since = ohlcvs[-1][0] + (60 * 1000) 
        time.sleep(exchange.rateLimit / 1000)

    except ccxt.NetworkError as e:
        print(f"네트워크 오류 발생: {e}. 5초 후 재시도합니다.")
        time.sleep(5)
    except ccxt.ExchangeError as e:
        print(f"거래소 오류 발생: {e}. 10초 후 재시도합니다.")
        time.sleep(10)
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
        break

print("데이터 다운로드 완료. CSV 파일로 변환을 시작합니다...")

# 5. 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# 6. 타임스탬프를 사람이 읽을 수 있는 날짜 형식으로 변환
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 7. 중복 데이터 제거
df.drop_duplicates(subset='timestamp', inplace=True)

# 8. CSV 파일(.csv)로 저장  <- 이 부분이 변경되었습니다!
file_name = f"{symbol.replace('/', '_')}_{timeframe}_1year_data.csv"
df.to_csv(file_name, index=False, encoding='utf-8-sig')

print(f"✅ 모든 작업 완료! 파일이 '{file_name}'으로 저장되었습니다.")
print(f"총 {len(df)}개의 데이터가 저장되었습니다.")