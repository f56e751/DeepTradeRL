import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# --- 설정 ---
symbol = 'BTC/USDT'      # 원하는 암호화폐 심볼
timeframe = '1m'         # 1분봉
exchange_name = 'binance'# 거래소 이름
# --------------

print(f"스크립트를 시작합니다. {symbol} {timeframe} 전체 역사적 데이터를 다운로드합니다.")

# 1. 거래소 객체 생성
try:
    exchange = getattr(ccxt, exchange_name)()
    exchange.load_markets() # 시장 정보를 로드하여 심볼 유효성 검사
except AttributeError:
    print(f"오류: '{exchange_name}' 거래소를 찾을 수 없습니다.")
    exit()
except Exception as e:
    print(f"거래소 초기화 중 오류 발생: {e}")
    exit()

# 심볼 유효성 검사
if symbol not in exchange.symbols:
    print(f"오류: '{symbol}' 심볼이 '{exchange_name}' 거래소에 존재하지 않습니다.")
    exit()

# 2. 데이터 수집 시작점 설정
initial_since = exchange.parse8601('2017-07-17T00:00:00Z') # 바이낸스 BTC/USDT 상장일 부근
print(f"데이터 수집 시작점: {pd.to_datetime(initial_since, unit='ms')}")

# 현재 시간을 가져와서 데이터 수집의 종료점으로 사용합니다.
current_timestamp = exchange.milliseconds()

# 3. 모든 데이터를 저장할 리스트 생성
all_ohlcvs = []
since = initial_since # 반복문의 시작점을 찾은 초기 타임스탬프로 설정

# 4. 모든 데이터를 가져올 때까지 반복
while since < current_timestamp: # 현재 시간까지 데이터를 가져옵니다.
    try:
        print(f"{pd.to_datetime(since, unit='ms')} 부터 데이터 가져오는 중...")
        
        # fetch_ohlcv로 데이터 가져오기 (limit은 1000이 대부분의 거래소에서 최대)
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        
        if not ohlcvs:
            # 더 이상 가져올 데이터가 없거나, 현재 timestamp에 도달한 경우
            print("더 이상 가져올 데이터가 없거나, 현재 시점에 도달했습니다. 루프를 종료합니다.")
            break 
            
        all_ohlcvs.extend(ohlcvs)
        
        # 다음 fetch를 위한 since 값 업데이트:
        # 가져온 데이터 중 가장 마지막 봉의 타임스탬프에 1분(60초 * 1000ms = 60000ms)을 더합니다.
        # 이렇게 하면 다음 호출 시 중복 없이 다음 봉부터 데이터를 가져올 수 있습니다.
        since = ohlcvs[-1][0] + (60 * 1000) 
        
        # 거래소 API 호출 제한 (Rate Limit) 준수
        # Binance의 경우 1분봉은 rateLimit이 20ms 정도로 매우 빠를 수 있으나,
        # 너무 짧으면 IP 차단 가능성이 있으므로 최소 0.1초 정도의 sleep을 권장합니다.
        time.sleep(max(exchange.rateLimit / 1000, 0.1)) # 최소 0.1초 대기
        
        # 진행 상황 출력
        print(f"현재까지 총 {len(all_ohlcvs)}개의 데이터 포인트 수집 ({pd.to_datetime(ohlcvs[-1][0], unit='ms')} 까지)")

    except ccxt.NetworkError as e:
        print(f"네트워크 오류 발생: {e}. 5초 후 재시도합니다.")
        time.sleep(5)
    except ccxt.ExchangeError as e:
        print(f"거래소 오류 발생: {e}. 10초 후 재시도합니다.")
        # 만약 특정 시작점에서 계속 오류가 난다면, since 값을 조정해야 할 수도 있습니다.
        # (예: since = since + (60 * 1000) # 현재 실패한 봉을 건너뛰고 다음 봉부터 시도)
        time.sleep(10)
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
        break # 예상치 못한 오류 발생 시 루프 종료

print("데이터 다운로드 완료. CSV 파일로 변환을 시작합니다...")

# 5. 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# 6. 타임스탬프를 사람이 읽을 수 있는 날짜 형식으로 변환
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 7. 중복 데이터 제거
df.drop_duplicates(subset='timestamp', inplace=True)

# timestamp를 기준으로 오름차순 정렬 (가장 오래된 데이터부터)
df.sort_values(by='timestamp', inplace=True)
df.reset_index(drop=True, inplace=True) # 인덱스 재설정

# 8. CSV 파일(.csv)로 저장
file_name = f"src/db/{symbol.replace('/', '_')}_{timeframe}_all_history_data.csv"
df.to_csv(file_name, index=False, encoding='utf-8-sig')

print(f"✅ 모든 작업 완료! 파일이 '{file_name}'으로 저장되었습니다.")
print(f"총 {len(df)}개의 데이터가 저장되었습니다.")