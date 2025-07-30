# main.py
import queue
import time
import os

# kline_client.py 파일에서 필요한 클래스와 함수를 가져옵니다.
from src.backend.get_raw_data import BinanceKlineClient, get_raw_kline_message, save_raw_json_to_file

# --- 사용자 설정 ---
# 원하는 종목, 데이터 간격, 저장할 파일 경로를 설정합니다.
SYMBOL = "ethusdt"  # 예: 'btcusdt', 'ethusdt', 'bnbusdt' 등
INTERVAL = "1m"     # 예: '1m', '5m', '15m', '1h', '4h', '1d' 등

# 파일 이름에 심볼과 인터벌을 포함시켜 동적으로 생성합니다.
JSON_FILE = f"{SYMBOL}_{INTERVAL}_raw_kline_data.jsonl"

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # 1. 데이터 통신을 위한 큐(Queue) 생성
    message_queue = queue.Queue()

    # 2. BinanceKlineClient 인스턴스 생성
    # 설정한 값(종목, 간격)과 큐를 인자로 전달하여 클라이언트를 초기화합니다.
    kline_client = BinanceKlineClient(
        symbol=SYMBOL,
        interval=INTERVAL,
        data_queue=message_queue
    )

    try:
        # 3. WebSocket 스트림 시작
        kline_client.start_stream()

        print(f"\n'{SYMBOL.upper()}'의 '{INTERVAL}' K-line 원시 데이터를 '{JSON_FILE}'에 저장합니다.")
        print("프로그램을 중지하려면 Ctrl+C를 누르세요.")
        print("-" * 70)

        # 4. 메인 루프: 큐에서 데이터 수신 및 파일 저장
        # 이 루프는 사용자가 직접 중지(Ctrl+C)할 때까지 계속 실행됩니다.
        while True:
            # 큐에서 메시지 가져오기
            raw_message = get_raw_kline_message(message_queue)

            if raw_message:
                # 메시지가 있으면 파일에 저장
                save_raw_json_to_file(raw_message, JSON_FILE)
            
            # CPU 과부하 방지를 위한 짧은 대기
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 종료되었습니다 (Ctrl+C).")
    except Exception as e:
        print(f"\n메인 루프에서 예상치 못한 오류 발생: {e}")
    finally:
        # 5. 프로그램 종료 시 스트림을 안전하게 정리
        print("리소스 정리 중...")
        kline_client.stop_stream()
        print("프로그램이 완전히 종료되었습니다.")