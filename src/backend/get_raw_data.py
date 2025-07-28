import websocket
import json
import ssl
import threading
import queue
import time
import os
from datetime import datetime

# --- 설정 변수 ---
SYMBOL = "btcusdt"
INTERVAL = "1m"
WEBSOCKET_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_{INTERVAL}"
JSON_FILE = f"{SYMBOL}_{INTERVAL}_raw_kline_data.jsonl" # .jsonl 확장자는 JSON Lines를 의미

# --- WebSocket 클라이언트 클래스 정의 ---
class BinanceKlineClient:
    def __init__(self, data_queue: queue.Queue):
        self.data_queue = data_queue # 데이터를 저장하는 queue
        self.ws: websocket.WebSocketApp = None # WebSocketApp 인스턴스
        self.stream_thread: threading.Thread = None # 스트림을 실행할 스레드
        self._is_running = False # 스트림이 현재 실행 중인지 상태 추적

    def _on_open(self, ws):
        """WebSocket 연결이 성공적으로 열렸을 때 호출되는 내부 콜백."""
        print("WebSocket 연결이 열렸습니다.")
        print(f"종목: {SYMBOL.upper()}, 봉 간격: {INTERVAL}")

    def _on_message(self, ws, message):
        """WebSocket으로부터 메시지를 수신했을 때 호출되는 내부 콜백."""
        # 수신된 원시 메시지를 큐에 넣습니다.
        self.data_queue.put(message)
        # print(f"Raw message added to queue. Queue size: {self.data_queue.qsize()}") # 디버깅용

    def _on_error(self, ws, error):
        """WebSocket에서 오류가 발생했을 때 호출되는 내부 콜백."""
        print(f"WebSocket 오류 발생: {error}")
        # 여기에 재연결 로직을 추가할 수 있습니다.

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket 연결이 닫혔을 때 호출되는 내부 콜백."""
        print(f"WebSocket 연결이 닫혔습니다. 상태 코드: {close_status_code}, 메시지: {close_msg}")
        self._is_running = False # 스트림 상태 업데이트

    def _run_websocket_loop(self):
        """별도의 스레드에서 WebSocket 연결을 유지하는 내부 메서드."""
        try:
            # run_forever()는 블로킹 함수이므로 스레드에서 실행되어야 합니다.
            # sslopt={"cert_reqs": ssl.CERT_NONE}은 개발/테스트용이며, 운영 환경에서는 보안을 위해 제거하거나 ssl.create_default_context()를 사용해야 합니다.
            self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        except Exception as e:
            print(f"WebSocket run_forever 루프 실행 중 오류: {e}")
        finally:
            print("WebSocket 스트림 루프가 종료되었습니다.")

    def start_stream(self):
        """
        WebSocket 스트림을 시작하고, 수신된 원시 데이터를 큐에 넣는 역할을 합니다.
        이 메서드는 별도의 스레드를 생성하여 WebSocket 연결을 관리합니다.
        """
        if self._is_running:
            print("스트림이 이미 실행 중입니다. 다시 시작할 수 없습니다.")
            return

        print("WebSocket 스트림을 시작 요청합니다...")
        self.ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                         on_open=self._on_open,
                                         on_message=self._on_message,
                                         on_error=self._on_error,
                                         on_close=self._on_close)
        
        # WebSocket 연결을 별도의 스레드에서 실행
        self.stream_thread = threading.Thread(target=self._run_websocket_loop)
        self.stream_thread.daemon = True # 메인 프로그램 종료 시 스레드도 함께 종료되도록 설정
        self.stream_thread.start()
        self._is_running = True
        print("WebSocket 스트림 시작 요청 완료.")

    def stop_stream(self):
        """
        현재 실행 중인 WebSocket 스트림을 안전하게 종료합니다.
        """
        if not self._is_running:
            print("스트림이 실행 중이 아닙니다. 종료할 스트림이 없습니다.")
            return

        print("WebSocket 스트림을 종료합니다...")
        if self.ws:
            self.ws.close() # ws.close()를 호출하면 run_forever() 루프가 종료됩니다.
        
        if self.stream_thread and self.stream_thread.is_alive():
            # 스트림 스레드가 종료될 때까지 기다립니다 (최대 5초)
            self.stream_thread.join(timeout=5)
            if self.stream_thread.is_alive():
                print("경고: WebSocket 스레드가 5초 후에도 종료되지 않았습니다. 강제 종료될 수 있습니다.")
            else:
                print("WebSocket 스레드가 성공적으로 종료되었습니다.")
        
        self._is_running = False
        print("WebSocket 스트림 종료 요청 완료.")

# --- 데이터 처리 및 저장 함수 ---
# k: k-line candle 데이터
# t: Open Time, T: close Time
# o: open, h: high, l: low, c: close, v: volume
# x: is_final_bar
def get_raw_kline_message(data_queue: queue.Queue, timeout: float = 0.1) -> str or None:
    try:
        raw_message = data_queue.get(timeout=timeout)
        return raw_message
    except queue.Empty:
        return None
    except Exception as e:
        print(f"큐에서 메시지를 가져오는 중 오류 발생: {e}")
        return None

def save_raw_json_to_file(raw_json_string: str, file_path: str):
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(raw_json_string + '\n') # 각 JSON 객체 뒤에 줄바꿈 추가
        # print(f"JSON 데이터 파일에 추가됨: {datetime.now().strftime('%H:%M:%S')}") # 메시지가 너무 많아 주석 처리
    except Exception as e:
        print(f"JSON 파일 저장 오류: {e}")

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    message_queue = queue.Queue() # 메시지를 받을 큐 생성
    kline_client = BinanceKlineClient(message_queue) # 클라이언트 인스턴스 생성

    try:
        # 1. WebSocket 스트림 시작 함수 호출
        kline_client.start_stream()

        print(f"\n'{SYMBOL.upper()}'의 '{INTERVAL}' K-line 원시 데이터를 '{JSON_FILE}'에 저장합니다.")
        print("--------------------------------------------------------------------------------------------------")

        # 메인 루프에서 데이터 수신 및 저장 (데모를 위해 20초간 실행)
        run_duration_seconds = 20
        start_time = time.time()

        while time.time() - start_time < run_duration_seconds:
            raw_message = get_raw_kline_message(message_queue) # 큐에서 메시지 가져오기

            if raw_message:
                save_raw_json_to_file(raw_message, JSON_FILE)
            
            time.sleep(0.001) # CPU 과부하 방지를 위한 짧은 대기

        print(f"\n{run_duration_seconds}초 동안 데이터 수신 및 저장 완료.")

    except KeyboardInterrupt:
        print("\n사용자가 프로그램을 종료했습니다 (Ctrl+C).")
    except Exception as e:
        print(f"\n메인 루프에서 예상치 못한 오류 발생: {e}")
    finally:
        # 2. WebSocket 스트림 종료 함수 호출 (프로그램 종료 시 항상 실행)
        kline_client.stop_stream()
        print("프로그램이 종료됩니다.")