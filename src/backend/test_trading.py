import os
import logging
import time
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# --- 1. 기본 설정 ---

# 로깅 설정: 모든 기록을 'trading_bot.log' 파일에 남깁니다. (인코딩 UTF-8 설정)
logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8')


def create_binance_client():
    """환경 변수에서 API 키를 불러와 바이낸스 클라이언트를 생성합니다."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        logging.error("환경 변수에서 API 키를 찾을 수 없습니다. (BINANCE_API_KEY, BINANCE_API_SECRET)")
        raise ValueError("API 키가 설정되지 않았습니다.")

    # testnet=True로 모의투자 서버에 접속합니다.
    client = Client(api_key, api_secret, testnet=True)
    logging.info("바이낸스 테스트넷 클라이언트 생성 완료.")
    return client


# --- 2. 계좌 관련 함수 (개선된 버전) ---

def get_my_assets(client):
    """
    보유한 자산(USDT 및 코인)의 잔고만 조회하여 보기 쉽게 출력합니다.
    0개인 자산은 무시합니다.
    """
    try:
        account_info = client.get_account()
        balances = account_info.get('balances', [])

        usdt_balance = None
        owned_coins = []

        for asset in balances:
            # 사용 가능 수량(free)과 주문 중인 수량(locked)을 더해 총 보유량 계산
            total_balance = float(asset['free']) + float(asset['locked'])

            # 총 보유량이 0보다 큰 유의미한 자산만 필터링
            if total_balance > 0.00000001:
                if asset['asset'] == 'USDT':
                    usdt_balance = {
                        'asset': 'USDT',
                        'total': total_balance,
                        'free': float(asset['free'])
                    }
                # BTC만 출력
                else:
                    if asset['asset'] == 'BTC':
                        owned_coins.append({
                            'asset': asset['asset'],
                            'total': total_balance,
                            'free': float(asset['free'])
                        })

        print("\n--- 자금 현황 ---")
        logging.info("--- 자금 현황 ---")

        # 보유 현금(USDT) 출력
        if usdt_balance:
            usdt_msg = f"💵 보유 현금 (USDT): {usdt_balance['total']:.8f}"
            print(usdt_msg)
            logging.info(usdt_msg)
        else:
            print("💵 보유 현금 (USDT): 0.00")
            logging.info("보유 현금 (USDT): 0.00")

        # 보유 코인 목록 출력
        if owned_coins:
            print("🪙 보유 코인 목록:")
            logging.info("보유 코인 목록:")
            for coin in owned_coins:
                # 소수점 8자리까지 표시하여 작은 수량도 잘 보이게 함
                coin_msg = f"  - {coin['asset']:<6} | 총 보유: {coin['total']:.8f}"
                print(coin_msg)
                logging.info(coin_msg)
        else:
            print("🪙 보유 코인 목록: 없음")
            logging.info("보유 코인 목록: 없음")

        return {'usdt': usdt_balance, 'coins': owned_coins}

    except BinanceAPIException as e:
        log_msg = f"자산 정보 조회 실패 (API 오류): {e}"
        logging.error(log_msg)
        print(f"❌ 에러 발생: {log_msg}")
    except Exception as e:
        log_msg = f"자산 정보 조회 실패 (일반 오류): {e}"
        logging.error(log_msg)
        print(f"❌ 에러 발생: {log_msg}")
    return None


# --- 3. 매매 관련 함수 ---

def place_buy_order(client, symbol, quantity):
    """지정된 수량만큼 시장가 매수 주문을 실행합니다."""
    try:
        logging.info(f"시장가 매수 주문 시도: 심볼={symbol}, 수량={quantity}")
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        logging.info(f"매수 주문 성공: {order}")
        print(f"✅ 매수 주문 성공: {order['fills'][0]['price']} 가격에 {quantity}개 체결")
        return order
    except BinanceAPIException as e:
        log_msg = f"매수 주문 실패 (API 오류): {e}"
        logging.error(log_msg)
        print(f"❌ 에러 발생: {log_msg}")
    except Exception as e:
        log_msg = f"매수 주문 실패 (일반 오류): {e}"
        logging.error(log_msg)
        print(f"❌ 에러 발생: {log_msg}")
    return None


def place_sell_order(client, symbol, quantity):
    """지정된 수량만큼 시장가 매도 주문을 실행합니다."""
    try:
        logging.info(f"시장가 매도 주문 시도: 심볼={symbol}, 수량={quantity}")
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        logging.info(f"매도 주문 성공: {order}")
        print(f"✅ 매도 주문 성공: {order['fills'][0]['price']} 가격에 {quantity}개 체결")
        return order
    except BinanceAPIException as e:
        log_msg = f"매도 주문 실패 (API 오류): {e}"
        logging.error(log_msg)
        print(f"❌ 에러 발생: {log_msg}")
    except Exception as e:
        log_msg = f"매도 주문 실패 (일반 오류): {e}"
        logging.error(log_msg)
        print(f"❌ 에러 발생: {log_msg}")
    return None


# --- 4. 메인 실행 함수 ---

def main():
    """프로그램의 메인 로직을 실행합니다."""
    print("🤖 바이낸스 자동매매 봇을 시작합니다. (모의투자)")
    
    try:
        # 1. 바이낸스 클라이언트 생성
        binance_client = create_binance_client()
        
        # 2. 거래 시작 전, 현재 자산 현황 확인
        get_my_assets(binance_client)
        
        # 3. 간단한 매매 로직 실행 (예시)
        symbol_to_trade = 'BTCUSDT'
        quantity_to_trade = 0.001
        
        print(f"\n--- {symbol_to_trade} 자동 거래 테스트 시작 ---")
        # 예시: BTCUSDT 0.001개 매수 후 잠시 후 매도
        if place_buy_order(binance_client, symbol_to_trade, quantity_to_trade):
            time.sleep(3) # 실제 봇에서는 단순 대기보다 다른 방식을 사용해야 합니다.
            place_sell_order(binance_client, symbol_to_trade, quantity_to_trade)
        
        # 4. 거래 종료 후, 최종 자산 현황 확인
        print("\n--- 거래 후 최종 자산 현황 ---")
        get_my_assets(binance_client)

    except ValueError as e:
        print(f"프로그램 시작 실패: {e}")
    except Exception as e:
        print(f"예상치 못한 오류로 프로그램이 종료됩니다: {e}")
        logging.critical(f"프로그램 비정상 종료: {e}")

if __name__ == "__main__":
    main()