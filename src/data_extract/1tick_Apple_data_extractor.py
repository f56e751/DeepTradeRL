# ======================================
# 여기에 API 키를 붙여넣어 주세요
# ======================================
API_KEY = "db-3vH5GBqA47mjgJcY6mviWvxrGtRCY"  # <- 이 부분에 실제 API 키를 붙여넣으세요

import databento as db
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AppleDataExtractor:
    """애플 주식 데이터 추출 클래스"""
    
    def __init__(self, api_key, symbol="AAPL", data_dir="src/db"):
        # Databento Historical 클라이언트 초기화 
        # 여러 방법을 시도해서 호환성 확보
        try:
            # 방법 1: 위치 인수로 API 키 전달
            self.client = db.Historical(api_key)
        except TypeError:
            try:
                # 방법 2: key 파라미터로 전달
                self.client = db.Historical(key=api_key)
            except TypeError:
                # 방법 3: 환경변수 설정 후 기본 초기화
                import os
                os.environ['DATABENTO_API_KEY'] = api_key
                self.client = db.Historical()
        
        self.symbol = symbol
        self.data_dir = data_dir
        
        # 디렉토리 생성
        self._create_data_directory()
        
    def _create_data_directory(self):
        """데이터 저장 디렉토리 생성"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"디렉토리 생성: {self.data_dir}")
        else:
            logger.info(f"기존 디렉토리 사용: {self.data_dir}")
    
    def _get_file_path(self, filename):
        """파일 전체 경로 반환"""
        return os.path.join(self.data_dir, filename)
        
    def extract_tick_data(self, start_date, end_date):
        """OHLCV 데이터 추출 (요청된 틱 데이터 형식)"""
        try:
            logger.info("=== OHLCV 데이터 추출 시작 ===")
            
            # OHLCV 1분 데이터 추출
            tick_data = self.client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=[self.symbol],
                schema="ohlcv-1m",  # 1분 간격 OHLCV
                start=start_date,
                end=end_date,
                stype_in="raw_symbol"
            )
            
            tick_df = tick_data.to_df()
            logger.info(f"OHLCV 데이터 추출 완료: {len(tick_df)} 건")
            
            # 요청된 형식으로 변환: Date,Ticker,Open,High,Low,Close,Adj Close,Volume
            if not tick_df.empty:
                # 날짜 컬럼 생성 (timestamp에서 날짜 추출)
                tick_df['Date'] = tick_df.index.date
                tick_df['Ticker'] = self.symbol
                
                # 컬럼 매핑 및 순서 조정
                formatted_df = pd.DataFrame({
                    'Date': tick_df['Date'],
                    'Ticker': tick_df['Ticker'],
                    'Open': tick_df['open'],
                    'High': tick_df['high'],
                    'Low': tick_df['low'],
                    'Close': tick_df['close'],
                    'Adj Close': tick_df['close'],  # Adj Close는 Close와 동일하게 설정
                    'Volume': tick_df['volume']
                })
            else:
                formatted_df = pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
            
            # 데이터 저장
            if isinstance(start_date, datetime):
                date_str = start_date.strftime("%Y-%m-%d_%H%M")
            else:
                date_str = str(start_date)
            filename = f"{self.symbol}_ohlcv_data_{date_str}.csv"
            filepath = self._get_file_path(filename)
            formatted_df.to_csv(filepath, index=False)
            logger.info(f"OHLCV 데이터 저장: {filepath}")
            
            return formatted_df
            
        except Exception as e:
            logger.error(f"OHLCV 데이터 추출 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def extract_orderbook_data(self, start_date, end_date):
        """오더북 데이터 추출 (요청된 형식으로 변환)"""
        try:
            logger.info("=== 오더북 데이터 추출 시작 ===")
            
            # MBP-10 스키마로 10 레벨 오더북 데이터 추출
            try:
                orderbook_data = self.client.timeseries.get_range(
                    dataset="XNAS.ITCH",
                    symbols=[self.symbol],
                    schema="mbp-10",  # Market By Price 10 레벨
                    start=start_date,
                    end=end_date,
                    stype_in="raw_symbol"
                )
                schema_type = "mbp-10"
                
            except Exception:
                logger.warning("MBP-10 스키마 실패, MBP-1로 재시도...")
                orderbook_data = self.client.timeseries.get_range(
                    dataset="XNAS.ITCH",
                    symbols=[self.symbol],
                    schema="mbp-1",
                    start=start_date,
                    end=end_date,
                    stype_in="raw_symbol"
                )
                schema_type = "mbp-1"
            
            orderbook_df = orderbook_data.to_df()
            logger.info(f"오더북 데이터 ({schema_type}) 추출 완료: {len(orderbook_df)} 건")
            
            # 요청된 형식으로 변환
            if not orderbook_df.empty:
                formatted_df = self._format_orderbook(orderbook_df, schema_type)
            else:
                # 빈 데이터프레임인 경우 컬럼만 생성
                formatted_df = self._create_empty_orderbook_df()
            
            # 데이터 저장
            if isinstance(start_date, datetime):
                date_str = start_date.strftime("%Y-%m-%d_%H%M")
            else:
                date_str = str(start_date)
            filename = f"{self.symbol}_orderbook_{schema_type}_data_{date_str}.csv"
            filepath = self._get_file_path(filename)
            formatted_df.to_csv(filepath, index=False)
            logger.info(f"오더북 데이터 저장: {filepath}")
            
            return formatted_df
            
        except Exception as e:
            logger.error(f"오더북 데이터 추출 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def _format_orderbook(self, df, schema_type):
        """오더북 데이터를 요청된 형식으로 변환"""
        # 타임스탬프 컬럼 생성
        formatted_data = {
            'timestamp': df.index
        }
        
        if schema_type == "mbp-10":
            # 10 레벨 데이터 처리
            for i in range(10):
                # Bid 데이터
                formatted_data[f'bid_px_{i:02d}'] = df.get(f'bid_px_{i:02d}', None)
                formatted_data[f'bid_sz_{i:02d}'] = df.get(f'bid_sz_{i:02d}', None)
                formatted_data[f'bid_ct_{i:02d}'] = df.get(f'bid_ct_{i:02d}', None)
                
                # Ask 데이터
                formatted_data[f'ask_px_{i:02d}'] = df.get(f'ask_px_{i:02d}', None)
                formatted_data[f'ask_sz_{i:02d}'] = df.get(f'ask_sz_{i:02d}', None)
                formatted_data[f'ask_ct_{i:02d}'] = df.get(f'ask_ct_{i:02d}', None)
        else:
            # MBP-1인 경우 첫 번째 레벨만 채우고 나머지는 NaN
            for i in range(10):
                if i == 0:
                    # 첫 번째 레벨은 실제 데이터
                    formatted_data[f'bid_px_{i:02d}'] = df.get('bid_px_00', None)
                    formatted_data[f'bid_sz_{i:02d}'] = df.get('bid_sz_00', None)
                    formatted_data[f'bid_ct_{i:02d}'] = df.get('bid_ct_00', None)
                    
                    formatted_data[f'ask_px_{i:02d}'] = df.get('ask_px_00', None)
                    formatted_data[f'ask_sz_{i:02d}'] = df.get('ask_sz_00', None)
                    formatted_data[f'ask_ct_{i:02d}'] = df.get('ask_ct_00', None)
                else:
                    # 나머지 레벨은 NaN
                    formatted_data[f'bid_px_{i:02d}'] = None
                    formatted_data[f'bid_sz_{i:02d}'] = None
                    formatted_data[f'bid_ct_{i:02d}'] = None
                    
                    formatted_data[f'ask_px_{i:02d}'] = None
                    formatted_data[f'ask_sz_{i:02d}'] = None
                    formatted_data[f'ask_ct_{i:02d}'] = None
        
        return pd.DataFrame(formatted_data)
    
    def _create_empty_orderbook_df(self):
        """빈 오더북 데이터프레임 생성"""
        columns = ['timestamp']
        
        # bid/ask 각각 10레벨의 px, sz, ct 컬럼 생성
        for i in range(10):
            columns.extend([
                f'bid_px_{i:02d}', f'bid_sz_{i:02d}', f'bid_ct_{i:02d}',
                f'ask_px_{i:02d}', f'ask_sz_{i:02d}', f'ask_ct_{i:02d}'
            ])
        
        return pd.DataFrame(columns=columns)
    
    def extract_bbo_data(self, start_date, end_date):
        """BBO 데이터 추출"""
        try:
            logger.info("=== BBO 데이터 추출 시작 ===")
            
            bbo_data = self.client.timeseries.get_range(
                dataset="XNAS.QBBO",
                symbols=[self.symbol],
                schema="bbo-1s",
                start=start_date,
                end=end_date,
                stype_in="raw_symbol"
            )
            
            bbo_df = bbo_data.to_df()
            logger.info(f"BBO 데이터 추출 완료: {len(bbo_df)} 건")
            
            # 데이터 저장
            filename = f"{self.symbol}_bbo_data_{start_date}.csv"
            filepath = self._get_file_path(filename)
            bbo_df.to_csv(filepath, index=False)
            logger.info(f"BBO 데이터 저장: {filepath}")
            
            return bbo_df
            
        except Exception as e:
            logger.error(f"BBO 데이터 추출 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def analyze_top_bottom_levels(self, orderbook_df, num_levels=10):
        """상위/하위 호가 분석"""
        if orderbook_df.empty:
            logger.warning("분석할 오더북 데이터가 없습니다.")
            return
        
        logger.info(f"=== 상위/하위 {num_levels}개 호가 분석 ===")
        
        # 컬럼명 확인 및 표준화
        if 'side' in orderbook_df.columns:
            # 매수/매도 구분
            buy_orders = orderbook_df[orderbook_df['side'] == 'B'].copy()
            sell_orders = orderbook_df[orderbook_df['side'] == 'A'].copy()
            
            logger.info(f"총 매수 주문: {len(buy_orders):,}건")
            logger.info(f"총 매도 주문: {len(sell_orders):,}건")
            
            # 가격별 분석
            if 'price' in orderbook_df.columns and 'size' in orderbook_df.columns:
                # 매수 호가 상위 10개 (높은 가격순)
                if not buy_orders.empty:
                    buy_summary = buy_orders.groupby('price')['size'].sum().sort_index(ascending=False)
                    logger.info(f"\n매수 호가 상위 {num_levels}개:")
                    for i, (price, size) in enumerate(buy_summary.head(num_levels).items(), 1):
                        logger.info(f"{i:2d}. 가격: ${price:8.2f}, 수량: {size:,}")
                
                # 매도 호가 하위 10개 (낮은 가격순)
                if not sell_orders.empty:
                    sell_summary = sell_orders.groupby('price')['size'].sum().sort_index(ascending=True)
                    logger.info(f"\n매도 호가 하위 {num_levels}개:")
                    for i, (price, size) in enumerate(sell_summary.head(num_levels).items(), 1):
                        logger.info(f"{i:2d}. 가격: ${price:8.2f}, 수량: {size:,}")
        else:
            logger.warning("'side' 컬럼을 찾을 수 없습니다. 오더북 구조를 확인하세요.")
            logger.info("사용 가능한 컬럼:", list(orderbook_df.columns))
    
    def display_data_sample(self, df, data_type, num_rows=5):
        """데이터 샘플 출력"""
        if not df.empty:
            logger.info(f"\n{data_type} 데이터 샘플 ({num_rows}건):")
            logger.info(f"전체 행 수: {len(df):,}")
            logger.info(f"컬럼: {list(df.columns)}")
            print(df.head(num_rows).to_string())
        else:
            logger.warning(f"{data_type} 데이터가 비어있습니다.")

def main():
    """메인 함수"""
    
    # API 키 검증
    if API_KEY == "YOUR_API_KEY_HERE":
        logger.error("API 키를 설정해주세요!")
        return
    
    # 데이터 추출 대상 및 기간 설정 (30분 데이터만)
    symbol = "AAPL"
    
    # 어제 오후 2시부터 2시 30분까지 (시장이 열려있는 시간)
    yesterday = datetime.now().date() - timedelta(days=1)
    start_time = datetime.combine(yesterday, datetime.min.time().replace(hour=14, minute=0))  # 2:00 PM
    end_time = start_time + timedelta(minutes=30)  # 2:30 PM
    
    logger.info(f"데이터 추출 시작")
    logger.info(f"심볼: {symbol}")
    logger.info(f"기간: {start_time} ~ {end_time} (30분)")
    
    # 데이터 추출기 초기화
    extractor = AppleDataExtractor(API_KEY, symbol)
    
    # 데이터 추출 실행
    logger.info("\n" + "="*50)
    
    # 1. OHLCV 데이터 추출 (요청된 틱 데이터 형식)
    ohlcv_df = extractor.extract_tick_data(start_time, end_time)
    logger.info("\nOHLCV 데이터 샘플:")
    if not ohlcv_df.empty:
        print(ohlcv_df.head())
        logger.info(f"컬럼: {list(ohlcv_df.columns)}")
    
    # 2. 오더북 데이터 추출 (요청된 형식)
    orderbook_df = extractor.extract_orderbook_data(start_time, end_time)
    logger.info("\n오더북 데이터 샘플:")
    if not orderbook_df.empty:
        print(orderbook_df.head())
        logger.info(f"컬럼: {list(orderbook_df.columns)}")
    
    # 3. BBO 데이터 추출 (기존 형식 유지)
    #bbo_df = extractor.extract_bbo_data(start_time, end_time)
    #extractor.display_data_sample(bbo_df, "BBO")
    
    # 추출 완료 요약
    logger.info("\n" + "="*50)
    logger.info("=== 데이터 추출 완료 ===")
    logger.info(f"데이터 기간: 30분")
    logger.info(f"OHLCV 데이터: {len(ohlcv_df):,} 건")
    logger.info(f"오더북 데이터: {len(orderbook_df):,} 건")
    #logger.info(f"BBO 데이터: {len(bbo_df):,} 건")
    
    # 저장된 파일 목록
    data_dir_display = os.path.abspath(extractor.data_dir)
    date_str = start_time.strftime("%Y-%m-%d_%H%M")
    files = [
        f"{symbol}_ohlcv_data_{date_str}.csv",
        f"{symbol}_orderbook_*_data_{date_str}.csv",
    ]
    
    logger.info(f"\n저장된 디렉토리: {data_dir_display}")
    logger.info("저장된 파일:")
    for file in files:
        logger.info(f"- {file}")
    
    logger.info("\n생성된 데이터 형식:")
    logger.info("1. OHLCV: Date,Ticker,Open,High,Low,Close,Adj Close,Volume")
    logger.info("2. OrderBook: timestamp + bid/ask px/sz/ct 각 10레벨")
    #logger.info("3. BBO: Best Bid/Offer 데이터")

if __name__ == "__main__":
    main()