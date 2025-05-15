import pandas as pd

class LOBCSVProcessor:
    """
    CSV 형식의 LOB 데이터를 TickStockTradingEnv에 입력 가능한
    pandas.DataFrame으로 가공하는 클래스
    - 가격(price) 컬럼과 잔량(volume) 컬럼을 모두 포함
    - 가격 컬럼은 원본 이름 유지, 잔량 컬럼은 0,1,2...로 정수 인덱스 명 지정
    - 반환된 DataFrame 컬럼 순서:
        [bid_px_00...bid_px_{L-1},
         ask_px_00...ask_px_{L-1},
         0,1,...,2*L-1]
    """
    def __init__(self, lob_levels: int = 10):
        self.lob_levels = lob_levels

    def load_and_process(self, csv_path: str) -> pd.DataFrame:
        """
        1) CSV 파일 로드 (timestamp 파싱)
        2) price 컬럼과 volume 컬럼 분리
        3) volume 컬럼명을 정수 인덱스로 변환
        4) 두 부분을 결합하여 반환
        """
        # 1) CSV 로드 및 timestamp 파싱
        df_raw = pd.read_csv(csv_path, parse_dates=['timestamp'])

        # 2) 가격(Price) 및 잔량(Size) 컬럼명 리스트 생성
        bid_px_cols = [f"bid_px_{i:02d}" for i in range(self.lob_levels)]
        ask_px_cols = [f"ask_px_{i:02d}" for i in range(self.lob_levels)]
        bid_sz_cols = [f"bid_sz_{i:02d}" for i in range(self.lob_levels)]
        ask_sz_cols = [f"ask_sz_{i:02d}" for i in range(self.lob_levels)]

        # 3) 필요한 컬럼만 추출
        df_prices = df_raw[bid_px_cols + ask_px_cols].copy()
        df_vol = df_raw[bid_sz_cols + ask_sz_cols].copy()

        # 4) volume 컬럼명을 정수 인덱스로 변환
        df_vol.columns = list(range(len(df_vol.columns)))

        # 5) 가격과 잔량을 합친 최종 DataFrame
        df_processed = pd.concat([df_prices, df_vol], axis=1).reset_index(drop=True)
        return df_processed

# 사용 예시:
# processor = LOBCSVProcessor(lob_levels=10)
# df_for_env = processor.load_and_process('lob_data.csv')

