import pandas as pd
from enum import Enum, auto
from typing import Union



class DFProcessMode(Enum):
    FillWithLast = auto() # 데이터가 비어있을 시 이전 스텝의 데이터 사용
    Skip = auto() # 데이터가 비어있을 시 해당 스텝의 데이터를 df로 변환하지 않음

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

    def load_and_process(self, csv_path: str, process_mode: DFProcessMode = DFProcessMode.FillWithLast) -> pd.DataFrame:
        """
        1) CSV 파일 로드 (timestamp 파싱)
        2) price 컬럼과 volume 컬럼 분리
        3) volume 컬럼명을 정수 인덱스로 변환
        4) 두 부분을 결합하여 반환

        
        input 컬럼 형식
        timestamp,bid_px_00,bid_sz_00,bid_ct_00,ask_px_00,ask_sz_00,ask_ct_00,bid_px_01,bid_sz_01,bid_ct_01,ask_px_01,ask_sz_01,ask_ct_01,bid_px_02,bid_sz_02,bid_ct_02,ask_px_02,ask_sz_02,ask_ct_02,bid_px_03,bid_sz_03,bid_ct_03,ask_px_03,ask_sz_03,ask_ct_03,bid_px_04,bid_sz_04,bid_ct_04,ask_px_04,ask_sz_04,ask_ct_04,bid_px_05,bid_sz_05,bid_ct_05,ask_px_05,ask_sz_05,ask_ct_05,bid_px_06,bid_sz_06,bid_ct_06,ask_px_06,ask_sz_06,ask_ct_06,bid_px_07,bid_sz_07,bid_ct_07,ask_px_07,ask_sz_07,ask_ct_07,bid_px_08,bid_sz_08,bid_ct_08,ask_px_08,ask_sz_08,ask_ct_08,bid_px_09,bid_sz_09,bid_ct_09,ask_px_09,ask_sz_09,ask_ct_09
        2025-05-13 14:00:00.006248867+00:00,209.92,61,3

        output 컬럼 형식
           bid_px_00  bid_px_01  bid_px_02  bid_px_03  bid_px_04  bid_px_05  bid_px_06  bid_px_07  bid_px_08  bid_px_09  ask_px_00  ask_px_01  ask_px_02  ask_px_03  ask_px_04  ask_px_05  ask_px_06  ...    3    4    5    6      7   8    9   10   11   12   13   14   15   16   17   18   19
     0     209.92     209.91      209.9     209.89     209.88     209.87     209.86     209.85     209.84     209.83     209.94     209.95     209.96     209.97     209.98     209.99      210.0  ...  755  267  261  210  10066  56  158  100  129  232  175  150  225  176  176  306  156
     1     209.92     209.91      209.9     209.89     209.88     209.87     209.86     209.85     209.84     209.83     209.94     209.95     209.96     209.97     209.98     209.99      210.0  ...  755  267  261  210  10066  56  158  100  129  232  175  150  225  176  176  306  156
        """
        # 1) CSV 로드 및 timestamp 파싱
        df_raw = pd.read_csv(csv_path, parse_dates=['timestamp'])

        # 빈 값이 있으면 이전 row 값으로 채우기
        if process_mode == DFProcessMode.FillWithLast:
            df_raw.fillna(method='ffill', inplace=True)

        # timestamp 컬럼을 따로 보관
        df_timestamp = df_raw[['timestamp']].copy()

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
        df_processed = pd.concat([df_timestamp, df_prices, df_vol], axis=1).reset_index(drop=True)

        return df_processed
    

class OHLCVCSVProcessor:
    """
    CSV 형식의 OHLCV 데이터를 TickStockTradingEnv 등에 입력 가능한
    pandas.DataFrame으로 가공하는 클래스
    - timestamp, open, high, low, close, volume 컬럼을 모두 포함
    - process_mode:
        - FillWithLast: 가격 컬럼의 결측치는 직전 값으로 채우고,
                        volume 결측치는 0으로 채움
    - 반환된 DataFrame 컬럼 순서:
        [timestamp, open, high, low, close, volume]
    """
    def __init__(self):
        pass

    def load_and_process(
        self,
        csv_path: str,
        process_mode: DFProcessMode = DFProcessMode.FillWithLast
    ) -> pd.DataFrame:
        """
        1) CSV 파일 로드 (timestamp 파싱)
        2) process_mode에 따라 결측치 처리
        3) 필요한 컬럼만 추출하여 순서 고정 후 반환
        """
        # 1) CSV 로드 및 timestamp 파싱
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])

        # 2) 결측치 처리
        if process_mode == DFProcessMode.FillWithLast:
            # 가격 컬럼(open, high, low, close)
            price_cols = ['open', 'high', 'low', 'close']
            # 1) 위로 채우기
            df[price_cols] = df[price_cols].fillna(method='ffill')
            # 2) 맨 앞 구간(첫 값이 NaN인 경우) 아래 첫 유효값으로 채우기
            df[price_cols] = df[price_cols].fillna(method='bfill')

            # volume 처리: NaN은 0으로
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)
            else:
                df['volume'] = 0

        # (필요 시 다른 process_mode 분기 추가)

        # 3) 컬럼 순서 고정
        output_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_processed = df[output_cols].reset_index(drop=True)

        return df_processed

class ExtendedOHLCVCSVProcessor:
    """
    CSV 형식의 OHLCV + 기술지표 데이터를 TickStockTradingEnv 등에 입력 가능한
    pandas.DataFrame으로 가공하는 클래스
    
    - timestamp, open, high, low, close, volume, dispersion20, slowK, slowD, MA5, MA20 컬럼 포함
    - process_mode:
        - FillWithLast: 가격 및 지표 컬럼의 결측치는 직전 값으로 채우고,
                        volume 결측치는 0으로 채움
        - Skip: 결측치가 있는 행은 통째로 제거
    - 반환된 DataFrame 컬럼 순서:
        [timestamp, open, high, low, close, volume,
         dispersion20, slowK, slowD, MA5, MA20]
    """
    def __init__(self):
        # 처리할 기술지표 컬럼명 정의
        self.tech_cols = ['dispersion20', 'slowK', 'slowD', 'MA5', 'MA20']

    def load_and_process(
        self,
        csv_path: str,
        process_mode: DFProcessMode = DFProcessMode.FillWithLast
    ) -> pd.DataFrame:
        # 1) CSV 로드 및 timestamp 파싱
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])

        # 2) 결측치 처리
        if process_mode == DFProcessMode.FillWithLast:
            # 가격(open, high, low, close) + 지표 컬럼을 모두 이전 값으로 채우기
            price_and_tech = ['open', 'high', 'low', 'close'] + self.tech_cols
            df[price_and_tech] = df[price_and_tech].fillna(method='ffill')
            df[price_and_tech] = df[price_and_tech].fillna(method='bfill')
            # volume은 0으로
            df['volume'] = df.get('volume', 0).fillna(0)
        elif process_mode == DFProcessMode.Skip:
            # 한 행이라도 NaN이 있으면 제거
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'] + self.tech_cols)

        # 3) 필요한 컬럼만 추출하여 순서 고정
        output_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + self.tech_cols
        df_processed = df[output_cols].reset_index(drop=True)

        return df_processed

class MarketDataMerger:
    """
    OHLCVCSVProcessor와 LOBCSVProcessor를 사용해
    두 개의 DataFrame을 timestamp 기준으로 병합하고,
    timestamp 일치 여부를 검증합니다.
    """
    def __init__(
        self,
        lob_processor: LOBCSVProcessor,
        ohlcv_processor: Union[OHLCVCSVProcessor, ExtendedOHLCVCSVProcessor]
    ):
        self.lob_proc   = lob_processor
        self.ohlcv_proc = ohlcv_processor

    def merge(
        self,
        lob_csv_path: str,
        ohlcv_csv_path: str,
        lob_mode: DFProcessMode   = DFProcessMode.FillWithLast,
        ohlcv_mode: DFProcessMode = DFProcessMode.FillWithLast
    ) -> pd.DataFrame:
        """
        1) 두 CSV를 각각 처리
        2) timestamp 기준으로 inner join
        3) timestamp 일치 검증
        4) 합친 DataFrame 반환 (timestamp 포함)
        """
        # 1) 로드 & 전처리
        df_lob   = self.lob_proc.load_and_process(lob_csv_path,   process_mode=lob_mode)
        df_ohlcv = self.ohlcv_proc.load_and_process(ohlcv_csv_path, process_mode=ohlcv_mode)

        # 2) 병합
        df_merged = pd.merge(
            df_ohlcv,
            df_lob,
            on='timestamp',
            how='inner',
            validate='one_to_one'  # timestamp 당 한 행씩만 있어야 함
        )

        # 3) timestamp 검증
        # 원본 각각의 timestamp 개수와 병합된 갯수가 모두 같아야 완전 일치
        count_ohlcv = df_ohlcv['timestamp'].nunique()
        count_lob   = df_lob['timestamp'].nunique()
        count_merged= df_merged['timestamp'].nunique()

        if not (count_ohlcv == count_lob == count_merged):
            raise ValueError(
                f"Timestamp mismatch:\n"
                f"  OHLCV rows: {count_ohlcv}\n"
                f"  LOB   rows: {count_lob}\n"
                f"  Merged rows: {count_merged}\n"
                "  -> 모든 timestamp가 양쪽에 동일하게 존재해야 합니다."
            )
        else:
            print("timestamp 일치")

        return df_merged


class DataSplitter:
    """
    DataFrame을 train/val/test로 분할하는 클래스
    - train_ratio, val_ratio, test_ratio 합이 1.0이어야 함
    - 시계열 순서를 유지하여 분할
    """
    def __init__(self, train_ratio: float, val_ratio: float, test_ratio: float):
        total = train_ratio + val_ratio + test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame):
        """
        DataFrame을 비율에 따라 순서대로 분할
        Returns: (df_train, df_val, df_test)
        """
        n = len(df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        # 남은 것은 test
        n_test = n - n_train - n_val

        df_train = df.iloc[:n_train].reset_index(drop=True)
        df_val   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        df_test  = df.iloc[n_train + n_val:].reset_index(drop=True)

        return df_train, df_val, df_test


def validate_no_missing(df: pd.DataFrame) -> bool:
        """
        주어진 DataFrame에 NaN(빈 값)이 하나라도 있는지 검사합니다.
        - NaN이 있는 컬럼별 개수를 출력
        - NaN이 없으면 True, 하나라도 있으면 False 반환
        """
        missing_counts = df.isna().sum()
        total_missing = missing_counts.sum()
        if total_missing == 0:
            print("✔️ No missing values detected.")
            return True
        else:
            # 컬럼별로 빈 값이 있는 경우만 보여줌
            print("❌ Missing values found:")
            for col, cnt in missing_counts.items():
                if cnt > 0:
                    print(f"  - {col!r}: {cnt} missing")
            return False


def test_LOBCSVProcessor(lob_csv_path):
    """
    LOBCSVProcessor의 기본 동작을 검증하는 테스트 함수.
    1) 지정된 CSV 경로로부터 LOB 데이터를 로드 및 가공
    2) 결측치가 없는지 검증
    3) 가공 결과의 상위 5개 행을 출력
    """
    # 1) 프로세서 생성 (LOB 레벨은 필요에 따라 조정)
    processor = LOBCSVProcessor(lob_levels=10)
    
    # 2) CSV 로드 및 가공
    df_for_env = processor.load_and_process(lob_csv_path)

    # 3) 빈 값(결측치)이 남아 있지 않은지 확인
    validate_no_missing(df_for_env)

    # 4) 결과 확인 (상위 5개 행 출력)
    print("=== LOBProcessor Output (first 5 rows) ===")
    print(df_for_env.head())
    print("\n")


def test_OHLCVCSVProcessor(ohlcv_csv_path):
    """
    OHLCVCSVProcessor의 기본 동작을 검증하는 테스트 함수.
    1) 지정된 CSV 경로로부터 OHLCV 데이터를 로드 및 가공
    2) 결측치가 없는지 검증
    3) 가공 결과의 상위 5개 행을 출력
    """
    # 1) 프로세서 생성
    processor = OHLCVCSVProcessor()
    
    # 2) CSV 로드 및 가공 (FillWithLast 모드 사용)
    df_ohlcv = processor.load_and_process(
        ohlcv_csv_path,
        process_mode=DFProcessMode.FillWithLast
    )
    
    # 3) 빈 값(결측치)이 남아 있지 않은지 확인
    validate_no_missing(df_ohlcv)
    
    # 4) 결과 확인 (상위 5개 행 출력)
    print("=== OHLCVProcessor Output (first 5 rows) ===")
    print(df_ohlcv.head())
    print("\n")


def merge_lob_and_ohlcv(lob_csv_path, ohlcv_csv_path):
    """
    LOB 데이터와 OHLCV 데이터를 timestamp 기준으로 병합하는 함수.
    1) 각각 LOBCSVProcessor, OHLCVCSVProcessor로 데이터를 가공
    2) MarketDataMerger를 사용해 내부 조인(인터섹션) 수행
    3) 병합된 결과의 상위 5개 행을 출력 후 반환
    """
    # 1) 프로세서 인스턴스 생성
    lob_processor   = LOBCSVProcessor(lob_levels=10)
    ohlcv_processor = OHLCVCSVProcessor()

    # 2) 병합기 생성 및 데이터 병합
    merger = MarketDataMerger(lob_processor, ohlcv_processor)
    df_all = merger.merge(
        lob_csv_path   = lob_csv_path,
        ohlcv_csv_path = ohlcv_csv_path
    )
    
    # 3) 결과 확인 (상위 5개 행 출력)
    print("=== Merged Market Data (first 5 rows) ===")
    print(df_all.head())
    print("\n")
    
    return df_all

def merge_lob_and_ohlcv_extended(lob_csv_path, ohlcv_extended_csv_path):
    # 1) 프로세서 인스턴스 생성
    lob_processor   = LOBCSVProcessor(lob_levels=10)
    ohlcv_processor = ExtendedOHLCVCSVProcessor()

    # 2) 병합기 생성 및 데이터 병합
    merger = MarketDataMerger(lob_processor, ohlcv_processor)
    df_all = merger.merge(
        lob_csv_path   = lob_csv_path,
        ohlcv_csv_path = ohlcv_extended_csv_path
    )
    
    # 3) 결과 확인 (상위 5개 행 출력)
    print("=== Merged Market Data (first 5 rows) ===")
    pd.set_option('display.max_columns', None)
    print(df_all.head())
    print("\n")
    
    return df_all

def load_ohlcv_df(
    csv_path: str,
    process_mode: DFProcessMode = DFProcessMode.FillWithLast
) -> pd.DataFrame:
    """
    주어진 CSV 경로에서 OHLCV 데이터를 로드하여 전처리된 DataFrame으로 반환합니다.
    - timestamp, open, high, low, close, volume 컬럼만 포함
    - process_mode에 따라 결측치 처리
    """
    processor = OHLCVCSVProcessor()
    return processor.load_and_process(csv_path, process_mode)


def test_DataSplitter():
    # 앞서 merge_lob_and_ohlcv 로 생성한 df_all 사용 가정
    df_all = merge_lob_and_ohlcv(lob_csv_path, ohlcv_csv_path)

    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)

    print("Train:", df_train.shape)
    print("Val:  ", df_val.shape)
    print("Test: ", df_test.shape)


if __name__ == "__main__":
    # csv_path = "src/db/AAPL_orderbook_mbp-10_data_2025-05-13_1400.csv"
    lob_csv_path = "src/db/AAPL_minute_orderbook_2019_01-07_combined.csv"
    ohlcv_csv_path = "src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv"
    ohlcv_extended_csv_path = "src/db/indicator/AAPL_with_indicators_v2.csv"
    test_LOBCSVProcessor(lob_csv_path)
    test_OHLCVCSVProcessor(ohlcv_csv_path)
    # merge_lob_and_ohlcv(lob_csv_path, ohlcv_csv_path)
    merge_lob_and_ohlcv_extended(lob_csv_path, ohlcv_extended_csv_path)
    # test_DataSplitter()


