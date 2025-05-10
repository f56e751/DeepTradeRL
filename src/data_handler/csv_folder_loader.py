import os
import pandas as pd
from pathlib import Path

class CSVFolderLoader:
    """
    폴더 내 CSV 파일을 모두 읽어
    MultiDataHandler에 들어가기 적합한 long-format DataFrame을 반환합니다.
    가정: 각 CSV 파일에 이미 Date, Ticker, Open, High, Low, Close, Adj Close, Volume 컬럼이 존재합니다.
    """
    def __init__(self, folder_path: str, date_column: str = 'Date'):
        self.folder_path = Path(folder_path)
        self.date_column = date_column

    def load(self) -> pd.DataFrame:
        """
        폴더 내 모든 CSV를 읽어 하나의 DataFrame으로 합칩니다.
        반환 형식: Date, Ticker, Open, High, Low, Close, Adj Close, Volume 컬럼
        """
        dfs = []
        # 지정된 폴더에서 .csv 파일을 모두 순회
        for file_path in sorted(self.folder_path.glob('*.csv')):
            df = pd.read_csv(file_path, parse_dates=[self.date_column])
            # 필수 컬럼 검증
            required = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            missing = set(required) - set(df.columns)
            if missing:
                raise ValueError(f"{file_path.name}에서 누락된 컬럼: {missing}")
            # 필요한 컬럼만 선택
            df = df[required]
            dfs.append(df)

        if not dfs:
            raise ValueError(f"CSV 파일을 찾을 수 없습니다: {self.folder_path}")

        # 모든 DataFrame 합치기 및 날짜, 티커 순 정렬
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        return combined
