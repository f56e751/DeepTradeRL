from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

def get_samsung_daily_data_pykrx():
    """
    pykrx 라이브러리를 사용하여 삼성전자 4년치 일봉 데이터를 추출하는 함수
    파일명: samsung_pykrx_daily.py
    """
    # 삼성전자 종목코드: 005930
    samsung_code = "005930"
    
    # 4년치 데이터 기간 설정
    # 데이터 기간 설정: 2014-01-01 ~ 2017-12-31
    start_date = "20140101"
    end_date = "20171231"
    
    print(f"데이터 추출 기간: {start_date} ~ {end_date}")
    
    # KRX에서 삼성전자 일봉 데이터 가져오기
    samsung_data = stock.get_market_ohlcv_by_date(start_date, end_date, samsung_code)
    
    # 데이터 확인
    print(f"추출된 데이터 수: {len(samsung_data)} 행")
    print(samsung_data.head())
    
    # 디렉토리 확인 및 생성
    # 현재 파일 경로를 기준으로 상대 경로 설정
    db_dir = Path(__file__).parent.parent / "db"
    
    # db 디렉토리가 없으면 생성
    os.makedirs(db_dir, exist_ok=True)
    
    # 파일 경로 및 이름 설정
    output_path = db_dir / "samsung_data_2014_2017.csv"
    
    # CSV 파일로 저장
    samsung_data.to_csv(output_path)

    
    print(f"삼성전자 4년치 일봉 데이터가 '{output_path}'에 성공적으로 저장되었습니다.")
    return samsung_data
    
# 함수 실행
if __name__ == "__main__":
    get_samsung_daily_data_pykrx()