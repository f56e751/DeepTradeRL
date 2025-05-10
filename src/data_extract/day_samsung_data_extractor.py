from pykrx import stock
import pandas as pd
import os
from pathlib import Path

def get_samsung_daily_data_pykrx():
    """
    pykrx 라이브러리를 사용하여 삼성전자 4년치 일봉 데이터를
    Date, Ticker, Open, High, Low, Close, Adj Close, Volume 컬럼으로
    CSV에 저장하고, tickers.txt에 '티커: 회사명' 형식으로 기록하는 함수
    """
    ticker = "005930"
    start_date = "20140101"
    end_date   = "20171231"
    print(f"데이터 추출 기간: {start_date} ~ {end_date}")

    # 1) 원본 OHLCV 데이터 가져오기
    df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
    if df.empty:
        raise ValueError(f"No data for {ticker} between {start_date} and {end_date}")

    # 2) 인덱스를 컬럼으로 올리기 & 컬럼명 영문으로 변경
    df = (
        df.reset_index()
          .rename(columns={
              "날짜":  "Date",  # reset_index()가 만든 컬럼명
              "시가":    "Open",
              "고가":    "High",
              "저가":    "Low",
              "종가":    "Close",
              "거래량":  "Volume"
          })
    )

    # 3) Ticker, Adj Close 컬럼 추가
    df["Ticker"]    = ticker
    print("Before assignment, df columns:", df.columns.tolist())
    df["Adj Close"] = df["Close"]   # (필요시 조정 종가 계산 로직으로 교체)

    # 4) 원하는 순서로 정리
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    # 5) 저장할 경로 준비
    db_dir = Path(__file__).parent.parent / "db/day"
    os.makedirs(db_dir, exist_ok=True)
    output_path = db_dir / f"{ticker}_{start_date}_{end_date}.csv"

    # 6) CSV로 저장 (index 제외)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"CSV로 저장 완료: {output_path} ({len(df)} rows)")

    # ──────────────────────────────────────────────────
    # 7) tickers.txt에 '티커: 회사명' 기록
    try:
        company_name = stock.get_market_ticker_name(ticker)
    except Exception:
        company_name = "Unknown"

    ticker_log_path = db_dir / "tickers.txt"
    # 이미 동일한 티커가 기록되어 있는지 확인
    existing = ticker_log_path.read_text(encoding="utf-8-sig") if ticker_log_path.exists() else ""
    entry = f"{ticker}: {company_name}"
    if entry not in existing:
        with open(ticker_log_path, "a", encoding="utf-8-sig") as f:
            f.write(entry + "\n")
        print(f"Tickers log updated: {entry}")
    else:
        print(f"Ticker already logged: {ticker}")

    return df

if __name__ == "__main__":
    get_samsung_daily_data_pykrx()
