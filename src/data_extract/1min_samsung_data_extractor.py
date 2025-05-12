import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()
        self.ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveTrData.connect(self._receive_tr_data)

    def comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def _event_connect(self, err_code):
        if err_code == 0:
            print("로그인 성공")
        else:
            print(f"로그인 실패 - 에러 코드 : {err_code}")
        self.login_event_loop.exit()

    def get_connect_state(self):
        ret = self.dynamicCall("GetConnectState()")
        return ret
        
    def set_input_value(self, id, value):
        self.dynamicCall("SetInputValue(QString, QString)", id, value)
        
    def comm_rq_data(self, rqname, trcode, next, screen_no):
        self.dynamicCall("CommRqData(QString, QString, int, QString)", 
                         rqname, trcode, next, screen_no)
        self.tr_event_loop = QEventLoop()
        self.tr_event_loop.exec_()
        
    def _get_repeat_cnt(self, trcode, rqname):
        ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        return ret
        
    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4):
        if next == '2':
            self.remained_data = True
        else:
            self.remained_data = False
            
        if rqname == "opt10080_req":
            self._opt10080(rqname, trcode)
            
        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass
    
    def _validate_price(self, price_str):
        """가격 데이터 검증 및 변환"""
        try:
            price = int(price_str.strip())
            # 음수인 경우 절댓값 사용
            return abs(price)
        except ValueError:
            print(f"가격 변환 오류: {price_str}")
            return 0
    
    def _opt10080(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)
        
        print(f"데이터 개수: {data_cnt}")
        
        for i in range(data_cnt):
            date = self.dynamicCall("GetCommData(QString, QString, int, QString)", 
                                    trcode, rqname, i, "체결시간")
            open_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", 
                                         trcode, rqname, i, "시가")
            high_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", 
                                         trcode, rqname, i, "고가")
            low_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", 
                                        trcode, rqname, i, "저가")
            close_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", 
                                          trcode, rqname, i, "현재가")
            volume = self.dynamicCall("GetCommData(QString, QString, int, QString)", 
                                     trcode, rqname, i, "거래량")
            
            # 음수 검증 및 절댓값 적용
            open_val = self._validate_price(open_price)
            high_val = self._validate_price(high_price)
            low_val = self._validate_price(low_price)
            close_val = self._validate_price(close_price)
            volume_val = abs(int(volume.strip())) if volume.strip() else 0
            
            # 가격이 모두 0이 아닌 경우만 추가 (잘못된 데이터 필터링)
            if open_val > 0 and high_val > 0 and low_val > 0 and close_val > 0:
                self.ohlcv['date'].append(date.strip())
                self.ohlcv['open'].append(open_val)
                self.ohlcv['high'].append(high_val)
                self.ohlcv['low'].append(low_val)
                self.ohlcv['close'].append(close_val)
                self.ohlcv['volume'].append(volume_val)
            else:
                print(f"잘못된 데이터 스킵: {date.strip()} O:{open_val} H:{high_val} L:{low_val} C:{close_val}")
    
    def get_min_data(self, code, tick_range, start_date=None, end_date=None):
        """
        분봉 데이터 요청
        :param code: 종목코드
        :param tick_range: 분봉 범위 (1: 1분, 5: 5분, 15: 15분, 30: 30분, 60: 60분, 240: 240분)
        :param start_date: 조회 시작일 (YYYYMMDD 형식)
        :param end_date: 조회 종료일 (YYYYMMDD 형식)
        :return: DataFrame
        """
        print(f"데이터 요청 시작: {code}, 분봉:{tick_range}분")
        self.ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        self.set_input_value("종목코드", code)
        self.set_input_value("틱범위", str(tick_range))
        self.set_input_value("수정주가구분", "1")  # 1: 수정주가, 0: 원주가
        
        # 키움 API 최대 조회 가능한 기간으로 설정
        if not start_date:
            # 약 8개월 전으로 설정 (키움 API 최대 제공 기간)
            eight_months_ago = (datetime.now() - timedelta(days=240))
            start_date = eight_months_ago.strftime("%Y%m%d")
        
        # 조회 기간 설정
        if end_date:
            self.set_input_value("시작일자", end_date)
        
        rqname = "opt10080_req"
        trcode = "opt10080"
        next = 0
        screen_no = "0101"
        
        # 최초 조회
        print("첫 번째 조회 중...")
        self.comm_rq_data(rqname, trcode, next, screen_no)
        
        # 데이터가 더 있는 경우 연속 조회 (최대 50번으로 증가)
        count = 1
        while self.remained_data and count < 50:  # 더 많은 데이터를 위해 최대 50회 조회
            count += 1
            print(f"{count}번째 조회 중... (누적 데이터: {len(self.ohlcv['date'])}개)")
            
            # 연속 조회 시 딜레이 증가 (안정성 향상)
            time.sleep(0.5)
            
            self.set_input_value("종목코드", code)
            self.set_input_value("틱범위", str(tick_range))
            self.set_input_value("수정주가구분", "1")
            
            self.comm_rq_data(rqname, trcode, 2, screen_no)
        
        if not self.ohlcv['date']:
            print("데이터가 없습니다.")
            return pd.DataFrame()
        
        # 데이터 프레임으로 변환
        df = pd.DataFrame(self.ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # 날짜 형식 변환
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S')
        
        # 한국 주식시장 거래시간 필터링 (09:00 - 15:35)
        print("거래시간 필터링 적용 중...")
        df = df[
            (df['date'].dt.hour >= 9) & 
            ((df['date'].dt.hour < 15) | 
             ((df['date'].dt.hour == 15) & (df['date'].dt.minute <= 35)))
        ]
        
        # 주말 제외 (평일만)
        df = df[df['date'].dt.weekday < 5]
        
        # 시간 순서대로 정렬 (오래된 데이터가 위에)
        df = df.sort_values('date', ascending=True)
        df = df.reset_index(drop=True)
        
        # 날짜 필터링 (지정된 기간만)
        if start_date:
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            df = df[df['date'] >= start_dt]
            print(f"시작일 {start_date} 이후 데이터 필터링 완료")
        
        if end_date:
            end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(days=1)  # 종료일 포함을 위해 다음날 자정까지
            df = df[df['date'] < end_dt]
            print(f"종료일 {end_date} 이전 데이터 필터링 완료")
        
        # 기본적인 데이터 검증
        if len(df) > 0:
            print(f"필터링 후 데이터 범위: {df['date'].min()} ~ {df['date'].max()}")
            print(f"총 데이터 수: {len(df)}")
            
            # 기간별 데이터 수 확인
            start_date_actual = df['date'].min().strftime('%Y-%m-%d')
            end_date_actual = df['date'].max().strftime('%Y-%m-%d')
            print(f"실제 데이터 기간: {start_date_actual} ~ {end_date_actual}")
            
            # OHLC 검증
            invalid_ohlc = df[(df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)]
            if len(invalid_ohlc) > 0:
                print(f"경고: {len(invalid_ohlc)}개의 잘못된 OHLC 데이터가 발견되었습니다.")
        
        return df


if __name__ == "__main__":
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom.comm_connect()
    
    # 연결 상태 확인
    if kiwoom.get_connect_state() == 1:
        print("서버 연결 성공")
        
        # 특정 기간 설정
        start_date = "20241113"  # 2024년 11월 13일
        end_date = "20250509"    # 2025년 5월 9일
        
        print(f"조회 기간: {start_date} ~ {end_date}")
        print("1분봉 데이터 수집 시작...")
        
        # 삼성전자(005930) 1분봉 데이터 가져오기
        samsung_data = kiwoom.get_min_data("005930", 1, start_date, end_date)
        
        if len(samsung_data) > 0:
            print(f"\n수집 완료! 총 데이터 수: {len(samsung_data)}")
            
            # 실제 수집된 기간 확인
            actual_start = samsung_data['date'].min().strftime('%Y-%m-%d')
            actual_end = samsung_data['date'].max().strftime('%Y-%m-%d')
            print(f"실제 데이터 기간: {actual_start} ~ {actual_end}")
            
            # 월별 데이터 통계
            monthly_stat = samsung_data.groupby([samsung_data['date'].dt.year, samsung_data['date'].dt.month]).size()
            print(f"\n월별 데이터 분포:")
            for (year, month), count in monthly_stat.items():
                print(f"  {year}년 {month}월: {count}개")
            
            # 가장 오래된 데이터와 최신 데이터 확인
            print(f"\n가장 오래된 데이터:")
            print(samsung_data.head(3))
            print(f"\n가장 최신 데이터:")
            print(samsung_data.tail(3))
            
            # 거래일 수 계산
            trading_days = samsung_data['date'].dt.date.nunique()
            print(f"\n총 거래일 수: {trading_days}일")
            print(f"일평균 분봉 개수: {len(samsung_data) / trading_days:.0f}개")
            
            # 음수 데이터 확인
            negative_data = samsung_data[(samsung_data['open'] < 0) | 
                                       (samsung_data['high'] < 0) | 
                                       (samsung_data['low'] < 0) | 
                                       (samsung_data['close'] < 0)]
            
            if len(negative_data) > 0:
                print(f"\n경고: {len(negative_data)}개의 음수 데이터가 발견되었습니다:")
                print(negative_data.head())
            else:
                print("\n✓ 모든 가격 데이터가 정상입니다.")
            
            # 데이터 파일로 저장
            db_dir = Path(__file__).parent.parent / "db"
            db_dir.mkdir(exist_ok=True)
            
            # date 컬럼을 보기 좋은 형식으로 변환
            samsung_data_copy = samsung_data.copy()
            samsung_data_copy['date'] = samsung_data_copy['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 파일명 설정
            output_path = db_dir / f"samsung_1min_{start_date}_{end_date}.csv"
            
            samsung_data_copy.to_csv(output_path, index=False)
            print(f"\n데이터 저장 완료: {output_path}")
            print(f"파일 크기: {output_path.stat().st_size / (1024*1024):.2f} MB")
            
            # 샘플 데이터 확인
            print(f"\n저장된 데이터 샘플:")
            print(samsung_data_copy.head())
        else:
            print("데이터를 가져오지 못했습니다.")
            print("키움 API 제한으로 인해 해당 기간의 데이터를 가져올 수 없을 수 있습니다.")
    else:
        print("서버 연결 실패")
        
    app.exec_()