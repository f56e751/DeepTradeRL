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
            
    def _opt10080(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)
        
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
            
            self.ohlcv['date'].append(date.strip())
            self.ohlcv['open'].append(int(open_price.strip()))
            self.ohlcv['high'].append(int(high_price.strip()))
            self.ohlcv['low'].append(int(low_price.strip()))
            self.ohlcv['close'].append(int(close_price.strip()))
            self.ohlcv['volume'].append(int(volume.strip()))
    
    def get_min_data(self, code, tick_range, start_date=None, end_date=None):
        """
        분봉 데이터 요청
        :param code: 종목코드
        :param tick_range: 분봉 범위 (5: 5분, 15: 15분, 30: 30분, 60: 60분, 240: 240분)
        :param start_date: 조회 시작일 (없으면 가장 최근 데이터)
        :return: DataFrame
        """
        self.ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        self.set_input_value("종목코드", code)
        self.set_input_value("틱범위", tick_range)
        self.set_input_value("수정주가구분", 1)
        
        if start_date:
            self.set_input_value("시작일자", start_date)
        
        rqname = "opt10080_req"
        trcode = "opt10080"
        next = 0
        screen_no = "0101"
        
        # 최초 조회
        self.comm_rq_data(rqname, trcode, next, screen_no)
        
        # 데이터가 더 있는 경우 연속 조회
        while self.remained_data:
            # 연속 조회 시 딜레이 
            time.sleep(0.2)
            self.set_input_value("종목코드", code)
            self.set_input_value("틱범위", tick_range)
            self.set_input_value("수정주가구분", 1)
            self.comm_rq_data(rqname, trcode, 2, screen_no)
        
        # 데이터 프레임으로 변환
        df = pd.DataFrame(self.ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # 날짜 형식 변환
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S')
        
        # 최신 데이터가 위에 오도록 정렬
        df = df.sort_values('date', ascending=False)
        df = df.reset_index(drop=True)
        
        # 6개월 필터링 (필요시)
        if end_date:
            df = df[df['date'] <= end_date]
        
        if start_date:
            six_months_ago = pd.to_datetime(start_date, format='%Y%m%d')
            df = df[df['date'] >= six_months_ago]
        
        return df

if __name__ == "__main__":
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom.comm_connect()
    
    # 연결 상태 확인
    if kiwoom.get_connect_state() == 1:
        print("서버 연결 성공")
        
        # 오늘 날짜 기준 6개월 전 날짜 계산
        today = datetime.today().strftime("%Y%m%d")
        six_months_ago = (datetime.today() - timedelta(days=365)).strftime("%Y%m%d")
        
        # 삼성전자(005930) 3분봉 데이터 가져오기
        samsung_data = kiwoom.get_min_data("005930", "3", six_months_ago)
        
        print(f"데이터 수: {len(samsung_data)}")
        print(samsung_data.head())
        
        # 데이터 파일로 저장
        # 디렉토리 확인 및 생성
        # 현재 파일 경로를 기준으로 상대 경로 설정
        db_dir = Path(__file__).parent.parent / "db"
        output_path = db_dir / "samsung_3min_data.csv"
        samsung_data.to_csv(output_path, index=False)
        print("데이터가 samsung_3min_data.csv 파일로 저장되었습니다.")
    else:
        print("서버 연결 실패")
        
    app.exec_()