import gym
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    """
    기본 뼈대 주식 거래 환경
    - OpenAI Gym API 호환
    - reset, step, render 메서드 포함
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, initial_balance=1e6):
        super(StockTradingEnv, self).__init__()
        # 입력 데이터 (예: pandas DataFrame)
        # TODO 여기서 입력 데이터를 에이전트를 training할 때 사용하는 데이터 형식으로 입력?
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        # 행동 공간: 0 = 보유 유지, 1 = 매수, 2 = 매도
        self.action_space = spaces.Discrete(3)

        # 관측 공간: [현금잔고, 보유주식수, 가격, ...]
        obs_low = np.array([0, 0, -np.inf])
        obs_high = np.array([np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 내부 상태 초기화
        self.reset()

    def reset(self):
        # 초기 상태 설정
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0

        # 초기 관측 객체 반환
        return self._get_observation()

    # TODO 이 부분을 action을 enum으로 바꾸기
    def step(self, action):
        # action: 0=Hold, 1=Buy, 2=Sell
        done = False
        reward = 0
        price = self._get_price()

        # 간단한 매수/매도 로직
        if action == 1:  # Buy
            self.shares_held += 1
            self.balance -= price
        elif action == 2 and self.shares_held > 0:  # Sell
            self.shares_held -= 1
            self.balance += price

        # 다음 스텝 이동
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        # 보상: 포트폴리오 가치 변화
        portfolio_value = self.balance + self.shares_held * price
        reward = portfolio_value - self.initial_balance

        obs = self._get_observation()
        info = {"balance": self.balance, "shares_held": self.shares_held}

        return obs, reward, done, info

    def render(self, mode="human"):
        # 단순 출력 예시
        price = self._get_price()
        print(f"Step: {self.current_step}, Price: {price:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held}")

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    
    
    # TODO 이 부분을 OCP를 준수할 수 있게 작성하기
    # ============= 일봉으로 ============
    # 1. 한 주식에 대해서만 적용 가능하게
    # 2. 여러 주식이 있는 환경에 대해서 적용 가능하게
    # ============= 분봉으로 ============
    # 3. 분봉으로 여러 주식이 있는 환경에 대해서 적용 가능하게


    # observation으로 추출할 데이터
    # 1. 
    def _get_observation(self):
        # 예시 관측: [balance, shares_held, current_price]
        price = self._get_price()
        return np.array([self.balance, self.shares_held, price], dtype=np.float32)

    def _get_price(self):
        # 현재 스텝의 종가 반환
        return self.df.loc[self.current_step, "Close"]
