import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NormalizationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        print("NormalizationWrapper: 관측 데이터의 통계치를 수집합니다...")
        obs_list = []
        obs, info = self.env.reset()
        if obs is not None:
            obs_list.append(obs)
        
        done = False
        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if obs is not None:
                obs_list.append(obs)
        
        if not obs_list:
             self.obs_min = 0
             self.scale = 1
        else:
            obs_matrix = np.array(obs_list)
            self.obs_min = obs_matrix.min(axis=0)
            obs_max = obs_matrix.max(axis=0)
            self.scale = obs_max - self.obs_min
            self.scale[self.scale == 0] = 1.0

        print("NormalizationWrapper: 통계치 수집 완료.")

    def _normalize(self, obs):
        return np.clip((obs - self.obs_min) / self.scale, 0.0, 1.0).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return (self._normalize(obs), info) if obs is not None else (None, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return None, reward, terminated, truncated, info
        return self._normalize(obs), reward, terminated, truncated, info

class PortfolioScalingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        print("\n[경고] PortfolioScalingWrapper: 현재 스케일링 로직은 OHLCV와 같은 가격 기반의 시계열 데이터에 최적화되어 있습니다.")
        print("      - 만약 관측값에 거래량(volume)과 같은 수량 기반 데이터가 포함될 경우, 이 값들도 포트폴리오 가치로 나누어지므로 의도치 않은 결과가 발생할 수 있습니다.")
        print("      - 향후 가격 정보가 아닌 데이터를 시계열 관측에 포함하려면 스케일링 로직 수정이 필요합니다.\n")
        self.env = env
        self._pnl_idx = -1
        self._pos_idx = -1
        self._cash_idx = -1
        self._pf_val_idx = -1
        self._volume_col_idx = -1
        self._close_col_idx = -1
        self._ohlcv_indices_in_df = []
        self._df_feats_len = 0
        self._locate_indices()

    def _locate_indices(self):
        unwrapped_env = self.env.unwrapped
        if not hasattr(unwrapped_env, 'include_pnl'):
            print("경고: PortfolioScalingWrapper는 include_pnl 등과 같은 속성을 가진 환경을 위해 설계되었습니다.")
            return
        self._df_feats_len = unwrapped_env.df.shape[1] * unwrapped_env.lookback
        current_idx = self._df_feats_len
        if unwrapped_env.include_pnl:
            self._pnl_idx = current_idx
            current_idx += 1
        if unwrapped_env.include_position:
            self._pos_idx = current_idx
            current_idx += 1
        if hasattr(unwrapped_env, 'include_cash') and unwrapped_env.include_cash:
            self._cash_idx = current_idx
            current_idx += 1
        if unwrapped_env.include_portfolio_value:
            self._pf_val_idx = current_idx
            current_idx += 1
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df_cols = unwrapped_env.df.columns.tolist()
        col_indices = {col: i for i, col in enumerate(df_cols)}
        self._ohlcv_indices_in_df = [col_indices.get(c) for c in ohlcv_cols]
        self._ohlcv_indices_in_df = [i for i in self._ohlcv_indices_in_df if i is not None]
        self._volume_col_idx = col_indices.get('volume', -1)
        self._close_col_idx = col_indices.get('close', -1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if obs is None:
            return None, info
        return self._scale_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return None, reward, terminated, truncated, info
        return self._scale_obs(obs), reward, terminated, truncated, info

    def _scale_obs(self, obs):
        if self._pf_val_idx == -1:
            return obs
        portfolio_value = obs[self._pf_val_idx]
        if portfolio_value <= 1e-9:
            return obs
        scaled_obs = obs.copy()
        unwrapped_env = self.env.unwrapped
        num_df_feats = unwrapped_env.df.shape[1]
        for i in range(unwrapped_env.lookback):
            close_price_obs_idx = i * num_df_feats + self._close_col_idx
            if self._close_col_idx == -1 or close_price_obs_idx >= len(obs):
                continue
            close_price = obs[close_price_obs_idx]
            for col_idx in self._ohlcv_indices_in_df:
                obs_idx = i * num_df_feats + col_idx
                if col_idx == self._volume_col_idx:
                    volume = obs[obs_idx]
                    transaction_amount = volume * close_price
                    scaled_obs[obs_idx] = transaction_amount / portfolio_value
                else:
                    scaled_obs[obs_idx] = obs[obs_idx] / portfolio_value
        if self._pnl_idx != -1:
            pnl = obs[self._pnl_idx]
            scaled_obs[self._pnl_idx] = pnl / portfolio_value
        if self._pos_idx != -1:
            price = unwrapped_env.get_price()
            position_qty = unwrapped_env.inventory.get_position('TICKER')
            position_value = position_qty * price
            scaled_obs[self._pos_idx] = position_value / portfolio_value
        if self._cash_idx != -1:
            cash = obs[self._cash_idx]
            scaled_obs[self._cash_idx] = cash / portfolio_value
        if self._pf_val_idx != -1:
            scaled_obs[self._pf_val_idx] = 1.0
        return scaled_obs

class ReshapeObservationWrapper(gym.Wrapper):
    """
    관측치를 (lookback, num_features) 형태로 변환하는 래퍼.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        unwrapped_env = self.env.unwrapped
        self.lookback = unwrapped_env.lookback
        
        df_shape = unwrapped_env.df.shape
        num_df_feats = df_shape[1]
        
        non_sequence_dim = self.observation_space.shape[0] - (self.lookback * num_df_feats)
        
        self.num_features = num_df_feats
        self.non_sequence_dim = non_sequence_dim

        self.observation_space = spaces.Dict({
            "sequence": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.lookback, self.num_features), 
                dtype=np.float32
            ),
            "instant": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.non_sequence_dim,), 
                dtype=np.float32
            )
        })

    def _reshape_obs(self, observation: np.ndarray) -> dict:
        """관측치 형태를 변환하는 내부 메소드"""
        sequence_part_flat = observation[:self.lookback * self.num_features]
        instant_part = observation[self.lookback * self.num_features:]
        sequence_part = sequence_part_flat.reshape(self.lookback, self.num_features)
        return {"sequence": sequence_part, "instant": instant_part}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._reshape_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if obs is None:
            return None, reward, terminated, truncated, info
        
        return self._reshape_obs(obs), reward, terminated, truncated, info