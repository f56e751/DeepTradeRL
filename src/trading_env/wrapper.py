import gymnasium as gym
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
             # obs_list가 비어있는 경우 (예: 에피소드가 한 스텝만에 끝나는 경우)
             self.obs_min = 0
             self.scale = 1
        else:
            obs_matrix = np.array(obs_list)
            self.obs_min = obs_matrix.min(axis=0)
            obs_max = obs_matrix.max(axis=0)
            self.scale = obs_max - self.obs_min
            self.scale[self.scale == 0] = 1.0 # 0으로 나누는 것을 방지

        print("NormalizationWrapper: 통계치 수집 완료.")

    def _normalize(self, obs):
        return np.clip((obs - self.obs_min) / self.scale, 0.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return (self._normalize(obs), info) if obs is not None else (None, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return None, reward, terminated, truncated, info
        return self._normalize(obs), reward, terminated, truncated, info


# class FlattenDictWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
        
#         # 기존 Dict 스페이스의 각 값들의 shape을 가져와서 합산
#         total_dim = sum(
#             np.prod(space.shape) for space in self.observation_space.spaces.values()
#         )
        
#         self.observation_space = gym.spaces.Box(
#             low=-np.inf, high=np.inf, shape=(int(total_dim),), dtype=np.float32
#         )

#     def observation(self, observation):
#         # Dict의 값들을 순서대로 flatten하여 하나의 벡터로 합침
#         return np.concatenate(
#             [obs.flatten() for obs in observation.values()]
#         )

class PortfolioScalingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        print("\n[경고] PortfolioScalingWrapper: 현재 스케일링 로직은 OHLCV와 같은 가격 기반의 시계열 데이터에 최적화되어 있습니다.")
        print("      - 만약 관측값에 거래량(volume)과 같은 수량 기반 데이터가 포함될 경우, 이 값들도 포트폴리오 가치로 나누어지므로 의도치 않은 결과가 발생할 수 있습니다.")
        print("      - 향후 가격 정보가 아닌 데이터를 시계열 관측에 포함하려면 스케일링 로직 수정이 필요합니다.\n")
        self.env = env

        # 관측 공간의 형태는 바뀌지 않지만, 일부 요소의 의미는 변경됩니다.
        # 값의 범위 또한 변경될 수 있지만, 여전히 [-inf, inf] 내에 있다고 가정합니다.
        # 값들은 비율이 될 것입니다.
        
        # 빠른 접근을 위해 인덱스 저장
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
        
        # 이 Wrapper는 UnifiedTradingEnv를 위해 설계되었습니다.
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
        
        # 환경의 데이터프레임에서 OHLCV 컬럼 인덱스 가져오기
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df_cols = unwrapped_env.df.columns.tolist()
        col_indices = {col: i for i, col in enumerate(df_cols)}
        self._ohlcv_indices_in_df = [col_indices.get(c) for c in ohlcv_cols]
        self._ohlcv_indices_in_df = [i for i in self._ohlcv_indices_in_df if i is not None]

        self._volume_col_idx = col_indices.get('volume', -1)
        self._close_col_idx = col_indices.get('close', -1)

        print("INFO: PortfolioScalingWrapper 인덱스 초기화 완료")
        print(f"      - PnL 인덱스: {self._pnl_idx}")
        print(f"      - 포지션 인덱스: {self._pos_idx}")
        print(f"      - 현금 인덱스: {self._cash_idx}")
        print(f"      - 포트폴리오 가치 인덱스: {self._pf_val_idx}")

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
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
            return obs # 포트폴리오 가치 없이는 스케일링 불가

        portfolio_value = obs[self._pf_val_idx]

        if portfolio_value <= 1e-9: # 0으로 나누기 방지
            return obs

        scaled_obs = obs.copy()
        unwrapped_env = self.env.unwrapped

        # 1. OHLCV 데이터 스케일링
        num_df_feats = unwrapped_env.df.shape[1]
        for i in range(unwrapped_env.lookback):
            # 현 시점(i)의 종가 가져오기
            close_price_obs_idx = i * num_df_feats + self._close_col_idx
            if self._close_col_idx == -1 or close_price_obs_idx >= len(obs):
                continue # 종가 없이는 volume 스케일링 불가
            close_price = obs[close_price_obs_idx]

            for col_idx in self._ohlcv_indices_in_df:
                obs_idx = i * num_df_feats + col_idx
                
                # volume은 거래대금으로 변환 후 스케일링
                if col_idx == self._volume_col_idx:
                    volume = obs[obs_idx]
                    transaction_amount = volume * close_price
                    scaled_obs[obs_idx] = transaction_amount / portfolio_value
                # 다른 OHLC 값들은 그대로 스케일링
                else:
                    scaled_obs[obs_idx] = obs[obs_idx] / portfolio_value
        
        # 2. PnL 스케일링
        if self._pnl_idx != -1:
            pnl = obs[self._pnl_idx]
            scaled_obs[self._pnl_idx] = pnl / portfolio_value

        # 3. 포지션 가치 스케일링 (기존 position을 대체)
        if self._pos_idx != -1:
            price = unwrapped_env.get_price()
            position_qty = unwrapped_env.inventory.get_position('TICKER')
            position_value = position_qty * price
            scaled_obs[self._pos_idx] = position_value / portfolio_value

        # 4. 현금 스케일링 (기존 cash를 대체)
        if self._cash_idx != -1:
            cash = obs[self._cash_idx]
            scaled_obs[self._cash_idx] = cash / portfolio_value

        # 5. 포트폴리오 가치 스케일링 (기존 portfolio_value를 대체, 항상 1.0이 됨)
        if self._pf_val_idx != -1:
            scaled_obs[self._pf_val_idx] = 1.0   
        return scaled_obs