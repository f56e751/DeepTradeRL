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
