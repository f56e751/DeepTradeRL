import gym
import torch
import numpy as np
import sys
import os

from ..data_handler import merge_lob_and_ohlcv, merge_lob_and_ohlcv_extended, DataSplitter, Sc201OHLCVHandler, Sc202OHLCVHandler, Sc203OHLCVHandler, Sc203OHLCVTechHandler
from ..trading_env import MinutelyOrderbookOHLCVEnv, InputType
from ..deeplob import deeplob



class LSTMObsWrapper(gym.Wrapper):
    def __init__(self, env, pretrained_lstm: torch.nn.Module,
                 train_seq_len: int,  # e.g. 100
                 device='cpu'):
        super().__init__(env)
        self.pretrained = pretrained_lstm.to(device)
        self.device = device
        self.train_seq_len = train_seq_len

        # env.observation_space이 Dict인 걸 확인
        assert isinstance(env.observation_space, gym.spaces.Dict)
        # LSTM 입력용 시퀀스 크기와 MLP 입력 벡터 크기 가져오기
        T_raw, D_snap = env.observation_space.spaces['lstm_snapshots'].shape
        D_mlp         = env.observation_space.spaces['mlp_input'].shape[0]
        
        # DeepLOB 내부 LSTM hidden_size
        H = self.pretrained.lstm.hidden_size
        # 최종 반환 차원 = LSTM 피처(H) + MLP 피처(D_mlp)
        total_dim     = H + D_mlp




        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,), dtype=np.float32
        )
        self.action_space = env.action_space
        self.temp_env = env

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        return self._make_obs(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        return self._make_obs(raw_obs), reward, done, info

    def _make_obs(self, obs_dict):
        # --- 1) 패딩하여 (T₀, D_snap) 로 맞추기 ---
        snaps = obs_dict['lstm_snapshots']            # (T_raw, D_snap)
        T_raw, D_snap = snaps.shape
        if T_raw < self.train_seq_len:
            pad_len = self.train_seq_len - T_raw
            pad = np.zeros((pad_len, D_snap), dtype=snaps.dtype)
            snaps_padded = np.vstack([pad, snaps])
        else:
            snaps_padded = snaps[-self.train_seq_len:]  # 길면 오른쪽 끝만

        # --- 2) conv/inception/LSTM pipeline ---
        x = torch.as_tensor(snaps_padded, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(1)  # (1,1,T₀,D_snap)
        with torch.no_grad():
            # conv block 1
            x = self.pretrained.conv1(x)
            # conv block 2
            x = self.pretrained.conv2(x)
            # conv block 3
            x = self.pretrained.conv3(x)

            # inception branches
            x1 = self.pretrained.inp1(x)
            x2 = self.pretrained.inp2(x)
            x3 = self.pretrained.inp3(x)

            # concat → (1,192,T′,1)
            x_cat = torch.cat([x1, x2, x3], dim=1)

            # LSTM 시퀀스 차원으로 정리 → (1, T′, 192)
            x_seq = x_cat.squeeze(-1).permute(0, 2, 1)

            # LSTM forward → (1, T′, H)
            lstm_out, _ = self.pretrained.lstm(x_seq)

        # 마지막 타임스텝만 떼어내 (H,)
        feats_lstm = lstm_out[:, -1, :].squeeze(0).cpu().numpy()

        # --- 3) others flatten ---
        vec_mlp = obs_dict['mlp_input']

        # --- 4) concat & 반환 ---
        return np.concatenate([feats_lstm, vec_mlp], axis=0)
    
    def _get_mid_price(self):
        return self.temp_env._get_mid_price()


def load_pretrained_lstm():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_lstm = torch.load(
        "src/deeplob/best_pretrained_deeplob",
        map_location=device,
        weights_only=False
    )
    pretrained_lstm.to(device).eval()

    # 파라미터 업데이트 방지
    for param in pretrained_lstm.parameters():
        param.requires_grad = False

    return pretrained_lstm


# 사용 예:
# env = MinutelyOrderbookOHLCVEnv(..., input_type=InputType.LSTM)
# pretrained_lstm = MyPretrainedLSTM()  # .output_dim 속성 필요
# wrapped = LSTMObsWrapper(env, pretrained_lstm, device=device)
# model = PPO('MlpPolicy', wrapped, ...)
if __name__ == "__main__":
    # 1) 데이터 준비
    lob_csv_path = "src/db/AAPL_minute_orderbook_2019_01-07_combined.csv"
    ohlcv_csv_path = "src/db/AAPL_minute_ohlcv_2019_01-07_combined.csv"
    df_all = merge_lob_and_ohlcv(lob_csv_path, ohlcv_csv_path)
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    df_train, df_val, df_test = splitter.split(df_all)

    # 2) 환경 생성 (LSTM 모드)
    env = MinutelyOrderbookOHLCVEnv(
        df=df_train,
        handler_cls=Sc203OHLCVHandler,
        initial_cash=1e10,
        lob_levels=10,
        lookback=9,
        window_size=9,
        input_type=InputType.LSTM,
        transaction_fee=0.0023,
        h_max=100,
        hold_threshold=0.2,
    )

    # 3) Pretrained LSTM 모델 로드 (모델 전체)
    pretrained_lstm = load_pretrained_lstm()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # 4) Wrapper 적용
    wrapped_env = LSTMObsWrapper(env, 
                                 pretrained_lstm, 
                                 train_seq_len=100, 
                                 device=device)

    # 5) 테스트
    obs = wrapped_env.reset()
    print(">>> Wrapped obs shape after reset:", obs.shape)

    for i in range(5):
        action = wrapped_env.action_space.sample()
        obs, reward, done, info = wrapped_env.step(action)
        print(f"Step {i} | obs shape: {obs.shape} | reward: {reward:.2f}")
        if done:
            break