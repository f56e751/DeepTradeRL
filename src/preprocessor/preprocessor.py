import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

# ---------------------------
# 1. 데이터 로딩 및 정리
# ---------------------------
class LOBDataset(Dataset):
    def __init__(self, csv_path, window_size=100):
        self.df = pd.read_csv(csv_path)

        # DeepLOB 입력: 40개 피처 선택 (10레벨 bid/ask 가격+수량)
        self.features = [
            f"{side}_px_{i:02d}" for i in range(10) for side in ['ask', 'bid']
        ] + [
            f"{side}_sz_{i:02d}" for i in range(10) for side in ['ask', 'bid']
        ]

        # 결측치 보간: forward fill 후 backward fill
        self.df[self.features] = self.df[self.features].fillna(method='ffill').fillna(method='bfill')

        # z-score 정규화
        self.scaler = StandardScaler()
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

        self.X = self.df[self.features].values
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.window_size]  # shape: (window_size, 40)
        return torch.tensor(x, dtype=torch.float32)


# ---------------------------
# 2. DeepLOB 모델 정의
# ---------------------------
class DeepLOBFeatureExtractor(nn.Module):
    def __init__(self, input_features=40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, 16, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01)
        )
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)      # (B, F, T) for Conv1d
        x = self.conv(x)            # (B, 16, T)
        x = x.permute(0, 2, 1)      # (B, T, 16) for LSTM
        _, (h_n, _) = self.lstm(x)  # h_n: (1, B, 64)
        return h_n.squeeze(0)       # (B, 64)


# ---------------------------
# 3. 전처리 파이프라인 실행 함수
# ---------------------------
def generate_features(csv_path, output_dir, window_size=100, batch_size=256):
    print(f"[INFO] Loading dataset from: {csv_path}")
    dataset = LOBDataset(csv_path, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("[INFO] Initializing DeepLOB model...")
    model = DeepLOBFeatureExtractor()
    model.eval()

    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            features = model(batch)
            all_features.append(features.numpy())

    final_array = np.concatenate(all_features, axis=0)  # (N, 64)

    os.makedirs(output_dir, exist_ok=True)

    # csv로 저장
    csv_path = os.path.join(output_dir, 'preprocessed_LOB.csv')
    pd.DataFrame(final_array, columns=[f"lob_feat_{i}" for i in range(final_array.shape[1])]).to_csv(csv_path, index=False)


# ---------------------------
# 4. 실행 예시
# ---------------------------
if __name__ == '__main__':
    generate_features(
        csv_path='src/db/AAPL_minute_orderbook_2019_01-07_combined.csv',
        output_dir='src/db/preprocessed'
    )
