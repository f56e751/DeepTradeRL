import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------
# DeepLOB Model Definition
# -------------------------------
class deeplob(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len
        self.dropout = nn.Dropout(p=0.2)  

        # Convolution Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32)
        )

        # Convolution Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32)
        )

        # Convolution Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32)
        )

        # Inception Modules
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64)
        )

        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64)
        )

        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64)
        )

        # LSTM & FC
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        # Concatenate: (B, 192, T, 1)
        x = torch.cat([x_inp1, x_inp2, x_inp3], dim=1)

        # Reshape for LSTM: (B, T, 192)
        x = x.squeeze(-1).permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 마지막 timestep
        x = self.fc1(x)

        return x  # CrossEntropyLoss에 softmax 포함되어 있으므로 여기선 미적용

# -------------------------------
# Load and preprocess data
# -------------------------------
def load_input_data(csv_path, T=100):
    df = pd.read_csv(csv_path)
    features = []
    for lvl in range(10):
        lvl_str = f"{lvl:02d}"
        features += [f"ask_px_{lvl_str}", f"ask_sz_{lvl_str}",
                    f"bid_px_{lvl_str}", f"bid_sz_{lvl_str}"]
    
    # NaN 보간 또는 대체
    df[features] = df[features].interpolate(method='linear', limit_direction='both')

    # 또는 평균/0으로 대체 (보간 불가능한 경우 대비)
    df[features] = df[features].fillna(method='bfill').fillna(method='ffill').fillna(0)

    X = df[features].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq = np.zeros((len(X_scaled) - T + 1, T, X_scaled.shape[1]), dtype=np.float32)
    for i in range(T, len(X_scaled) + 1):
        X_seq[i - T] = X_scaled[i - T:i]

    return torch.tensor(X_seq).unsqueeze(1)  # shape: (N, 1, T, 40)

# -------------------------------
# Load pretrained model
# -------------------------------
def load_model(weight_path, device='cpu'):
    model = torch.load('best_pretrained_deeplob', map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


# -------------------------------
# Predict and save logits
# -------------------------------
from torch.utils.data import DataLoader, TensorDataset

def predict_logits(model, x_input, device='cuda', batch_size=512):
    model.eval()
    loader = DataLoader(TensorDataset(x_input), batch_size=batch_size)
    results = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy()
            results.append(probs)

    return np.concatenate(results, axis=0)


def save_logits_to_csv(logits, path='logit_predictions.csv'):
    df = pd.DataFrame(logits, columns=['logit_up', 'logit_stay', 'logit_down'])
    df['pred_label'] = np.argmax(logits, axis=1)
    df.to_csv(path, index_label='timestep')

# -------------------------------
# Run when executed directly
# -------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = 'AAPL_minute_orderbook_2019_01-07_combined.csv'
    model_path = 'best_pretrained_deeplob'

    print("[INFO] Loading data...")
    x_input = load_input_data(data_path)

    print("[INFO] Loading model...")
    model = load_model(model_path, device)

    print("[INFO] Predicting logits...")
    logits = predict_logits(model, x_input, device)

    print("[INFO] Saving to CSV...")
    save_logits_to_csv(logits)

    print("[INFO] Done.")
