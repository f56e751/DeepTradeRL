# This is a majorly revised version of the model trainer.
# Key improvements:
# 1. The model now predicts logarithmic returns, aligning with the paper's methodology.
# 2. Extended features (high, low, volume) are also transformed to returns/changes.
# 3. Evaluation and visualization are adapted to plot predicted price vs. actual price.
# 4. The script now evaluates and saves plots/metrics for BOTH the final and best models.
# 5. Evaluation metrics now include NMSE as per the paper.
# 6. Added a zoomed-in plot for the first 200 test steps.
# All previous features like model selection, logging, and checkpointing are retained.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import math
import copy
import argparse
import time
import yaml
from tqdm import tqdm
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DNN(nn.Module):
    """Defines a standard MLP (DNN) model with configurable layers and dropout."""
    def __init__(self, input_size, hidden_layer_sizes=[200, 100], dropout_rate=0.5):
        super(DNN, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence for batch_first=True."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """A standard Transformer Encoder model for regression."""
    def __init__(self, input_features: int, d_model: int, n_head: int, n_layers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, d_model * 4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

def create_sequences(data, seq_length):
    """Creates sequences from the time series data."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length), :])
        ys.append(data[i + seq_length, 0])
    return np.array(xs), np.array(ys)

def load_and_prepare_data(args):
    """Loads data, calculates log returns, scales, creates sequences, and splits."""
    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"Error: The file '{args.file_path}' was not found.")

    df = pd.read_csv(args.file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    original_df = df.copy() # Keep original df for price plotting

    if args.use_extended_features:
        print("Using extended features and converting to log returns/changes.")
        feature_df = pd.DataFrame()
        feature_df['close_return'] = np.log(df['close'] / df['close'].shift(1))
        feature_df['high_return'] = np.log(df['high'] / df['high'].shift(1))
        feature_df['low_return'] = np.log(df['low'] / df['low'].shift(1))
        feature_df['volume_change'] = np.log(df['volume'] + 1e-6).diff()
    else:
        print("Using single feature: close log return")
        feature_df = pd.DataFrame()
        feature_df['close_return'] = np.log(df['close'] / df['close'].shift(1))
    
    feature_df = feature_df.dropna().reset_index(drop=True)
    # Align original_df by dropping the first row as well
    original_df = original_df.iloc[1:].reset_index(drop=True)

    time_series = feature_df.values.astype(np.float32)
    
    train_split_index = int(len(time_series) * args.train_size)
    train_series = time_series[:train_split_index]
    
    scaler = StandardScaler()
    scaler.fit(train_series)
    scaled_series = scaler.transform(time_series)

    X, y = create_sequences(scaled_series, args.sequence_length)
    
    # Also create a price sequence from the original df for evaluation
    original_prices_for_y = create_sequences(original_df[['close']].values, args.sequence_length)[1]


    if args.model_type != 'TRANSFORMER':
        X = X.reshape(X.shape[0], -1)
        print(f"Data shapes (after flattening for MLP):\nX shape: {X.shape}")
    else:
        print(f"Data shapes (sequential for Transformer):\nX shape: {X.shape}")

    train_split_idx = int(len(X) * args.train_size)
    val_split_idx = int(len(X) * (args.train_size + args.validation_size))

    X_train, y_train = X[:train_split_idx], y[:train_split_idx]
    X_val, y_val = X[train_split_idx:val_split_idx], y[train_split_idx:val_split_idx]
    X_test, y_test = X[val_split_idx:], y[val_split_idx:]
    
    # Slice the original prices to match the test set
    y_test_prices = original_prices_for_y[val_split_idx:]

    return X_train, y_train.reshape(-1, 1), X_val, y_val.reshape(-1, 1), X_test, y_test.reshape(-1, 1), scaler, y_test_prices

def train(model, train_loader, val_loader, save_dir, args):
    """Handles the training loop. Returns the model in its final state."""
    log_path = os.path.join(save_dir, 'training_log.csv')
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    model = model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    with open(log_path, 'w') as f: f.write('epoch,train_loss,val_loss,learning_rate\n')
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for inputs, targets in train_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        avg_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                avg_val_loss += criterion(model(inputs), targets).item()
        avg_val_loss /= len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_pbar.set_postfix(train_loss=f"{avg_train_loss:.6f}", val_loss=f"{avg_val_loss:.6f}", lr=f"{current_lr:.6f}")
        
        with open(log_path, 'a') as f: f.write(f'{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{current_lr}\n')
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
        writer.add_scalar('Learning_Rate', current_lr, epoch + 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs.")
            break
            
    writer.close()
    return model

def calculate_nmse(y_true, y_pred):
    """Calculates Mean Squared Error and Normalized Mean Squared Error."""
    mse = mean_squared_error(y_true, y_pred)
    var_true = np.var(y_true)
    if var_true == 0:
        return mse, np.inf
    nmse = mse / var_true
    return mse, nmse

def evaluate_and_plot(model, model_name, X_test, y_test_returns, y_test_prices, scaler, save_dir, ar_model=None, model_type='DNN_ONLY'):
    """Evaluates a model, saves metrics and plots prices."""
    print(f"\n--- Evaluating {model_name} Model on Test Set ---")
    
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    model.eval()
    with torch.no_grad():
        dnn_predictions_scaled = model(X_test_tensor).cpu().numpy()

    if model_type == 'AR_DNN':
        ar_predictions_scaled = ar_model.predict(X_test)
        final_predictions_scaled = ar_predictions_scaled + dnn_predictions_scaled
    else:
        final_predictions_scaled = dnn_predictions_scaled
    
    y_test_actual_returns = scaler.inverse_transform(np.pad(y_test_returns, ((0,0), (0, scaler.n_features_in_-1)), 'constant'))[:,0]
    final_predictions_actual_returns = scaler.inverse_transform(np.pad(final_predictions_scaled, ((0,0), (0, scaler.n_features_in_-1)), 'constant'))[:,0]
    
    # --- Metrics on Log Returns ---
    mse_final, nmse_final = calculate_nmse(y_test_actual_returns, final_predictions_actual_returns)
    rmse_final = np.sqrt(mse_final)
    actual_direction = (y_test_actual_returns > 0).astype(int)
    final_pred_direction = (final_predictions_actual_returns > 0).astype(int)
    final_direction_acc = np.mean(actual_direction == final_pred_direction)

    print(f"--- Performance of {model_name} Model (on Log Returns) ---")
    print(f"NMSE: {nmse_final:.6f}")
    print(f"RMSE: {rmse_final:.6f}")
    print(f"Direction Accuracy: {final_direction_acc:.4f}")
    
    metrics = {
        'nmse': float(nmse_final),
        'rmse': float(rmse_final),
        'direction_accuracy': float(final_direction_acc)
    }
    metrics_path = os.path.join(save_dir, f'test_metrics_{model_name.lower()}.yaml')
    with open(metrics_path, 'w') as f: yaml.dump(metrics, f, sort_keys=False)
    print(f"📄 {model_name} model metrics saved to '{metrics_path}'")

    # --- Convert predicted returns to prices for plotting ---
    previous_close_prices = y_test_prices[:-1].flatten()
    actual_close_prices = y_test_prices[1:].flatten()

    min_len = min(len(previous_close_prices), len(final_predictions_actual_returns))
    predicted_prices = previous_close_prices[:min_len] * np.exp(final_predictions_actual_returns[:min_len])

    # --- Visualization of Full Prices ---
    plt.figure(figsize=(15, 7))
    plt.plot(actual_close_prices[:min_len], label='Actual Price', color='blue', alpha=0.8)
    plt.plot(predicted_prices, label=f'{model_name} Predicted Price', color='red', linestyle='--')
    
    plt.title(f'Price Prediction ({model_name} Model): Actual vs. Predicted (Test Set)')
    plt.xlabel(f'Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    output_filename = os.path.join(save_dir, f'price_visualization_{model_name.lower()}.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"📈 {model_name} model price visualization saved to '{output_filename}'")

    # --- Zoomed-in Visualization of first 200 steps ---
    zoom_len = 200
    if len(actual_close_prices) >= zoom_len:
        print(f"--- Generating Zoomed-in Visualization (first {zoom_len} steps) ---")
        plt.figure(figsize=(15, 7))
        plt.plot(actual_close_prices[:zoom_len], label='Actual Price', color='blue', alpha=0.8)
        plt.plot(predicted_prices[:zoom_len], label=f'{model_name} Predicted Price', color='red', linestyle='--')
        
        plt.title(f'Zoomed-in Price Prediction ({model_name} Model): First {zoom_len} Steps of Test Set')
        plt.xlabel(f'Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        zoom_output_filename = os.path.join(save_dir, f'price_visualization_{model_name.lower()}_zoom.png')
        plt.savefig(zoom_output_filename)
        plt.close()
        print(f"📈 Zoomed-in visualization saved to '{zoom_output_filename}'")


def main(args):
    """Main function to run the entire training and evaluation pipeline."""
    run_name = f"{int(time.time())}_{args.tag}" if args.tag else f"{int(time.time())}"
    save_dir = os.path.join('predictor_runs', run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 All results will be saved in: {save_dir}")

    with open(os.path.join(save_dir, 'parameters.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_prices = load_and_prepare_data(args)
    scaler_path = os.path.join(save_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    ar_model = None
    input_size = X_train.shape[1]
    
    if args.model_type == 'TRANSFORMER':
        num_features = scaler.n_features_in_
        model_instance = TransformerModel(input_features=num_features, d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layers, dropout=args.transformer_dropout).to(device)
    else:
        model_instance = DNN(input_size=input_size, hidden_layer_sizes=args.hidden_layer_sizes, dropout_rate=args.dropout_rate).to(device)

    if args.model_type == 'AR_DNN':
        print("\n--- Training AR_DNN model ---")
        ar_model = LinearRegression()
        ar_model.fit(X_train, y_train)
        ar_model_path = os.path.join(save_dir, 'ar_model.pkl')
        with open(ar_model_path, 'wb') as f: joblib.dump(ar_model, f)
        print(f"✅ AR model saved to {ar_model_path}")
        
        train_residuals = y_train - ar_model.predict(X_train)
        val_residuals = y_val - ar_model.predict(X_val)
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(train_residuals).float()), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(val_residuals).float()), batch_size=args.batch_size)
    
    else:
        print(f"\n--- Training {args.model_type} model ---")
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()), batch_size=args.batch_size)

    final_model = train(model_instance, train_loader, val_loader, save_dir, args)
    
    # --- Evaluation ---
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(final_model.state_dict(), final_model_path)
    print(f"💾 Final model saved to {final_model_path}")
    evaluate_and_plot(final_model, "Final", X_test, y_test, y_test_prices, scaler, save_dir, ar_model, args.model_type)

    print("\n--- Loading and Evaluating Best Model ---")
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        if args.model_type == 'TRANSFORMER':
            num_features = scaler.n_features_in_
            best_model = TransformerModel(input_features=num_features, d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layers, dropout=args.transformer_dropout).to(device)
        else:
            best_model = DNN(input_size=input_size, hidden_layer_sizes=args.hidden_layer_sizes, dropout_rate=args.dropout_rate).to(device)
        
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        evaluate_and_plot(best_model, "Best", X_test, y_test, y_test_prices, scaler, save_dir, ar_model, args.model_type)
    else:
        print("⚠️ Best model checkpoint not found. Skipping evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model to predict logarithmic returns.')
    
    parser.add_argument('--file_path', type=str, default='src/db/BTC_USDT_15min_1year_data.csv')
    parser.add_argument('--target_column', type=str, default='close')
    parser.add_argument('--use_extended_features', action='store_true', help='Use close, high, low, volume as features.')
    parser.add_argument('--model_type', type=str, default='DNN_ONLY', choices=['AR_DNN', 'DNN_ONLY', 'TRANSFORMER'])
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[200, 100])
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--transformer_dropout', type=float, default=0.2)
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--validation_size', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--tag', type=str, default='log_return_experiment')

    args = parser.parse_args()
    main(args)

