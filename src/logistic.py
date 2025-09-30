# single_asset_trend_logreg.py
# CSV: timestamp,open,high,low,close,volume
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import logging
logging.basicConfig(level=logging.INFO)

def drop_first_g_each_day(df, g, ts_col):
    d = df.copy()
    d["__day__"] = pd.to_datetime(d[ts_col]).dt.tz_localize(None).dt.date
    d["__rank__"] = d.groupby("__day__").cumcount()
    d = d[d["__rank__"] >= g].drop(columns=["__day__", "__rank__"])
    return d

def build_lag_feats(r, g):
    # r: 1D array of normalized returns
    X = []
    for t in range(g-1, len(r)-1):  # predict t+1
        X.append([r[t-k] for k in range(g)])  # [r_t, r_{t-1}, ..., r_{t-g+1}]
    return np.asarray(X, dtype=np.float32)

def make_labels(next_r, eps):
    y = np.zeros_like(next_r, dtype=np.int32)
    y[next_r > eps] = 1
    y[next_r < -eps] = -1
    return y

def norm_train_test(train, test):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--timestamp_col", default="timestamp")
    ap.add_argument("--g", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--use_pca", action="store_true")
    ap.add_argument("--pca_dim", type=int, default=10)  # g<=10이면 의미 없음
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values(args.timestamp_col).reset_index(drop=True)
    logging.info("Loaded dataframe shape: %s (rows=%d, cols=%d)", df.shape, df.shape[0], df.shape[1])
    
    # log-return from close
    close = df["close"].to_numpy(dtype=np.float64)
    rt = np.zeros_like(close)
    rt[1:] = np.log(close[1:] / close[:-1])

    # attach into frame for per-day dropping of first g bars
    df_r = df[[args.timestamp_col]].copy()
    df_r["r"] = rt
    df_r = drop_first_g_each_day(df_r, args.g, args.timestamp_col)

    r = df_r["r"].to_numpy(dtype=np.float32)

    # build lag features and targets
    X_all = build_lag_feats(r, args.g)                # shape [N, g]
    next_r = r[args.g:]                               # aligned r_{t+1}
    y_all = make_labels(next_r, args.eps)             # {-1,0,+1}


    
    # time split 80/20
    N = X_all.shape[0]
    ntr = int(N * 0.8)
    tr_idx = np.arange(ntr)
    te_idx = np.arange(ntr, N)

    # normalize using TRAIN stats (논문 규칙)
    X_tr_raw, X_te_raw = norm_train_test(X_all[tr_idx], X_all[te_idx])

    # keep only ±1
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]
    mtr = y_tr != 0
    mte = y_te != 0

    Xtr = X_tr_raw[mtr]
    Xte = X_te_raw[mte]
    ytr = y_tr[mtr]
    yte = y_te[mte]

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0 or len(np.unique(ytr)) < 2:
        print("유효 샘플 부족.")
        return

    # optional PCA (RawData가 기본. 논문 요약: RawData ≈ PCA380)
    if args.use_pca:
        pca_k = min(args.pca_dim, Xtr.shape[1])
        pca = PCA(n_components=pca_k, svd_solver="full", whiten=False, random_state=0)
        pca.fit(Xtr)
        Xtr = pca.transform(Xtr).astype(np.float32)
        Xte = pca.transform(Xte).astype(np.float32)

    # Logistic Regression
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=1000, random_state=0)
    clf.fit(Xtr, ytr)

    acc_tr = (clf.predict(Xtr) == ytr).mean()
    y_pred_te = clf.predict(Xte)
    acc_te = (y_pred_te == yte).mean()
    ref = max((yte == 1).mean(), (yte == -1).mean())

    f1 = f1_score(yte, y_pred_te, average="binary", pos_label=1)

    print(f"Samples  train={Xtr.shape[0]}  test={Xte.shape[0]}")
    print(f"Accuracy train={acc_tr:.4f}  test={acc_te:.4f}  reference={ref:.4f}  f1={f1:.4f}")

if __name__ == "__main__":
    main()
