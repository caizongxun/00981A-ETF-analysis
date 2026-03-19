#!/usr/bin/env python3
"""
訓練 Random Forest 選股模型，訓練完存出 .pkl 供回測使用
執行方式： python model/stock_selector.py
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "roe", "pe_ratio", "pb_ratio", "gross_margin",
    "operating_margin", "debt_to_equity",
    "price_mom_3m", "price_mom_6m", "volatility_60d",
    "market_cap_rank"
]

def prepare_dataset():
    features_df = pd.read_csv(DATA_DIR / "features.csv")
    holdings_df = pd.read_csv(DATA_DIR / "holdings_sample.csv")
    in_etf = set(holdings_df["stock_id"].astype(str))
    features_df["label"] = features_df["ticker"].astype(str).isin(in_etf).astype(int)
    df = features_df.dropna(subset=FEATURE_COLS)
    if len(df) == 0:
        raise ValueError("features.csv 內有效樣本為 0，請先跑 feature_engineering.py")
    return df[FEATURE_COLS].values, df["label"].values, df

def train_and_save(X, y):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        class_weight="balanced", random_state=42
    )
    if len(np.unique(y)) >= 2 and y.sum() >= 2:
        cv = StratifiedKFold(n_splits=min(5, int(y.sum())), shuffle=True, random_state=42)
        scores = cross_val_score(rf, X_sc, y, cv=cv, scoring="roc_auc")
        print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    else:
        print("[WARN] 正樣本過少，跳過交叉驗證")

    rf.fit(X_sc, y)
    with open(DATA_DIR / "rf_model.pkl",  "wb") as f: pickle.dump(rf, f)
    with open(DATA_DIR / "scaler.pkl",    "wb") as f: pickle.dump(scaler, f)
    print(f"模型已儲存: {DATA_DIR}/rf_model.pkl")
    return rf, scaler

def plot_feature_importance(model):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(imp)), imp[idx])
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in idx], rotation=45, ha="right")
    ax.set_title("00981A 選股特徵重要性")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    out = DATA_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150)
    print(f"圖表已存: {out}")

if __name__ == "__main__":
    print("準備訓練資料...")
    X, y, df = prepare_dataset()
    print(f"樣本數: {len(X)}, 正樣本(入選)數: {y.sum()}")
    model, scaler = train_and_save(X, y)
    plot_feature_importance(model)
    print("\n完成！接下來執行 python backtest/backtest.py")
