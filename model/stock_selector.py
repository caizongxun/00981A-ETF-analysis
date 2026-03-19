#!/usr/bin/env python3
"""
機器學習選股模型：學習 00981A 選股邏輯，訓練完後儲存模型供回測使用
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "roe", "pe_ratio", "pb_ratio", "gross_margin",
    "operating_margin", "debt_to_equity",
    "price_mom_3m", "price_mom_6m", "volatility_60d",
    "market_cap_rank"
]

def prepare_dataset(features_path="../data/features.csv",
                    holdings_path="../data/holdings_sample.csv"):
    features_df = pd.read_csv(features_path)
    holdings_df = pd.read_csv(holdings_path)
    in_etf = set(holdings_df["stock_id"].astype(str))
    features_df["label"] = features_df["ticker"].astype(str).isin(in_etf).astype(int)
    df = features_df.dropna(subset=FEATURE_COLS)
    return df[FEATURE_COLS].values, df["label"].values, df

def train_and_save(X, y, out_dir="../data"):
    os.makedirs(out_dir, exist_ok=True)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_sc, y, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    rf.fit(X_sc, y)

    # 儲存模型與 scaler（供 backtest.py 載入）
    with open(f"{out_dir}/rf_model.pkl",  "wb") as f: pickle.dump(rf, f)
    with open(f"{out_dir}/scaler.pkl",    "wb") as f: pickle.dump(scaler, f)
    print("模型已儲存: data/rf_model.pkl, data/scaler.pkl")
    return rf, scaler

def plot_feature_importance(model, output_path="../data/feature_importance.png"):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(imp)), imp[idx])
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in idx], rotation=45, ha="right")
    ax.set_title("00981A 選股特徵重要性")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"特徵重要性圖已存: {output_path}")

if __name__ == "__main__":
    print("準備訓練資料...")
    X, y, df = prepare_dataset()
    print(f"樣本數: {len(X)}, 正樣本: {y.sum()}")
    model, scaler = train_and_save(X, y)
    plot_feature_importance(model)
    print("\n完成！接下來執行 backtest/backtest.py 進行回測")
