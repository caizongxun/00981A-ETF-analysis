#!/usr/bin/env python3
"""
機器學習選股模型：嘗試學習 00981A 經理人的選股邏輯
策略：以歷史持股為正樣本，未持股為負樣本，訓練二元分類器
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "roe", "pe_ratio", "pb_ratio", "gross_margin",
    "operating_margin", "debt_to_equity",
    "price_mom_3m", "price_mom_6m", "volatility_60d",
    "market_cap_rank"
]

def prepare_dataset(features_path="../data/features.csv",
                    holdings_path="../data/holdings_sample.csv") -> tuple:
    """
    合併特徵矩陣與持股標籤，構建訓練資料集
    """
    features_df = pd.read_csv(features_path)
    holdings_df = pd.read_csv(holdings_path)
    
    in_etf_tickers = set(holdings_df["stock_id"].astype(str))
    features_df["label"] = features_df["ticker"].astype(str).isin(in_etf_tickers).astype(int)
    
    df = features_df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    return X, y, df

def train_model(X, y):
    """
    訓練 Random Forest 分類器，並做交叉驗證
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="roc_auc")
    
    print(f"Random Forest CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    rf.fit(X_scaled, y)
    return rf, scaler

def plot_feature_importance(model, feature_names):
    """
    畫出特徵重要性，揭示哪些指標最能區別是否入選
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("00981A 選股特徵重要性（Random Forest）")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig("../data/feature_importance.png", dpi=150)
    print("圖表已儲存: data/feature_importance.png")

def predict_new_stocks(model, scaler, candidates_df: pd.DataFrame):
    """
    預測新股票被選入 00981A 的機率
    """
    X = candidates_df[FEATURE_COLS].fillna(candidates_df[FEATURE_COLS].median()).values
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    candidates_df = candidates_df.copy()
    candidates_df["select_prob"] = proba
    return candidates_df.sort_values("select_prob", ascending=False)

if __name__ == "__main__":
    print("準備訓練資料...")
    X, y, df = prepare_dataset()
    print(f"資料集大小: {len(X)} 筆, 正樣本: {y.sum()} 筆")
    
    print("\n訓練 Random Forest 模型...")
    model, scaler = train_model(X, y)
    
    print("\n特徵重要性分析...")
    plot_feature_importance(model, FEATURE_COLS)
    
    print("\n完成！可使用 predict_new_stocks() 預測新股票入選機率")
