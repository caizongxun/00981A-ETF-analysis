#!/usr/bin/env python3
"""
日頻異常偵測模型訓練

輸入:  data/daily_features.csv
輸出:  models/daily_up_model.pkl
       models/daily_down_model.pkl
       models/daily_feature_cols.json

策略:
  - 以時間切分 walk-forward (80% train / 20% val)
  - 用 LightGBM 分類器 (binary)
  - SMOTE 處理樣本不均衡
  - 輸出 feature importance
"""
import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

IN_CSV     = DATA_DIR / "daily_features.csv"
UP_MODEL   = MODEL_DIR / "daily_up_model.pkl"
DOWN_MODEL = MODEL_DIR / "daily_down_model.pkl"
FEAT_JSON  = MODEL_DIR / "daily_feature_cols.json"

FEATURE_COLS = [
    "vol_ratio_5d", "vol_ratio_20d", "vol_zscore_20d",
    "ret_1d", "ret_5d", "high_low_pct",
    "close_vs_ma5", "close_vs_ma20",
    "inst_net_ratio", "margin_chg_pct",
]


def train_lgbm(X_train, y_train, X_val, y_val, label_name: str):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] pip install lightgbm")
        sys.exit(1)

    pos_rate = y_train.mean()
    scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    from sklearn.metrics import roc_auc_score, classification_report
    pred = model.predict(X_val)
    prob = model.predict_proba(X_val)[:, 1]
    auc  = roc_auc_score(y_val, prob)
    print(f"[{label_name}] AUC={auc:.4f}")
    print(classification_report(y_val, pred, zero_division=0))

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print(f"Top 5 features:")
    print(imp.nlargest(5).to_string())
    return model


def main():
    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    # 只保留有足夠資料的列
    avail = [c for c in FEATURE_COLS if c in df.columns]
    df.dropna(subset=avail + ["label_up5", "label_down5"], inplace=True)
    print(f"有效樣本: {len(df):,}")

    # 時間切分
    cutoff = df["date"].quantile(0.8)
    train  = df[df["date"] <= cutoff]
    val    = df[df["date"] >  cutoff]
    print(f"Train: {len(train):,} ({train['date'].min().date()} ~ {train['date'].max().date()})")
    print(f"Val  : {len(val):,}   ({val['date'].min().date()} ~ {val['date'].max().date()})")

    X_train = train[avail].values.astype(np.float32)
    X_val   = val[avail].values.astype(np.float32)

    # 上漲異常
    print("\n--- 訓練上漲異常模型 ---")
    up_model = train_lgbm(X_train, train["label_up5"].values,
                          X_val,   val["label_up5"].values,
                          "label_up5")
    with open(UP_MODEL, "wb") as f:
        pickle.dump(up_model, f)
    print(f"儲存: {UP_MODEL}")

    # 下跌異常
    print("\n--- 訓練下跌異常模型 ---")
    down_model = train_lgbm(X_train, train["label_down5"].values,
                            X_val,   val["label_down5"].values,
                            "label_down5")
    with open(DOWN_MODEL, "wb") as f:
        pickle.dump(down_model, f)
    print(f"儲存: {DOWN_MODEL}")

    # 儲存特徵欄位
    with open(FEAT_JSON, "w") as f:
        json.dump(avail, f)
    print(f"儲存特徵清單: {FEAT_JSON}")
    print("接下來執行: python daily/predict_today.py")


if __name__ == "__main__":
    main()
