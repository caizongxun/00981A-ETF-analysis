#!/usr/bin/env python3
"""
日頻異常偵測模型訓練
執行: python daily/train_model.py
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


def train_lgbm(X_train, y_train, X_val, y_val, feat_names: list, label_name: str):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] pip install lightgbm")
        sys.exit(1)
    from sklearn.metrics import roc_auc_score, classification_report

    pos_rate  = y_train.mean()
    scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    print(f"  pos_rate={pos_rate:.3f}  scale_pos_weight={scale_pos:.1f}")

    # 用 DataFrame 传入，避免 sklearn 警告
    X_tr = pd.DataFrame(X_train, columns=feat_names)
    X_vl = pd.DataFrame(X_val,   columns=feat_names)

    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,      # 降低，對少數類更敢分
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        class_weight="balanced",   # 雙保険
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_train,
        eval_set=[(X_vl, y_val)],
        callbacks=[
            lgb.early_stopping(60, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    # 使用最佳 prob threshold
    prob = model.predict_proba(X_vl)[:, 1]
    auc  = roc_auc_score(y_val, prob)
    print(f"[{label_name}] AUC={auc:.4f}")

    # 找最佳 F1 threshold
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.7, 0.02):
        pred_t = (prob >= t).astype(int)
        f1 = f1_score(y_val, pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  best threshold={best_t:.2f}  F1={best_f1:.4f}")
    print(classification_report(y_val, (prob >= best_t).astype(int), zero_division=0))

    imp = pd.Series(model.feature_importances_, index=feat_names)
    print(f"  Top 5: {imp.nlargest(5).to_dict()}")
    return model, best_t


def main():
    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    avail = [c for c in FEATURE_COLS if c in df.columns]
    df.dropna(subset=avail + ["label_up5", "label_down5"], inplace=True)
    print(f"有效樣本: {len(df):,}")

    cutoff = df["date"].quantile(0.8)
    train  = df[df["date"] <= cutoff]
    val    = df[df["date"] >  cutoff]
    print(f"Train: {len(train):,} ({train['date'].min().date()} ~ {train['date'].max().date()})")
    print(f"Val  : {len(val):,}   ({val['date'].min().date()} ~ {val['date'].max().date()})")

    X_train = train[avail].values.astype(np.float32)
    X_val   = val[avail].values.astype(np.float32)

    print("\n--- 訓練上漲異常模型 ---")
    up_model, up_thresh = train_lgbm(
        X_train, train["label_up5"].values,
        X_val,   val["label_up5"].values,
        avail, "label_up5"
    )
    with open(UP_MODEL, "wb") as f:
        pickle.dump(up_model, f)
    print(f"儲存: {UP_MODEL}")

    print("\n--- 訓練下跌異常模型 ---")
    down_model, down_thresh = train_lgbm(
        X_train, train["label_down5"].values,
        X_val,   val["label_down5"].values,
        avail, "label_down5"
    )
    with open(DOWN_MODEL, "wb") as f:
        pickle.dump(down_model, f)
    print(f"儲存: {DOWN_MODEL}")

    meta = {"feature_cols": avail, "up_thresh": up_thresh, "down_thresh": down_thresh}
    with open(FEAT_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"儲存特徵 + 間値: {FEAT_JSON}")
    print("接下來執行: python daily/predict_today.py")


if __name__ == "__main__":
    main()
