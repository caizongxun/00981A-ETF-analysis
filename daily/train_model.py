#!/usr/bin/env python3
"""
日頻異常偵測模型訓練 - T+0 架構
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


def train_lgbm(X_tr: pd.DataFrame, y_train, X_vl: pd.DataFrame, y_val,
               label_name: str):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] pip install lightgbm"); sys.exit(1)
    from sklearn.metrics import roc_auc_score, classification_report, f1_score

    pos_rate  = y_train.mean()
    scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    print(f"  pos_rate={pos_rate:.3f}  scale_pos_weight={scale_pos:.1f}")

    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        class_weight="balanced",
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

    prob = model.predict_proba(X_vl)[:, 1]
    auc  = roc_auc_score(y_val, prob)
    print(f"[{label_name}] AUC={auc:.4f}")

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.75, 0.02):
        f1 = f1_score(y_val, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  best threshold={best_t:.2f}  F1={best_f1:.4f}")
    print(classification_report(y_val, (prob >= best_t).astype(int), zero_division=0))

    imp = pd.Series(model.feature_importances_, index=X_tr.columns)
    print(f"  Top 5: {imp.nlargest(5).to_dict()}")
    return model, float(best_t)


def main():
    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    # 相容舊版 label 名稱 (label_up5 -> label_up1)
    if "label_up1" not in df.columns and "label_up5" in df.columns:
        print("[WARN] 偵測到舊版 label_up5，請先執行 python daily/build_features.py 重建特徵")
        sys.exit(1)

    avail = [c for c in FEATURE_COLS if c in df.columns]
    df.dropna(subset=avail + ["label_up1", "label_down1"], inplace=True)
    print(f"有效樣本: {len(df):,}")

    cutoff = df["date"].quantile(0.8)
    train  = df[df["date"] <= cutoff]
    val    = df[df["date"] >  cutoff]
    print(f"Train: {len(train):,} ({train['date'].min().date()} ~ {train['date'].max().date()})")
    print(f"Val  : {len(val):,}   ({val['date'].min().date()} ~ {val['date'].max().date()})")

    X_tr = pd.DataFrame(train[avail].values.astype(np.float32), columns=avail)
    X_vl = pd.DataFrame(val[avail].values.astype(np.float32),   columns=avail)

    print("\n--- 訓練隣日上漲模型 ---")
    up_model, up_thresh = train_lgbm(X_tr, train["label_up1"].values,
                                     X_vl, val["label_up1"].values, "label_up1")
    with open(UP_MODEL, "wb") as f: pickle.dump(up_model, f)
    print(f"儲存: {UP_MODEL}")

    print("\n--- 訓練隣日下跌模型 ---")
    down_model, down_thresh = train_lgbm(X_tr, train["label_down1"].values,
                                         X_vl, val["label_down1"].values, "label_down1")
    with open(DOWN_MODEL, "wb") as f: pickle.dump(down_model, f)
    print(f"儲存: {DOWN_MODEL}")

    meta = {"feature_cols": avail, "up_thresh": up_thresh, "down_thresh": down_thresh}
    with open(FEAT_JSON, "w") as f: json.dump(meta, f, indent=2)
    print(f"儲存特徵 + 閾値: {FEAT_JSON}")
    print("接下來執行: python daily/predict_today.py")


if __name__ == "__main__":
    main()
