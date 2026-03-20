#!/usr/bin/env python3
"""
日頻異常偉測模型訓練 - T+0 架構
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

# 無論 build_features.py 新增何種特徵，這裡自動收錄
# 排除的欄位: meta 資訊、label、從 targets
EXCLUDE_COLS = {
    "date", "stock_id", "close", "adj_close",
    "fwd_ret_1d", "label_up1", "label_down1",
}

NAN_THRESH = 0.50   # NaN 比例超過此閾値的特徵直接排除


def auto_feature_cols(df: pd.DataFrame) -> list[str]:
    """從 dataframe 自動選出有效特徵欄位"""
    candidates = [c for c in df.columns if c not in EXCLUDE_COLS]
    nan_rate   = df[candidates].isna().mean()
    keep = [c for c in candidates if nan_rate[c] <= NAN_THRESH]
    dropped = [c for c in candidates if nan_rate[c] > NAN_THRESH]
    if dropped:
        print(f"[INFO] NaN>{NAN_THRESH*100:.0f}% 排除特徵: {dropped}")
    print(f"[INFO] 使用 {len(keep)} 個特徵: {keep}")
    return keep


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
    print(f"  Top 10: {imp.nlargest(10).to_dict()}")
    return model, float(best_t)


def main():
    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    if "label_up1" not in df.columns:
        print("[ERROR] 找不到 label_up1，請先執行 python daily/build_features.py")
        sys.exit(1)

    feat_cols = auto_feature_cols(df)

    # dropna 只看 label，特徵的 NaN 由 LightGBM 內部處理
    df.dropna(subset=["label_up1", "label_down1"], inplace=True)
    print(f"有效樣本: {len(df):,}")

    cutoff = df["date"].quantile(0.8)
    train  = df[df["date"] <= cutoff]
    val    = df[df["date"] >  cutoff]
    print(f"Train: {len(train):,} ({train['date'].min().date()} ~ {train['date'].max().date()})")
    print(f"Val  : {len(val):,}   ({val['date'].min().date()} ~ {val['date'].max().date()})")

    X_tr = train[feat_cols].astype(np.float32)
    X_vl = val[feat_cols].astype(np.float32)

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

    meta = {"feature_cols": feat_cols, "up_thresh": up_thresh, "down_thresh": down_thresh}
    with open(FEAT_JSON, "w") as f: json.dump(meta, f, indent=2)
    print(f"儲存特徵 + 閾値: {FEAT_JSON}")
    print("接下來執行: python daily/predict_today.py")


if __name__ == "__main__":
    main()
