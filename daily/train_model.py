#!/usr/bin/env python3
"""
日頻模型訓練 v3 - 回歸周相對報酬

目標: 直接預測 fwd_rel_5d (個股5日報酬 - 0050同期5日報酬)
模型: LGBMRegressor

執行: python daily/train_model.py
"""
import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

IN_CSV    = DATA_DIR / "daily_features.csv"
REG_MODEL = MODEL_DIR / "daily_reg_model.pkl"
FEAT_JSON = MODEL_DIR / "daily_feature_cols.json"

EXCLUDE_COLS = {
    "date", "stock_id", "close", "adj_close",
    "fwd_ret_1d", "fwd_ret_5d", "fwd_rel_5d", "bm_fwd_ret_5d",
    "fwd_rel_3d", "bm_fwd_ret_3d", "fwd_ret_3d",
    "label_up1", "label_down1",
}

NAN_THRESH = 0.50


def auto_feature_cols(df: pd.DataFrame) -> list:
    candidates = [c for c in df.columns if c not in EXCLUDE_COLS]
    nan_rate   = df[candidates].isna().mean()
    keep    = [c for c in candidates if nan_rate[c] <= NAN_THRESH]
    dropped = [c for c in candidates if nan_rate[c] >  NAN_THRESH]
    if dropped:
        print(f"[INFO] NaN>{NAN_THRESH*100:.0f}% 排除: {dropped}")
    print(f"[INFO] 使用 {len(keep)} 個特徵: {keep}")
    return keep


def train_lgbm_reg(X_tr, y_tr, X_vl, y_vl):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] pip install lightgbm"); sys.exit(1)
    from sklearn.metrics import mean_absolute_error

    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=6,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    pred = model.predict(X_vl)
    mae  = mean_absolute_error(y_vl, pred)
    corr = float(pd.Series(pred).corr(pd.Series(y_vl)))
    print(f"  MAE={mae*100:.3f}%  IC(corr)={corr:.4f}")

    # 每周 IC (以 5 天為周期)
    vl_df = pd.DataFrame({"pred": pred, "actual": y_vl.values,
                          "date": X_vl.index.map(
                              lambda i: pd.NaT  # placeholder
                          )})

    imp = pd.Series(model.feature_importances_, index=X_tr.columns)
    print(f"  Top 15 features:")
    for feat, val in imp.nlargest(15).items():
        print(f"    {feat}: {val}")
    return model


def main():
    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    target = "fwd_rel_5d"
    if target not in df.columns:
        print(f"[ERROR] 找不到 {target}，請先執行 python daily/build_features.py")
        sys.exit(1)

    feat_cols = auto_feature_cols(df)
    df.dropna(subset=[target] + feat_cols, inplace=True)
    print(f"有效樣本: {len(df):,}")

    cutoff = df["date"].quantile(0.8)
    train  = df[df["date"] <= cutoff]
    val    = df[df["date"] >  cutoff]
    print(f"Train: {len(train):,} ({train['date'].min().date()} ~ {train['date'].max().date()})")
    print(f"Val  : {len(val):,}   ({val['date'].min().date()} ~ {val['date'].max().date()})")

    X_tr = train[feat_cols].astype(np.float32)
    X_vl = val[feat_cols].astype(np.float32)
    y_tr = train[target].values.astype(np.float32)
    y_vl = val[target]

    print(f"\n--- 訓練周相對報酬回歸模型 (目標: {target}) ---")
    reg_model = train_lgbm_reg(X_tr, y_tr, X_vl, y_vl)

    with open(REG_MODEL, "wb") as f:
        pickle.dump(reg_model, f)
    print(f"儲存: {REG_MODEL}")

    meta = {"feature_cols": feat_cols, "target": target, "model_type": "regressor"}
    with open(FEAT_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"儲存特徵清單: {FEAT_JSON}")
    print("接下來執行: python daily/backtest.py")


if __name__ == "__main__":
    main()
