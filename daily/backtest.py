#!/usr/bin/env python3
"""
日頻模型回測系統 - Walk-Forward Out-of-Sample

架構:
  1. 將全部資料按時間切成 N 個 fold
  2. 每個 fold:
       train = fold 開始之前所有資料 (对当前 fold 而言是未來)
       test  = 此 fold 期間 (模型完全沒見過)
  3. 拼接每個 fold 的預測結果得到完整 OOS 回測曲線

策略:
  - 每日收盤後用 T 特徵預測 T+1 報酬
  - 選 up_prob 前 5 名買入，權重等重
  - T+1 收盤出場 (保持 1 天)
  - 停損: 實際報酬 < -3% 則按 -3% 計算
  - 基準: 每日持有宇宙全部股票均等權重

執行: python daily/backtest.py
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

IN_CSV     = DATA_DIR / "daily_features.csv"
FEAT_JSON  = MODEL_DIR / "daily_feature_cols.json"
OUT_TRADES = DATA_DIR / "backtest_trades.csv"
OUT_EQUITY = DATA_DIR / "backtest_equity.csv"

# 回測參數
N_FOLDS       = 6      # walk-forward fold 數 (= 回測山月數)
MIN_TRAIN_MONTHS = 12  # 每個 fold 至少需要 12 個月訓練資料
TOP_N         = 5
STOP_LOSS     = -0.03
COST_RT       = 0.001425

FEATURE_COLS = [
    "vol_ratio_5d", "vol_ratio_20d", "vol_zscore_20d",
    "ret_1d", "ret_5d", "high_low_pct",
    "close_vs_ma5", "close_vs_ma20",
    "inst_net_ratio", "margin_chg_pct",
]


def load_feat_cols():
    if not FEAT_JSON.exists():
        print(f"[ERROR] 找不到 {FEAT_JSON}，請先執行 python daily/train_model.py")
        sys.exit(1)
    with open(FEAT_JSON) as f: meta = json.load(f)
    return meta["feature_cols"] if isinstance(meta, dict) else meta


def train_fold(X_tr: pd.DataFrame, y_tr: np.ndarray) -> object:
    """train LightGBM on given fold data"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] pip install lightgbm"); sys.exit(1)

    pos_rate  = y_tr.mean()
    scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
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
    model.fit(X_tr, y_tr)
    return model


def run_backtest():
    feat_cols = load_feat_cols()

    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values(["date", "stock_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "fwd_ret_1d" not in df.columns or "label_up1" not in df.columns:
        print("[ERROR] 請先執行 python daily/build_features.py")
        sys.exit(1)

    avail = [c for c in feat_cols if c in df.columns]
    df.dropna(subset=avail + ["label_up1", "fwd_ret_1d"], inplace=True)

    all_dates = sorted(df["date"].unique())
    total_days = len(all_dates)
    print(f"資料範圍: {all_dates[0].date()} ~ {all_dates[-1].date()} ({total_days} 交易日)")

    # 建構 fold 邊界: 將後 6 個月的交易日分成 N_FOLDS 個等長區間
    min_train_days = MIN_TRAIN_MONTHS * 21
    if total_days <= min_train_days + N_FOLDS:
        print(f"[ERROR] 資料太少，至少需要 {min_train_days + N_FOLDS + 1} 天資料")
        sys.exit(1)

    bt_dates  = all_dates[min_train_days:]   # 可用于回測的日期
    fold_size = len(bt_dates) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start_idx = i * fold_size
        end_idx   = (i + 1) * fold_size if i < N_FOLDS - 1 else len(bt_dates)
        folds.append((bt_dates[start_idx], bt_dates[end_idx - 1]))

    print(f"\nWalk-Forward Folds ({N_FOLDS} 個):")
    for i, (s, e) in enumerate(folds):
        train_end = all_dates[all_dates.index(s) - 1]
        n_train   = len([d for d in all_dates if d <= train_end])
        print(f"  Fold {i+1}: train=~{train_end.date()} ({n_train}天)  test={s.date()}~{e.date()}")

    # 逐 fold 訓練 + 預測
    all_trades  = []
    all_eq_rows = []
    cum_ret = 0.0
    bm_cum  = 0.0

    for fold_i, (fold_start, fold_end) in enumerate(folds):
        print(f"\n[Fold {fold_i+1}/{N_FOLDS}] 訓練中...", end=" ", flush=True)

        train_mask = df["date"] < fold_start
        test_mask  = (df["date"] >= fold_start) & (df["date"] <= fold_end)

        train_df = df[train_mask]
        test_df  = df[test_mask]

        X_tr = pd.DataFrame(train_df[avail].values.astype(np.float32), columns=avail)
        y_tr = train_df["label_up1"].values

        model = train_fold(X_tr, y_tr)
        print(f"完成  (訓練 {len(X_tr):,} 筆 | 測試 {len(test_df):,} 筆)")

        fold_dates = sorted(test_df["date"].unique())
        for dt in fold_dates:
            day_df = test_df[test_df["date"] == dt].dropna(subset=avail)
            if len(day_df) < TOP_N:
                continue

            X_day = pd.DataFrame(day_df[avail].values.astype(np.float32), columns=avail)
            day_df = day_df.copy()
            day_df["up_prob"] = model.predict_proba(X_day)[:, 1]

            picks = day_df.nlargest(TOP_N, "up_prob")
            rets  = picks["fwd_ret_1d"].values
            rets  = np.clip(rets, STOP_LOSS, None)
            rets  = rets - COST_RT * 2
            port_ret = float(np.mean(rets))

            univ_rets = test_df[test_df["date"] == dt]["fwd_ret_1d"].dropna()
            bm_ret    = float(univ_rets.mean()) if len(univ_rets) > 0 else 0.0

            cum_ret += np.log1p(port_ret)
            bm_cum  += np.log1p(bm_ret)

            all_eq_rows.append({
                "date":       dt,
                "fold":       fold_i + 1,
                "port_ret":   round(port_ret, 6),
                "bm_ret":     round(bm_ret,   6),
                "cum_ret":    round(float(np.expm1(cum_ret)), 6),
                "bm_cum_ret": round(float(np.expm1(bm_cum)),  6),
            })

            for _, row in picks.iterrows():
                r = float(row["fwd_ret_1d"])
                r_adj = max(r, STOP_LOSS) - COST_RT * 2
                all_trades.append({
                    "fold":        fold_i + 1,
                    "signal_date": dt,
                    "stock_id":    row["stock_id"],
                    "up_prob":     round(float(row["up_prob"]), 4),
                    "entry_close": round(float(row["close"]),  2),
                    "raw_ret":     round(r,     6),
                    "adj_ret":     round(r_adj, 6),
                    "stopped":     int(r < STOP_LOSS),
                })

    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(all_eq_rows)

    trades_df.to_csv(OUT_TRADES, index=False, encoding="utf-8-sig")
    equity_df.to_csv(OUT_EQUITY, index=False, encoding="utf-8-sig")

    # 結果展示
    n_trades  = len(trades_df)
    win_rate  = (trades_df["adj_ret"] > 0).mean() * 100
    avg_ret   = trades_df["adj_ret"].mean() * 100
    total_ret = equity_df["cum_ret"].iloc[-1] * 100
    bm_ret_t  = equity_df["bm_cum_ret"].iloc[-1] * 100
    excess    = total_ret - bm_ret_t
    stop_n    = trades_df["stopped"].sum()

    eq = equity_df["cum_ret"].values + 1
    roll_max = np.maximum.accumulate(eq)
    dd = (eq - roll_max) / roll_max
    max_dd = float(dd.min()) * 100

    daily_rets = equity_df["port_ret"].values
    sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) \
             if np.std(daily_rets) > 0 else 0.0

    bt_start_date = equity_df["date"].min().date()
    bt_end_date   = equity_df["date"].max().date()

    print("\n" + "="*55)
    print(f"Walk-Forward OOS 回測結果")
    print(f"回測期間: {bt_start_date} ~ {bt_end_date}  Folds: {N_FOLDS}")
    print(f"選股數: top {TOP_N}  停損: {STOP_LOSS*100:.0f}%  每邊成本: {COST_RT*100:.4f}%")
    print("="*55)
    print(f"總交易筆數  : {n_trades}")
    print(f"勝率       : {win_rate:.1f}%")
    print(f"平均單筆報酬: {avg_ret:+.3f}%")
    print(f"單日停損筆數: {stop_n} ({stop_n/n_trades*100:.1f}%)")
    print(f"策略總報酬  : {total_ret:+.2f}%")
    print(f"基準總報酬  : {bm_ret_t:+.2f}%")
    print(f"超額報酬    : {excess:+.2f}%")
    print(f"最大回撤    : {max_dd:.2f}%")
    print(f"年化 Sharpe : {sharpe:.3f}")
    print("="*55)

    equity_df["month"] = pd.to_datetime(equity_df["date"]).dt.to_period("M")
    monthly = equity_df.groupby("month").agg(
        port=pd.NamedAgg("port_ret", lambda x: (1 + x).prod() - 1),
        bm=pd.NamedAgg("bm_ret",   lambda x: (1 + x).prod() - 1),
    )
    monthly["excess"] = monthly["port"] - monthly["bm"]
    print("\n月分装報酬 (%):")
    print((monthly * 100).round(2).to_string())
    print(f"\n輸出: {OUT_TRADES}")
    print(f"輸出: {OUT_EQUITY}")

    return equity_df, trades_df


if __name__ == "__main__":
    run_backtest()
