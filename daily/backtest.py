#!/usr/bin/env python3
"""
日頻模型回測系統 - Walk-Forward Out-of-Sample

架構:
  1. 將全部資料按時間切成 N 個 fold
  2. 每個 fold: train = fold 開始之前所有資料, test = 此 fold 期間
  3. 拼接每個 fold 的預測得到完整 OOS 曲線

策略:
  - 每日收盤後用 T 特徵預測 T+1 報酬
  - 選 up_prob 前 5 名買入，權重等重
  - T+1 收盤出場 (保持 1 天)
  - 單日報酬上限 +9.7% / 下限 -3%
  - 基準: 0050 實際日報酬

執行: python daily/backtest.py
"""
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"

IN_CSV     = DATA_DIR / "daily_features.csv"
FEAT_JSON  = MODEL_DIR / "daily_feature_cols.json"
OUT_TRADES = DATA_DIR / "backtest_trades.csv"
OUT_EQUITY = DATA_DIR / "backtest_equity.csv"

N_FOLDS          = 6
MIN_TRAIN_MONTHS = 12
TOP_N            = 5
STOP_LOSS        = -0.03
UP_CAP           =  0.097
COST_RT          = 0.001425

FEATURE_COLS = [
    "vol_ratio_5d", "vol_ratio_20d", "vol_zscore_20d",
    "ret_1d", "ret_5d", "high_low_pct",
    "close_vs_ma5", "close_vs_ma20",
    "inst_net_ratio", "margin_chg_pct",
]


def load_feat_cols():
    if not FEAT_JSON.exists():
        print(f"[ERROR] 找不到 {FEAT_JSON}")
        sys.exit(1)
    with open(FEAT_JSON) as f: meta = json.load(f)
    return meta["feature_cols"] if isinstance(meta, dict) else meta


def fetch_0050_returns(start: str, end: str) -> pd.Series:
    """從 yfinance 拉 0050 日報酬，索引為日期"""
    print("下載 0050 基準資料...", end=" ", flush=True)
    h = yf.download("0050.TW",
                    start=(pd.Timestamp(start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                    end=(pd.Timestamp(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True)
    if h.empty:
        print("失敗，將使用 0 作為基準")
        return pd.Series(dtype=float)
    close = h["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    ret = close.pct_change().dropna()
    print(f"OK ({len(ret)} 天)")
    return ret


def train_fold(X_tr: pd.DataFrame, y_tr: np.ndarray):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] pip install lightgbm"); sys.exit(1)
    pos_rate  = y_tr.mean()
    scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03,
        num_leaves=31, max_depth=5,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def summarize(trades_df, equity_df, n_folds, top_n):
    n_trades  = len(trades_df)
    win_rate  = (trades_df["adj_ret"] > 0).mean() * 100
    avg_ret   = trades_df["adj_ret"].mean() * 100
    total_ret = equity_df["cum_ret"].iloc[-1] * 100
    bm_ret_t  = equity_df["bm_cum_ret"].iloc[-1] * 100
    excess    = total_ret - bm_ret_t
    stop_n    = trades_df["stopped"].sum()
    capped_n  = trades_df["capped"].sum()

    eq       = equity_df["cum_ret"].values + 1
    roll_max = np.maximum.accumulate(eq)
    max_dd   = float(((eq - roll_max) / roll_max).min()) * 100

    daily  = equity_df["port_ret"].values
    sharpe = float(np.mean(daily) / np.std(daily) * np.sqrt(252)) \
             if np.std(daily) > 0 else 0.0

    n_days  = len(equity_df)
    ann_ret = float((1 + total_ret / 100) ** (252 / n_days) - 1) * 100
    bm_ann  = float((1 + bm_ret_t  / 100) ** (252 / n_days) - 1) * 100

    bt_s = equity_df["date"].min().date()
    bt_e = equity_df["date"].max().date()

    print("\n" + "="*55)
    print("Walk-Forward OOS 回測結果  (基準: 0050)")
    print(f"回測期間: {bt_s} ~ {bt_e}  Folds: {n_folds}")
    print(f"選股數: top {top_n}  停損: {STOP_LOSS*100:.0f}%  上限: {UP_CAP*100:.1f}%  每邊成本: {COST_RT*100:.4f}%")
    print("="*55)
    print(f"總交易筆數    : {n_trades}")
    print(f"勝率         : {win_rate:.1f}%")
    print(f"平均單筆報酬  : {avg_ret:+.3f}%")
    print(f"單日停損筆數  : {stop_n}  ({stop_n/n_trades*100:.1f}%)")
    print(f"單日觸上限筆數: {capped_n}  ({capped_n/n_trades*100:.1f}%)")
    print(f"策略總報酬    : {total_ret:+.2f}%")
    print(f"策略年化報酬  : {ann_ret:+.2f}%")
    print(f"0050 總報酬    : {bm_ret_t:+.2f}%")
    print(f"0050 年化報酬  : {bm_ann:+.2f}%")
    print(f"超額報酬 (vs 0050): {excess:+.2f}%")
    print(f"最大回撤      : {max_dd:.2f}%")
    print(f"年化 Sharpe   : {sharpe:.3f}")
    print("="*55)

    eq2 = equity_df.copy()
    eq2["month"] = pd.to_datetime(eq2["date"]).dt.to_period("M")
    monthly = eq2.groupby("month").agg(
        port=pd.NamedAgg("port_ret", lambda x: (1+x).prod()-1),
        bm  =pd.NamedAgg("bm_ret",  lambda x: (1+x).prod()-1),
    )
    monthly["excess"] = monthly["port"] - monthly["bm"]
    print("\n月分装報酬 (%) vs 0050:")
    print((monthly * 100).round(2).to_string())


def run_backtest():
    feat_cols = load_feat_cols()

    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values(["date", "stock_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "fwd_ret_1d" not in df.columns or "label_up1" not in df.columns:
        print("[ERROR] 請先執行 python daily/build_features.py")
        sys.exit(1)

    # 單日報酬 cap
    df["fwd_ret_1d"] = df["fwd_ret_1d"].clip(STOP_LOSS, UP_CAP)

    avail = [c for c in feat_cols if c in df.columns]
    df.dropna(subset=avail + ["label_up1", "fwd_ret_1d"], inplace=True)

    all_dates  = sorted(df["date"].unique())
    total_days = len(all_dates)
    print(f"資料範圍: {all_dates[0].date()} ~ {all_dates[-1].date()} ({total_days} 交易日)")

    min_train_days = MIN_TRAIN_MONTHS * 21
    if total_days <= min_train_days + N_FOLDS:
        print("[ERROR] 資料太少")
        sys.exit(1)

    bt_dates  = all_dates[min_train_days:]
    fold_size = len(bt_dates) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        s = i * fold_size
        e = (i+1)*fold_size if i < N_FOLDS-1 else len(bt_dates)
        folds.append((bt_dates[s], bt_dates[e-1]))

    # 一次性下載 0050 回測期間內的日報酬
    bm_ret_series = fetch_0050_returns(
        str(folds[0][0].date()), str(folds[-1][1].date())
    )

    print(f"\nWalk-Forward Folds ({N_FOLDS} 個):")
    for i, (s, e) in enumerate(folds):
        train_end = all_dates[all_dates.index(s) - 1]
        n_tr = len([d for d in all_dates if d <= train_end])
        print(f"  Fold {i+1}: train=~{train_end.date()} ({n_tr}天)  test={s.date()}~{e.date()}")

    all_trades  = []
    all_eq_rows = []
    cum_ret = 0.0
    bm_cum  = 0.0

    for fold_i, (fold_start, fold_end) in enumerate(folds):
        print(f"\n[Fold {fold_i+1}/{N_FOLDS}] 訓練中...", end=" ", flush=True)
        train_df = df[df["date"] < fold_start]
        test_df  = df[(df["date"] >= fold_start) & (df["date"] <= fold_end)]

        X_tr  = pd.DataFrame(train_df[avail].values.astype(np.float32), columns=avail)
        model = train_fold(X_tr, train_df["label_up1"].values)
        print(f"完成  (訓練 {len(X_tr):,} 筆 | 測試 {len(test_df):,} 筆)")

        for dt in sorted(test_df["date"].unique()):
            day_df = test_df[test_df["date"] == dt].dropna(subset=avail)
            if len(day_df) < TOP_N:
                continue

            X_day = pd.DataFrame(day_df[avail].values.astype(np.float32), columns=avail)
            day_df = day_df.copy()
            day_df["up_prob"] = model.predict_proba(X_day)[:, 1]

            picks    = day_df.nlargest(TOP_N, "up_prob")
            raw_rets = picks["fwd_ret_1d"].values
            adj_rets = raw_rets - COST_RT * 2
            port_ret = float(np.mean(adj_rets))

            # 0050 基準報酬，對齊到同一交易日 (T+1)
            dt_next = dt + pd.Timedelta(days=1)
            # 找最接近的交易日
            bm_candidates = bm_ret_series[
                (bm_ret_series.index >= dt) &
                (bm_ret_series.index <= dt + pd.Timedelta(days=5))
            ]
            bm_ret = float(bm_candidates.iloc[0]) if len(bm_candidates) > 0 else 0.0

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
                r     = float(row["fwd_ret_1d"])
                r_adj = r - COST_RT * 2
                all_trades.append({
                    "fold":        fold_i + 1,
                    "signal_date": dt,
                    "stock_id":    row["stock_id"],
                    "up_prob":     round(float(row["up_prob"]), 4),
                    "entry_close": round(float(row["close"]),  2),
                    "raw_ret":     round(r,     6),
                    "adj_ret":     round(r_adj, 6),
                    "stopped":     int(r <= STOP_LOSS),
                    "capped":      int(r >= UP_CAP),
                })

    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(all_eq_rows)

    trades_df.to_csv(OUT_TRADES, index=False, encoding="utf-8-sig")
    equity_df.to_csv(OUT_EQUITY, index=False, encoding="utf-8-sig")

    summarize(trades_df, equity_df, N_FOLDS, TOP_N)
    print(f"\n輸出: {OUT_TRADES}")
    print(f"輸出: {OUT_EQUITY}")
    return equity_df, trades_df


if __name__ == "__main__":
    run_backtest()
