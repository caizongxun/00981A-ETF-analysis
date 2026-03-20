#!/usr/bin/env python3
"""
日頻模型回測系統 - Walk-Forward Out-of-Sample v2

選股逻輯改善:
  - 綜合評分 score = up_prob - down_prob (排除双向高波動股)
  - ATR 過濾: 排除當日宇宙中 atr_pct > 80th percentile 的股票
  - top N 由 score 排序選出

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
ATR_PCT_FILTER   = 0.80   # 排除 atr_pct > 80th percentile 的高波動股


def load_feat_cols():
    if not FEAT_JSON.exists():
        print(f"[ERROR] 找不到 {FEAT_JSON}")
        sys.exit(1)
    with open(FEAT_JSON) as f: meta = json.load(f)
    return meta["feature_cols"] if isinstance(meta, dict) else meta


def fetch_0050_returns(start: str, end: str) -> pd.Series:
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


def train_fold(X_tr: pd.DataFrame, y_up: np.ndarray, y_dn: np.ndarray):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] pip install lightgbm"); sys.exit(1)

    def _fit(y):
        pos_rate  = y.mean()
        scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        m = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03,
            num_leaves=31, max_depth=5,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        m.fit(X_tr, y)
        return m

    return _fit(y_up), _fit(y_dn)


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
    print(f"ATR 過濾: >{ATR_PCT_FILTER*100:.0f}th pct 排除")
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
    print("\n月分裝報酬 (%) vs 0050:")
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

    df["fwd_ret_1d"] = df["fwd_ret_1d"].clip(STOP_LOSS, UP_CAP)

    avail = [c for c in feat_cols if c in df.columns]
    has_down = "label_down1" in df.columns
    df.dropna(subset=avail + ["label_up1", "fwd_ret_1d"], inplace=True)

    all_dates  = sorted(df["date"].unique())
    total_days = len(all_dates)
    print(f"資料範圍: {all_dates[0].date()} ~ {all_dates[-1].date()} ({total_days} 交易日)")

    min_train_days = MIN_TRAIN_MONTHS * 21
    bt_dates  = all_dates[min_train_days:]
    fold_size = len(bt_dates) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        s = i * fold_size
        e = (i+1)*fold_size if i < N_FOLDS-1 else len(bt_dates)
        folds.append((bt_dates[s], bt_dates[e-1]))

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

        X_tr = pd.DataFrame(train_df[avail].values.astype(np.float32), columns=avail)
        up_model, dn_model = train_fold(
            X_tr,
            train_df["label_up1"].values,
            train_df["label_down1"].values if has_down else np.zeros(len(train_df)),
        )
        print(f"完成  (訓練 {len(X_tr):,} 筆 | 測試 {len(test_df):,} 筆)")

        # 預先計算此 fold 的 atr_pct 80th percentile
        atr_thresh = None
        if "atr_pct" in test_df.columns:
            atr_thresh = test_df["atr_pct"].quantile(ATR_PCT_FILTER)

        for dt in sorted(test_df["date"].unique()):
            day_df = test_df[test_df["date"] == dt].dropna(subset=avail).copy()

            # ATR 過濾: 排除當日高波動股
            if atr_thresh is not None and "atr_pct" in day_df.columns:
                day_df = day_df[day_df["atr_pct"] <= atr_thresh]

            if len(day_df) < TOP_N:
                continue

            X_day = pd.DataFrame(day_df[avail].values.astype(np.float32), columns=avail)
            day_df["up_prob"] = up_model.predict_proba(X_day)[:, 1]
            day_df["dn_prob"] = dn_model.predict_proba(X_day)[:, 1]
            # 綜合評分: 上漲機率 - 下跌機率
            day_df["score"]   = day_df["up_prob"] - day_df["dn_prob"]

            picks    = day_df.nlargest(TOP_N, "score")
            raw_rets = picks["fwd_ret_1d"].values
            adj_rets = raw_rets - COST_RT * 2
            port_ret = float(np.mean(adj_rets))

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
                    "dn_prob":     round(float(row["dn_prob"]), 4),
                    "score":       round(float(row["score"]),   4),
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
