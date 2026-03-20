#!/usr/bin/env python3
"""
日頻模型回測系統 - Walk-Forward Out-of-Sample v4

選股逻輯:
  1. score > 0 過濾 (up_prob > down_prob)
  2. up_prob 排序，選 top N
  3. 支援多日持有 HOLD_DAYS (1/3/5)

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
HOLD_DAYS        = 3    # 持有天數: 1=当日出場, 3=續抜3天, 5=續抜5天


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
        print("失敗")
        return pd.Series(dtype=float)
    close = h["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    ret = close.pct_change().dropna()
    print(f"OK ({len(ret)} 天)")
    return ret


def train_fold(X_tr, y_up, y_dn):
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


def compute_hold_ret(date_idx: dict, dates: list, start_dt, hold_days: int,
                     df: pd.DataFrame) -> dict:
    """
    預先計算每支股票從入場日起持有 hold_days 天的累積報酬。
    回傳 {stock_id: hold_ret}
    """
    # 找出出場日 index
    si = date_idx.get(start_dt)
    if si is None: return {}
    exit_i = min(si + hold_days, len(dates) - 1)
    exit_dt = dates[exit_i]

    # 從 start_dt 到 exit_dt 所有日子準將 fwd_ret_1d 展開再複利
    window = df[(df["date"] > start_dt) & (df["date"] <= exit_dt)]
    result = {}
    for sid, g in window.groupby("stock_id"):
        cum = float((1 + g["fwd_ret_1d"]).prod() - 1)
        result[sid] = cum
    return result


def summarize(trades_df, equity_df, n_folds, top_n, hold_days):
    n_trades  = len(trades_df)
    win_rate  = (trades_df["adj_ret"] > 0).mean() * 100
    avg_ret   = trades_df["adj_ret"].mean() * 100
    total_ret = equity_df["cum_ret"].iloc[-1] * 100
    bm_ret_t  = equity_df["bm_cum_ret"].iloc[-1] * 100
    excess    = total_ret - bm_ret_t
    stop_n    = (trades_df["raw_ret"] <= STOP_LOSS * hold_days).sum()

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
    print(f"選股數: top {top_n}  持有: {hold_days}天  停損: {STOP_LOSS*100:.0f}%  每邊成本: {COST_RT*100:.4f}%")
    print("="*55)
    print(f"總交易筆數    : {n_trades}")
    print(f"勝率         : {win_rate:.1f}%")
    print(f"平均單筆報酬  : {avg_ret:+.3f}%")
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

    # clip 單日報酬 (多日持有用於計算累積，不再 clip hold_days)
    df["fwd_ret_1d"] = df["fwd_ret_1d"].clip(STOP_LOSS, UP_CAP)

    avail = [c for c in feat_cols if c in df.columns]
    has_down = "label_down1" in df.columns
    df.dropna(subset=avail + ["label_up1", "fwd_ret_1d"], inplace=True)

    all_dates  = sorted(df["date"].unique())
    total_days = len(all_dates)
    date_idx   = {d: i for i, d in enumerate(all_dates)}
    print(f"資料範圍: {all_dates[0].date()} ~ {all_dates[-1].date()} ({total_days} 交易日)")
    print(f"持有天數: {HOLD_DAYS} 天")

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
            train_df["label_down1"].values if has_down else np.zeros(len(train_df), dtype=np.float32),
        )
        print(f"完成  (訓練 {len(X_tr):,} 筆 | 測試 {len(test_df):,} 筆)")

        test_dates = sorted(test_df["date"].unique())

        # HOLD_DAYS=1: 與舊版相同，用 fwd_ret_1d
        # HOLD_DAYS>1: 每天基於 signal 日選出股票，累積後續 hold_days 天的報酬
        # 等重: 每天選新的組，不重離
        skip_until = {}  # stock_id -> 封鎖到的 date index

        for dt_i, dt in enumerate(test_dates):
            day_df = test_df[test_df["date"] == dt].dropna(subset=avail).copy()
            if len(day_df) < TOP_N:
                continue

            X_day = pd.DataFrame(day_df[avail].values.astype(np.float32), columns=avail)
            day_df["up_prob"] = up_model.predict_proba(X_day)[:, 1]
            day_df["dn_prob"] = dn_model.predict_proba(X_day)[:, 1]
            day_df["score"]   = day_df["up_prob"] - day_df["dn_prob"]

            candidates = day_df[day_df["score"] > 0]
            if len(candidates) < TOP_N:
                candidates = day_df

            picks = candidates.nlargest(TOP_N, "up_prob")

            if HOLD_DAYS == 1:
                raw_rets = picks["fwd_ret_1d"].values
            else:
                # 累積 hold_days 天的報酬
                raw_rets = []
                for _, row in picks.iterrows():
                    sid = row["stock_id"]
                    si  = date_idx.get(dt)
                    if si is None: raw_rets.append(np.nan); continue
                    exit_i   = min(si + HOLD_DAYS, len(all_dates) - 1)
                    hold_df  = df[(df["stock_id"] == sid) &
                                  (df["date"] > dt) &
                                  (df["date"] <= all_dates[exit_i])]
                    if hold_df.empty: raw_rets.append(np.nan); continue
                    cum = float((1 + hold_df["fwd_ret_1d"]).prod() - 1)
                    raw_rets.append(cum)
                raw_rets = np.array(raw_rets)

            raw_rets = raw_rets[~np.isnan(raw_rets)]
            if len(raw_rets) == 0: continue
            adj_rets = raw_rets - COST_RT * 2
            port_ret = float(np.mean(adj_rets))

            # 基準: 0050 持有同樣天數累積
            si = date_idx.get(dt)
            exit_i = min(si + HOLD_DAYS, len(all_dates) - 1) if si is not None else si
            bm_window = bm_ret_series[
                (bm_ret_series.index > dt) &
                (bm_ret_series.index <= all_dates[exit_i])
            ] if exit_i is not None else pd.Series(dtype=float)
            bm_ret = float((1 + bm_window).prod() - 1) if len(bm_window) > 0 else 0.0

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

            for j, (_, row) in enumerate(picks.iterrows()):
                r     = float(raw_rets[j]) if j < len(raw_rets) else np.nan
                if np.isnan(r): continue
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
                    "capped":      int(r >= UP_CAP * HOLD_DAYS),
                })

    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(all_eq_rows)

    trades_df.to_csv(OUT_TRADES, index=False, encoding="utf-8-sig")
    equity_df.to_csv(OUT_EQUITY, index=False, encoding="utf-8-sig")

    summarize(trades_df, equity_df, N_FOLDS, TOP_N, HOLD_DAYS)
    print(f"\n輸出: {OUT_TRADES}")
    print(f"輸出: {OUT_EQUITY}")
    return equity_df, trades_df


if __name__ == "__main__":
    run_backtest()
