#!/usr/bin/env python3
"""
日頻模型回測系統 - Walk-Forward Out-of-Sample v9

核心改變 (vs v8):
  - 模型改為 LGBMRegressor，直接預測 fwd_rel_3d
  - 選股邏輯: predicted_rel 最高的 top N
  - 空手條件: 所有候選股 predicted_rel < 0 且市場空頭時跳過
  - 持倉報酬使用 fwd_ret_3d (實際個股報酬)

執行: python daily/backtest.py
"""
import sys
import json
import random
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
STOP_LOSS        = -0.10
UP_CAP           =  0.15
COST_RT          = 0.001425
SLIPPAGE_RT      = 0.003
LIQUIDITY_FAIL   = 0.05
HOLD_DAYS        = 3
INIT_CAPITAL     = 10_000
MIN_PRED_REL     = 0.0     # predicted_rel 低於此值不進場 (空頭保護)
RANDOM_SEED      = 42


def load_feat_cols():
    if not FEAT_JSON.exists():
        print(f"[ERROR] 找不到 {FEAT_JSON}，請先執行 train_model.py")
        sys.exit(1)
    with open(FEAT_JSON) as f:
        meta = json.load(f)
    cols = meta["feature_cols"] if isinstance(meta, dict) else meta
    model_type = meta.get("model_type", "classifier") if isinstance(meta, dict) else "classifier"
    return cols, model_type


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


def is_market_bear(bm_ret_series: pd.Series, dt: pd.Timestamp, window: int = 20) -> bool:
    hist = bm_ret_series[bm_ret_series.index < dt].tail(window)
    if len(hist) < window // 2:
        return False
    return float((1 + hist).prod() - 1) < 0


def train_fold_reg(X_tr, y_tr):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] pip install lightgbm"); sys.exit(1)

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def summarize(trades_df, equity_df, n_folds, top_n, hold_days):
    n_trades   = len(trades_df)
    win_rate   = (trades_df["adj_ret"] > 0).mean() * 100
    avg_ret    = trades_df["adj_ret"].mean() * 100
    total_ret  = equity_df["cum_ret"].iloc[-1] * 100
    bm_ret_t   = equity_df["bm_cum_ret"].iloc[-1] * 100
    excess     = total_ret - bm_ret_t
    liq_fail_n = trades_df["liq_failed"].sum()
    skip_days  = int(equity_df["skipped"].sum()) if "skipped" in equity_df.columns else 0

    eq       = equity_df["cum_ret"].values + 1
    roll_max = np.maximum.accumulate(eq)
    max_dd   = float(((eq - roll_max) / roll_max).min()) * 100

    bt_s = equity_df["date"].min()
    bt_e = equity_df["date"].max()
    n_calendar_days = (pd.Timestamp(bt_e) - pd.Timestamp(bt_s)).days
    n_years = n_calendar_days / 365.25

    ann_ret = float((1 + total_ret / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    bm_ann  = float((1 + bm_ret_t  / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    eq_rets = equity_df["port_ret"].values / hold_days
    sharpe  = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(252)) \
              if np.std(eq_rets) > 0 else 0.0

    final_capital = equity_df["capital"].iloc[-1]

    print("\n" + "="*60)
    print("Walk-Forward OOS 回測結果  (基準: 0050)")
    print(f"回測期間: {pd.Timestamp(bt_s).date()} ~ {pd.Timestamp(bt_e).date()}  "
          f"({n_calendar_days}日/{n_years:.2f}年)  Folds: {n_folds}")
    print(f"選股數: top {top_n}  持有: {hold_days}天  MIN_PRED_REL: {MIN_PRED_REL}")
    print(f"交易成本: {(COST_RT+SLIPPAGE_RT)*100:.4f}%/邊  流動性失敗: {LIQUIDITY_FAIL*100:.0f}%  (共 {liq_fail_n} 筆)")
    print(f"空手跳過次數: {skip_days}")
    print("="*60)
    print(f"起始資金      : {INIT_CAPITAL:,.0f} NTD")
    print(f"最終資金      : {final_capital:,.0f} NTD")
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
    print("="*60)

    eq2 = equity_df.copy()
    eq2["month"] = pd.to_datetime(eq2["date"]).dt.to_period("M")
    monthly = eq2.groupby("month").agg(
        port=pd.NamedAgg("port_ret", lambda x: (1+x).prod()-1),
        bm  =pd.NamedAgg("bm_ret",  lambda x: (1+x).prod()-1),
        capital_end=("capital", "last"),
    )
    monthly["excess"] = monthly["port"] - monthly["bm"]

    print("\n月分資金與報酬 (vs 0050):")
    print(f"{'month':<10} {'port%':>8} {'bm%':>8} {'excess%':>9} {'capital(NTD)':>14}")
    print("-" * 52)
    for m, row in monthly.iterrows():
        print(f"{str(m):<10} {row['port']*100:>+8.2f} {row['bm']*100:>+8.2f} "
              f"{row['excess']*100:>+9.2f} {row['capital_end']:>14,.0f}")


def run_backtest():
    rng = random.Random(RANDOM_SEED)
    feat_cols, model_type = load_feat_cols()

    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values(["date", "stock_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    fwd_col = "fwd_ret_3d" if "fwd_ret_3d" in df.columns else "fwd_ret_1d"
    rel_col = "fwd_rel_3d"

    if fwd_col not in df.columns:
        print("[ERROR] 請先執行 python daily/build_features.py")
        sys.exit(1)

    df[fwd_col] = df[fwd_col].clip(STOP_LOSS, UP_CAP)

    avail = [c for c in feat_cols if c in df.columns]
    df.dropna(subset=avail + [fwd_col], inplace=True)

    all_dates  = sorted(df["date"].unique())
    total_days = len(all_dates)
    date_idx   = {d: i for i, d in enumerate(all_dates)}
    print(f"資料範圍: {all_dates[0].date()} ~ {all_dates[-1].date()} ({total_days} 交易日)")
    print(f"持有天數: {HOLD_DAYS}天  MIN_PRED_REL: {MIN_PRED_REL}  滑價: {SLIPPAGE_RT*100:.2f}%")
    print(f"流動性失敗: {LIQUIDITY_FAIL*100:.0f}%  起始資金: {INIT_CAPITAL:,} NTD")
    print(f"模型類型: {model_type}  使用特徵: {len(avail)} 個")

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
    capital = float(INIT_CAPITAL)
    carry_over = {}

    for fold_i, (fold_start, fold_end) in enumerate(folds):
        print(f"\n[Fold {fold_i+1}/{N_FOLDS}] 訓練中...", end=" ", flush=True)
        train_df = df[df["date"] < fold_start]
        test_df  = df[(df["date"] >= fold_start) & (df["date"] <= fold_end)]

        X_tr = pd.DataFrame(train_df[avail].values.astype(np.float32), columns=avail)

        # 回歸模型: 目標為 fwd_rel_3d (若有)，否則 fwd_ret_3d
        y_col = rel_col if rel_col in train_df.columns else fwd_col
        y_tr  = train_df[y_col].fillna(0).values.astype(np.float32)

        model = train_fold_reg(X_tr, y_tr)
        print(f"完成  (訓練 {len(X_tr):,} 筆 | 測試 {len(test_df):,} 筆)")

        test_dates     = sorted(test_df["date"].unique())
        signal_indices = list(range(0, len(test_dates), HOLD_DAYS))

        for sig_i in signal_indices:
            dt = test_dates[sig_i]
            bear = is_market_bear(bm_ret_series, dt, window=20)

            day_df = test_df[test_df["date"] == dt].dropna(subset=avail).copy()
            if len(day_df) < 1:
                continue

            X_day = pd.DataFrame(day_df[avail].values.astype(np.float32), columns=avail)
            day_df["predicted_rel"] = model.predict(X_day)

            # 空手條件: 空頭 + 所有預測值都 < MIN_PRED_REL
            if bear and (day_df["predicted_rel"].max() < MIN_PRED_REL):
                all_eq_rows.append({
                    "date": dt, "fold": fold_i+1,
                    "port_ret": 0.0, "bm_ret": 0.0,
                    "cum_ret": round(float(np.expm1(cum_ret)), 6),
                    "bm_cum_ret": round(float(np.expm1(bm_cum)), 6),
                    "capital": round(capital, 2),
                    "skipped": 1,
                })
                continue

            picks = day_df.nlargest(min(TOP_N, len(day_df)), "predicted_rel")

            raw_rets  = []
            liq_flags = []

            for _, row in picks.iterrows():
                sid = row["stock_id"]

                if rng.random() < LIQUIDITY_FAIL:
                    fallback = carry_over.pop(sid, 0.0)
                    raw_rets.append(fallback)
                    liq_flags.append(1)
                    continue

                si = date_idx.get(dt)
                if si is None:
                    raw_rets.append(np.nan); liq_flags.append(0); continue

                exit_i  = min(si + HOLD_DAYS, len(all_dates) - 1)

                # 優先使用 fwd_ret_3d (信號當天已計算好的欄位)
                cur = df[(df["stock_id"] == sid) & (df["date"] == dt)]
                if len(cur) > 0 and fwd_col in cur.columns:
                    cum = float(cur[fwd_col].values[0])
                else:
                    hold_df = df[(df["stock_id"] == sid) &
                                 (df["date"] >  dt) &
                                 (df["date"] <= all_dates[exit_i])]
                    cum = float((1 + hold_df["fwd_ret_1d"]).prod() - 1) \
                          if not hold_df.empty else 0.0

                if rng.random() < LIQUIDITY_FAIL:
                    carry_over[sid] = cum
                    raw_rets.append(0.0)
                    liq_flags.append(1)
                else:
                    carry_over.pop(sid, None)
                    raw_rets.append(cum)
                    liq_flags.append(0)

            raw_rets  = np.array(raw_rets, dtype=float)
            liq_flags = np.array(liq_flags, dtype=int)
            valid     = ~np.isnan(raw_rets)
            if valid.sum() == 0:
                continue

            raw_rets_v = raw_rets[valid]
            liq_v      = liq_flags[valid]

            cost_per_trade = (COST_RT + SLIPPAGE_RT) * 2
            adj_rets = raw_rets_v - np.where(liq_v == 0, cost_per_trade, 0.0)
            port_ret = float(np.mean(adj_rets))

            capital = capital * (1 + port_ret)

            si     = date_idx.get(dt)
            exit_i = min(si + HOLD_DAYS, len(all_dates) - 1)
            bm_window = bm_ret_series[
                (bm_ret_series.index >  dt) &
                (bm_ret_series.index <= all_dates[exit_i])
            ]
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
                "capital":    round(capital, 2),
                "skipped":    0,
            })

            valid_idx = np.where(valid)[0]
            for k, j in enumerate(valid_idx):
                row_p = picks.iloc[j]
                r     = float(raw_rets_v[k])
                r_adj = r - (cost_per_trade if liq_v[k] == 0 else 0.0)
                all_trades.append({
                    "fold":          fold_i + 1,
                    "signal_date":   dt,
                    "stock_id":      row_p["stock_id"],
                    "predicted_rel": round(float(row_p["predicted_rel"]), 4),
                    "entry_close":   round(float(row_p["close"]), 2),
                    "raw_ret":       round(r,     6),
                    "adj_ret":       round(r_adj, 6),
                    "liq_failed":    int(liq_v[k]),
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
