#!/usr/bin/env python3
"""
日頻模型回測系統 - Walk-Forward Out-of-Sample v8

優化:
  - MIN_SCORE: 最低 score 門檻，過濾低信心選股
  - ATR 動態停損: 停損 = -ATR_MULT * atr_pct (取代固定 -3%)
  - SLIPPAGE_RT: 滑價模擬
  - LIQUIDITY_FAIL: 流動性失敗模擬
  - INIT_CAPITAL: 起始資金 10,000 NTD

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
ATR_MULT         = 1.5     # 動態停損: -ATR_MULT * atr_pct，替代固定 STOP_LOSS
UP_CAP           = 0.097
COST_RT          = 0.001425
SLIPPAGE_RT      = 0.003
LIQUIDITY_FAIL   = 0.05
HOLD_DAYS        = 3
INIT_CAPITAL     = 10_000
MIN_SCORE        = 0.10    # score < 0.1 不選
RANDOM_SEED      = 42


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
            n_estimators=600, learning_rate=0.025,
            num_leaves=31, max_depth=5,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        m.fit(X_tr, y)
        return m

    return _fit(y_up), _fit(y_dn)


def summarize(trades_df, equity_df, n_folds, top_n, hold_days):
    n_trades   = len(trades_df)
    win_rate   = (trades_df["adj_ret"] > 0).mean() * 100
    avg_ret    = trades_df["adj_ret"].mean() * 100
    total_ret  = equity_df["cum_ret"].iloc[-1] * 100
    bm_ret_t   = equity_df["bm_cum_ret"].iloc[-1] * 100
    excess     = total_ret - bm_ret_t
    liq_fail_n = trades_df["liq_failed"].sum()
    stopped_n  = trades_df["stopped"].sum()

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
    print(f"選股數: top {top_n}  持有: {hold_days}天  MIN_SCORE: {MIN_SCORE}")
    print(f"停損: ATR x{ATR_MULT}  交易成本: {(COST_RT+SLIPPAGE_RT)*100:.4f}%/邊")
    print(f"流動性失敗: {LIQUIDITY_FAIL*100:.0f}%  (共 {liq_fail_n} 筆)  ATR停損觸發: {stopped_n} 筆")
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
    feat_cols = load_feat_cols()

    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values(["date", "stock_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "fwd_ret_1d" not in df.columns or "label_up1" not in df.columns:
        print("[ERROR] 請先執行 python daily/build_features.py")
        sys.exit(1)

    # 動態停損: 用 atr_pct_raw 計算每筆的停損下限 (不預先 clip)
    has_atr_raw = "atr_pct_raw" in df.columns

    avail = [c for c in feat_cols if c in df.columns]
    has_down = "label_down1" in df.columns
    df.dropna(subset=avail + ["label_up1", "fwd_ret_1d"], inplace=True)

    all_dates  = sorted(df["date"].unique())
    total_days = len(all_dates)
    date_idx   = {d: i for i, d in enumerate(all_dates)}
    print(f"資料範圍: {all_dates[0].date()} ~ {all_dates[-1].date()} ({total_days} 交易日)")
    print(f"持有天數: {HOLD_DAYS}天  MIN_SCORE: {MIN_SCORE}  ATR停損倍數: x{ATR_MULT}")
    print(f"滑價: {SLIPPAGE_RT*100:.2f}%  流動性失敗: {LIQUIDITY_FAIL*100:.0f}%  起始資金: {INIT_CAPITAL:,} NTD")

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
        up_model, dn_model = train_fold(
            X_tr,
            train_df["label_up1"].values,
            train_df["label_down1"].values if has_down else np.zeros(len(train_df), dtype=np.float32),
        )
        print(f"完成  (訓練 {len(X_tr):,} 筆 | 測試 {len(test_df):,} 筆)")

        test_dates     = sorted(test_df["date"].unique())
        signal_indices = list(range(0, len(test_dates), HOLD_DAYS))

        for sig_i in signal_indices:
            dt = test_dates[sig_i]
            day_df = test_df[test_df["date"] == dt].dropna(subset=avail).copy()
            if len(day_df) < 1:
                continue

            X_day = pd.DataFrame(day_df[avail].values.astype(np.float32), columns=avail)
            day_df["up_prob"] = up_model.predict_proba(X_day)[:, 1]
            day_df["dn_prob"] = dn_model.predict_proba(X_day)[:, 1]
            day_df["score"]   = day_df["up_prob"] - day_df["dn_prob"]

            # MIN_SCORE 過濾 + score > 0
            candidates = day_df[(day_df["score"] >= MIN_SCORE)]
            if len(candidates) == 0:
                candidates = day_df[day_df["score"] > 0]
            if len(candidates) == 0:
                candidates = day_df

            picks = candidates.nlargest(min(TOP_N, len(candidates)), "up_prob")

            raw_rets  = []
            liq_flags = []
            stop_flags = []

            for _, row in picks.iterrows():
                sid = row["stock_id"]

                # 動態停損門檻
                atr_val = float(row["atr_pct_raw"]) if has_atr_raw and not pd.isna(row.get("atr_pct_raw", np.nan)) else 0.03
                dyn_stop = -ATR_MULT * max(atr_val, 0.005)  # 最低 -0.5% * 1.5 = -0.75%

                if rng.random() < LIQUIDITY_FAIL:
                    fallback = carry_over.pop(sid, 0.0)
                    raw_rets.append(fallback)
                    liq_flags.append(1)
                    stop_flags.append(0)
                    continue

                si = date_idx.get(dt)
                if si is None:
                    raw_rets.append(np.nan); liq_flags.append(0); stop_flags.append(0); continue

                exit_i  = min(si + HOLD_DAYS, len(all_dates) - 1)
                hold_df = df[(df["stock_id"] == sid) &
                             (df["date"] >  dt) &
                             (df["date"] <= all_dates[exit_i])]

                if hold_df.empty:
                    r = float(row["fwd_ret_1d"])
                    cum = max(r, dyn_stop * HOLD_DAYS)
                    raw_rets.append(cum); liq_flags.append(0)
                    stop_flags.append(int(r < dyn_stop * HOLD_DAYS))
                    continue

                # 逐日模擬：觸及 ATR 停損即出場
                cum = 0.0
                stopped = False
                for _, day_r in hold_df.iterrows():
                    daily_r = float(day_r["fwd_ret_1d"])
                    cum = (1 + cum) * (1 + daily_r) - 1
                    if cum <= dyn_stop:
                        stopped = True
                        cum = dyn_stop
                        break
                cum = min(cum, UP_CAP * HOLD_DAYS)

                if rng.random() < LIQUIDITY_FAIL:
                    carry_over[sid] = cum
                    raw_rets.append(0.0)
                    liq_flags.append(1)
                    stop_flags.append(int(stopped))
                else:
                    carry_over.pop(sid, None)
                    raw_rets.append(cum)
                    liq_flags.append(0)
                    stop_flags.append(int(stopped))

            raw_rets   = np.array(raw_rets,   dtype=float)
            liq_flags  = np.array(liq_flags,  dtype=int)
            stop_flags = np.array(stop_flags, dtype=int)
            valid      = ~np.isnan(raw_rets)
            if valid.sum() == 0: continue

            raw_rets_v  = raw_rets[valid]
            liq_v       = liq_flags[valid]
            stop_v      = stop_flags[valid]

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
            })

            valid_idx = np.where(valid)[0]
            for k, j in enumerate(valid_idx):
                row_p = picks.iloc[j]
                r     = float(raw_rets_v[k])
                r_adj = r - (cost_per_trade if liq_v[k] == 0 else 0.0)
                all_trades.append({
                    "fold":        fold_i + 1,
                    "signal_date": dt,
                    "stock_id":    row_p["stock_id"],
                    "up_prob":     round(float(row_p["up_prob"]), 4),
                    "dn_prob":     round(float(row_p["dn_prob"]), 4),
                    "score":       round(float(row_p["score"]),   4),
                    "entry_close": round(float(row_p["close"]),  2),
                    "raw_ret":     round(r,     6),
                    "adj_ret":     round(r_adj, 6),
                    "liq_failed":  int(liq_v[k]),
                    "stopped":     int(stop_v[k]),
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
