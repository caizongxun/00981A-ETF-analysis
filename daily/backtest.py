#!/usr/bin/env python3
"""
日頻模型回測系統 - Walk-Forward Out-of-Sample v11

核心改變 (vs v10):
  - REBAL_FREQ=5: 每 5 天（周）重新評分一次
  - 換股邏輯不變: 只換需要換的，續抱不收費
  - 持倉報酬使用 fwd_ret_5d (實際5日個股報酬)
  - 空頭條件不變

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
REBAL_FREQ       = 5       # 每幾個交易日重新評分一次
STOP_LOSS        = -0.15
UP_CAP           =  0.20
COST_RT          = 0.001425
SLIPPAGE_RT      = 0.003
INIT_CAPITAL     = 10_000
MIN_PRED_REL     = 0.0


def load_feat_cols():
    if not FEAT_JSON.exists():
        print(f"[ERROR] 找不到 {FEAT_JSON}，請先執行 train_model.py")
        sys.exit(1)
    with open(FEAT_JSON) as f:
        meta = json.load(f)
    cols       = meta["feature_cols"] if isinstance(meta, dict) else meta
    model_type = meta.get("model_type", "regressor") if isinstance(meta, dict) else "regressor"
    target     = meta.get("target", "fwd_rel_5d")    if isinstance(meta, dict) else "fwd_rel_5d"
    return cols, model_type, target


def fetch_0050_returns(start: str, end: str) -> pd.Series:
    print("下載 0050 基準資料...", end=" ", flush=True)
    h = yf.download("0050.TW",
                    start=(pd.Timestamp(start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                    end=(pd.Timestamp(end)   + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True)
    if h.empty:
        print("失敗"); return pd.Series(dtype=float)
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
        n_estimators=1000, learning_rate=0.02,
        num_leaves=63, max_depth=6,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.5,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def summarize(trades_df, equity_df, n_folds, top_n, rebal_freq):
    buy_df    = trades_df[trades_df["action"] == "buy"] if "action" in trades_df.columns else trades_df
    n_rebal   = len(buy_df)
    total_ret = equity_df["cum_ret"].iloc[-1] * 100
    bm_ret_t  = equity_df["bm_cum_ret"].iloc[-1] * 100
    excess    = total_ret - bm_ret_t
    skip_days = int(equity_df["skipped"].sum()) if "skipped" in equity_df.columns else 0

    eq       = equity_df["cum_ret"].values + 1
    roll_max = np.maximum.accumulate(eq)
    max_dd   = float(((eq - roll_max) / roll_max).min()) * 100

    bt_s = equity_df["date"].min()
    bt_e = equity_df["date"].max()
    n_cal  = (pd.Timestamp(bt_e) - pd.Timestamp(bt_s)).days
    n_years = n_cal / 365.25

    ann_ret = float((1 + total_ret/100)**(1/n_years)-1)*100 if n_years > 0 else 0.0
    bm_ann  = float((1 + bm_ret_t /100)**(1/n_years)-1)*100 if n_years > 0 else 0.0

    eq_rets = equity_df["port_ret"].values
    sharpe  = float(np.mean(eq_rets)/np.std(eq_rets)*np.sqrt(252)) \
              if np.std(eq_rets) > 0 else 0.0

    final_capital = equity_df["capital"].iloc[-1]

    # 換股比例統計
    if "action" in trades_df.columns:
        rebal_days = trades_df[trades_df["action"].isin(["buy","sell"])]["date"].nunique()
        total_test_days = equity_df["date"].nunique()
        turnover_rate = rebal_days / max(total_test_days, 1) * 100
    else:
        turnover_rate = 0

    print("\n" + "="*60)
    print("Walk-Forward OOS 回測結果  (基準: 0050)")
    print(f"回測期間: {pd.Timestamp(bt_s).date()} ~ {pd.Timestamp(bt_e).date()}  "
          f"({n_cal}日/{n_years:.2f}年)  Folds: {n_folds}")
    print(f"選股: top {top_n}  重評頻率: 每 {rebal_freq} 天  MIN_PRED_REL: {MIN_PRED_REL}")
    print(f"交易成本: {(COST_RT+SLIPPAGE_RT)*100:.4f}%/邊  空手跳過: {skip_days} 次")
    print(f"換股天數占比: {turnover_rate:.1f}%  買入筆數: {n_rebal}")
    print("="*60)
    print(f"起始資金      : {INIT_CAPITAL:,.0f} NTD")
    print(f"最終資金      : {final_capital:,.0f} NTD")
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
    feat_cols, model_type, target = load_feat_cols()

    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values(["date", "stock_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    fwd_col = "fwd_ret_5d" if "fwd_ret_5d" in df.columns else "fwd_ret_1d"
    if fwd_col not in df.columns:
        print("[ERROR] 請先執行 python daily/build_features.py")
        sys.exit(1)

    df[fwd_col] = df[fwd_col].clip(STOP_LOSS, UP_CAP)

    avail = [c for c in feat_cols if c in df.columns]
    df.dropna(subset=avail + [fwd_col], inplace=True)

    # 建立 (stock_id, date) 快速查詢
    df_idx = df.set_index(["stock_id", "date"])
    df_idx.sort_index(inplace=True)

    all_dates = sorted(df["date"].unique())
    print(f"資料範圍: {all_dates[0].date()} ~ {all_dates[-1].date()} ({len(all_dates)} 交易日)")
    print(f"重評頻率: 每 {REBAL_FREQ} 天  TOP_N={TOP_N}  MIN_PRED_REL={MIN_PRED_REL}")
    print(f"起始資金: {INIT_CAPITAL:,} NTD  模型: {model_type}  目標: {target}  特徵: {len(avail)} 個")

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
    cum_ret  = 0.0
    bm_cum   = 0.0
    capital  = float(INIT_CAPITAL)
    holdings = set()
    cost_one = (COST_RT + SLIPPAGE_RT) * 2

    for fold_i, (fold_start, fold_end) in enumerate(folds):
        print(f"\n[Fold {fold_i+1}/{N_FOLDS}] 訓練中...", end=" ", flush=True)
        train_df = df[df["date"] < fold_start]
        test_df  = df[(df["date"] >= fold_start) & (df["date"] <= fold_end)]

        X_tr = train_df[avail].astype(np.float32)
        y_tr = train_df[target].fillna(0).values.astype(np.float32)
        model = train_fold_reg(X_tr, y_tr)
        print(f"完成  (訓練 {len(X_tr):,} 筆 | 測試 {len(test_df):,} 筆)")

        test_dates = sorted(test_df["date"].unique())

        # 計算每一天的預測分數 (重評日再計算)
        rebal_dates = set(test_dates[i] for i in range(0, len(test_dates), REBAL_FREQ))
        current_top = set()

        for day_i, dt in enumerate(test_dates):
            bear = is_market_bear(bm_ret_series, dt, window=20)

            # 重評日: 重新選股
            if dt in rebal_dates:
                day_df = test_df[test_df["date"] == dt].dropna(subset=avail).copy()
                if len(day_df) > 0:
                    X_day = day_df[avail].astype(np.float32)
                    day_df = day_df.copy()
                    day_df["predicted_rel"] = model.predict(X_day)

                    # 空頭條件
                    if bear and day_df["predicted_rel"].max() < MIN_PRED_REL:
                        # 清倉
                        n_sell = len(holdings)
                        if n_sell > 0:
                            sell_cost = cost_one * n_sell / max(TOP_N, 1)
                            capital  *= (1 - sell_cost)
                            for sid in holdings:
                                all_trades.append({
                                    "fold": fold_i+1, "date": dt, "stock_id": sid,
                                    "action": "sell_bear", "predicted_rel": None, "adj_ret": -cost_one,
                                })
                            holdings = set()
                        current_top = set()

                        bm_r = float(bm_ret_series.get(dt, 0.0))
                        bm_cum += np.log1p(bm_r)
                        all_eq_rows.append({
                            "date": dt, "fold": fold_i+1, "port_ret": 0.0,
                            "bm_ret": round(bm_r, 6),
                            "cum_ret": round(float(np.expm1(cum_ret)), 6),
                            "bm_cum_ret": round(float(np.expm1(bm_cum)), 6),
                            "capital": round(capital, 2), "skipped": 1,
                        })
                        continue

                    new_top = set(
                        day_df.nlargest(min(TOP_N, len(day_df)), "predicted_rel")["stock_id"]
                    )

                    # 換股
                    to_sell = holdings - new_top
                    to_buy  = new_top  - holdings
                    n_sell  = len(to_sell)
                    n_buy   = len(to_buy)

                    # 計算持倉報酬: 是用 fwd_ret_5d (5天累計, 庺位平均)
                    # 【重評日使用這期持倉的報酬】
                    port_ret = 0.0
                    if len(holdings) > 0:
                        rets = []
                        for sid in holdings:
                            try:
                                r = float(df_idx.loc[(sid, dt), fwd_col])
                            except KeyError:
                                r = 0.0
                            rets.append(r)
                        port_ret = float(np.mean(rets))

                    # 換股成本
                    if len(new_top) > 0:
                        port_ret -= cost_one * (n_sell + n_buy) / len(new_top)

                    holdings  = new_top
                    current_top = new_top
                    capital  *= (1 + port_ret)
                    cum_ret  += np.log1p(port_ret)

                    pred_map = dict(zip(day_df["stock_id"], day_df["predicted_rel"]))
                    for sid in to_sell:
                        all_trades.append({
                            "fold": fold_i+1, "date": dt, "stock_id": sid,
                            "action": "sell", "predicted_rel": round(pred_map.get(sid, 0.0), 4),
                            "adj_ret": -cost_one,
                        })
                    for sid in to_buy:
                        all_trades.append({
                            "fold": fold_i+1, "date": dt, "stock_id": sid,
                            "action": "buy", "predicted_rel": round(pred_map.get(sid, 0.0), 4),
                            "adj_ret": -cost_one,
                        })

                    bm_r = float(bm_ret_series.get(dt, 0.0))
                    bm_cum += np.log1p(bm_r)
                    all_eq_rows.append({
                        "date": dt, "fold": fold_i+1,
                        "port_ret": round(port_ret, 6), "bm_ret": round(bm_r, 6),
                        "cum_ret": round(float(np.expm1(cum_ret)), 6),
                        "bm_cum_ret": round(float(np.expm1(bm_cum)), 6),
                        "capital": round(capital, 2), "skipped": 0,
                    })

            else:
                # 非重評日: 持倉不動，記錄当天个股日報酬
                port_ret = 0.0
                if len(holdings) > 0:
                    rets = []
                    for sid in holdings:
                        try:
                            r = float(df_idx.loc[(sid, dt), "fwd_ret_1d"])
                        except KeyError:
                            r = 0.0
                        rets.append(r)
                    port_ret = float(np.mean(rets))

                capital *= (1 + port_ret)
                cum_ret += np.log1p(port_ret)

                bm_r = float(bm_ret_series.get(dt, 0.0))
                bm_cum += np.log1p(bm_r)
                all_eq_rows.append({
                    "date": dt, "fold": fold_i+1,
                    "port_ret": round(port_ret, 6), "bm_ret": round(bm_r, 6),
                    "cum_ret": round(float(np.expm1(cum_ret)), 6),
                    "bm_cum_ret": round(float(np.expm1(bm_cum)), 6),
                    "capital": round(capital, 2), "skipped": 0,
                })

    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(all_eq_rows)

    trades_df.to_csv(OUT_TRADES, index=False, encoding="utf-8-sig")
    equity_df.to_csv(OUT_EQUITY, index=False, encoding="utf-8-sig")

    summarize(trades_df, equity_df, N_FOLDS, TOP_N, REBAL_FREQ)
    print(f"\n輸出: {OUT_TRADES}")
    print(f"輸出: {OUT_EQUITY}")
    return equity_df, trades_df


if __name__ == "__main__":
    run_backtest()
