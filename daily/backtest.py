#!/usr/bin/env python3
"""
日頻模型回測系統

策略:
  - 每日收盤後用 T 特徵預測 T+1 報酬
  - 選 up_prob 前 5 名買入，權重等重
  - T+1 收盤出場 (保持 1 天)
  - 停損: 實際報酬 < -3% 則按 -3% 計算 (close-to-close 近似)
  - 基準: 每日持有相同宇宙均等權重 (naive benchmark)

輸入: data/daily_features.csv  (build_features.py 產生)
       models/daily_up_model.pkl  (train_model.py 產生)
輸出: data/backtest_trades.csv   每筆交易明細
       data/backtest_equity.csv   每日累計報酬

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

IN_CSV      = DATA_DIR / "daily_features.csv"
UP_MODEL    = MODEL_DIR / "daily_up_model.pkl"
FEAT_JSON   = MODEL_DIR / "daily_feature_cols.json"
OUT_TRADES  = DATA_DIR / "backtest_trades.csv"
OUT_EQUITY  = DATA_DIR / "backtest_equity.csv"

# 回測參數
BT_MONTHS   = 6          # 回測期間
TOP_N       = 5          # 每日買入前 N 名
STOP_LOSS   = -0.03      # 單日停損線
COST_RT     = 0.001425   # 單邊交易成本 (等周道手續費 + 證交税)


def load_model_meta():
    for p in [UP_MODEL, FEAT_JSON]:
        if not p.exists():
            print(f"[ERROR] 找不到 {p}，請先執行 python daily/train_model.py")
            sys.exit(1)
    with open(UP_MODEL, "rb") as f: model = pickle.load(f)
    with open(FEAT_JSON)       as f: meta  = json.load(f)
    feat_cols = meta["feature_cols"] if isinstance(meta, dict) else meta
    up_thresh = meta.get("up_thresh", 0.45) if isinstance(meta, dict) else 0.45
    return model, feat_cols, up_thresh


def run_backtest():
    model, feat_cols, up_thresh = load_model_meta()

    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    df.sort_values(["date", "stock_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 確認有 T+1 報酬欄位
    if "fwd_ret_1d" not in df.columns:
        print("[ERROR] 找不到 fwd_ret_1d，請先執行 python daily/build_features.py")
        sys.exit(1)

    # 回測時間範圍: 最新 6 個月的訓練時間之外
    max_date  = df["date"].max()
    bt_start  = max_date - pd.DateOffset(months=BT_MONTHS)
    bt_df     = df[df["date"] >= bt_start].copy()
    all_dates = sorted(bt_df["date"].unique())
    print(f"回測期間: {all_dates[0].date()} ~ {all_dates[-1].date()} ({len(all_dates)} 交易日)")

    avail = [c for c in feat_cols if c in bt_df.columns]

    trades  = []
    eq_rows = []
    cum_ret = 0.0   # 累計報酬 (log scale)
    bm_cum  = 0.0

    for dt in all_dates:
        day_df = bt_df[bt_df["date"] == dt].dropna(subset=avail)
        if len(day_df) < TOP_N:
            continue

        # 預測
        X = pd.DataFrame(day_df[avail].values.astype(np.float32), columns=avail)
        day_df = day_df.copy()
        day_df["up_prob"] = model.predict_proba(X)[:, 1]

        # 選前 N 名
        picks = day_df.nlargest(TOP_N, "up_prob")

        # 實際 T+1 報酬 (已在 features 裡, = fwd_ret_1d)
        rets = picks["fwd_ret_1d"].values
        rets = np.clip(rets, STOP_LOSS, None)   # 停損
        rets = rets - COST_RT * 2               # 去除成本 (買 + 賣)
        port_ret = float(np.mean(rets))         # 等權重

        # benchmark: 全宇宙等權重当日報酬
        univ_rets = bt_df[bt_df["date"] == dt]["fwd_ret_1d"].dropna()
        bm_ret    = float(univ_rets.mean()) if len(univ_rets) > 0 else 0.0

        cum_ret += np.log1p(port_ret)
        bm_cum  += np.log1p(bm_ret)

        eq_rows.append({
            "date":       dt,
            "port_ret":   round(port_ret, 6),
            "bm_ret":     round(bm_ret,   6),
            "cum_ret":    round(float(np.expm1(cum_ret)), 6),
            "bm_cum_ret": round(float(np.expm1(bm_cum)),  6),
        })

        for _, row in picks.iterrows():
            r = float(row["fwd_ret_1d"])
            r_adj = max(r, STOP_LOSS) - COST_RT * 2
            trades.append({
                "signal_date": dt,
                "stock_id":    row["stock_id"],
                "up_prob":     round(float(row["up_prob"]), 4),
                "entry_close": round(float(row["close"]),  2),
                "raw_ret":     round(r,     6),
                "adj_ret":     round(r_adj, 6),
                "stopped":     int(r < STOP_LOSS),
            })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(eq_rows)

    trades_df.to_csv(OUT_TRADES, index=False, encoding="utf-8-sig")
    equity_df.to_csv(OUT_EQUITY, index=False, encoding="utf-8-sig")

    # 結果展示
    print("\n" + "="*50)
    print(f"回測期間: {BT_MONTHS} 個月  選股數: top {TOP_N}  停損: {STOP_LOSS*100:.0f}%")
    print("="*50)

    n_trades  = len(trades_df)
    win_rate  = (trades_df["adj_ret"] > 0).mean() * 100
    avg_ret   = trades_df["adj_ret"].mean() * 100
    total_ret = equity_df["cum_ret"].iloc[-1] * 100
    bm_ret_t  = equity_df["bm_cum_ret"].iloc[-1] * 100
    excess    = total_ret - bm_ret_t
    stop_n    = trades_df["stopped"].sum()

    # 最大回撤
    eq = equity_df["cum_ret"].values + 1
    roll_max = np.maximum.accumulate(eq)
    dd = (eq - roll_max) / roll_max
    max_dd = float(dd.min()) * 100

    # 年化 Sharpe
    daily_rets = equity_df["port_ret"].values
    sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) \
             if np.std(daily_rets) > 0 else 0.0

    print(f"總交易筆數  : {n_trades}")
    print(f"勝率       : {win_rate:.1f}%")
    print(f"平均單筆報酬: {avg_ret:+.3f}%")
    print(f"單日停損筆數: {stop_n} ({stop_n/n_trades*100:.1f}%)")
    print(f"策略總報酬  : {total_ret:+.2f}%")
    print(f"基準總報酬  : {bm_ret_t:+.2f}%")
    print(f"超額報酬    : {excess:+.2f}%")
    print(f"最大回撤    : {max_dd:.2f}%")
    print(f"年化 Sharpe : {sharpe:.3f}")
    print("="*50)
    print(f"輸出: {OUT_TRADES}")
    print(f"輸出: {OUT_EQUITY}")

    # 按月分装報酬
    equity_df["month"] = equity_df["date"].dt.to_period("M")
    monthly = equity_df.groupby("month").agg(
        port=pd.NamedAgg("port_ret", lambda x: (1 + x).prod() - 1),
        bm=pd.NamedAgg("bm_ret",   lambda x: (1 + x).prod() - 1),
    )
    monthly["excess"] = monthly["port"] - monthly["bm"]
    print("\n月分装報酬:")
    print((monthly * 100).round(2).to_string())

    return equity_df, trades_df


if __name__ == "__main__":
    run_backtest()
