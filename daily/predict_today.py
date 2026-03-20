#!/usr/bin/env python3
"""
每日收盤後執行，輸出今日異常名單

輸入:  models/daily_up_model.pkl
       models/daily_down_model.pkl
       models/daily_feature_cols.json
輸出:  data/anomaly_today.csv  (每次覆寫)
       data/anomaly_log.csv    (歷史累積)

執行: python daily/predict_today.py
"""
import sys
import json
import pickle
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from finmind_client import get_client
from fetch_history import UNIVERSE, get_valid_symbol

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"

UP_MODEL   = MODEL_DIR / "daily_up_model.pkl"
DOWN_MODEL = MODEL_DIR / "daily_down_model.pkl"
FEAT_JSON  = MODEL_DIR / "daily_feature_cols.json"
OUT_TODAY  = DATA_DIR / "anomaly_today.csv"
OUT_LOG    = DATA_DIR / "anomaly_log.csv"

TOP_N = 20       # 輸出前 N 名
UP_PROB_THRESH   = 0.55
DOWN_PROB_THRESH = 0.55


def load_models():
    for p in [UP_MODEL, DOWN_MODEL, FEAT_JSON]:
        if not p.exists():
            print(f"[ERROR] 找不到 {p}，請先執行 python daily/train_model.py")
            sys.exit(1)
    with open(UP_MODEL,   "rb") as f: up_m   = pickle.load(f)
    with open(DOWN_MODEL, "rb") as f: down_m = pickle.load(f)
    with open(FEAT_JSON)         as f: feats  = json.load(f)
    return up_m, down_m, feats


def fetch_recent_ohlcv(sym: str, lookback: int = 40) -> pd.DataFrame:
    """拉最近 lookback 天的 OHLCV"""
    h = yf.download(sym, period=f"{lookback}d", progress=False, auto_adjust=True)
    if h.empty:
        return pd.DataFrame()
    close  = h["Close"] if "Close" in h else h.iloc[:, 0]
    volume = h["Volume"] if "Volume" in h else pd.Series(dtype=float)
    high   = h["High"]   if "High"  in h else close
    low    = h["Low"]    if "Low"   in h else close
    for s in [close, volume, high, low]:
        if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return pd.DataFrame({
        "date":   close.index,
        "close":  close.values,
        "volume": volume.reindex(close.index).values,
        "high":   high.reindex(close.index).values,
        "low":    low.reindex(close.index).values,
    })


def compute_today_features(g: pd.DataFrame, inst_today: float = np.nan,
                           margin_chg: float = np.nan) -> dict | None:
    """用最近 40 天資料算出今日特徵 (最後一列)"""
    if len(g) < 21:
        return None
    c = g["close"].astype(float)
    v = g["volume"].astype(float)
    h = g["high"].astype(float)
    lo= g["low"].astype(float)

    ma5_v  = v.rolling(5,  min_periods=3).mean().iloc[-1]
    ma20_v = v.rolling(20, min_periods=10).mean().iloc[-1]
    vz20   = ((v - v.rolling(20, min_periods=10).mean()) /
              v.rolling(20, min_periods=10).std().replace(0, np.nan)).iloc[-1]
    ret1   = float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) >= 2 else np.nan
    ret5   = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else np.nan
    hlpct  = float((h.iloc[-1] - lo.iloc[-1]) / c.iloc[-1]) if c.iloc[-1] != 0 else np.nan
    ma5_c  = c.rolling(5,  min_periods=3).mean().iloc[-1]
    ma20_c = c.rolling(20, min_periods=10).mean().iloc[-1]

    return {
        "vol_ratio_5d":   float(v.iloc[-1] / ma5_v)  if ma5_v  and ma5_v  != 0 else np.nan,
        "vol_ratio_20d":  float(v.iloc[-1] / ma20_v) if ma20_v and ma20_v != 0 else np.nan,
        "vol_zscore_20d": float(vz20),
        "ret_1d":         ret1,
        "ret_5d":         ret5,
        "high_low_pct":   hlpct,
        "close_vs_ma5":   float(c.iloc[-1] / ma5_c  - 1) if ma5_c  and ma5_c  != 0 else np.nan,
        "close_vs_ma20":  float(c.iloc[-1] / ma20_c - 1) if ma20_c and ma20_c != 0 else np.nan,
        "inst_net_ratio": float(inst_today / ma20_v) if (not np.isnan(inst_today)) and ma20_v else np.nan,
        "margin_chg_pct": float(margin_chg),
    }


def main():
    up_m, down_m, feat_cols = load_models()
    client = get_client()
    today  = date.today().strftime("%Y-%m-%d")
    start  = (pd.Timestamp(today) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    print(f"預測日期: {today}")
    print(f"載入宇宙 {len(UNIVERSE)} 支...")

    valid = {}
    for tk in UNIVERSE:
        r = get_valid_symbol(tk)
        if r:
            valid[tk] = r[0]

    rows = []
    for i, (tk, sym) in enumerate(valid.items()):
        g = fetch_recent_ohlcv(sym, lookback=45)
        if g.empty:
            continue

        # 三大法人今日
        inst_rows = client.get("TaiwanStockInstitutionalInvestorsBuySell", tk, start)
        inst_today = np.nan
        if inst_rows:
            idf = pd.DataFrame(inst_rows)
            idf["date"] = pd.to_datetime(idf["date"])
            today_inst = idf[idf["date"] == today]
            if not today_inst.empty and {"buy", "sell"}.issubset(idf.columns):
                inst_today = float(
                    today_inst["buy"].astype(float).sum() -
                    today_inst["sell"].astype(float).sum()
                )

        # 融資近期變化
        margin_chg = np.nan
        m_rows = client.get("TaiwanStockMarginPurchaseShortSale", tk, start)
        if m_rows:
            mdf = pd.DataFrame(m_rows)
            mdf["date"] = pd.to_datetime(mdf["date"])
            if {"MarginPurchaseBuy", "MarginPurchaseSell"}.issubset(mdf.columns):
                mdf["net"] = mdf["MarginPurchaseBuy"].astype(float) - mdf["MarginPurchaseSell"].astype(float)
                margin_chg = float(mdf["net"].tail(5).sum())

        feat = compute_today_features(g, inst_today, margin_chg)
        if feat is None:
            continue

        X = np.array([[feat.get(c, np.nan) for c in feat_cols]], dtype=np.float32)
        up_prob   = float(up_m.predict_proba(X)[0, 1])
        down_prob = float(down_m.predict_proba(X)[0, 1])

        rows.append({
            "date":      today,
            "stock_id":  tk,
            "up_prob":   round(up_prob,   4),
            "down_prob": round(down_prob, 4),
            "close":     round(float(g["close"].iloc[-1]), 2),
            **{k: round(v, 4) if not np.isnan(v) else None for k, v in feat.items()},
        })

        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(valid)}]")

    result = pd.DataFrame(rows)
    if result.empty:
        print("[WARN] 今日無資料")
        return

    # 排序輸出
    result.sort_values("up_prob", ascending=False, inplace=True)
    result.to_csv(OUT_TODAY, index=False, encoding="utf-8-sig")
    print(f"\n=== 今日異常上漲候選 (top {TOP_N}) ===")
    print(result[result["up_prob"] > UP_PROB_THRESH][["stock_id", "up_prob", "vol_ratio_20d", "inst_net_ratio", "close"]]
          .head(TOP_N).to_string(index=False))
    print(f"\n=== 今日異常下跌候選 (top {TOP_N}) ===")
    print(result.sort_values("down_prob", ascending=False)
          [result["down_prob"] > DOWN_PROB_THRESH][["stock_id", "down_prob", "vol_ratio_20d", "close"]]
          .head(TOP_N).to_string(index=False))

    # 累積 log
    if OUT_LOG.exists():
        log = pd.read_csv(OUT_LOG)
        log = pd.concat([log, result], ignore_index=True)
    else:
        log = result
    log.to_csv(OUT_LOG, index=False, encoding="utf-8-sig")
    print(f"\n儲存: {OUT_TODAY}")
    print(f"累積 log: {OUT_LOG}")


if __name__ == "__main__":
    main()
