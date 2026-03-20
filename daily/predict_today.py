#!/usr/bin/env python3
"""
每日收盤後執行，輸出今日異常名單
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

TOP_N = 20


def load_models():
    for p in [UP_MODEL, DOWN_MODEL, FEAT_JSON]:
        if not p.exists():
            print(f"[ERROR] 找不到 {p}，請先執行 python daily/train_model.py")
            sys.exit(1)
    with open(UP_MODEL,   "rb") as f: up_m   = pickle.load(f)
    with open(DOWN_MODEL, "rb") as f: down_m = pickle.load(f)
    with open(FEAT_JSON)        as f: meta   = json.load(f)

    # 相容新版 (dict) 與舊版 (list) FEAT_JSON
    if isinstance(meta, dict):
        feat_cols   = meta["feature_cols"]
        up_thresh   = meta.get("up_thresh",   0.45)
        down_thresh = meta.get("down_thresh", 0.45)
    else:
        feat_cols   = meta
        up_thresh   = 0.45
        down_thresh = 0.45

    print(f"特徵數: {len(feat_cols)}  up_thresh={up_thresh:.2f}  down_thresh={down_thresh:.2f}")
    return up_m, down_m, feat_cols, up_thresh, down_thresh


def _squeeze(s):
    """yfinance 新版可能回傳 MultiIndex DataFrame，強制轉為 1D Series"""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def fetch_recent_ohlcv(sym: str, lookback: int = 45) -> pd.DataFrame:
    h = yf.download(sym, period=f"{lookback}d", progress=False, auto_adjust=True)
    if h.empty:
        return pd.DataFrame()
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    close  = _squeeze(h.get("Close",  h.iloc[:, 0]))
    volume = _squeeze(h.get("Volume", pd.Series(np.nan, index=h.index)))
    high   = _squeeze(h.get("High",   close))
    low    = _squeeze(h.get("Low",    close))
    close.index = pd.to_datetime(close.index).tz_localize(None)
    idx = close.index
    return pd.DataFrame({
        "date":   idx,
        "close":  close.values,
        "volume": volume.reindex(idx).values,
        "high":   high.reindex(idx).values,
        "low":    low.reindex(idx).values,
    })


def compute_today_features(g: pd.DataFrame, inst_today: float = np.nan,
                           margin_chg: float = np.nan) -> dict | None:
    if len(g) < 21:
        return None
    c  = g["close"].astype(float).reset_index(drop=True)
    v  = g["volume"].astype(float).reset_index(drop=True)
    h  = g["high"].astype(float).reset_index(drop=True)
    lo = g["low"].astype(float).reset_index(drop=True)

    ma5_v   = v.rolling(5,  min_periods=3).mean().iloc[-1]
    ma20_v  = v.rolling(20, min_periods=10).mean().iloc[-1]
    std20_v = v.rolling(20, min_periods=10).std().iloc[-1]
    vz20    = (v.iloc[-1] - v.rolling(20, min_periods=10).mean().iloc[-1]) / std20_v \
              if std20_v and std20_v != 0 else np.nan

    ret1  = float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) >= 2 else np.nan
    ret5  = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else np.nan
    hlpct = float((h.iloc[-1] - lo.iloc[-1]) / c.iloc[-1]) if c.iloc[-1] != 0 else np.nan
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
        "inst_net_ratio": float(inst_today / ma20_v) if not np.isnan(inst_today) and ma20_v else np.nan,
        "margin_chg_pct": float(margin_chg) if not np.isnan(margin_chg) else np.nan,
    }


def main():
    up_m, down_m, feat_cols, up_thresh, down_thresh = load_models()
    client = get_client()
    today  = date.today().strftime("%Y-%m-%d")
    start  = (pd.Timestamp(today) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    print(f"預測日期: {today}")
    print(f"確認 yfinance 後綴...")
    valid = {}
    for tk in UNIVERSE:
        r = get_valid_symbol(tk)
        if r:
            valid[tk] = r[0]
    print(f"有效宇宙: {len(valid)} 支")

    rows = []
    for i, (tk, sym) in enumerate(valid.items()):
        g = fetch_recent_ohlcv(sym)
        if g.empty or len(g) < 21:
            continue

        inst_today = np.nan
        try:
            inst_rows = client.get("TaiwanStockInstitutionalInvestorsBuySell", tk, start)
            if inst_rows:
                idf = pd.DataFrame(inst_rows)
                idf["date"] = pd.to_datetime(idf["date"])
                today_inst = idf[idf["date"].dt.strftime("%Y-%m-%d") == today]
                if not today_inst.empty and {"buy", "sell"}.issubset(idf.columns):
                    inst_today = float(
                        today_inst["buy"].astype(float).sum() -
                        today_inst["sell"].astype(float).sum()
                    )
        except Exception:
            pass

        margin_chg = np.nan
        try:
            m_rows = client.get("TaiwanStockMarginPurchaseShortSale", tk, start)
            if m_rows:
                mdf = pd.DataFrame(m_rows)
                mdf["date"] = pd.to_datetime(mdf["date"])
                if {"MarginPurchaseBuy", "MarginPurchaseSell"}.issubset(mdf.columns):
                    mdf["net"] = mdf["MarginPurchaseBuy"].astype(float) - mdf["MarginPurchaseSell"].astype(float)
                    margin_chg = float(mdf["net"].tail(5).sum())
        except Exception:
            pass

        feat = compute_today_features(g, inst_today, margin_chg)
        if feat is None:
            continue

        # 用 DataFrame 傳入，避免 sklearn feature name warning
        X = pd.DataFrame([[feat.get(c, np.nan) for c in feat_cols]], columns=feat_cols)
        up_prob   = float(up_m.predict_proba(X)[0, 1])
        down_prob = float(down_m.predict_proba(X)[0, 1])

        rows.append({
            "date":      today,
            "stock_id":  tk,
            "up_prob":   round(up_prob,   4),
            "down_prob": round(down_prob, 4),
            "close":     round(float(g["close"].iloc[-1]), 2),
            **{k: round(float(fv), 4) if not (isinstance(fv, float) and np.isnan(fv)) else None
               for k, fv in feat.items()},
        })

        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(valid)}]")

    result = pd.DataFrame(rows)
    if result.empty:
        print("[WARN] 今日無資料")
        return

    result.sort_values("up_prob", ascending=False, inplace=True)
    result.to_csv(OUT_TODAY, index=False, encoding="utf-8-sig")

    print(f"\n=== 今日異常上漲候選 (top {TOP_N}, 閾值>{up_thresh:.2f}) ===")
    up_cands  = result[result["up_prob"] > up_thresh]
    show_up   = [c for c in ["stock_id","up_prob","vol_ratio_20d","inst_net_ratio","close"] if c in result.columns]
    print(up_cands[show_up].head(TOP_N).to_string(index=False))

    print(f"\n=== 今日異常下跌候選 (top {TOP_N}, 閾值>{down_thresh:.2f}) ===")
    down_cands = result.sort_values("down_prob", ascending=False)
    down_cands = down_cands[down_cands["down_prob"] > down_thresh]
    show_down  = [c for c in ["stock_id","down_prob","vol_ratio_20d","close"] if c in result.columns]
    print(down_cands[show_down].head(TOP_N).to_string(index=False))

    if OUT_LOG.exists():
        log = pd.concat([pd.read_csv(OUT_LOG), result], ignore_index=True)
    else:
        log = result
    log.to_csv(OUT_LOG, index=False, encoding="utf-8-sig")
    print(f"\n儲存: {OUT_TODAY}")
    print(f"累積 log: {OUT_LOG} ({len(log)} 筆)")


if __name__ == "__main__":
    main()
