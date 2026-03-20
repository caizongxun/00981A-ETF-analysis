#!/usr/bin/env python3
"""
每日收盤後執行 (18:00+)，輸出明日候選股票

預測架構:
  X = 今日收盤後已知特徵 (T)
  y = 明日 (T+1) 報酬 > 2% / < -2%

執行: python daily/predict_today.py
"""
import sys
import json
import pickle
from pathlib import Path
from datetime import date, datetime

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
DATA_READY_HOUR = 18


def load_models():
    for p in [UP_MODEL, DOWN_MODEL, FEAT_JSON]:
        if not p.exists():
            print(f"[ERROR] 找不到 {p}，請先執行 python daily/train_model.py")
            sys.exit(1)
    with open(UP_MODEL,   "rb") as f: up_m   = pickle.load(f)
    with open(DOWN_MODEL, "rb") as f: down_m = pickle.load(f)
    with open(FEAT_JSON)        as f: meta   = json.load(f)
    if isinstance(meta, dict):
        feat_cols   = meta["feature_cols"]
        up_thresh   = meta.get("up_thresh",   0.45)
        down_thresh = meta.get("down_thresh", 0.45)
    else:
        feat_cols = meta; up_thresh = down_thresh = 0.45
    print(f"特徵數: {len(feat_cols)}  up_thresh={up_thresh:.2f}  down_thresh={down_thresh:.2f}")
    return up_m, down_m, feat_cols, up_thresh, down_thresh


def _squeeze(s):
    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    return s


def fetch_recent_ohlcv(sym: str, lookback: int = 80) -> pd.DataFrame:
    """lookback=80 確保 60MA / ret_20d 有足夠數據"""
    h = yf.download(sym, period=f"{lookback}d", progress=False,
                    auto_adjust=False)  # 取 adj_close 計算特徵
    if h.empty: return pd.DataFrame()
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    close     = _squeeze(h.get("Close",     h.iloc[:, 0]))
    adj_close = _squeeze(h.get("Adj Close", close))
    volume    = _squeeze(h.get("Volume",    pd.Series(np.nan, index=h.index)))
    high      = _squeeze(h.get("High",      close))
    low       = _squeeze(h.get("Low",       close))
    close.index = pd.to_datetime(close.index).tz_localize(None)
    idx = close.index
    return pd.DataFrame({
        "date":      idx,
        "close":     close.values,
        "adj_close": adj_close.reindex(idx).values,
        "volume":    volume.reindex(idx).values,
        "high":      high.reindex(idx).values,
        "low":       low.reindex(idx).values,
    })


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    r     = (100 - 100 / (1 + rs)).iloc[-1]
    return float(r) if not np.isnan(r) else np.nan


def compute_features_today(g: pd.DataFrame,
                           inst_net: float = np.nan,
                           margin_chg_pct: float = np.nan,
                           short_ratio: float = np.nan) -> dict | None:
    """
    計算今日特徵 (= T 的收盤資訊)。
    全部使用 adj_close 計算特徵，與 build_features.py 一致。
    """
    if len(g) < 25: return None
    c  = g["close"].astype(float).reset_index(drop=True)
    ac = g["adj_close"].astype(float).reset_index(drop=True)
    v  = g["volume"].astype(float).reset_index(drop=True)
    h  = g["high"].astype(float).reset_index(drop=True)
    lo = g["low"].astype(float).reset_index(drop=True)

    def last(s): return float(s.iloc[-1]) if not np.isnan(s.iloc[-1]) else np.nan

    # 量能
    ma5_v   = v.rolling(5,  min_periods=3).mean()
    ma20_v  = v.rolling(20, min_periods=10).mean()
    std20_v = v.rolling(20, min_periods=10).std()
    vz20    = (v - ma20_v) / std20_v.replace(0, np.nan)
    sign    = np.sign(ac.diff().fillna(0))
    obv     = (sign * v).cumsum()
    obv_slope = (obv.diff(5) / ma20_v.replace(0, np.nan))

    # 報酬動能 (adj)
    ret1   = ac.pct_change(1)
    ret5   = ac.pct_change(5)
    ret20  = ac.pct_change(20)
    mom_acc = ret5 - ac.pct_change(10).shift(5)

    # 價格結構 (adj)
    ma5_c  = ac.rolling(5,  min_periods=3).mean()
    ma20_c = ac.rolling(20, min_periods=10).mean()
    ma60_c = ac.rolling(60, min_periods=30).mean()
    bb_std = ac.rolling(20, min_periods=10).std()
    bb_upper = ma20_c + 2 * bb_std
    bb_lower = ma20_c - 2 * bb_std
    bb_pos   = (ac - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # ATR
    tr = pd.concat([
        h - lo,
        (h - ac.shift(1)).abs(),
        (lo - ac.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14   = tr.rolling(14, min_periods=7).mean()
    atr_pct = atr14 / ac.replace(0, np.nan)

    # RSI
    rsi14 = rsi(ac, 14)

    # 成交金額
    turnover     = ac * v
    turnover_ma5  = turnover.rolling(5,  min_periods=3).mean()
    turnover_ma20 = turnover.rolling(20, min_periods=10).mean()
    turnover_ratio = turnover_ma5 / turnover_ma20.replace(0, np.nan)

    # 三大法人
    mv20 = last(ma20_v)
    inst_net_ratio    = float(inst_net / mv20) if not np.isnan(inst_net) and mv20 else np.nan
    # inst_net_ratio_5d 當日無法計算 5 日累積，用單日代替
    inst_net_ratio_5d = inst_net_ratio

    hlpct = (h - lo) / ac.replace(0, np.nan)

    return {
        "vol_ratio_5d":      last(v / ma5_v.replace(0, np.nan)),
        "vol_ratio_20d":     last(v / ma20_v.replace(0, np.nan)),
        "vol_zscore_20d":    last(vz20),
        "obv_slope":         last(obv_slope),
        "ret_1d":            last(ret1),
        "ret_5d":            last(ret5),
        "ret_20d":           last(ret20),
        "mom_acc":           last(mom_acc),
        "high_low_pct":      last(hlpct),
        "close_vs_ma5":      last(ac / ma5_c.replace(0, np.nan) - 1),
        "close_vs_ma20":     last(ac / ma20_c.replace(0, np.nan) - 1),
        "close_vs_ma60":     last(ac / ma60_c.replace(0, np.nan) - 1),
        "bb_pos":            last(bb_pos),
        "atr_pct":           last(atr_pct),
        "rsi_14":            rsi14,
        "inst_net_ratio":    inst_net_ratio,
        "inst_net_ratio_5d": inst_net_ratio_5d,
        "margin_chg_pct":    margin_chg_pct,
        "short_ratio":       short_ratio,
        "turnover_ratio":    last(turnover_ratio),
    }


def main():
    now = datetime.now()
    if now.hour < DATA_READY_HOUR:
        print(f"[WARN] 現在 {now.strftime('%H:%M')}，三大法人/融資資料最早 18:00 後才就緒\n")

    up_m, down_m, feat_cols, up_thresh, down_thresh = load_models()
    client = get_client()
    today  = date.today().strftime("%Y-%m-%d")
    start  = (pd.Timestamp(today) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    print(f"以今日 ({today}) 收盤資訊預測明日候選...")
    print("確認 yfinance 後綴...")
    valid = {}
    for tk in UNIVERSE:
        r = get_valid_symbol(tk)
        if r: valid[tk] = r[0]
    print(f"有效宇宙: {len(valid)} 支")

    rows = []
    null_inst = null_margin = 0
    for i, (tk, sym) in enumerate(valid.items()):
        g = fetch_recent_ohlcv(sym, lookback=80)
        if g.empty or len(g) < 25: continue

        # 三大法人
        inst_net = np.nan
        try:
            inst_rows = client.get("TaiwanStockInstitutionalInvestorsBuySell", tk, start)
            if inst_rows:
                idf = pd.DataFrame(inst_rows)
                idf["date"] = pd.to_datetime(idf["date"])
                td = idf[idf["date"].dt.strftime("%Y-%m-%d") == today]
                if not td.empty and {"buy", "sell"}.issubset(idf.columns):
                    inst_net = float(td["buy"].astype(float).sum() -
                                     td["sell"].astype(float).sum())
        except Exception: pass
        if np.isnan(inst_net): null_inst += 1

        # 融資
        margin_chg_pct = np.nan
        short_ratio    = np.nan
        try:
            m_rows = client.get("TaiwanStockMarginPurchaseShortSale", tk, start)
            if m_rows:
                mdf = pd.DataFrame(m_rows)
                mdf["date"] = pd.to_datetime(mdf["date"])
                if {"MarginPurchaseBuy", "MarginPurchaseSell"}.issubset(mdf.columns):
                    mb  = mdf["MarginPurchaseBuy"].astype(float)
                    ms  = mdf["MarginPurchaseSell"].astype(float)
                    net = mb - ms
                    mb_mean = mb.rolling(20).mean().iloc[-1]
                    margin_chg_pct = float(net.tail(5).sum() / mb_mean) \
                                     if mb_mean and mb_mean != 0 else np.nan
                if {"ShortSaleSell", "MarginPurchaseBuy"}.issubset(mdf.columns):
                    ss5 = mdf["ShortSaleSell"].astype(float).tail(5).sum()
                    mg5 = mdf["MarginPurchaseBuy"].astype(float).tail(5).sum()
                    short_ratio = float(ss5 / mg5) if mg5 and mg5 != 0 else np.nan
        except Exception: pass
        if np.isnan(margin_chg_pct): null_margin += 1

        feat = compute_features_today(g, inst_net, margin_chg_pct, short_ratio)
        if feat is None: continue

        X = pd.DataFrame([[feat.get(c, np.nan) for c in feat_cols]], columns=feat_cols)
        up_prob   = float(up_m.predict_proba(X)[0, 1])
        down_prob = float(down_m.predict_proba(X)[0, 1])
        score     = round(up_prob - down_prob, 4)

        rows.append({
            "date":      today,
            "stock_id":  tk,
            "up_prob":   round(up_prob,   4),
            "down_prob": round(down_prob, 4),
            "score":     score,
            "close":     round(float(g["close"].iloc[-1]), 2),
            **{k: (round(float(fv), 4) if isinstance(fv, float) and not np.isnan(fv) else None)
               for k, fv in feat.items()},
        })
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(valid)}]")

    total = len(valid)
    print(f"\n資料完整性: inst_net 缺失 {null_inst}/{total} | margin 缺失 {null_margin}/{total}")

    result = pd.DataFrame(rows)
    if result.empty:
        print("[WARN] 今日無資料"); return

    result.sort_values("up_prob", ascending=False, inplace=True)
    result.to_csv(OUT_TODAY, index=False, encoding="utf-8-sig")

    print(f"\n=== 明日上漲候選 (top {TOP_N}, 閾値>{up_thresh:.2f}) ===")
    up_cands = result[result["up_prob"] > up_thresh]
    show_up  = [c for c in ["stock_id","up_prob","down_prob","score",
                             "vol_ratio_20d","rsi_14","ret_1d","close"]
                if c in result.columns]
    print(up_cands[show_up].head(TOP_N).to_string(index=False))

    print(f"\n=== 明日下跌候選 (top {TOP_N}, 閾値>{down_thresh:.2f}) ===")
    down_cands = result.sort_values("down_prob", ascending=False)
    down_cands = down_cands[down_cands["down_prob"] > down_thresh]
    show_down  = [c for c in ["stock_id","down_prob","score","rsi_14","ret_1d","close"]
                  if c in result.columns]
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
