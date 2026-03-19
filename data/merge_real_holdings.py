#!/usr/bin/env python3
"""
將手動下載的 real_holdings_raw.csv 和 yfinance 特徵合併
輸出 holdings_history.csv 供 walk_forward.py 使用

執行： python data/merge_real_holdings.py
"""
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CANDIDATE_POOL = {
    "2330": ".TW", "4552": ".TW", "2308": ".TW", "6669": ".TW",
    "2368": ".TW", "2383": ".TW", "2454": ".TW", "3008": ".TW",
    "2317": ".TW", "2382": ".TW", "3711": ".TW", "2357": ".TW",
    "2379": ".TW", "3691": ".TWO", "8299": ".TWO",
    "6274": ".TWO", "6223": ".TWO", "5274": ".TWO",
}
INDUSTRY_MAP = {
    "2330": "半導體", "4552": "通訊網路", "2308": "電子零組件", "6669": "電腦週邊",
    "2368": "電子零組件", "2383": "電子零組件", "2454": "半導體", "3008": "光電",
    "2317": "電腦週邊", "2382": "電腦週邊", "3711": "通訊網路", "2357": "電腦週邊",
    "2379": "半導體", "3691": "電子零組件", "8299": "半導體",
    "6274": "電子零組件", "6223": "半導體", "5274": "半導體",
}

def get_features(pool, label_set):
    records = []
    for tk, suffix in pool.items():
        sym = f"{tk}{suffix}"
        try:
            t = yf.Ticker(sym)
            info = t.info
            hist = t.history(period="9mo")
            if hist.empty: continue
            closes = hist["Close"]
            mom3  = (closes.iloc[-1] / closes.iloc[-63]  - 1) if len(closes) >= 63  else np.nan
            mom6  = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
            vol60 = closes.pct_change().tail(60).std() * np.sqrt(252)
            records.append({
                "stock_id":         str(tk),
                "industry":         INDUSTRY_MAP.get(tk, "不明"),
                "roe":              info.get("returnOnEquity",   np.nan),
                "pe_ratio":         info.get("trailingPE",       np.nan),
                "pb_ratio":         info.get("priceToBook",      np.nan),
                "gross_margin":     info.get("grossMargins",     np.nan),
                "operating_margin": info.get("operatingMargins", np.nan),
                "debt_to_equity":   info.get("debtToEquity",     np.nan),
                "market_cap":       info.get("marketCap",        np.nan),
                "price_mom_3m":     mom3,
                "price_mom_6m":     mom6,
                "volatility_60d":   vol60,
                "in_etf":           int(tk in label_set),
            })
        except Exception as e:
            print(f"  [WARN] {sym}: {e}")
    df = pd.DataFrame(records)
    if not df.empty:
        df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df


if __name__ == "__main__":
    raw_path = DATA_DIR / "real_holdings_raw.csv"
    if not raw_path.exists():
        print(f"[ERROR] 找不到 {raw_path}")
        print("請參考 docs/manual_download.md 手動補充資料")
        exit(1)

    raw = pd.read_csv(raw_path, dtype={"stock_id": str})
    periods = sorted(raw["period"].unique())
    print(f"讀到 {len(raw)} 筆資料，{len(periods)} 個時期: {periods}")

    all_dfs = []
    for ym in periods:
        holdings_set = set(raw[raw["period"] == ym]["stock_id"].astype(str))
        print(f"\n[{ym}] 持股: {sorted(holdings_set)}")
        feat_df = get_features(CANDIDATE_POOL, holdings_set)
        if not feat_df.empty:
            feat_df["period"] = ym
            all_dfs.append(feat_df)

    if all_dfs:
        out = DATA_DIR / "holdings_history.csv"
        full = pd.concat(all_dfs, ignore_index=True)
        full.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n完成! 已儲存 {out} ({len(full)} 筆)")
        print("接下來: python backtest/walk_forward.py")
    else:
        print("[ERROR] 沒有有效資料")
