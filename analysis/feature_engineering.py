#!/usr/bin/env python3
"""
特徵工程：為每支成分股計算財報特徵
執行方式：從專案根目錄執行  python analysis/feature_engineering.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from pathlib import Path
from typing import List

# 從任何目錄執行都能找對 data/
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "roe", "pe_ratio", "pb_ratio", "gross_margin",
    "operating_margin", "debt_to_equity",
    "price_mom_3m", "price_mom_6m", "volatility_60d",
    "market_cap_rank"
]

# Yahoo Finance 對台股代碼測試後綴 (.TW 或 .TWO)
def get_valid_ticker(stock_id: str):
    for suffix in [".TW", ".TWO"]:
        t = yf.Ticker(f"{stock_id}{suffix}")
        try:
            hist = t.history(period="5d")
            if not hist.empty:
                return t, suffix
        except Exception:
            pass
    return None, None

def get_stock_info(stock_id: str) -> dict:
    ticker, suffix = get_valid_ticker(stock_id)
    if ticker is None:
        print(f"  [SKIP] {stock_id} 在 Yahoo Finance 找不到")
        return {"ticker": stock_id}
    try:
        info = ticker.info
        hist = ticker.history(period="9mo")
        closes = hist["Close"]

        mom3  = (closes.iloc[-1] / closes.iloc[-63]  - 1) if len(closes) >= 63  else np.nan
        mom6  = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
        vol60 = closes.pct_change().tail(60).std() * np.sqrt(252)

        return {
            "ticker":           stock_id,
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
        }
    except Exception as e:
        print(f"  [WARN] {stock_id}: {e}")
        return {"ticker": stock_id}

def build_feature_matrix(tickers: List[str]) -> pd.DataFrame:
    records = []
    for tk in tickers:
        print(f"  Processing {tk}...")
        records.append(get_stock_info(tk))
    df = pd.DataFrame(records)
    df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df

if __name__ == "__main__":
    # 00981A 近期持股樣本股票清單
    tickers = [
        "2330",  # 台積電
        "3691",  # 奇鋐 (上櫃小市場用 .TWO)
        "4552",  # 智邦
        "2308",  # 台達電
        "6669",  # 緯穎
        "2368",  # 金像電
        "8299",  # 群聯 (上櫃小市場用 .TWO)
        "6274",  # 台燿 (.TWO)
        "2383",  # 台光電
        "6223",  # 旺矽 (.TWO)
        "2454",  # 聯發科
        "3008",  # 大立光
        "2317",  # 鴿海
        "2382",  # 廣達
        "3711",  # 朗潤
    ]

    print("開始計算特徵矩陣...")
    df = build_feature_matrix(tickers)
    out = DATA_DIR / "features.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n特徵矩陣已儲存至 {out}")
    print(df[["ticker", "roe", "pe_ratio", "price_mom_3m", "market_cap_rank"]].to_string(index=False))
