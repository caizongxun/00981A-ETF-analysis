#!/usr/bin/env python3
"""
特徵工程：為每支成分股計算財報特徵，用於分析選股邏輯
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List

# 目標特徵清單
FEATURES = [
    "market_cap",        # 市值
    "market_cap_rank",   # 市值排名
    "roe",               # 股東權益報酬率
    "revenue_growth_yoy",# 營收年增率
    "eps_growth_yoy",    # EPS 年增率
    "pe_ratio",          # 本益比
    "pb_ratio",          # 股價淨值比
    "gross_margin",      # 毛利率
    "operating_margin",  # 營業利益率
    "debt_to_equity",    # 負債/股東權益
    "price_mom_3m",      # 3個月動能
    "price_mom_6m",      # 6個月動能
    "volatility_60d",    # 60日波動率
    "industry",          # 產業別
]

def get_stock_info(ticker: str, suffix=".TW") -> dict:
    """
    使用 yfinance 取得個股基本財報指標
    """
    try:
        t = yf.Ticker(f"{ticker}{suffix}")
        info = t.info
        hist = t.history(period="6mo")
        
        roe = info.get("returnOnEquity", np.nan)
        pe = info.get("trailingPE", np.nan)
        pb = info.get("priceToBook", np.nan)
        market_cap = info.get("marketCap", np.nan)
        gross_margin = info.get("grossMargins", np.nan)
        operating_margin = info.get("operatingMargins", np.nan)
        de_ratio = info.get("debtToEquity", np.nan)
        
        # 動能計算
        if len(hist) >= 60:
            mom_3m = (hist["Close"].iloc[-1] / hist["Close"].iloc[-63] - 1) if len(hist) >= 63 else np.nan
            mom_6m = (hist["Close"].iloc[-1] / hist["Close"].iloc[-126] - 1) if len(hist) >= 126 else np.nan
            vol_60d = hist["Close"].pct_change().tail(60).std() * np.sqrt(252)
        else:
            mom_3m = mom_6m = vol_60d = np.nan
        
        return {
            "ticker": ticker,
            "market_cap": market_cap,
            "roe": roe,
            "pe_ratio": pe,
            "pb_ratio": pb,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "debt_to_equity": de_ratio,
            "price_mom_3m": mom_3m,
            "price_mom_6m": mom_6m,
            "volatility_60d": vol_60d,
        }
    except Exception as e:
        print(f"[WARN] {ticker} 資料取得失敗: {e}")
        return {"ticker": ticker}

def build_feature_matrix(tickers: List[str]) -> pd.DataFrame:
    """
    批次取得所有持股的特徵矩陣
    """
    records = []
    for ticker in tickers:
        print(f"Processing {ticker}...")
        record = get_stock_info(ticker)
        records.append(record)
    
    df = pd.DataFrame(records)
    # 市值排名（越小代表市值越大）
    df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df

if __name__ == "__main__":
    # 00981A 近期成分股
    tickers = ["2330", "3691", "4552", "2308", "6669",
               "2368", "8299", "6274", "2383", "6223"]
    
    print("開始計算特徵矩陣...")
    df = build_feature_matrix(tickers)
    df.to_csv("../data/features.csv", index=False, encoding="utf-8-sig")
    print("特徵矩陣已儲存至 data/features.csv")
    print(df.to_string())
