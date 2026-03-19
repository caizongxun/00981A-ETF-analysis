#!/usr/bin/env python3
"""
合成歷史持股資料生成器
執行方式： python data/generate_synthetic_holdings.py
"""
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CANDIDATE_POOL = {
    "2330": ".TW", "4552": ".TW", "2308": ".TW", "6669": ".TW",
    "2368": ".TW", "2383": ".TW", "2454": ".TW", "3008": ".TW",
    "2317": ".TW", "2382": ".TW", "3711": ".TW", "2357": ".TW",
    "2379": ".TW", "3691": ".TWO", "8299": ".TWO",
    "6274": ".TWO", "6223": ".TWO", "5274": ".TWO",
}

STOCK_NAMES = {
    "2330": "台積電", "4552": "智邦", "2308": "台達電", "6669": "緯穎",
    "2368": "金像電", "2383": "台光電", "2454": "聯發科", "3008": "大立光",
    "2317": "鴿海", "2382": "廣達", "3711": "朗潤", "2357": "華硕",
    "2379": "瑞昑", "3691": "奇鋐", "8299": "群聯",
    "6274": "台燿", "6223": "旺矽", "5274": "嘉澤科",
}

INDUSTRY_MAP = {
    "2330": "半導體", "4552": "通訊網路", "2308": "電子零組件", "6669": "電腦週邊",
    "2368": "電子零組件", "2383": "電子零組件", "2454": "半導體", "3008": "光電",
    "2317": "電腦週邊", "2382": "電腦週邊", "3711": "通訊網路", "2357": "電腦週邊",
    "2379": "半導體", "3691": "電子零組件", "8299": "半導體",
    "6274": "電子零組件", "6223": "半導體", "5274": "半導體",
}

N_MONTHS   = 9
START_YEAR = 2025
START_MON  = 5   # 00981A 上市於 2025-05

def period_label(year, month):
    """month 超過 12 時自動进位"""
    while month > 12:
        month -= 12
        year  += 1
    return f"{year}-{month:02d}"

def get_features_snapshot(pool, label_set):
    records = []
    for tk, suffix in pool.items():
        sym = f"{tk}{suffix}"
        try:
            t    = yf.Ticker(sym)
            info = t.info
            hist = t.history(period="9mo")
            if hist.empty:
                continue
            closes = hist["Close"]
            mom3  = (closes.iloc[-1] / closes.iloc[-63]  - 1) if len(closes) >= 63  else np.nan
            mom6  = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
            vol60 = closes.pct_change().tail(60).std() * np.sqrt(252)
            records.append({
                "stock_id":         str(tk),
                "stock_name":       STOCK_NAMES.get(tk, tk),
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
            print(f"  [OK] {sym}")
        except Exception as e:
            print(f"  [WARN] {sym}: {e}")
    df = pd.DataFrame(records)
    if not df.empty:
        df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df


def generate_monthly_holdings():
    all_tickers = list(CANDIDATE_POOL.keys())
    np.random.seed(42)
    core = {"2368", "2383", "6223", "5274", "6274", "2308", "8299", "2330", "4552", "6669"}
    result = [core.copy()]
    for _ in range(N_MONTHS - 1):
        prev = result[-1].copy()
        out_cands = list(prev)
        in_cands  = [t for t in all_tickers if t not in prev]
        n_change  = np.random.randint(1, 3)
        for _ in range(min(n_change, len(out_cands), len(in_cands))):
            prev.discard(np.random.choice(out_cands))
            prev.add(np.random.choice(in_cands))
        result.append(prev)
    return result


if __name__ == "__main__":
    print("取得特徵快照（這可能需要幾分鐘）...")
    monthly_holdings = generate_monthly_holdings()

    all_dfs = []
    for i, holdings_set in enumerate(monthly_holdings):
        year  = START_YEAR
        month = START_MON + i
        label = period_label(year, month)   # 正確歸位，不會出現 2025-13
        print(f"\n=== 時期 {label} | 持股數: {len(holdings_set)} ===")
        df = get_features_snapshot(CANDIDATE_POOL, holdings_set)
        df["period"] = label
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    out = DATA_DIR / "holdings_history.csv"
    full_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n歷史持股資料已儲存: {out}")
    print(f"總樣本數: {len(full_df)}，共 {full_df['period'].nunique()} 個時期")
    print("時期列表:", sorted(full_df['period'].unique()))
