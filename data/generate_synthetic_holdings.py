#!/usr/bin/env python3
"""
合成歷史持股資料生成器
由於 00981A 僅上市不到一年，持股明細樣本僅有 1 筆。
本腳本模擬「如果每個月持股對應不同財報狀態」的多概持股快照資料集，
供 Walk-Forward 訓練使用。真實使用時請替換為從口袋證券/CMoney 手動下載的歷史資料。
"""
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

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

def get_features_snapshot(tickers_with_suffix: dict, label_set: set) -> pd.DataFrame:
    """取得當前特徵，和標籤（是否入選）"""
    records = []
    for tk, suffix in tickers_with_suffix.items():
        sym = f"{tk}{suffix}"
        try:
            t = yf.Ticker(sym)
            info = t.info
            hist = t.history(period="9mo")
            if hist.empty:
                continue
            closes = hist["Close"]
            mom3  = (closes.iloc[-1] / closes.iloc[-63]  - 1) if len(closes) >= 63  else np.nan
            mom6  = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
            vol60 = closes.pct_change().tail(60).std() * np.sqrt(252)
            records.append({
                "stock_id":         tk,
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


def generate_monthly_snapshots():
    """
    模擬每月 00981A 持股情境（以 2025-05 上市為基準，共 9 個快照）
    主要模擬：每期持股有微小變動（添加/排除股票）
    """
    all_tickers = list(CANDIDATE_POOL.keys())
    np.random.seed(42)

    # 初始持股池（已知 00981A 前10大）
    core_holdings = {"2368", "2383", "6223", "5274", "6274", "2308", "8299", "2330", "4552", "6669"}
    
    snapshots = []
    monthly_holdings = [core_holdings.copy()]
    
    # 模擬 9 個月的持股變化
    for i in range(8):
        prev = monthly_holdings[-1].copy()
        # 每期隨機添加 1-2 支新股 / 移除 1-2 支舊股
        out_candidates = list(prev)
        in_candidates  = [t for t in all_tickers if t not in prev]
        n_change = np.random.randint(1, 3)
        for _ in range(min(n_change, len(out_candidates), len(in_candidates))):
            prev.discard(np.random.choice(out_candidates))
            prev.add(np.random.choice(in_candidates))
        monthly_holdings.append(prev)
    
    return monthly_holdings


if __name__ == "__main__":
    print("取得目前特徵快照（這可能需要幾分鐘）...")
    monthly = generate_monthly_snapshots()
    
    all_dfs = []
    for month_idx, holdings_set in enumerate(monthly):
        period = f"2025-{(5 + month_idx):02d}"
        print(f"\n=== 時期 {period} | 持股數: {len(holdings_set)} ===")
        df = get_features_snapshot(CANDIDATE_POOL, holdings_set)
        df["period"] = period
        all_dfs.append(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    out = DATA_DIR / "holdings_history.csv"
    full_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n歷史持股資料已儲存: {out}")
    print(f"總樣本數: {len(full_df)}，共 {full_df['period'].nunique()} 個時期")
