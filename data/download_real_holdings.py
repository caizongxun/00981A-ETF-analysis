#!/usr/bin/env python3
"""
00981A 真實持股下載器

API 回傳格式：
  {"Title": ["日期","標的代號","標的名稱","權重(%)","持有數","單位"],
   "Data":  [["20260319","2330","台積電","8.66","4271000","股"], ...]}

限制：此 API 只提供最新一期持股，無歷史資料。
歷史資料需自行每期執行一次、手動補充，參考 docs/manual_download.md

執行： python data/download_real_holdings.py
"""
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import json
import time
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

COOKIE_FILE = ROOT / ".cookie"
API_URL = (
    "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
    "?action=getdtnodata&DtNo=59449513"
    "&ParamStr=AssignID%3D00981A%3B"
    "&FilterNo=0"
)

INDUSTRY_MAP = {
    "2330": "半導體", "4552": "通訊網路", "2308": "電子零組件", "6669": "電腦週邊",
    "2368": "電子零組件", "2383": "電子零組件", "2454": "半導體", "3008": "光電",
    "2317": "電腦週邊", "2382": "電腦週邊", "3711": "通訊網路", "2357": "電腦週邊",
    "2379": "半導體", "3691": "電子零組件", "8299": "半導體",
    "6274": "電子零組件", "6223": "半導體", "5274": "半導體",
    "3017": "電子零組件", "3653": "電子零組件", "2345": "通訊網路",
    "3037": "半導體", "3665": "電子零組件",
}


def fetch_current_holdings(cookie: str) -> list[dict]:
    """\u62ff取最新一期 00981A 持股，回傳 [{stock_id, stock_name, weight}, ...]"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.pocket.tw/etf/tw/00981A/fundholding",
        "Cookie": cookie,
    }
    r = requests.get(API_URL, headers=headers, timeout=20)
    r.raise_for_status()
    data = json.loads(r.text)
    rows = data.get("Data", [])
    # 欄位: [date, stock_id, name, weight%, shares, unit]
    result = []
    for row in rows:
        if len(row) < 3:
            continue
        result.append({
            "stock_id":   str(row[1]).strip(),
            "stock_name": str(row[2]).strip(),
            "weight":     float(row[3]) if len(row) > 3 else 0.0,
            "shares":     str(row[4]) if len(row) > 4 else "",
            "date":       str(row[0]),
        })
    return result


def get_suffix(stock_id: str) -> str:
    """\u4e0a櫃用 .TWO，上市用 .TW"""
    # 上櫃股一般為 4 位且開頭為 5xxx 6xxx
    # 更準確做法是查 TWSE vs TPEX，這裡用簡单規則
    ots = {"3691", "8299", "6274", "6223", "5274", "6446", "3661",
           "4552", "3037", "3665", "3653", "3017"}
    return ".TWO" if stock_id in ots else ".TW"


def get_features(holdings: list[dict]) -> pd.DataFrame:
    """\u7576期全候選池 = \u6240有持股股票"""
    holding_ids = {h["stock_id"] for h in holdings}
    records = []
    for h in holdings:
        tk     = h["stock_id"]
        suffix = get_suffix(tk)
        sym    = f"{tk}{suffix}"
        try:
            t      = yf.Ticker(sym)
            info   = t.info
            hist   = t.history(period="9mo")
            if hist.empty:
                print(f"  [SKIP] {sym} 無股價")
                continue
            closes = hist["Close"]
            mom3  = (closes.iloc[-1] / closes.iloc[-63]  - 1) if len(closes) >= 63  else np.nan
            mom6  = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
            vol60 = closes.pct_change().tail(60).std() * np.sqrt(252)
            records.append({
                "stock_id":         tk,
                "stock_name":       h["stock_name"],
                "industry":         INDUSTRY_MAP.get(tk, "不明"),
                "weight":           h["weight"],
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
                "in_etf":           1,  # 持股明細內都是入選
            })
            print(f"  [OK] {sym} weight={h['weight']}%")
        except Exception as e:
            print(f"  [WARN] {sym}: {e}")
    df = pd.DataFrame(records)
    if not df.empty:
        df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df


if __name__ == "__main__":
    if not COOKIE_FILE.exists():
        print("[ERROR] 請先執行: python data/input_cookie.py")
        exit(1)

    cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()
    today  = date.today().strftime("%Y-%m")

    print("下載 00981A 最新持股...")
    holdings = fetch_current_holdings(cookie)
    print(f"\n持股數: {len(holdings)} 支")
    for h in holdings:
        print(f"  {h['stock_id']} {h['stock_name']:8s} {h['weight']}%")

    # 儲存原始持股
    raw_path = DATA_DIR / "real_holdings_raw.csv"
    existing = pd.read_csv(raw_path, dtype={"stock_id": str}) if raw_path.exists() else pd.DataFrame()

    new_rows = pd.DataFrame([{
        "period":     today,
        "stock_id":   h["stock_id"],
        "stock_name": h["stock_name"],
        "weight":     h["weight"],
    } for h in holdings])

    # 移除同期舊資料，再合併
    if not existing.empty and today in existing["period"].values:
        existing = existing[existing["period"] != today]
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n原始持股已存: {raw_path}")
    print(f"目前共有 {combined['period'].nunique()} 期資料: {sorted(combined['period'].unique())}")

    # 取得特徵
    print(f"\n取得 {today} 特徵（這可能需要幾分鐘）...")
    feat_df = get_features(holdings)
    feat_df["period"] = today

    # 合併歷史特徵
    hist_path = DATA_DIR / "holdings_history.csv"
    existing_feat = pd.read_csv(hist_path, dtype={"stock_id": str}) if hist_path.exists() else pd.DataFrame()
    if not existing_feat.empty and today in existing_feat.get("period", pd.Series()).values:
        existing_feat = existing_feat[existing_feat["period"] != today]
    full = pd.concat([existing_feat, feat_df], ignore_index=True)
    full.to_csv(hist_path, index=False, encoding="utf-8-sig")

    print(f"\n完成！ holdings_history.csv 已存 ({len(full)} 筆, {full['period'].nunique()} 期)")
    print("\n請注意: 此 API 只有最新一期資料")
    print("建議: 每個月初執行一次此腳本，累積建立多期持股資料")
    print("       累積 3 期以上後再執行 walk_forward.py 才有意義")
