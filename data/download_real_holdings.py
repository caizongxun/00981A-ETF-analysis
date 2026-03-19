#!/usr/bin/env python3
"""
00981A 真實持股下載器 - 使用 pocket.tw 內部 API

執行前需要設定 COOKIE：
  1. 瀏覽器開啟 https://www.pocket.tw/etf/tw/00981A/fundholding
  2. F12 > Application > Cookies > www.pocket.tw
  3. 將整行 cookie 字串複製到下方 COOKIE 變數，或存為 .env 檔

執行方式：
  python data/download_real_holdings.py
  python data/download_real_holdings.py --cookie "your_cookie_string"

輸出：
  data/real_holdings_raw.csv
  data/holdings_history.csv  (合併特徵後)
"""
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import json
import time
import re
import argparse
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── 請先從瀏覽器複製 cookie 貴入此處 ──
COOKIE = ""  # 例："ASP.NET_SessionId=xxx; ..."

# API 設定
BASE   = "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"

# DtNo 59449513 / M722 = 00981A 每期持股明細
# DtNo 61495191 / M066 = 亀山股持股信息（備用）
DTNO_HOLDING  = "59449513"
DTNO_DETAIL   = "61495191"
ASSIGN_00981A = "00981A"
ASSIGN_DETAIL = "98642180"

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


def make_headers(cookie: str) -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/122 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.pocket.tw/etf/tw/00981A/fundholding",
        "Cookie": cookie,
    }


def build_url(dtno: str, assign_id: str, period: int = 0,
              dt_range: int = 1, dt_order: int = 1,
              major_table: str = "M722") -> str:
    """
    period: 0=最新, 1=上期, 2=上上期 ...
    dt_range: 1=每期持股明細
    """
    param = (
        f"AssignID%3D{assign_id}%3B"
        f"MTPeriod%3D{period}%3B"
        f"DTMode%3D0%3B"
        f"DTRange%3D{dt_range}%3B"
        f"DTOrder%3D{dt_order}%3B"
        f"MajorTable%3D{major_table}%3B"
    )
    return (
        f"{BASE}?action=getdtnodata"
        f"&DtNo={dtno}"
        f"&ParamStr={param}"
        f"&FilterNo=0"
    )


def fetch_one_period(period_offset: int, cookie: str) -> list[dict]:
    """
    period_offset: 0=最新期, 1=上一期, ...
    回傳 [{stock_id, stock_name, weight, ...}, ...]
    """
    url = build_url(
        dtno=DTNO_HOLDING,
        assign_id=ASSIGN_00981A,
        period=period_offset,
        major_table="M722"
    )
    try:
        r = requests.get(url, headers=make_headers(cookie), timeout=20)
        if r.status_code != 200:
            print(f"  [HTTP {r.status_code}] offset={period_offset}")
            return []

        raw = r.text.strip()
        # pocket.tw 可能回傳純文字或 JSON
        if raw.startswith("[") or raw.startswith("{"):
            data = json.loads(raw)
        else:
            # 嘗試從 HTML 中提取股票代號
            codes = re.findall(r'\b(\d{4,6})\b', raw)
            return [{"stock_id": c} for c in set(codes)]

        # JSON 格式對應：列表 or {data: [...]}
        rows = data if isinstance(data, list) else data.get("data", [])
        result = []
        for row in rows:
            # 嘗試常見欄位名稱
            sid = (
                row.get("StockID") or row.get("stock_id") or
                row.get("Code")    or row.get("code")     or
                row.get("c1")      or ""
            )
            sname = (
                row.get("StockName") or row.get("stock_name") or
                row.get("Name")      or row.get("name")       or
                row.get("c2")        or ""
            )
            weight = (
                row.get("Weight") or row.get("weight") or
                row.get("Ratio")  or row.get("ratio")  or
                row.get("c3")     or 0
            )
            sid = str(sid).strip()
            if re.match(r'^\d{4,6}$', sid):
                result.append({
                    "stock_id":   sid,
                    "stock_name": str(sname).strip(),
                    "weight":     weight,
                })
        return result

    except Exception as e:
        print(f"  [ERR] offset={period_offset}: {e}")
        return []


def probe_periods(cookie: str, max_periods: int = 12) -> dict[int, list]:
    """
    依序試 period_offset 0,1,2,...
    回傳 {offset: [rows...]}
    """
    result = {}
    for offset in range(max_periods):
        print(f"  查詢 offset={offset}...", end=" ", flush=True)
        rows = fetch_one_period(offset, cookie)
        if not rows:
            print("無資料，停止")
            break
        stocks = [r["stock_id"] for r in rows if r.get("stock_id")]
        print(f"找到 {len(rows)} 筆 | 候選池交集: {sorted(set(stocks) & set(CANDIDATE_POOL))}")
        result[offset] = rows
        time.sleep(0.8)
    return result


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


# ── offset -> 月份標籤轉換 ──
def offset_to_period(offset: int) -> str:
    """
    offset=0 = 最新期（實際月份需從 API 回傳內容判斷，這裡先用簡单逻輯）
    如果 API 回傳中有日期欄位可更準確判斷
    """
    from datetime import date
    from dateutil.relativedelta import relativedelta
    base = date.today().replace(day=1)
    d    = base - relativedelta(months=offset)
    return d.strftime("%Y-%m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cookie", default="",
                        help="pocket.tw 登入 Cookie 字串")
    parser.add_argument("--max-periods", type=int, default=10,
                        help="最多查詢期數（預設 10）")
    args = parser.parse_args()

    cookie = args.cookie or COOKIE
    if not cookie:
        print("""
[ERROR] 需要 pocket.tw 登入 Cookie！

取得步驟：
  1. 瀏覽器登入 https://www.pocket.tw
  2. 開啟 F12 > Application > Cookies > www.pocket.tw
  3. 將所有 cookie 整行複製
  4. 執行：
     python data/download_real_holdings.py --cookie "ASP.NET_SessionId=xxx; ..."
""")
        exit(1)

    print(f"開始查詢 00981A 持股，最多 {args.max_periods} 期...\n")
    period_data = probe_periods(cookie, max_periods=args.max_periods)

    if not period_data:
        print("\n[FAIL] 沒有取得任何資料，請檢查 Cookie 是否正確")
        exit(1)

    # 儲存原始持股資料
    raw_rows = []
    for offset, rows in period_data.items():
        period = offset_to_period(offset)
        for r in rows:
            raw_rows.append({
                "period":     period,
                "stock_id":   r.get("stock_id", ""),
                "stock_name": r.get("stock_name", ""),
                "weight":     r.get("weight", ""),
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_path = DATA_DIR / "real_holdings_raw.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n原始持股已存: {raw_path} ({len(raw_df)} 筆)")
    print(raw_df.groupby("period")["stock_id"].apply(list).to_string())

    # 合併特徵
    print("\n取得 yfinance 特徵（這可能需要幾分鐘）...")
    all_dfs = []
    for offset, rows in period_data.items():
        period  = offset_to_period(offset)
        holdings_set = {str(r["stock_id"]) for r in rows
                        if re.match(r'^\d{4,6}$', str(r.get("stock_id","")))}
        holdings_set &= set(CANDIDATE_POOL.keys())
        if not holdings_set:
            print(f"  [{period}] 候選池交集為空，跳過")
            continue
        print(f"  [{period}] 持股: {sorted(holdings_set)}")
        feat_df = get_features(CANDIDATE_POOL, holdings_set)
        if not feat_df.empty:
            feat_df["period"] = period
            all_dfs.append(feat_df)
        time.sleep(1)

    if all_dfs:
        out = DATA_DIR / "holdings_history.csv"
        full = pd.concat(all_dfs, ignore_index=True)
        full.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n完成! holdings_history.csv 已存: {out} ({len(full)} 筆)")
        print("接下來執行: python backtest/walk_forward.py")
    else:
        print("[ERROR] 沒有任何候選池交集資料")
