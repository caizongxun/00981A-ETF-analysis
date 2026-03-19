#!/usr/bin/env python3
"""
00981A 真實持股資料下載器

資料來源優先順序：
  1. 投信投顧公會 (SITCA) - 公開每月投資組合新資訊中有前10大持股
  2. 統一投信網站 - ETF 每日公布持股明細 PDF
  3. Selenium 爬號口袋證券動態頁面（需安裝 chromedirver）

執行方式：
  python data/download_real_holdings.py

輸出：
  data/real_holdings_raw.csv   - 原始資料
  data/holdings_history.csv    - 合並特徵後，直接供 walk_forward.py 使用
"""
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import json
import time
import re
from pathlib import Path
from datetime import datetime

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

FEATURE_COLS = [
    "roe", "pe_ratio", "pb_ratio", "gross_margin",
    "operating_margin", "debt_to_equity",
    "price_mom_3m", "price_mom_6m", "volatility_60d", "market_cap_rank"
]


# ===========================================================
# 來源 1：投信投顧公會 SITCA
# 公開 API：每月下旬公布其內ETF持股明細 JSON
# ===========================================================
def fetch_sitca_holdings(year_month: str) -> set:
    """
    year_month: 'YYYY-MM'
    回傳該月 00981A 持股的股票代號 set
    """
    year, month = year_month.split("-")
    # SITCA ETF 資料公開 API—返回指定復剀日期的 ETF 影指資料
    url = (
        f"https://www.sitca.org.tw/ROC/SITCA_ETF/etf_statement.aspx"
        f"?Year={year}&Month={month}&Page=1&PageSize=100"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Referer": "https://www.sitca.org.tw/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        # SITCA 返回 HTML，需用 pandas 解析表格
        tables = pd.read_html(r.text, encoding="utf-8")
        for t in tables:
            # 找包含 00981A 的表
            t_str = t.to_string()
            if "00981A" in t_str or "00981" in t_str:
                print(f"  [SITCA] {year_month} 找到表格，行數={len(t)}")
                # 提取包含 4~6 位數字的股票代號
                codes = set()
                for cell in t.values.flatten():
                    if isinstance(cell, str):
                        m = re.findall(r'\b(\d{4,6})\b', cell)
                        codes.update(m)
                return codes
    except Exception as e:
        print(f"  [SITCA] {year_month} 失敗: {e}")
    return set()


# ===========================================================
# 來源 2：統一投信官方 API
# 00981A 每日公布为「主動式 ETF」，每周公布持股明細 JSON
# ===========================================================
def fetch_upamc_holdings(year_month: str) -> set:
    """
    統一投信公開的 ETF 持股 API
    year_month: 'YYYY-MM'
    """
    year, month = year_month.split("-")
    # 統一投信 ETF 持股 JSON API 端點
    base_urls = [
        f"https://www.upamc.com.tw/api/etf/holding?code=00981A&year={year}&month={month}",
        f"https://api.upamc.com.tw/ETF/GetETFHolding?FundCode=00981A&Date={year}{month}01",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }
    for url in base_urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                print(f"  [UPAMC] {year_month} 回傳: {str(data)[:200]}")
                # 提取股票代號
                codes = set()
                def extract(obj):
                    if isinstance(obj, dict):
                        for v in obj.values(): extract(v)
                    elif isinstance(obj, list):
                        for i in obj: extract(i)
                    elif isinstance(obj, str):
                        m = re.findall(r'\b(\d{4})\b', obj)
                        codes.update(m)
                extract(data)
                if codes:
                    return codes
        except Exception as e:
            print(f"  [UPAMC] {url[:60]} 失敗: {e}")
    return set()


# ===========================================================
# 來源 3：口袋證券 pocket.tw 靜態資料備用 API
# ===========================================================
def fetch_pocket_holdings(year_month: str) -> set:
    """
    回傳該月最新公開持股資料
    year_month: 'YYYY-MM'
    """
    # pocket.tw 內部 API - 可能選對資料日期
    date_str = f"{year_month}-01"
    urls = [
        f"https://api.pocket.tw/etf/tw/00981A/fundholding?date={date_str}",
        f"https://www.pocket.tw/api/etf/tw/00981A/holding?date={date_str}",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Referer": "https://www.pocket.tw/",
    }
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200 and r.text.strip().startswith("["):
                rows = r.json()
                codes = {str(row.get("stock_id", "") or row.get("stockNo", ""))
                         for row in rows if isinstance(row, dict)}
                codes = {c for c in codes if re.match(r'^\d{4,6}$', c)}
                if codes:
                    print(f"  [pocket] {year_month} 找到 {len(codes)} 支: {codes}")
                    return codes
        except Exception as e:
            print(f"  [pocket] {year_month} {e}")
    return set()


# ===========================================================
# 整合：依序試各來源
# ===========================================================
def get_real_holdings(year_month: str) -> set:
    """
    依序試三個來源，回傳該月實際持股股票代號 set
    若全部失敗，回傳 None
    """
    print(f"\n查詢 {year_month} 持股...")

    # 來源 1：口袋
    codes = fetch_pocket_holdings(year_month)
    if codes:
        # 過濾為候選池中的股票
        filtered = codes & set(CANDIDATE_POOL.keys())
        if filtered:
            print(f"  ==> 口袋 成功: {sorted(filtered)}")
            return filtered

    # 來源 2：統一投信
    codes = fetch_upamc_holdings(year_month)
    if codes:
        filtered = codes & set(CANDIDATE_POOL.keys())
        if filtered:
            print(f"  ==> UPAMC 成功: {sorted(filtered)}")
            return filtered

    # 來源 3： SITCA
    codes = fetch_sitca_holdings(year_month)
    if codes:
        filtered = codes & set(CANDIDATE_POOL.keys())
        if filtered:
            print(f"  ==> SITCA 成功: {sorted(filtered)}")
            return filtered

    print(f"  [WARN] {year_month} 所有來源均失敗")
    return None


# ===========================================================
# 特徵擷取
# ===========================================================
def get_features_snapshot(pool: dict, label_set: set) -> pd.DataFrame:
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
        except Exception as e:
            print(f"  [WARN] {sym}: {e}")
    df = pd.DataFrame(records)
    if not df.empty:
        df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df


# ===========================================================
# 主流程
# ===========================================================
if __name__ == "__main__":
    # 00981A 上市日 2025-05-20，人工產生到 2026-01
    months = [
        "2025-06", "2025-07", "2025-08", "2025-09",
        "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"
    ]

    raw_records = []  # 純持股資料
    all_dfs     = []  # 含特徵的完整資料
    failed      = []  # 來源全失敗的月份

    for ym in months:
        holdings = get_real_holdings(ym)
        time.sleep(1)  # 防止被封溺

        if holdings is None:
            failed.append(ym)
            # 用上一期持股代替（如果有）
            if all_dfs:
                prev_df = all_dfs[-1].copy()
                prev_df["period"] = ym
                all_dfs.append(prev_df)
                print(f"  [FALLBACK] {ym} 使用上一期持股")
            continue

        # 記錄原始持股
        for tk in holdings:
            raw_records.append({"period": ym, "stock_id": tk,
                                "stock_name": STOCK_NAMES.get(tk, tk)})

        # 取得特徵
        print(f"  取得 {ym} 特徵中...")
        feat_df = get_features_snapshot(CANDIDATE_POOL, holdings)
        if not feat_df.empty:
            feat_df["period"] = ym
            all_dfs.append(feat_df)
        time.sleep(2)

    # --- 儲存 ---
    raw_path = DATA_DIR / "real_holdings_raw.csv"
    pd.DataFrame(raw_records).to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n原始持股資料: {raw_path}  ({len(raw_records)} 筆)")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        hist_path = DATA_DIR / "holdings_history.csv"
        full_df.to_csv(hist_path, index=False, encoding="utf-8-sig")
        print(f"含特徵持股資料: {hist_path}  ({len(full_df)} 筆, {full_df['period'].nunique()} 期)")
    else:
        print("[ERROR] 沒有任何成功資料，請改用手動下載模式")

    if failed:
        print(f"\n失敗的月份: {failed}")
        print("請參考 docs/manual_download.md 手動補充")

    print("\n完成! 接下來執行: python backtest/walk_forward.py")
