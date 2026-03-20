#!/usr/bin/env python3
"""
日頻資料下載
  - OHLCV (yfinance)
  - 三大法人買賣超 (FinMind: TaiwanStockInstitutionalInvestorsBuySell)
  - 融資融券 (FinMind: TaiwanStockMarginPurchaseShortSale)

輸出: data/daily_raw.csv
執行: python daily/fetch_daily.py
"""
import sys
import time
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from finmind_client import get_client
from fetch_history import UNIVERSE, get_valid_symbol

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_CSV  = DATA_DIR / "daily_raw.csv"

START_DATE = "2023-01-01"


def fetch_institutional(stock_id: str, client) -> pd.DataFrame:
    rows = client.get("TaiwanStockInstitutionalInvestorsBuySell", stock_id, START_DATE)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # 聚合三大法人淨買超
    net_cols = [c for c in df.columns if "buy" in c.lower() or "sell" in c.lower()]
    # FinMind 欄位: buy, sell (各 name 有不同 row)
    if {"buy", "sell", "name"}.issubset(df.columns):
        df["net"] = df["buy"].astype(float) - df["sell"].astype(float)
        agg = df.groupby("date")["net"].sum().reset_index()
        agg.rename(columns={"net": "inst_net"}, inplace=True)
        return agg
    return pd.DataFrame()


def fetch_margin(stock_id: str, client) -> pd.DataFrame:
    rows = client.get("TaiwanStockMarginPurchaseShortSale", stock_id, START_DATE)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    keep = {"date", "MarginPurchaseBuy", "MarginPurchaseSell",
            "ShortSaleBuy", "ShortSaleSell"}
    df = df[[c for c in df.columns if c in keep]]
    return df


def main():
    client = get_client()

    # 取得有效 symbol
    print("確認 yfinance 後綴...")
    valid = {}
    for tk in UNIVERSE:
        r = get_valid_symbol(tk)
        if r:
            valid[tk] = r[0]
        time.sleep(0.05)
    print(f"有效宇宙: {len(valid)} 支")

    # 下載 OHLCV
    print("下載日頻 OHLCV...")
    all_price = []
    for tk, sym in valid.items():
        try:
            h = yf.download(sym, start=START_DATE, progress=False, auto_adjust=True)
            if h.empty:
                continue
            close  = h["Close"] if "Close" in h else h.iloc[:, 0]
            volume = h["Volume"] if "Volume" in h else pd.Series(dtype=float)
            high   = h["High"]   if "High"  in h else pd.Series(dtype=float)
            low    = h["Low"]    if "Low"   in h else pd.Series(dtype=float)
            if isinstance(close, pd.DataFrame): close  = close.iloc[:, 0]
            if isinstance(volume,pd.DataFrame): volume = volume.iloc[:, 0]
            if isinstance(high,  pd.DataFrame): high   = high.iloc[:, 0]
            if isinstance(low,   pd.DataFrame): low    = low.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None)
            tmp = pd.DataFrame({
                "date": close.index,
                "stock_id": tk,
                "close":  close.values,
                "volume": volume.reindex(close.index).values,
                "high":   high.reindex(close.index).values   if not high.empty else np.nan,
                "low":    low.reindex(close.index).values    if not low.empty  else np.nan,
            })
            all_price.append(tmp)
        except Exception as e:
            print(f"  {sym} [ERR] {e}")
        time.sleep(0.05)

    price_df = pd.concat(all_price, ignore_index=True)
    price_df["date"] = pd.to_datetime(price_df["date"])
    print(f"OHLCV: {len(price_df):,} 筆")

    # 下載三大法人 + 融資
    print("下載三大法人 + 融資 (FinMind)...")
    inst_list, margin_list = [], []
    for i, tk in enumerate(valid.keys()):
        inst   = fetch_institutional(tk, client)
        margin = fetch_margin(tk, client)
        if not inst.empty:
            inst["stock_id"] = tk
            inst_list.append(inst)
        if not margin.empty:
            margin["stock_id"] = tk
            margin_list.append(margin)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(valid)}]")
        time.sleep(0.1)

    inst_df   = pd.concat(inst_list,   ignore_index=True) if inst_list   else pd.DataFrame()
    margin_df = pd.concat(margin_list, ignore_index=True) if margin_list else pd.DataFrame()
    print(f"三大法人: {len(inst_df):,} 筆, 融資: {len(margin_df):,} 筆")

    # 合併
    df = price_df.copy()
    if not inst_df.empty:
        inst_df["date"] = pd.to_datetime(inst_df["date"])
        df = df.merge(inst_df, on=["date", "stock_id"], how="left")
    if not margin_df.empty:
        margin_df["date"] = pd.to_datetime(margin_df["date"])
        df = df.merge(margin_df, on=["date", "stock_id"], how="left")

    df.sort_values(["stock_id", "date"], inplace=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"完成: {OUT_CSV} ({len(df):,} 筆)")
    print("接下來執行: python daily/build_features.py")


if __name__ == "__main__":
    main()
