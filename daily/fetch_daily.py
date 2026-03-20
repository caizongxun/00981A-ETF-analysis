#!/usr/bin/env python3
"""
日頻資料下載
  - OHLCV (yfinance, auto_adjust=False 未還原權値價)
  - 三大法人買賣超 (FinMind: TaiwanStockInstitutionalInvestorsBuySell)
  - 融資融券 (FinMind: TaiwanStockMarginPurchaseShortSale)

重要: auto_adjust=False 保留原始成交價，除權息日報酬不包含還原跳升
       -> fwd_ret_1d 就是真實的隨日奖利計算前报酬

輸出: data/daily_raw.csv
執行: python daily/fetch_daily.py
"""
import sys
import time
from pathlib import Path

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
    return df[[c for c in df.columns if c in keep]]


def main():
    client = get_client()

    print("確認 yfinance 後綴...")
    valid = {}
    for tk in UNIVERSE:
        r = get_valid_symbol(tk)
        if r:
            valid[tk] = r[0]
        time.sleep(0.05)
    print(f"有效宇宙: {len(valid)} 支")

    print("下載日頻 OHLCV (auto_adjust=False)...")
    all_price = []
    for tk, sym in valid.items():
        try:
            # auto_adjust=False: 保留原始成交價，除權息日報酬 = 真實報酬
            h = yf.download(sym, start=START_DATE, progress=False,
                            auto_adjust=False)
            if h.empty:
                continue
            # auto_adjust=False 時 Close = 未還原收盤價, Adj Close = 還原權値價
            # 我們用 Close 計算 fwd_ret_1d，用 Adj Close 計算均線/動能特徵
            close     = h["Close"]
            adj_close = h["Adj Close"]
            volume    = h["Volume"]
            high      = h["High"]
            low       = h["Low"]
            for s in [close, adj_close, volume, high, low]:
                if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            if isinstance(close,     pd.DataFrame): close     = close.iloc[:, 0]
            if isinstance(adj_close, pd.DataFrame): adj_close = adj_close.iloc[:, 0]
            if isinstance(volume,    pd.DataFrame): volume    = volume.iloc[:, 0]
            if isinstance(high,      pd.DataFrame): high      = high.iloc[:, 0]
            if isinstance(low,       pd.DataFrame): low       = low.iloc[:, 0]

            close.index = pd.to_datetime(close.index).tz_localize(None)
            tmp = pd.DataFrame({
                "date":      close.index,
                "stock_id":  tk,
                "close":     close.values,        # 未還原，用於計算 fwd_ret_1d
                "adj_close": adj_close.reindex(close.index).values,  # 還原，用於均線特徵
                "volume":    volume.reindex(close.index).values,
                "high":      high.reindex(close.index).values,
                "low":       low.reindex(close.index).values,
            })
            all_price.append(tmp)
        except Exception as e:
            print(f"  {sym} [ERR] {e}")
        time.sleep(0.05)

    price_df = pd.concat(all_price, ignore_index=True)
    price_df["date"] = pd.to_datetime(price_df["date"])
    print(f"OHLCV: {len(price_df):,} 筆")

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
