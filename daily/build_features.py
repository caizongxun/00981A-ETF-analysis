#!/usr/bin/env python3
"""
日頻異常特徵建構

輸入:  data/daily_raw.csv
輸出:  data/daily_features.csv

特徵設計 (per stock per day):
  量能異常:
    vol_ratio_5d    成交量 / 5日均量
    vol_ratio_20d   成交量 / 20日均量
    vol_zscore_20d  量能 Z-score (20日)
  價格型態:
    ret_1d          當日報酬
    ret_5d          5日報酬
    high_low_pct    (high-low)/close  當日振幅
    close_vs_ma5    close / ma5 - 1
    close_vs_ma20   close / ma20 - 1
  三大法人:
    inst_net_ratio  inst_net / avg_volume_20d (標準化)
  融資:
    margin_chg_pct  融資餘額變化率 (5日)
  目標:
    label_up5       未來5日報酬 > 3%  -> 1 (異常上漲訊號)
    label_down5     未來5日報酬 < -3% -> 1 (異常下跌訊號)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IN_CSV   = DATA_DIR / "daily_raw.csv"
OUT_CSV  = DATA_DIR / "daily_features.csv"

UP_THRESH   =  0.03
DOWN_THRESH = -0.03


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["stock_id", "date"], inplace=True)

    records = []
    for sid, g in df.groupby("stock_id"):
        g = g.copy().reset_index(drop=True)
        c = g["close"].astype(float)
        v = g["volume"].astype(float)
        h = g["high"].astype(float)  if "high"   in g else c
        lo= g["low"].astype(float)   if "low"    in g else c

        # 量能
        ma5_v  = v.rolling(5,  min_periods=3).mean()
        ma20_v = v.rolling(20, min_periods=10).mean()
        vz20   = (v - v.rolling(20, min_periods=10).mean()) / \
                 (v.rolling(20, min_periods=10).std().replace(0, np.nan))

        # 價格
        ret1  = c.pct_change(1)
        ret5  = c.pct_change(5)
        hlpct = (h - lo) / c.replace(0, np.nan)
        ma5_c  = c.rolling(5,  min_periods=3).mean()
        ma20_c = c.rolling(20, min_periods=10).mean()

        # 三大法人
        inst = g["inst_net"].astype(float) if "inst_net" in g else pd.Series(np.nan, index=g.index)
        inst_ratio = inst / ma20_v.replace(0, np.nan)

        # 融資 (MarginPurchaseBuy - MarginPurchaseSell 當作淨增)
        if {"MarginPurchaseBuy", "MarginPurchaseSell"}.issubset(g.columns):
            margin_net = g["MarginPurchaseBuy"].astype(float) - g["MarginPurchaseSell"].astype(float)
            margin_chg = margin_net.rolling(5, min_periods=3).sum()
            margin_chg_pct = margin_chg / (g["MarginPurchaseBuy"].astype(float).rolling(20).mean().replace(0, np.nan))
        else:
            margin_chg_pct = pd.Series(np.nan, index=g.index)

        # 未來5日報酬 (label)
        fwd5 = c.shift(-5) / c - 1

        feat = pd.DataFrame({
            "date":            g["date"],
            "stock_id":        sid,
            "close":           c.values,
            "vol_ratio_5d":    (v / ma5_v.replace(0, np.nan)).values,
            "vol_ratio_20d":   (v / ma20_v.replace(0, np.nan)).values,
            "vol_zscore_20d":  vz20.values,
            "ret_1d":          ret1.values,
            "ret_5d":          ret5.values,
            "high_low_pct":    hlpct.values,
            "close_vs_ma5":    (c / ma5_c.replace(0, np.nan) - 1).values,
            "close_vs_ma20":   (c / ma20_c.replace(0, np.nan) - 1).values,
            "inst_net_ratio":  inst_ratio.values,
            "margin_chg_pct": margin_chg_pct.values,
            "fwd_ret_5d":      fwd5.values,
            "label_up5":       (fwd5 > UP_THRESH).astype(int).values,
            "label_down5":     (fwd5 < DOWN_THRESH).astype(int).values,
        })
        records.append(feat)

    out = pd.concat(records, ignore_index=True)
    out.dropna(subset=["ret_1d", "vol_ratio_20d"], inplace=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"完成: {OUT_CSV} ({len(out):,} 筆)")
    print(f"  label_up5=1: {out['label_up5'].sum():,} ({out['label_up5'].mean()*100:.1f}%)")
    print(f"  label_down5=1: {out['label_down5'].sum():,} ({out['label_down5'].mean()*100:.1f}%)")
    return out


if __name__ == "__main__":
    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV)
    build(df)
    print("接下來執行: python daily/train_model.py")
