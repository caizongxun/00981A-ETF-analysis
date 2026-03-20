#!/usr/bin/env python3
"""
日頻特徵建構 - T+0 架構

輸入:  data/daily_raw.csv
輸出:  data/daily_features.csv

設計原則:
  所有特徵都是 T-1 (前一交易日收盤後已知資訊)
  標籤是 T+1 (隣日報酬) -> 可直接用昨日收盤資訊選今日股

特徵 (T-1 收盤後已知):
  vol_ratio_5d    昨日專量 / 5日均量
  vol_ratio_20d   昨日專量 / 20日均量
  vol_zscore_20d  昨日量能 Z-score
  ret_1d          昨日報酬
  ret_5d          過去5日報酬
  high_low_pct    昨日振幅
  close_vs_ma5    昨日收盤 vs MA5
  close_vs_ma20   昨日收盤 vs MA20
  inst_net_ratio  昨日三大法人淨買 / 均量
  margin_chg_pct  融資近5日變化

標籤 (T+1 隣日報酬):
  label_up1       隣日報酬 > 2%  -> 1
  label_down1     隣日報酬 < -2% -> 1
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IN_CSV   = DATA_DIR / "daily_raw.csv"
OUT_CSV  = DATA_DIR / "daily_features.csv"

UP_THRESH   =  0.02   # 隣日漲 > 2%
DOWN_THRESH = -0.02   # 隣日跌 > 2%


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["stock_id", "date"], inplace=True)

    records = []
    for sid, g in df.groupby("stock_id"):
        g = g.copy().reset_index(drop=True)
        c  = g["close"].astype(float)
        v  = g["volume"].astype(float)
        h  = g["high"].astype(float) if "high" in g else c
        lo = g["low"].astype(float)  if "low"  in g else c

        # 量能 (T-1 已知)
        ma5_v  = v.rolling(5,  min_periods=3).mean()
        ma20_v = v.rolling(20, min_periods=10).mean()
        vz20   = (v - v.rolling(20, min_periods=10).mean()) / \
                 v.rolling(20, min_periods=10).std().replace(0, np.nan)

        # 價格 (T-1 已知)
        ret1   = c.pct_change(1)          # 昨日報酬
        ret5   = c.pct_change(5)          # 過去5日報酬
        hlpct  = (h - lo) / c.replace(0, np.nan)
        ma5_c  = c.rolling(5,  min_periods=3).mean()
        ma20_c = c.rolling(20, min_periods=10).mean()

        # 三大法人 (T-1 已知)
        inst = g["inst_net"].astype(float) if "inst_net" in g \
               else pd.Series(np.nan, index=g.index)
        inst_ratio = inst / ma20_v.replace(0, np.nan)

        # 融資 (T-1 已知)
        if {"MarginPurchaseBuy", "MarginPurchaseSell"}.issubset(g.columns):
            margin_net = g["MarginPurchaseBuy"].astype(float) - \
                         g["MarginPurchaseSell"].astype(float)
            margin_chg = margin_net.rolling(5, min_periods=3).sum()
            margin_chg_pct = margin_chg / \
                             g["MarginPurchaseBuy"].astype(float).rolling(20).mean().replace(0, np.nan)
        else:
            margin_chg_pct = pd.Series(np.nan, index=g.index)

        # === T+1 隣日報酬 (label) ===
        # shift(-1): 今天的 label = 明天的收盤 / 今天收盤 - 1
        fwd1 = c.shift(-1) / c - 1

        # 全部特徵再 shift(1)，確保預測時用的都是前一日的數據
        # 意思: row[date=T] 的 X 就是 T-1 的特徵， y 就是 T 的報酬
        feat = pd.DataFrame({
            "date":           g["date"],
            "stock_id":       sid,
            "close":          c.shift(1).values,            # T-1 收盤 (參考用)
            "vol_ratio_5d":   (v / ma5_v.replace(0, np.nan)).shift(1).values,
            "vol_ratio_20d":  (v / ma20_v.replace(0, np.nan)).shift(1).values,
            "vol_zscore_20d": vz20.shift(1).values,
            "ret_1d":         ret1.shift(1).values,
            "ret_5d":         ret5.shift(1).values,
            "high_low_pct":   hlpct.shift(1).values,
            "close_vs_ma5":   (c / ma5_c.replace(0, np.nan) - 1).shift(1).values,
            "close_vs_ma20":  (c / ma20_c.replace(0, np.nan) - 1).shift(1).values,
            "inst_net_ratio": inst_ratio.shift(1).values,
            "margin_chg_pct": margin_chg_pct.shift(1).values,
            "fwd_ret_1d":     fwd1.values,
            "label_up1":      (fwd1 > UP_THRESH).astype(int).values,
            "label_down1":    (fwd1 < DOWN_THRESH).astype(int).values,
        })
        records.append(feat)

    out = pd.concat(records, ignore_index=True)
    # 拿掉缺少基本特徵的 row
    out.dropna(subset=["ret_1d", "vol_ratio_20d", "fwd_ret_1d"], inplace=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"完成: {OUT_CSV} ({len(out):,} 筆)")
    print(f"  label_up1=1  : {out['label_up1'].sum():,}  ({out['label_up1'].mean()*100:.1f}%)")
    print(f"  label_down1=1: {out['label_down1'].sum():,}  ({out['label_down1'].mean()*100:.1f}%)")
    return out


if __name__ == "__main__":
    print(f"讀取 {IN_CSV} ...")
    df = pd.read_csv(IN_CSV)
    build(df)
    print("接下來執行: python daily/train_model.py")
