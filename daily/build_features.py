#!/usr/bin/env python3
"""
日頻特徵建構 - T+0 架構

輸入:  data/daily_raw.csv  (必須包含 close, adj_close 兩欄)
輸出:  data/daily_features.csv

設計原則:
  close     = 未還原原始成交價  -> 計算 fwd_ret_1d (真實報酬)
  adj_close = 還原權値價          -> 計算均線/動能/勢能特徵 (連續性準確)

所有特徵是 T-1 收盤後已知資訊，label 是 T+1 報酬
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IN_CSV   = DATA_DIR / "daily_raw.csv"
OUT_CSV  = DATA_DIR / "daily_features.csv"

UP_THRESH   =  0.02
DOWN_THRESH = -0.02


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["stock_id", "date"], inplace=True)

    # adj_close 必須存在
    if "adj_close" not in df.columns:
        raise ValueError(
            "daily_raw.csv 缺少 adj_close 欄位，"
            "請重新執行 python daily/fetch_daily.py"
        )

    records = []
    for sid, g in df.groupby("stock_id"):
        g  = g.copy().reset_index(drop=True)
        c  = g["close"].astype(float)      # 未還原: 用於報酬計算
        ac = g["adj_close"].astype(float)  # 還原:   用於特徵計算
        v  = g["volume"].astype(float)
        h  = g["high"].astype(float)  if "high" in g.columns else ac
        lo = g["low"].astype(float)   if "low"  in g.columns else ac

        # 動能特徵: 用 adj_close 進行滾動計算，避免除權息日出現跟空
        ma5_v  = v.rolling(5,  min_periods=3).mean()
        ma20_v = v.rolling(20, min_periods=10).mean()
        vz20   = (v - v.rolling(20, min_periods=10).mean()) / \
                 v.rolling(20, min_periods=10).std().replace(0, np.nan)

        ret1  = ac.pct_change(1)   # 昨日報酬 (adj)
        ret5  = ac.pct_change(5)
        hlpct = (h - lo) / ac.replace(0, np.nan)
        ma5_c  = ac.rolling(5,  min_periods=3).mean()
        ma20_c = ac.rolling(20, min_periods=10).mean()

        inst = g["inst_net"].astype(float) if "inst_net" in g.columns \
               else pd.Series(np.nan, index=g.index)
        inst_ratio = inst / ma20_v.replace(0, np.nan)

        if {"MarginPurchaseBuy", "MarginPurchaseSell"}.issubset(g.columns):
            margin_net = g["MarginPurchaseBuy"].astype(float) - \
                         g["MarginPurchaseSell"].astype(float)
            margin_chg = margin_net.rolling(5, min_periods=3).sum()
            margin_chg_pct = margin_chg / \
                g["MarginPurchaseBuy"].astype(float).rolling(20).mean().replace(0, np.nan)
        else:
            margin_chg_pct = pd.Series(np.nan, index=g.index)

        # === T+1 隣日報酬: 用未還原 close 計算 ===
        # c.shift(-1)/c - 1 = 明日未還原收盤 / 今日未還原收盤 - 1
        # = 真實市場報酬，不包含除權息跳升
        fwd1 = c.shift(-1) / c - 1

        # 全部特徵 shift(1): row[date=T] 的 X = T-1 的特徵, y = T+1 報酬
        feat = pd.DataFrame({
            "date":           g["date"],
            "stock_id":       sid,
            "close":          c.values,              # T 的未還原收盤，供回測展示用
            "vol_ratio_5d":   (v / ma5_v.replace(0, np.nan)).shift(1).values,
            "vol_ratio_20d":  (v / ma20_v.replace(0, np.nan)).shift(1).values,
            "vol_zscore_20d": vz20.shift(1).values,
            "ret_1d":         ret1.shift(1).values,
            "ret_5d":         ret5.shift(1).values,
            "high_low_pct":   hlpct.shift(1).values,
            "close_vs_ma5":   (ac / ma5_c.replace(0, np.nan) - 1).shift(1).values,
            "close_vs_ma20":  (ac / ma20_c.replace(0, np.nan) - 1).shift(1).values,
            "inst_net_ratio": inst_ratio.shift(1).values,
            "margin_chg_pct": margin_chg_pct.shift(1).values,
            "fwd_ret_1d":     fwd1.values,           # T+1 未還原報酬 (真實)
            "label_up1":      (fwd1 > UP_THRESH).astype(int).values,
            "label_down1":    (fwd1 < DOWN_THRESH).astype(int).values,
        })
        records.append(feat)

    out = pd.concat(records, ignore_index=True)
    out.dropna(subset=["ret_1d", "vol_ratio_20d", "fwd_ret_1d"], inplace=True)

    # 异常報酬輸出 (QC)
    p99 = out["fwd_ret_1d"].quantile(0.99)
    p1  = out["fwd_ret_1d"].quantile(0.01)
    n_outlier = ((out["fwd_ret_1d"] > 0.097) | (out["fwd_ret_1d"] < -0.1)).sum()
    print(f"fwd_ret_1d p1={p1*100:.2f}%  p99={p99*100:.2f}%  "
          f"超上下限筆數: {n_outlier} ({n_outlier/len(out)*100:.1f}%)")

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
