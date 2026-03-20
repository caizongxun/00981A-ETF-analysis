#!/usr/bin/env python3
"""
日頻特徵建構 - T+0 架構 v4

輸入:  data/daily_raw.csv
輸出:  data/daily_features.csv

特徵清單 (22 個, 全部 T-1 收盤後已知):
  量能類: vol_ratio_5d, vol_ratio_20d, vol_zscore_20d, obv_slope
  報酬動能: ret_1d, ret_5d, ret_20d, mom_acc
  價格結構: high_low_pct, close_vs_ma5, close_vs_ma20, close_vs_ma60,
               bb_pos, atr_pct
  技術指標: rsi_14
  法人融資: inst_net_ratio, inst_net_ratio_5d, margin_chg_pct, short_ratio
  成交金額: turnover_ratio
標籤: label_up1 (T+1 報酬 > 2%), label_down1 (T+1 報酬 < -2%)
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


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["stock_id", "date"], inplace=True)

    if "adj_close" not in df.columns:
        raise ValueError(
            "daily_raw.csv 缺少 adj_close 欄位，"
            "請重新執行 python daily/fetch_daily.py"
        )

    records = []
    for sid, g in df.groupby("stock_id"):
        g  = g.copy().reset_index(drop=True)
        c  = g["close"].astype(float)
        ac = g["adj_close"].astype(float)
        v  = g["volume"].astype(float)
        h  = g["high"].astype(float) if "high" in g.columns else ac
        lo = g["low"].astype(float)  if "low"  in g.columns else ac

        ma5_v  = v.rolling(5,  min_periods=3).mean()
        ma20_v = v.rolling(20, min_periods=10).mean()
        vz20   = (v - ma20_v) / v.rolling(20, min_periods=10).std().replace(0, np.nan)

        sign      = np.sign(ac.diff().fillna(0))
        obv       = (sign * v).cumsum()
        obv_slope = obv.diff(5) / ma20_v.replace(0, np.nan)

        ret1    = ac.pct_change(1)
        ret5    = ac.pct_change(5)
        ret20   = ac.pct_change(20)
        mom_acc = ret5 - ac.pct_change(10).shift(5)

        hlpct  = (h - lo) / ac.replace(0, np.nan)
        ma5_c  = ac.rolling(5,  min_periods=3).mean()
        ma20_c = ac.rolling(20, min_periods=10).mean()
        ma60_c = ac.rolling(60, min_periods=30).mean()

        bb_mid   = ma20_c
        bb_std   = ac.rolling(20, min_periods=10).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_pos   = (ac - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        tr = pd.concat([
            h - lo,
            (h - ac.shift(1)).abs(),
            (lo - ac.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14   = tr.rolling(14, min_periods=7).mean()
        atr_pct = atr14 / ac.replace(0, np.nan)

        rsi14 = rsi(ac, 14)

        inst = g["inst_net"].astype(float) if "inst_net" in g.columns \
               else pd.Series(np.nan, index=g.index)
        inst_ratio    = inst / ma20_v.replace(0, np.nan)
        inst_ratio_5d = inst.rolling(5, min_periods=3).sum() / ma20_v.replace(0, np.nan)

        if {"MarginPurchaseBuy", "MarginPurchaseSell"}.issubset(g.columns):
            mb  = g["MarginPurchaseBuy"].astype(float)
            ms  = g["MarginPurchaseSell"].astype(float)
            margin_net     = mb - ms
            margin_chg     = margin_net.rolling(5, min_periods=3).sum()
            margin_chg_pct = margin_chg / mb.rolling(20).mean().replace(0, np.nan)
        else:
            margin_chg_pct = pd.Series(np.nan, index=g.index)

        if {"ShortSaleBuy", "ShortSaleSell", "MarginPurchaseBuy"}.issubset(g.columns):
            ss_bal = g["ShortSaleSell"].astype(float).rolling(5, min_periods=3).sum()
            mg_bal = g["MarginPurchaseBuy"].astype(float).rolling(5, min_periods=3).sum()
            short_ratio = ss_bal / mg_bal.replace(0, np.nan)
        else:
            short_ratio = pd.Series(np.nan, index=g.index)

        turnover       = ac * v
        turnover_ma5   = turnover.rolling(5,  min_periods=3).mean()
        turnover_ma20  = turnover.rolling(20, min_periods=10).mean()
        turnover_ratio = turnover_ma5 / turnover_ma20.replace(0, np.nan)

        fwd1 = c.shift(-1) / c - 1

        def s(x): return x.shift(1).values

        feat = pd.DataFrame({
            "date":              g["date"],
            "stock_id":          sid,
            "close":             c.values,
            "vol_ratio_5d":      s(v / ma5_v.replace(0, np.nan)),
            "vol_ratio_20d":     s(v / ma20_v.replace(0, np.nan)),
            "vol_zscore_20d":    s(vz20),
            "obv_slope":         s(obv_slope),
            "ret_1d":            s(ret1),
            "ret_5d":            s(ret5),
            "ret_20d":           s(ret20),
            "mom_acc":           s(mom_acc),
            "high_low_pct":      s(hlpct),
            "close_vs_ma5":      s(ac / ma5_c.replace(0, np.nan) - 1),
            "close_vs_ma20":     s(ac / ma20_c.replace(0, np.nan) - 1),
            "close_vs_ma60":     s(ac / ma60_c.replace(0, np.nan) - 1),
            "bb_pos":            s(bb_pos),
            "atr_pct":           s(atr_pct),
            "rsi_14":            s(rsi14),
            "inst_net_ratio":    s(inst_ratio),
            "inst_net_ratio_5d": s(inst_ratio_5d),
            "margin_chg_pct":    s(margin_chg_pct),
            "short_ratio":       s(short_ratio),
            "turnover_ratio":    s(turnover_ratio),
            "fwd_ret_1d":        fwd1.values,
            "label_up1":         (fwd1 > UP_THRESH).astype(int).values,
            "label_down1":       (fwd1 < DOWN_THRESH).astype(int).values,
        })
        records.append(feat)

    out = pd.concat(records, ignore_index=True)
    out.dropna(subset=["ret_1d", "vol_ratio_20d", "fwd_ret_1d"], inplace=True)

    p99   = out["fwd_ret_1d"].quantile(0.99)
    p1    = out["fwd_ret_1d"].quantile(0.01)
    n_out = ((out["fwd_ret_1d"] > 0.097) | (out["fwd_ret_1d"] < -0.1)).sum()
    nan_rate = out.drop(columns=["date","stock_id","close",
                                  "fwd_ret_1d","label_up1","label_down1"]).isna().mean()
    print(f"fwd_ret_1d p1={p1*100:.2f}%  p99={p99*100:.2f}%  "
          f"超上下限筆數: {n_out} ({n_out/len(out)*100:.1f}%)")
    high_nan = nan_rate[nan_rate > 0.3]
    if len(high_nan):
        print("高 NaN 特徵 (>30%):")
        for col, r in high_nan.items():
            print(f"  {col}: {r*100:.0f}%")

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
