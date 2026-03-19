#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from utils.font_helper import setup_font

setup_font()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def describe_holdings(holdings):
    print("=== 產業分佈 ===")
    print(holdings["industry"].value_counts())
    print("\n=== 前10大持股 ===")
    top10 = holdings.sort_values("weight_pct", ascending=False).head(10)
    print(top10[["stock_id", "stock_name", "weight_pct", "industry"]].to_string(index=False))

def compare_features(in_etf, not_in_etf):
    cols = ["roe", "pe_ratio", "pb_ratio", "gross_margin",
            "operating_margin", "price_mom_3m", "price_mom_6m", "volatility_60d"]
    results = []
    for col in cols:
        a = in_etf[col].dropna()
        b = not_in_etf[col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(a, b)
        results.append({"feature": col,
                         "in_etf_mean": round(a.mean(), 4),
                         "not_in_etf_mean": round(b.mean(), 4),
                         "p_value": round(p_val, 4),
                         "significant": "*" if p_val < 0.05 else ""})
    print("\n=== 成分股 vs 非成分股特徵比較 ===")
    print(pd.DataFrame(results).to_string(index=False))

def plot_weight_distribution(holdings):
    industry_weight = holdings.groupby("industry")["weight_pct"].sum()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(industry_weight, labels=industry_weight.index, autopct="%1.1f%%")
    ax.set_title("00981A 產業配置")
    plt.tight_layout()
    out = DATA_DIR / "industry_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"圖表已儲存: {out}")

if __name__ == "__main__":
    holdings = pd.read_csv(DATA_DIR / "holdings_sample.csv")
    describe_holdings(holdings)
    plot_weight_distribution(holdings)
    features_path = DATA_DIR / "features.csv"
    if features_path.exists():
        features_df = pd.read_csv(features_path)
        in_etf_ids = set(holdings["stock_id"].astype(str))
        compare_features(
            features_df[features_df["ticker"].astype(str).isin(in_etf_ids)],
            features_df[~features_df["ticker"].astype(str).isin(in_etf_ids)]
        )
