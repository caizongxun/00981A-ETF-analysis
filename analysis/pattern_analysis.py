#!/usr/bin/env python3
"""
選股邏輯分析：比較 00981A 成分股 vs 全市場，找出統計顯著差異
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_data(features_path="../data/features.csv",
              holdings_path="../data/holdings_sample.csv") -> tuple:
    features = pd.read_csv(features_path)
    holdings = pd.read_csv(holdings_path)
    return features, holdings

def describe_holdings(holdings: pd.DataFrame):
    """
    產業分佈、市值分佈分析
    """
    print("=== 產業分佈 ===")
    print(holdings["industry"].value_counts())
    print("\n=== 前10大持股 ===")
    top10 = holdings.sort_values("weight_pct", ascending=False).head(10)
    print(top10[["stock_id", "stock_name", "weight_pct", "industry"]].to_string(index=False))

def compare_features(in_etf: pd.DataFrame, not_in_etf: pd.DataFrame):
    """
    t-test 比較成分股 vs 非成分股的各項指標差異
    """
    numeric_cols = ["roe", "pe_ratio", "pb_ratio", "gross_margin",
                    "operating_margin", "price_mom_3m", "price_mom_6m", "volatility_60d"]
    
    results = []
    for col in numeric_cols:
        a = in_etf[col].dropna()
        b = not_in_etf[col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(a, b)
        results.append({
            "feature": col,
            "in_etf_mean": a.mean(),
            "not_in_etf_mean": b.mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "significant": "*" if p_val < 0.05 else ""
        })
    
    result_df = pd.DataFrame(results)
    print("\n=== 成分股 vs 全市場特徵比較 ===")
    print(result_df.to_string(index=False))
    return result_df

def plot_weight_distribution(holdings: pd.DataFrame):
    """
    畫出持股產業分佈圓餅圖
    """
    industry_weight = holdings.groupby("industry")["weight_pct"].sum()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(industry_weight, labels=industry_weight.index, autopct="%1.1f%%")
    ax.set_title("00981A 產業配置")
    plt.tight_layout()
    plt.savefig("../data/industry_distribution.png", dpi=150)
    print("圖表已儲存: data/industry_distribution.png")

if __name__ == "__main__":
    holdings = pd.read_csv("../data/holdings_sample.csv")
    describe_holdings(holdings)
    plot_weight_distribution(holdings)
    print("\n下一步: 執行 model/stock_selector.py 訓練選股模型")
