#!/usr/bin/env python3
"""
回測模組：使用訓練好的 Random Forest 模型，模擬每月換股並計算累積報酬
流程：
  1. 每月重新取得候選股特徵
  2. 模型預測入選機率 > threshold 則買進，否則賣出
  3. 計算持有期間報酬，與大盤(0050)比較
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler

# ── 參數 ────────────────────────────────────────────
START_DATE     = "2025-06-01"   # 回測起始日（00981A 上市後）
END_DATE       = "2026-03-01"   # 回測結束日
INITIAL_CASH   = 1_000_000      # 初始資金（元）
TOP_N          = 10             # 每期持有股數
THRESHOLD      = 0.5            # 模型入選機率門檻
REBALANCE_FREQ = "M"            # 換股頻率：M=每月
TRANS_COST     = 0.001425       # 手續費 0.1425%
TAX_RATE       = 0.003          # 證交稅 0.3%（賣方）

# 候選池：00981A 歷史成分股（可擴充）
CANDIDATE_POOL = [
    "2330", "3691", "4552", "2308", "6669",
    "2368", "8299", "6274", "2383", "6223",
    "2454", "3008", "2317", "2382", "3711",
    "6446", "2357", "2379", "5274", "3661"
]

FEATURE_COLS = [
    "roe", "pe_ratio", "pb_ratio", "gross_margin",
    "operating_margin", "debt_to_equity",
    "price_mom_3m", "price_mom_6m", "volatility_60d",
    "market_cap_rank"
]

# ── 工具函式 ─────────────────────────────────────────
def get_features(tickers, as_of_date=None):
    """取得候選股當下特徵，as_of_date 為基準日"""
    records = []
    for tk in tickers:
        try:
            t = yf.Ticker(f"{tk}.TW")
            info = t.info
            hist = t.history(period="9mo")
            if hist.empty:
                continue
            closes = hist["Close"]
            mom3  = (closes.iloc[-1] / closes.iloc[-63] - 1)  if len(closes) >= 63  else np.nan
            mom6  = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
            vol60 = closes.pct_change().tail(60).std() * np.sqrt(252)
            records.append({
                "ticker":           tk,
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
                "last_price":       closes.iloc[-1],
            })
        except Exception as e:
            print(f"[WARN] {tk}: {e}")
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df


def simulate_prices(tickers, start, end):
    """下載歷史股價，回傳 DataFrame (date x ticker)"""
    price_dict = {}
    for tk in tickers:
        try:
            hist = yf.download(f"{tk}.TW", start=start, end=end,
                               progress=False, auto_adjust=True)
            if not hist.empty:
                price_dict[tk] = hist["Close"]
        except Exception:
            pass
    return pd.DataFrame(price_dict).ffill()


# ── 主回測引擎 ────────────────────────────────────────
class Backtester:
    def __init__(self, model, scaler):
        self.model   = model
        self.scaler  = scaler
        self.cash    = float(INITIAL_CASH)
        self.holdings = {}   # {ticker: shares}
        self.nav_history = []  # [(date, nav)]
        self.trade_log   = []  # [(date, action, ticker, shares, price)]

    def _predict_top_n(self, feature_df, n=TOP_N):
        """模型預測，回傳機率最高前 N 支股票"""
        df = feature_df.dropna(subset=FEATURE_COLS).copy()
        if df.empty:
            return []
        X = self.scaler.transform(df[FEATURE_COLS].values)
        df["prob"] = self.model.predict_proba(X)[:, 1]
        df = df[df["prob"] >= THRESHOLD]
        top = df.nlargest(n, "prob")
        return list(top["ticker"])

    def _sell_all(self, price_row, date):
        """全數賣出現有持股"""
        for tk, shares in list(self.holdings.items()):
            if tk in price_row.index and not np.isnan(price_row[tk]):
                price = price_row[tk]
                revenue = shares * price * (1 - TAX_RATE - TRANS_COST)
                self.cash += revenue
                self.trade_log.append((date, "SELL", tk, shares, price))
        self.holdings = {}

    def _buy_equal_weight(self, tickers, price_row, date):
        """等權重買進指定股票"""
        valid = [tk for tk in tickers
                 if tk in price_row.index and not np.isnan(price_row[tk])]
        if not valid:
            return
        alloc = self.cash / len(valid)
        for tk in valid:
            price  = price_row[tk]
            shares = int(alloc * (1 - TRANS_COST) / price)
            if shares <= 0:
                continue
            cost = shares * price * (1 + TRANS_COST)
            self.cash -= cost
            self.holdings[tk] = self.holdings.get(tk, 0) + shares
            self.trade_log.append((date, "BUY", tk, shares, price))

    def calc_nav(self, price_row, date):
        nav = self.cash
        for tk, shares in self.holdings.items():
            if tk in price_row.index and not np.isnan(price_row[tk]):
                nav += shares * price_row[tk]
        self.nav_history.append({"date": date, "nav": nav})
        return nav

    def run(self, price_df):
        rebalance_dates = pd.date_range(START_DATE, END_DATE, freq=REBALANCE_FREQ)
        print(f"回測期間: {START_DATE} ~ {END_DATE}")
        print(f"換股日期: {len(rebalance_dates)} 次")

        for rb_date in rebalance_dates:
            date_str = rb_date.strftime("%Y-%m-%d")
            print(f"\n[{date_str}] 換股中...")

            # 取得當日或最近一個交易日的收盤價
            avail = price_df.index[price_df.index <= rb_date]
            if avail.empty:
                continue
            price_row = price_df.loc[avail[-1]]

            # 賣出舊持股
            self._sell_all(price_row, date_str)

            # 模型預測新持股
            feat_df = get_features(CANDIDATE_POOL)
            top_tickers = self._predict_top_n(feat_df)
            print(f"  入選股票: {top_tickers}")

            # 買進新持股
            self._buy_equal_weight(top_tickers, price_row, date_str)

            # 記錄 NAV
            nav = self.calc_nav(price_row, date_str)
            print(f"  NAV: {nav:,.0f} 元")

        return pd.DataFrame(self.nav_history).set_index("date")


# ── 績效分析 ─────────────────────────────────────────
def calc_performance(nav_df, benchmark_ticker="0050.TW"):
    nav_df["return"] = nav_df["nav"].pct_change()
    total_return  = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / len(nav_df)) - 1
    sharpe        = nav_df["return"].mean() / nav_df["return"].std() * np.sqrt(252)
    max_dd        = (nav_df["nav"] / nav_df["nav"].cummax() - 1).min()

    print("\n=== 回測績效 ===")
    print(f"總報酬率   : {total_return*100:.2f}%")
    print(f"年化報酬率  : {annual_return*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"最大回撤    : {max_dd*100:.2f}%")

    # 下載大盤對比
    bm = yf.download(benchmark_ticker, start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True)["Close"]
    bm_return = (bm.iloc[-1] / bm.iloc[0] - 1) * 100
    print(f"大盤(0050)報酬: {bm_return:.2f}%")

    return total_return, annual_return, sharpe, max_dd


def plot_nav(nav_df, output_path="../data/backtest_nav.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    nav_series = nav_df["nav"] / nav_df["nav"].iloc[0] * 100
    ax.plot(nav_series.index, nav_series.values, label="策略 NAV", linewidth=2)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("00981A 選股策略 回測 NAV 走勢")
    ax.set_ylabel("累積淨值 (起始=100)")
    ax.set_xlabel("日期")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"NAV 圖已存: {output_path}")


# ── 入口 ─────────────────────────────────────────────
if __name__ == "__main__":
    model_path  = "../data/rf_model.pkl"
    scaler_path = "../data/scaler.pkl"

    # 若已有訓練好的模型則直接載入，否則先跑 model/stock_selector.py
    if not os.path.exists(model_path):
        print("[ERROR] 找不到模型檔，請先執行 model/stock_selector.py")
        exit(1)

    with open(model_path,  "rb") as f: model  = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)

    # 下載候選池歷史股價
    print("下載候選股歷史股價...")
    price_df = simulate_prices(CANDIDATE_POOL, START_DATE, END_DATE)

    # 執行回測
    bt = Backtester(model, scaler)
    nav_df = bt.run(price_df)

    # 績效
    calc_performance(nav_df)
    plot_nav(nav_df)

    # 輸出交易紀錄
    trade_df = pd.DataFrame(bt.trade_log,
                            columns=["date", "action", "ticker", "shares", "price"])
    trade_df.to_csv("../data/trade_log.csv", index=False, encoding="utf-8-sig")
    print("交易紀錄已存: data/trade_log.csv")
