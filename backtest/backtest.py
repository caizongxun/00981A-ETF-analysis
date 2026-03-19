#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.font_helper import setup_font

setup_font()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE   = "2025-06-01"
END_DATE     = "2026-03-01"
INITIAL_CASH = 1_000_000
TOP_N        = 10
THRESHOLD    = 0.5
TRANS_COST   = 0.001425
TAX_RATE     = 0.003

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

def get_valid_ticker_obj(stock_id):
    for suffix in [".TW", ".TWO"]:
        t = yf.Ticker(f"{stock_id}{suffix}")
        try:
            h = t.history(period="5d")
            if not h.empty:
                return t
        except Exception:
            pass
    return None

def get_features(tickers):
    records = []
    for tk in tickers:
        t = get_valid_ticker_obj(tk)
        if t is None:
            continue
        try:
            info   = t.info
            hist   = t.history(period="9mo")
            closes = hist["Close"]
            mom3  = (closes.iloc[-1] / closes.iloc[-63]  - 1) if len(closes) >= 63  else np.nan
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
            print(f"  [WARN] {tk}: {e}")
    df = pd.DataFrame(records)
    if not df.empty:
        df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df

def download_prices(tickers):
    price_dict = {}
    for tk in tickers:
        for suffix in [".TW", ".TWO"]:
            try:
                hist = yf.download(f"{tk}{suffix}", start=START_DATE,
                                   end=END_DATE, progress=False, auto_adjust=True)
                if not hist.empty:
                    price_dict[tk] = hist["Close"]
                    break
            except Exception:
                pass
    return pd.DataFrame(price_dict).ffill()

class Backtester:
    def __init__(self, model, scaler):
        self.model    = model
        self.scaler   = scaler
        self.cash     = float(INITIAL_CASH)
        self.holdings = {}
        self.nav_history = []
        self.trade_log   = []

    def _predict_top_n(self, feat_df):
        df = feat_df.dropna(subset=FEATURE_COLS).copy()
        if df.empty:
            return []
        X = self.scaler.transform(df[FEATURE_COLS].values)
        df["prob"] = self.model.predict_proba(X)[:, 1]
        df = df[df["prob"] >= THRESHOLD]
        return list(df.nlargest(TOP_N, "prob")["ticker"])

    def _sell_all(self, price_row, date):
        for tk, shares in list(self.holdings.items()):
            p = price_row.get(tk, np.nan)
            if pd.isna(p):
                continue
            self.cash += shares * p * (1 - TAX_RATE - TRANS_COST)
            self.trade_log.append((date, "SELL", tk, shares, round(float(p), 2)))
        self.holdings = {}

    def _buy_equal(self, tickers, price_row, date):
        valid = [tk for tk in tickers if not pd.isna(price_row.get(tk, np.nan))]
        if not valid:
            return
        alloc = self.cash / len(valid)
        for tk in valid:
            p = float(price_row[tk])
            shares = int(alloc * (1 - TRANS_COST) / p)
            if shares <= 0:
                continue
            self.cash -= shares * p * (1 + TRANS_COST)
            self.holdings[tk] = self.holdings.get(tk, 0) + shares
            self.trade_log.append((date, "BUY", tk, shares, round(p, 2)))

    def calc_nav(self, price_row, date):
        nav = self.cash + sum(
            sh * float(price_row.get(tk, 0))
            for tk, sh in self.holdings.items()
        )
        self.nav_history.append({"date": date, "nav": nav})
        return nav

    def run(self, price_df):
        rebalance_dates = pd.date_range(START_DATE, END_DATE, freq="MS")
        print(f"回測期間: {START_DATE} ~ {END_DATE}")
        for rb_date in rebalance_dates:
            date_str = rb_date.strftime("%Y-%m-%d")
            avail = price_df.index[price_df.index <= rb_date]
            if avail.empty:
                continue
            price_row = price_df.loc[avail[-1]].to_dict()
            self._sell_all(price_row, date_str)
            print(f"[{date_str}] 取得特徵中...", end=" ", flush=True)
            feat_df     = get_features(CANDIDATE_POOL)
            top_tickers = self._predict_top_n(feat_df)
            print(f"入選: {top_tickers}")
            self._buy_equal(top_tickers, price_row, date_str)
            nav = self.calc_nav(price_row, date_str)
            print(f"  NAV: {nav:,.0f} 元  現金: {self.cash:,.0f} 元")
        return pd.DataFrame(self.nav_history).set_index("date")

def calc_performance(nav_df):
    ret    = nav_df["nav"].pct_change().dropna()
    total  = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
    annual = (1 + total) ** (12 / max(len(nav_df), 1)) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0
    max_dd = (nav_df["nav"] / nav_df["nav"].cummax() - 1).min()
    print("\n====== 回測績效 ======")
    print(f"總報酬率   : {total*100:.2f}%")
    print(f"年化報酬率  : {annual*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"最大回撤    : {max_dd*100:.2f}%")
    try:
        bm     = yf.download("0050.TW", start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)["Close"]
        bm_ret = (float(bm.iloc[-1]) / float(bm.iloc[0]) - 1) * 100
        print(f"大盤(0050)報酬: {bm_ret:.2f}%")
        print(f"Alpha       : {total*100 - bm_ret:.2f}%")
    except Exception:
        pass

def plot_nav(nav_df):
    nav_norm = nav_df["nav"] / nav_df["nav"].iloc[0] * 100
    fig, ax  = plt.subplots(figsize=(12, 5))
    ax.plot(nav_norm.index, nav_norm.values, label="策略 NAV", linewidth=2, color="steelblue")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="起始基準")
    ax.fill_between(nav_norm.index, 100, nav_norm.values,
                    where=(nav_norm.values >= 100), alpha=0.15, color="green")
    ax.fill_between(nav_norm.index, 100, nav_norm.values,
                    where=(nav_norm.values < 100),  alpha=0.15, color="red")
    ax.set_title("00981A 選股策略 回測 NAV")
    ax.set_ylabel("累積淨值 (起始=100)")
    ax.legend()
    plt.tight_layout()
    out = DATA_DIR / "backtest_nav.png"
    plt.savefig(out, dpi=150)
    print(f"NAV 圖已存: {out}")

if __name__ == "__main__":
    model_path  = DATA_DIR / "rf_model.pkl"
    scaler_path = DATA_DIR / "scaler.pkl"
    if not model_path.exists():
        print("[ERROR] 找不到模型，請先執行: python model/stock_selector.py")
        exit(1)
    with open(model_path,  "rb") as f: model  = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    print("下載候選股歷史股價...")
    price_df = download_prices(CANDIDATE_POOL)
    print(f"取得 {len(price_df.columns)} 支股票的歷史股價")
    bt     = Backtester(model, scaler)
    nav_df = bt.run(price_df)
    calc_performance(nav_df)
    plot_nav(nav_df)
    trade_df = pd.DataFrame(bt.trade_log,
                            columns=["date", "action", "ticker", "shares", "price"])
    out = DATA_DIR / "trade_log.csv"
    trade_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"交易紀錄已存: {out}")
