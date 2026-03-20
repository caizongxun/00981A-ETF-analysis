#!/usr/bin/env python3
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from utils.font_helper import setup_font

setup_font()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

FEATURE_COLS_ALL = [
    "mom_1m", "mom_3m", "mom_6m",
    "vol_20d", "vol_60d",
    "rs_vs_market", "above_ma60", "price_range",
    "price_52w_pct", "vol_ratio_20d", "log_turnover",
    "rev_mom_1m", "rev_mom_3m", "rev_yoy",
]

START_DATE   = "2024-01-01"
END_DATE     = "2026-04-01"
INITIAL_CASH = 1_000_000
TOP_N        = 8
THRESHOLD    = 0.35
TRANS_COST   = 0.001425
TAX_RATE     = 0.003

_SUFFIX_CACHE: dict = {}
def get_valid_symbol(stock_id):
    if stock_id in _SUFFIX_CACHE:
        s = _SUFFIX_CACHE[stock_id]
        return f"{stock_id}{s}" if s else None
    for suffix in (".TW", ".TWO"):
        try:
            h = yf.Ticker(f"{stock_id}{suffix}").history(period="5d")
            if not h.empty:
                _SUFFIX_CACHE[stock_id] = suffix
                return f"{stock_id}{suffix}"
        except:
            pass
    _SUFFIX_CACHE[stock_id] = None
    return None


def get_feature_cols(df):
    return [c for c in FEATURE_COLS_ALL if c in df.columns]


def impute(df, feature_cols, medians=None):
    """NaN 用訓練集中位數填漟，回傳 (df, medians_dict)"""
    df = df.copy()
    if medians is None:
        medians = {}
        for col in feature_cols:
            if col in df.columns:
                v = df[col].dropna()
                medians[col] = float(v.median()) if len(v) > 0 else 0.0
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0.0))
    return df, medians


def train_model(train_df, feature_cols):
    df, medians = impute(train_df, feature_cols)
    if df["in_etf"].nunique() < 2:
        return None, None, None, np.nan
    if df["in_etf"].value_counts().min() < 5:
        return None, None, None, np.nan
    X = df[feature_cols].values
    y = df["in_etf"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5,
        class_weight="balanced", random_state=42
    )
    rf.fit(Xs, y)
    try:
        auc = roc_auc_score(y, rf.predict_proba(Xs)[:, 1])
    except:
        auc = np.nan
    return rf, scaler, medians, auc


def predict_top(model, scaler, medians, feat_df, feature_cols):
    df, _ = impute(feat_df, feature_cols, medians)
    if df.empty or model is None:
        return []
    proba = model.predict_proba(scaler.transform(df[feature_cols].values))
    if proba.shape[1] < 2:
        return []
    df = df.copy()
    df["prob"] = proba[:, 1]
    df = df[df["prob"] >= THRESHOLD]
    return [str(t) for t in df.nlargest(TOP_N, "prob")["stock_id"].tolist()]


def download_prices(universe):
    price_dict = {}
    for tk in universe:
        sym = get_valid_symbol(tk)
        if sym is None:
            continue
        try:
            h = yf.download(sym, start=START_DATE, end=END_DATE,
                            progress=False, auto_adjust=True)
            if h.empty:
                continue
            close = h["Close"]
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None)
            price_dict[str(tk)] = close
        except:
            pass
    df = pd.DataFrame(price_dict)
    df.index = pd.to_datetime(df.index)
    return df.ffill()


def run_walk_forward(history_df, price_df):
    history_df = history_df.copy()
    history_df["stock_id"] = history_df["stock_id"].astype(str)
    feature_cols = get_feature_cols(history_df)
    print(f"使用特徵: {feature_cols}")

    periods = sorted(history_df["period"].unique())
    print(f"有效時期: {periods}")
    if len(periods) < 2:
        print("[ERROR] 需要至少 2 個時期")
        return None, None, None

    cash = float(INITIAL_CASH)
    holdings = {}
    nav_log, trade_log, oos_list = [], [], []

    for i in range(1, len(periods)):
        train_up_to = periods[i - 1]
        predict_for = periods[i]

        train_df = history_df[history_df["period"] <= train_up_to]
        model, scaler, medians, train_auc = train_model(train_df, feature_cols)
        if model is None:
            print(f"[SKIP] {predict_for}: 訓練資料不足")
            continue

        curr_feat = history_df[history_df["period"] == train_up_to].copy()
        selected  = predict_top(model, scaler, medians, curr_feat, feature_cols)

        actual_df, _ = impute(
            history_df[history_df["period"] == predict_for].copy(),
            feature_cols, medians
        )
        oos_auc = np.nan
        if not actual_df.empty and actual_df["in_etf"].nunique() >= 2:
            try:
                proba = model.predict_proba(scaler.transform(actual_df[feature_cols].values))
                if proba.shape[1] >= 2:
                    oos_auc = roc_auc_score(actual_df["in_etf"].values, proba[:, 1])
            except:
                pass

        rb_date = pd.Timestamp(predict_for + "-01")
        avail   = price_df.index[price_df.index <= rb_date]
        if avail.empty:
            print(f"[SKIP] {predict_for}: 沒有對應交易日")
            continue
        price_row = price_df.loc[avail[-1]].to_dict()

        for tk, shares in list(holdings.items()):
            p = price_row.get(tk, np.nan)
            if pd.isna(p): continue
            cash += shares * float(p) * (1 - TAX_RATE - TRANS_COST)
            trade_log.append((predict_for, "SELL", tk, shares, round(float(p), 2)))
        holdings = {}

        valid = [tk for tk in selected if not pd.isna(price_row.get(tk, np.nan))]
        if valid:
            alloc = cash / len(valid)
            for tk in valid:
                p      = float(price_row[tk])
                shares = int(alloc * (1 - TRANS_COST) / p)
                if shares <= 0: continue
                cash -= shares * p * (1 + TRANS_COST)
                holdings[tk] = holdings.get(tk, 0) + shares
                trade_log.append((predict_for, "BUY", tk, shares, round(p, 2)))

        nav = cash + sum(sh * float(price_row.get(tk, 0)) for tk, sh in holdings.items())
        nav_log.append({"period": predict_for, "nav": nav})
        oos_list.append({
            "period":    predict_for,
            "train_auc": round(float(train_auc), 4) if not np.isnan(train_auc) else np.nan,
            "oos_auc":   round(float(oos_auc),   4) if not np.isnan(oos_auc)   else np.nan,
            "selected":  ",".join(selected),
            "nav":       round(nav, 0),
        })
        t_str = f"{train_auc:.3f}" if not np.isnan(train_auc) else "N/A"
        o_str = f"{oos_auc:.3f}"   if not np.isnan(oos_auc)   else "N/A"
        print(f"[{predict_for}] 訓練AUC={t_str} | OOS AUC={o_str} | 入選: {selected}")
        print(f"           NAV: {nav:,.0f} 元")

    return (
        pd.DataFrame(nav_log),
        pd.DataFrame(oos_list),
        pd.DataFrame(trade_log, columns=["period","action","ticker","shares","price"]),
    )


def print_performance(nav_df, oos_df):
    if nav_df.empty:
        print("[ERROR] 沒有任何有效期數")
        return
    total  = nav_df["nav"].iloc[-1] / INITIAL_CASH - 1
    annual = (1 + total) ** (12 / max(len(nav_df), 1)) - 1
    ret    = nav_df["nav"].pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0
    max_dd = (nav_df["nav"] / nav_df["nav"].cummax() - 1).min()
    print("\n====== Walk-Forward OOS 績效 ======")
    print(f"總報酬率      : {total*100:.2f}%")
    print(f"年化報酬率   : {annual*100:.2f}%")
    print(f"Sharpe Ratio : {sharpe:.3f}")
    print(f"最大回撤     : {max_dd*100:.2f}%")
    valid_oos = oos_df["oos_auc"].dropna()
    if len(valid_oos):
        print(f"平均 OOS AUC : {valid_oos.mean():.3f}")
        gap = (oos_df["train_auc"] - oos_df["oos_auc"]).dropna()
        if len(gap):
            print(f"訓練/OOS gap  : {gap.mean():.3f}")
    print("\n各期 OOS 細節:")
    print(oos_df[["period","train_auc","oos_auc","nav","selected"]].to_string(index=False))
    try:
        bm  = yf.download("0050.TW", start="2025-06-01", end=END_DATE,
                          progress=False, auto_adjust=True)["Close"]
        bm0 = float(bm.iloc[0].iloc[0] if hasattr(bm.iloc[0], "iloc") else bm.iloc[0])
        bm1 = float(bm.iloc[-1].iloc[0] if hasattr(bm.iloc[-1], "iloc") else bm.iloc[-1])
        bm_ret = (bm1 / bm0 - 1) * 100
        print(f"\n大盤(0050)報酬 : {bm_ret:.2f}%")
        print(f"Alpha        : {total*100 - bm_ret:.2f}%")
    except:
        pass


def plot_results(nav_df, oos_df):
    if nav_df.empty:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    nav_norm = nav_df["nav"] / INITIAL_CASH * 100
    ax1.plot(range(len(nav_norm)), nav_norm.values, marker="o",
             color="steelblue", linewidth=2)
    ax1.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xticks(range(len(nav_df)))
    ax1.set_xticklabels(nav_df["period"].tolist(), rotation=30)
    ax1.set_title("Walk-Forward OOS 累積淨値")
    ax1.set_ylabel("淨値 (起始=100)")
    vals = nav_norm.values
    ax1.fill_between(range(len(vals)), 100, vals,
                     where=(vals >= 100), alpha=0.15, color="green")
    ax1.fill_between(range(len(vals)), 100, vals,
                     where=(vals < 100), alpha=0.15, color="red")
    x = range(len(oos_df))
    ax2.bar([i-0.2 for i in x], oos_df["train_auc"].fillna(0), width=0.4,
            label="訓練 AUC", color="steelblue", alpha=0.8)
    ax2.bar([i+0.2 for i in x], oos_df["oos_auc"].fillna(0), width=0.4,
            label="OOS AUC", color="tomato", alpha=0.8)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="隨機基準")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(oos_df["period"].tolist(), rotation=30)
    ax2.set_title("訓練 AUC vs OOS AUC")
    ax2.set_ylabel("AUC")
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    plt.tight_layout()
    out = DATA_DIR / "walk_forward_result.png"
    plt.savefig(out, dpi=150)
    print(f"圖表已存: {out}")


if __name__ == "__main__":
    hist_path = DATA_DIR / "holdings_history.csv"
    if not hist_path.exists():
        print("[ERROR] 請先執行: python data/fetch_history.py")
        exit(1)

    history_df = pd.read_csv(hist_path)
    history_df["stock_id"] = history_df["stock_id"].astype(str)
    print(f"載入 {len(history_df)} 筆持股資料，{history_df['period'].nunique()} 個時期")

    feat_cols = get_feature_cols(history_df)
    nan_rates = history_df[feat_cols].isna().mean()
    hi_nan = nan_rates[nan_rates > 0.05]
    if len(hi_nan):
        print("NaN 比例:")
        for col, r in hi_nan.items():
            print(f"  {col}: {r*100:.1f}%")

    universe = history_df["stock_id"].unique().tolist()
    print("\n下載歷史股價...")
    price_df = download_prices(universe)
    print(f"取得 {len(price_df.columns)} 支股票歷史股價")

    result = run_walk_forward(history_df, price_df)
    if result[0] is None:
        exit(1)
    nav_df, oos_df, trade_df = result

    print_performance(nav_df, oos_df)
    plot_results(nav_df, oos_df)

    oos_df.to_csv(DATA_DIR / "oos_results.csv",    index=False, encoding="utf-8-sig")
    trade_df.to_csv(DATA_DIR / "wf_trade_log.csv", index=False, encoding="utf-8-sig")
    print("輸出: data/oos_results.csv, data/wf_trade_log.csv, data/walk_forward_result.png")
