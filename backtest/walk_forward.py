#!/usr/bin/env python3
"""
Walk-Forward 驗證 + Out-of-Sample 回測

核心逿輯：
  - 每一期只用「當期之前」的持股資料訓練模型
  - 預測下一期選股（模型從未見過該期資料）
  - 輸出每期 OOS 報酬與累積表現

執行方式：
  # 先執行歷史持股資料生成
  python data/generate_synthetic_holdings.py
  # 再執行 Walk-Forward
  python backtest/walk_forward.py
"""

import sys
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

FEATURE_COLS = [
    "roe", "pe_ratio", "pb_ratio", "gross_margin",
    "operating_margin", "debt_to_equity",
    "price_mom_3m", "price_mom_6m", "volatility_60d",
    "market_cap_rank"
]

CANDIDATE_POOL = {
    "2330": ".TW", "4552": ".TW", "2308": ".TW", "6669": ".TW",
    "2368": ".TW", "2383": ".TW", "2454": ".TW", "3008": ".TW",
    "2317": ".TW", "2382": ".TW", "3711": ".TW", "2357": ".TW",
    "2379": ".TW", "3691": ".TWO", "8299": ".TWO",
    "6274": ".TWO", "6223": ".TWO", "5274": ".TWO",
}

START_DATE   = "2025-06-01"
END_DATE     = "2026-03-01"
INITIAL_CASH = 1_000_000
TOP_N        = 8
THRESHOLD    = 0.4   # walk-forward 訓練樣本少，關檻放寬
TRANS_COST   = 0.001425
TAX_RATE     = 0.003


# ─ 訓練工具 ───────────────────────────────────────────
def train_on_history(history_df: pd.DataFrame, up_to_period: str):
    """
    使用 up_to_period 之前（含）的資料訓練模型
    回傳 (model, scaler, auc) 或 None
    """
    train_df = history_df[history_df["period"] <= up_to_period].copy()
    train_df = train_df.dropna(subset=FEATURE_COLS)
    if len(train_df) < 5 or train_df["in_etf"].sum() < 2:
        return None, None, None

    X = train_df[FEATURE_COLS].values
    y = train_df["in_etf"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4,
        class_weight="balanced", random_state=42
    )
    rf.fit(X_sc, y)

    # in-sample AUC
    try:
        auc = roc_auc_score(y, rf.predict_proba(X_sc)[:, 1])
    except Exception:
        auc = np.nan

    return rf, scaler, auc


def predict_next(model, scaler, current_features: pd.DataFrame) -> list:
    """根據当期特徵預測下期入選名單"""
    df = current_features.dropna(subset=FEATURE_COLS).copy()
    if df.empty or model is None:
        return []
    X = scaler.transform(df[FEATURE_COLS].values)
    df["prob"] = model.predict_proba(X)[:, 1]
    df = df[df["prob"] >= THRESHOLD]
    return list(df.nlargest(TOP_N, "prob")["stock_id"])


# ─ 股價下載 ───────────────────────────────────────────
def download_prices():
    price_dict = {}
    for tk, suffix in CANDIDATE_POOL.items():
        sym = f"{tk}{suffix}"
        try:
            hist = yf.download(sym, start=START_DATE, end=END_DATE,
                               progress=False, auto_adjust=True)
            if hist.empty:
                continue
            close = hist["Close"].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist["Close"]
            price_dict[tk] = close
        except Exception:
            pass
    df = pd.DataFrame(price_dict)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.ffill()


# ─ Walk-Forward 回測 ───────────────────────────────────────
def run_walk_forward(history_df: pd.DataFrame, price_df: pd.DataFrame):
    periods = sorted(history_df["period"].unique())
    # 至少需要 1 期當訓練，所以從第 2 期開始預測
    if len(periods) < 2:
        print("[ERROR] 需要至少 2 個時期的資料")
        return

    cash     = float(INITIAL_CASH)
    holdings = {}
    nav_log  = []
    trade_log = []
    oos_results = []

    print(f"可用時期: {periods}")
    print(f"Walk-Forward 訓練從第 2 期開始預測\n")

    for i in range(1, len(periods)):
        train_up_to = periods[i - 1]  # 訓練到上一期
        predict_for = periods[i]      # 預測這一期

        # 訓練
        model, scaler, train_auc = train_on_history(history_df, train_up_to)
        if model is None:
            print(f"[SKIP] {predict_for}: 訓練資料不足")
            continue

        # 用上一期特徵預測這一期選股
        curr_feat = history_df[history_df["period"] == train_up_to].copy()
        selected  = predict_next(model, scaler, curr_feat)

        # 計算 OOS AUC（用這一期的實際持股來評估）
        actual_df = history_df[history_df["period"] == predict_for].dropna(subset=FEATURE_COLS)
        oos_auc = np.nan
        if not actual_df.empty and actual_df["in_etf"].sum() >= 1:
            try:
                X_oos = scaler.transform(actual_df[FEATURE_COLS].values)
                prob_oos = model.predict_proba(X_oos)[:, 1]
                oos_auc = roc_auc_score(actual_df["in_etf"].values, prob_oos)
            except Exception:
                pass

        # 回測交易：以 predict_for 期第一個交易日為基準
        rb_date = pd.Timestamp(predict_for + "-01")
        avail   = price_df.index[price_df.index <= rb_date]
        if avail.empty:
            continue
        price_row = price_df.loc[avail[-1]].to_dict()

        # 賣出舊持股
        for tk, shares in list(holdings.items()):
            p = price_row.get(tk, np.nan)
            if pd.isna(p): continue
            cash += shares * float(p) * (1 - TAX_RATE - TRANS_COST)
            trade_log.append((predict_for, "SELL", tk, shares, round(float(p), 2)))
        holdings = {}

        # 買進新持股
        valid = [tk for tk in selected if not pd.isna(price_row.get(tk, np.nan))]
        if valid:
            alloc = cash / len(valid)
            for tk in valid:
                p = float(price_row[tk])
                shares = int(alloc * (1 - TRANS_COST) / p)
                if shares <= 0: continue
                cash -= shares * p * (1 + TRANS_COST)
                holdings[tk] = holdings.get(tk, 0) + shares
                trade_log.append((predict_for, "BUY", tk, shares, round(p, 2)))

        nav = cash + sum(sh * float(price_row.get(tk, 0)) for tk, sh in holdings.items())
        nav_log.append({"period": predict_for, "nav": nav})
        oos_results.append({
            "period":     predict_for,
            "train_auc":  round(train_auc, 4) if train_auc else np.nan,
            "oos_auc":    round(oos_auc, 4),
            "selected":   ",".join(selected),
            "nav":        round(nav, 0),
        })

        print(f"[{predict_for}] 訓練AUC={train_auc:.3f} | OOS AUC={oos_auc:.3f} | 入選: {selected}")
        print(f"           NAV: {nav:,.0f} 元")

    return pd.DataFrame(nav_log), pd.DataFrame(oos_results), pd.DataFrame(
        trade_log, columns=["period", "action", "ticker", "shares", "price"]
    )


# ─ 績效分析 + 畫圖 ───────────────────────────────────────
def print_performance(nav_df, oos_df):
    print("\n====== Walk-Forward OOS 績效 ======")
    total  = nav_df["nav"].iloc[-1] / INITIAL_CASH - 1
    annual = (1 + total) ** (12 / max(len(nav_df), 1)) - 1
    ret    = nav_df["nav"].pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0
    max_dd = (nav_df["nav"] / nav_df["nav"].cummax() - 1).min()

    print(f"總報酬率      : {total*100:.2f}%")
    print(f"年化報酬率   : {annual*100:.2f}%")
    print(f"Sharpe Ratio : {sharpe:.3f}")
    print(f"最大回撤     : {max_dd*100:.2f}%")
    print(f"\n平均 OOS AUC : {oos_df['oos_auc'].mean():.3f} (越高越好，0.5=跟猜相同)")
    print(f"訓練/OOS AUC gap : {(oos_df['train_auc'] - oos_df['oos_auc']).mean():.3f} (越小越不易過擬合)")
    print("\n各期 OOS 細節:")
    print(oos_df[["period", "train_auc", "oos_auc", "nav", "selected"]].to_string(index=False))

    try:
        bm     = yf.download("0050.TW", start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)["Close"]
        bm_ret = (float(bm.iloc[-1].iloc[0]) / float(bm.iloc[0].iloc[0]) - 1) * 100
        print(f"\n大盤(0050)報酬 : {bm_ret:.2f}%")
        print(f"Alpha        : {total*100 - bm_ret:.2f}%")
    except Exception:
        pass


def plot_results(nav_df, oos_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    # NAV 走勢
    nav_norm = nav_df["nav"] / INITIAL_CASH * 100
    ax1.plot(range(len(nav_norm)), nav_norm.values, marker="o", color="steelblue", linewidth=2)
    ax1.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xticks(range(len(nav_df)))
    ax1.set_xticklabels(nav_df["period"].tolist(), rotation=30)
    ax1.set_title("Walk-Forward OOS 累積淨值")
    ax1.set_ylabel("淨值 (起始=100)")
    ax1.fill_between(range(len(nav_norm)), 100, nav_norm.values,
                     where=(nav_norm.values >= 100), alpha=0.15, color="green")
    ax1.fill_between(range(len(nav_norm)), 100, nav_norm.values,
                     where=(nav_norm.values < 100),  alpha=0.15, color="red")

    # AUC 比較
    x = range(len(oos_df))
    ax2.bar([i - 0.2 for i in x], oos_df["train_auc"], width=0.4,
            label="訓練 AUC", color="steelblue", alpha=0.8)
    ax2.bar([i + 0.2 for i in x], oos_df["oos_auc"],   width=0.4,
            label="OOS AUC",  color="tomato",    alpha=0.8)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="隨機基準")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(oos_df["period"].tolist(), rotation=30)
    ax2.set_title("訓練 AUC vs OOS AUC（gap 越小越好）")
    ax2.set_ylabel("AUC")
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    out = DATA_DIR / "walk_forward_result.png"
    plt.savefig(out, dpi=150)
    print(f"\n圖表已存: {out}")


# ─ 入口 ─────────────────────────────────────────────
if __name__ == "__main__":
    hist_path = DATA_DIR / "holdings_history.csv"
    if not hist_path.exists():
        print("[ERROR] 找不到 holdings_history.csv")
        print("        請先執行: python data/generate_synthetic_holdings.py")
        exit(1)

    history_df = pd.read_csv(hist_path)
    print(f"載入 {len(history_df)} 筆持股資料，{history_df['period'].nunique()} 個時期")

    print("\n下載歷史股價...")
    price_df = download_prices()
    print(f"取得 {len(price_df.columns)} 支股票歷史股價")

    nav_df, oos_df, trade_df = run_walk_forward(history_df, price_df)

    print_performance(nav_df, oos_df)
    plot_results(nav_df, oos_df)

    oos_df.to_csv(DATA_DIR / "oos_results.csv",   index=False, encoding="utf-8-sig")
    trade_df.to_csv(DATA_DIR / "wf_trade_log.csv", index=False, encoding="utf-8-sig")
    print("輸出檔案: data/oos_results.csv, data/wf_trade_log.csv, data/walk_forward_result.png")
