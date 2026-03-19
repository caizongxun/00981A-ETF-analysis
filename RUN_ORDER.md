# 執行順序說明

## 環境安裝

```bash
pip install -r requirements.txt
```

---

## Step 1 — 爬取持股明細

```bash
python analysis/fetch_holdings.py
```

- 爬取 00981A 每日持股明細
- 輸出：`data/raw/holdings_YYYY-MM-DD.csv`

---

## Step 2 — 計算特徵矩陣

```bash
python analysis/feature_engineering.py
```

- 對每支持股計算 ROE、PE、動能、波動率等 10 個特徵
- 輸出：`data/features.csv`

---

## Step 3 — 訓練選股模型

```bash
python model/stock_selector.py
```

- 以 features.csv + holdings_sample.csv 訓練 Random Forest 分類器
- 輸出：`data/rf_model.pkl`、`data/scaler.pkl`、`data/feature_importance.png`
- CV AUC 分數會印出，確認模型有學到東西

---

## Step 4 — 選股邏輯分析

```bash
python analysis/pattern_analysis.py
```

- 統計分析成分股 vs 非成分股的特徵差異（t-test）
- 輸出：`data/industry_distribution.png`

---

## Step 5 — 回測

```bash
python backtest/backtest.py
```

- 載入訓練好的模型，模擬每月換股買賣
- **買進邏輯**：模型預測機率 ≥ 0.5 的前 10 支股票，等權重買進
- **賣出邏輯**：每月換股日全數賣出，重新選股買進
- 輸出：
  - `data/backtest_nav.png` — 策略 vs 大盤累積淨值走勢圖
  - `data/trade_log.csv` — 完整交易紀錄
  - 終端機印出：總報酬率、年化報酬率、Sharpe Ratio、最大回撤

---

## 資料流圖

```
fetch_holdings.py
       ↓
  data/raw/*.csv
       ↓
feature_engineering.py
       ↓
  data/features.csv
       ↓
 stock_selector.py → data/rf_model.pkl
                           ↓
                    backtest.py
                           ↓
               data/backtest_nav.png
               data/trade_log.csv
```
