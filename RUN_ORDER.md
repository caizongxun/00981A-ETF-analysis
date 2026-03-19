# 執行順序說明

> 所有指令都從 **專案根目錄** (`00981A-ETF-analysis/`) 執行

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

- 自動建立 `data/` 資料夾
- 對每支股票自動測試 `.TW` / `.TWO` 後綴
- 輸出：`data/features.csv`

---

## Step 3 — 訓練選股模型

```bash
python model/stock_selector.py
```

- 輸出：`data/rf_model.pkl`、`data/scaler.pkl`、`data/feature_importance.png`

---

## Step 4 — 選股邏輯分析（可選）

```bash
python analysis/pattern_analysis.py
```

---

## Step 5 — 回測

```bash
python backtest/backtest.py
```

- 輸出：`data/backtest_nav.png`、`data/trade_log.csv`
- 終端列出：總報酬率、年化報酬、Sharpe、最大回撤、Alpha vs 0050

---

## 常見錯誤處理

| 錯誤 | 原因 | 解決 |
|------|------|------|
| `OSError: non-existent directory` | 舊版路徑問題 | 已修正，自動建立 data/ |
| `Quote not found 3691.TW` | 上櫃小市場用 .TWO | 已修正，自動測試後綴 |
| `FileNotFoundError features.csv` | 尚未執行 Step 2 | 先執行 feature_engineering.py |
| `[ERROR] 找不到模型` | 尚未執行 Step 3 | 先執行 stock_selector.py |
