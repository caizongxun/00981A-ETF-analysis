# 執行順序說明

> 所有指令都從 **專案根目錄** (`00981A-ETF-analysis/`) 執行

---

## 《方案 A》 基礎回測（已完成）

```bash
pip install -r requirements.txt
python analysis/feature_engineering.py   # Step 2
python model/stock_selector.py           # Step 3
python backtest/backtest.py              # Step 5
```

---

## 《方案 B》 Walk-Forward OOS 回測（近似真實）

### Step 1 — 生成多期持股快照（需要一段時間）

```bash
python data/generate_synthetic_holdings.py
```

- 自動模擬 9 個月的持股變化資料
- 輸出：`data/holdings_history.csv`

### Step 2 — Walk-Forward 驗證

```bash
python backtest/walk_forward.py
```

- 每期只用「當期之前」資料訓練，預測下一期（模型從未見過該期）
- 輸出：
  - `data/walk_forward_result.png` — NAV 走勢 + AUC 比較圖
  - `data/oos_results.csv` — 各期 OOS AUC 、入選股票
  - `data/wf_trade_log.csv` — 交易紀錄

---

## 如何判斷結果可信度

| 指標 | 代表意義 |
|------|----------|
| OOS AUC > 0.6 | 模型對未見資料有預測力 |
| 訓練 AUC - OOS AUC < 0.15 | 沒有明顯過擬合 |
| Sharpe > 1.0 | 風險調整後報酬尚可 |
| 最大回撤 < -20% | 需加止損機制 |

## 常見錯誤處理

| 錯誤 | 解決 |
|------|------|
| `OSError: non-existent directory` | 已修正，自動建立 data/ |
| `Quote not found 3691.TW` | 上櫃小市場用 .TWO，已修正 |
| `FileNotFoundError features.csv` | 尚未執行 Step 2 |
| `[ERROR] 找不到模型` | 尚未執行 stock_selector.py |
| `[ERROR] 找不到 holdings_history.csv` | 尚未執行 generate_synthetic_holdings.py |
