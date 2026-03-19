# 00981A ETF 選股邏輯逆向工程分析

## 專案目的

透過演算法分析 **00981A 統一台股增長主動式ETF** 每日公開持股明細，嘗試逆向推導經理人選股邏輯，找出大盤下跌時仍能獲利的選股模型。

## ETF 基本資訊

| 項目 | 內容 |
|------|------|
| ETF 代號 | 00981A |
| 全名 | 統一台股增長主動式ETF |
| 掛牌日期 | 2025年5月 |
| 累積報酬 | 約62%（vs 0050的43%）|
| 規模 | 超過400億元 |
| 持股揭露 | 每日公開 |

## 選股策略摘要

1. **大型股優先**：至少60%投資台股市值前300大
2. **產業集中**：電子零組件(40%)、半導體(30%)、電腦週邊(10%)
3. **創新成長**：聚焦具技術護城河、高ROE、高營收成長率企業
4. **靈活調整**：每日監測，汰弱擇強

## 專案結構

```
00981A-ETF-analysis/
├── README.md                    # 專案說明
├── data/
│   └── holdings_sample.csv      # 持股明細範例資料
├── analysis/
│   ├── fetch_holdings.py        # 爬取每日持股明細
│   ├── feature_engineering.py  # 財報特徵工程
│   └── pattern_analysis.py     # 選股邏輯分析
└── model/
    └── stock_selector.py       # 機器學習選股模型
```

## 使用方式

```bash
# 安裝依賴
pip install -r requirements.txt

# 爬取持股明細
python analysis/fetch_holdings.py

# 特徵工程
python analysis/feature_engineering.py

# 分析選股邏輯
python analysis/pattern_analysis.py

# 訓練選股模型
python model/stock_selector.py
```

## 資料來源

- [CMoney 00981A 成分股](https://www.cmoney.tw/etf/tw/00981A)
- [口袋證券持股明細](https://www.pocket.tw/etf/tw/00981A/fundholding)
- [Yahoo 財經](https://tw.stock.yahoo.com/quote/00981A.TW)
- [鉅亨網](https://news.cnyes.com)
