# 手動下載持股資料指南

當自動下載失敗時，將手動下載的資料轉存成 `data/real_holdings_raw.csv`。

## 方式 A：口袋證券（最容易）

1. 開啟 https://www.pocket.tw/etf/tw/00981A/fundholding
2. 切換到各個月份，按 F12 → Network → XHR
3. 找 API 呼叫，複製 JSON 回傳值
4. 整理成以下格式：

```csv
period,stock_id,stock_name
2025-06,2330,台積電
2025-06,2368,金像電
...
```

## 方式 B： CMoney

1. 開啟 https://www.cmoney.tw/etf/tw/00981A/fundholding
2. 展開每個月份
3. 手動複製股票代號轉存 CSV

## 方式 C：統一投信官網直接下載

1. 開啟 https://www.upamc.com.tw/active-etf-holding.html
2. 搜尋 00981A
3. 下載各月 PDF/Excel
4. 將股票代號整理成 CSV

## 方式 D： SITCA 公開資料

1. 開啟 https://www.sitca.org.tw/ROC/SITCA_ETF/etf_statement.aspx
2. 選擇年/月 → 搜尋 00981A
3. 下載 Excel 檔 → 取得持股明細頁
4. 整理成 CSV 後放入 `data/real_holdings_raw.csv`

## 手動資料格式

```csv
period,stock_id,stock_name
2025-06,2330,台積電
2025-06,2368,金像電
2025-06,2383,台光電
2025-06,6223,旺矽
2025-06,5274,嘉澤科
2025-06,6274,台燿
2025-06,2308,台達電
2025-06,8299,群聯
2025-06,2330,台積電
2025-06,4552,智邦
```

## CSV 放好之後

```bash
# 合併真實持股和特徵
# 此脚本能讀取 real_holdings_raw.csv 並自動合併特徵
python data/merge_real_holdings.py

# 再跟 Walk-Forward 回測
python backtest/walk_forward.py
```
