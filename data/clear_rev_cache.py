#!/usr/bin/env python3
"""
清除營收快取中的 null 項目，讓下次執行會重試
執行: python data/clear_rev_cache.py
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REV_CACHE = DATA_DIR / "revenue_cache.json"

if not REV_CACHE.exists():
    print("快取檔不存在")
else:
    cache = json.loads(REV_CACHE.read_text(encoding="utf-8"))
    before = sum(len(v) for v in cache.values())
    # 移除全部 null 紀錄
    cleaned = {}
    for sid, months in cache.items():
        kept = {ym: v for ym, v in months.items() if v is not None}
        if kept:
            cleaned[sid] = kept
    after = sum(len(v) for v in cleaned.values())
    REV_CACHE.write_text(
        json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
    print(f"清除完成: {before} 筆 -> {after} 筆 (null 紀錄已刪除)")
    print("接下來執行: python data/debug_mops.py  (確認 API 回傳)")
    print("或直接: python data/fetch_history.py  (重試爬取營收)")
