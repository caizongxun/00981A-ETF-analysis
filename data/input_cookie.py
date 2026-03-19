#!/usr/bin/env python3
"""
互動輸入 pocket.tw Cookie，儲存後自動執行下載
執行： python data/input_cookie.py
"""
import subprocess
import sys
from pathlib import Path

print("="*50)
print("pocket.tw Cookie 輸入工具")
print("="*50)
print("請依序輸入每個 Cookie 的值（直接複製貼上，留空表示跳過）")
print()

FIELDS = [
    "csrftoken",
    "is_system_password",
    "refresh",
    "reminder",
    "remlno",
    "sessionid",
    "token",
]

cookies = {}
for field in FIELDS:
    val = input(f"{field}: ").strip()
    if val:
        cookies[field] = val

if not cookies:
    print("[ERROR] 沒有輸入任何值")
    sys.exit(1)

cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

print()
print("已組合的 Cookie 字串：")
print(cookie_str[:80] + "..." if len(cookie_str) > 80 else cookie_str)
print()

# 儲存到 .cookie 檔（不推到 git）
cookie_file = Path(__file__).resolve().parent.parent / ".cookie"
cookie_file.write_text(cookie_str, encoding="utf-8")
print(f"Cookie 已儲存: {cookie_file}")
print()

print("開始下載 00981A 持股資料...")
print("-" * 50)

script = Path(__file__).resolve().parent / "download_real_holdings.py"
result = subprocess.run(
    [sys.executable, str(script), "--cookie", cookie_str],
    cwd=str(Path(__file__).resolve().parent.parent)
)

if result.returncode != 0:
    print()
    print("[FAIL] 下載失敗。常見原因：")
    print("  1. Cookie 已過期 → 重新登入 pocket.tw 再執行此腳本")
    print("  2. token 或 sessionid 複製不完整 → 確認全部複製")
else:
    print()
    print("完成! 接下來執行 Walk-Forward 回測：")
    print("  python backtest/walk_forward.py")
