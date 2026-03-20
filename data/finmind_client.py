#!/usr/bin/env python3
"""
FinMind API client 含 token 輪轉功能

設定方式：將所有 token 寫入 .finmind_tokens，一行一個：
  eyJhbGci...(token1)
  eyJhbGci...(token2)

沒有 .finmind_tokens 時自動 fallback 到 .finmind_token (單一 token)
"""
import os
import time
import requests
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
_TOKENS_FILE = ROOT / ".finmind_tokens"
_TOKEN_FILE  = ROOT / ".finmind_token"


def _load_tokens() -> list[str]:
    if _TOKENS_FILE.exists():
        tokens = [
            line.strip() for line in _TOKENS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if tokens:
            return tokens
    if _TOKEN_FILE.exists():
        t = _TOKEN_FILE.read_text(encoding="utf-8").strip()
        if t:
            return [t]
    env_token = os.environ.get("FINMIND_TOKEN", "")
    if env_token:
        return [env_token]
    return [""]


class FinMindClient:
    FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
    # 402 = 超出每日额度，和 401/403/429 一樣認為需要換 token 或放棄
    RATE_LIMIT_CODES = {401, 402, 403, 429}

    def __init__(self):
        self._tokens = _load_tokens()
        self._idx    = 0
        self._lock   = Lock()
        self._cooldown: dict[int, float] = {}
        print(f"[FinMind] 載入 {len(self._tokens)} 個 token" +
              (" (token 輪轉已啟用)" if len(self._tokens) > 1 else ""))

    def _next_token(self, failed_idx: int) -> bool:
        with self._lock:
            self._cooldown[failed_idx] = time.time() + 3600
            now = time.time()
            for i in range(len(self._tokens)):
                idx = (failed_idx + 1 + i) % len(self._tokens)
                if self._cooldown.get(idx, 0) < now:
                    self._idx = idx
                    print(f"[FinMind] 切換到 token [{idx+1}/{len(self._tokens)}]")
                    return True
            # 所有 token 都超限，直接放棄
            print("[FinMind] 所有 token 超限，放棄此筆資料")
            return False

    def get(self, dataset: str, data_id: str, start_date: str,
            retries: int = 3) -> list[dict]:
        for attempt in range(retries * max(len(self._tokens), 1)):
            with self._lock:
                idx   = self._idx
                token = self._tokens[idx]
            try:
                r = requests.get(
                    self.FINMIND_API,
                    params={
                        "dataset":    dataset,
                        "data_id":    data_id,
                        "start_date": start_date,
                        "token":      token,
                    },
                    timeout=20,
                )
                # HTTP 層面的超限處理（包含 402）
                if r.status_code in self.RATE_LIMIT_CODES:
                    print(f"[FinMind] token[{idx+1}] 超限 HTTP {r.status_code}")
                    ok = self._next_token(idx)
                    if not ok:
                        return []   # 所有 token 都超限，直接回空
                    continue
                r.raise_for_status()
                data = r.json()
                # JSON 層面的超限處理
                if data.get("status") in (402, 403):
                    print(f"[FinMind] token[{idx+1}] 超出額度: {data.get('msg','')}")
                    ok = self._next_token(idx)
                    if not ok:
                        return []
                    continue
                if data.get("status") == 200:
                    return data.get("data", [])
                print(f"[FinMind] 異常 status={data.get('status')} msg={data.get('msg','')}")
                return []
            except requests.RequestException as e:
                print(f"[FinMind] 網路錯誤: {e}，等待 5s")
                time.sleep(5)
        return []


_client: FinMindClient | None = None


def get_client() -> FinMindClient:
    global _client
    if _client is None:
        _client = FinMindClient()
    return _client
