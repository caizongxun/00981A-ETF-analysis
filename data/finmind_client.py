#!/usr/bin/env python3
"""
FinMind API client 含 token 輪轉功能

設定方式：將所有 token 寫入 .finmind_tokens，一行一個：
  eyJhbGci...(token1)
  eyJhbGci...(token2)
  eyJhbGci...(token3)

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
    """載入所有可用的 token"""
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
    return [""]  # 未登入模式


class FinMindClient:
    """
    FinMind API client with token rotation.
    被限流時自動切換到下一個 token。
    """
    FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
    # 回傳這些錯誤碼時認為需要换 token
    RATE_LIMIT_CODES = {401, 403, 429}

    def __init__(self):
        self._tokens = _load_tokens()
        self._idx    = 0
        self._lock   = Lock()
        self._cooldown: dict[int, float] = {}  # idx -> 失效時間 (epoch)
        print(f"[FinMind] 載入 {len(self._tokens)} 個 token" +
              (" (token 輪轉已啟用)" if len(self._tokens) > 1 else ""))

    def _next_token(self, failed_idx: int):
        """failed_idx 這個 token 失敗，切換到下一個"""
        with self._lock:
            # 把失敗的 token 冷卻 1 小時
            self._cooldown[failed_idx] = time.time() + 3600
            # 尋找下一個可用的
            now = time.time()
            for i in range(len(self._tokens)):
                idx = (failed_idx + 1 + i) % len(self._tokens)
                if self._cooldown.get(idx, 0) < now:
                    self._idx = idx
                    print(f"[FinMind] 切換到 token [{idx+1}/{len(self._tokens)}]")
                    return True
            print("[FinMind] 所有 token 都已被限流，等待 60s...")
            time.sleep(60)
            # 重置冷卻記錄，再試一次
            self._cooldown.clear()
            return False

    def get(self, dataset: str, data_id: str, start_date: str,
            retries: int = 3) -> list[dict]:
        """
        拉取 FinMind 資料，被限流時自動换 token 重試
        """
        for attempt in range(retries * len(self._tokens)):
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
                if r.status_code in self.RATE_LIMIT_CODES:
                    print(f"[FinMind] token[{idx+1}] 被限流 (HTTP {r.status_code})")
                    self._next_token(idx)
                    continue
                r.raise_for_status()
                data = r.json()
                # FinMind 用 status=402 表示超限
                if data.get("status") in (402, 403):
                    print(f"[FinMind] token[{idx+1}] 超出额度: {data.get('msg','')}")
                    self._next_token(idx)
                    continue
                if data.get("status") == 200:
                    return data.get("data", [])
                # 其他錯誤
                print(f"[FinMind] 小異常 status={data.get('status')} msg={data.get('msg','')}")
                return []
            except requests.RequestException as e:
                print(f"[FinMind] 網路錯誤: {e}，等待 5s")
                time.sleep(5)
        return []


# 全域 singleton
_client: FinMindClient | None = None


def get_client() -> FinMindClient:
    global _client
    if _client is None:
        _client = FinMindClient()
    return _client
