from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str
    timeout_s: int = 60


class LLMTextClient:
    """Minimal LLM client using the same provider/model/env as LLMExplainer."""

    def __init__(self, config: Optional[LLMConfig] = None):
        provider = (os.getenv("LLM_PROVIDER", "openai") or "openai").lower()
        model = os.getenv("LLM_MODEL", "gpt-4o")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if not api_key:
            raise ValueError(f"API key not provided for {provider}")

        self.config = config or LLMConfig(provider=provider, api_key=api_key, model=model)

    def generate_markdown(self, prompt: str) -> str:
        if self.config.provider == "openai":
            return self._openai(prompt)
        if self.config.provider == "anthropic":
            return self._anthropic(prompt)
        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _openai(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You write careful, non-hallucinating business reports."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
        }
        resp = requests.post(url, headers=headers, json=data, timeout=self.config.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        return payload["choices"][0]["message"]["content"]

    def _anthropic(self, prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
            "temperature": 0.4,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=self.config.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        return payload["content"][0]["text"]
