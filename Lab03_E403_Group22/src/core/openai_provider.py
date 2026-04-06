import time
from typing import Dict, Any, Optional, Generator
from openai import OpenAI
from src.core.llm_provider import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=self.api_key)

    # openai_provider.py — sửa hàm generate()

    def generate(self, prompt: str, system_prompt: str = "", stop: list = None) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,   # ← Giảm xuống 0.2 để LLM bớt sáng tạo
        }
        if stop:
            params["stop"] = stop  # ← Truyền stop sequences vào API

        response = self.client.chat.completions.create(**params)
        return {"content": response.choices[0].message.content}

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
