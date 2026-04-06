"""OpenAI LLM client for chat completion."""

import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLM:
    """Client for interacting with OpenAI-compatible LLM APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.base_url = base_url

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """Send chat completion request and return response text.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: Whether to stream the response.

        Returns:
            The generated response text.
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if stream:
            return self._stream_response(params)

        response = self.client.chat.completions.create(**params, stream=False)
        return response.choices[0].message.content

    def _stream_response(self, params: dict) -> str:
        """Stream and collect the full response text."""
        stream = self.client.chat.completions.create(**params, stream=True)
        content_parts = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
        return "".join(content_parts)
    def invoke(self, prompt: str) -> str:
        """Example usage of LLMClient."""
        client = LLM()
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = client.chat_completion(messages)
        return response

if __name__ == "__main__":
    llm = LLM()
    print(llm.invoke("What is the capital of France?"))