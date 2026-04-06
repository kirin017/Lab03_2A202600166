"""OpenAI LLM client for chat completion."""

import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
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
def main():
        """Example usage of LLMClient."""
        client = LLMClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": """You are a weather search assistant. Your task is to find and provide accurate weather information for any location and time requested by the user.
If the request is unclear (e.g., missing location or time), ask follow-up questions.
Always present the results clearly and concisely."""},
        ]
        response = client.chat_completion(messages)
        print("Response:", response)
if __name__ == "__main__":
    main()