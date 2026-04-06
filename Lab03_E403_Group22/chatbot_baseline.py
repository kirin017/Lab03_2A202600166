from typing import Optional
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from src.core.llm_provider import LLMProvider


class SimpleChatbot:
    """
    Minimal baseline chatbot for Lab 3.
    Purpose: provide a direct-answer baseline without tool use.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def get_system_prompt(self) -> str:
        return (
            "You are a helpful chatbot. "
            "Answer directly using only your internal knowledge. "
            "Do not use tools. If you are unsure, say so briefly."
        )

    def run(self, user_input: str) -> str:
        logger.log_event("CHATBOT_START", {"input": user_input, "model": self.llm.model_name})
        result = self.llm.generate(user_input, system_prompt=self.get_system_prompt())

        usage = result.get("usage", {}) or {}
        latency_ms = result.get("latency_ms", 0)
        provider = result.get("provider", getattr(self.llm, "provider_name", "unknown"))
        tracker.track_request(provider=provider, model=self.llm.model_name, usage=usage, latency_ms=latency_ms)

        answer = (result.get("content") or "").strip()
        logger.log_event(
            "CHATBOT_END",
            {
                "input": user_input,
                "answer": answer,
                "usage": usage,
                "latency_ms": latency_ms,
            },
        )
        return answer
