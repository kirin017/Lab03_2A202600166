from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from typing import List, Dict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

class Chatbot:
    """
    "You are a weather assistant. "
    "Answer in 2-3 sentences maximum. Be direct and concise. "
    "If asked about real-time weather, say you have no live data in one sentence. "
        "No bullet points, no extra explanation."
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.history: List[Dict[str, str]] = []

    def get_system_prompt(self) -> str:
        return (
            "You are a weather assistant. "
            "Answer questions about weather, climate, and forecasts. "
            "IMPORTANT: You do NOT have access to real-time weather data or any external tools. "
            "If asked about current weather, you must clearly state that you cannot retrieve live data "
            "and can only provide general climate information based on your training knowledge."
        )

    def chat(self, user_input: str) -> str:
        logger.log_event("CHATBOT_INPUT", {
            "input": user_input,
            "model": self.llm.model_name
        })

        # Build prompt với history
        history_text = ""
        for turn in self.history:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        prompt = history_text + f"User: {user_input}"

        result = self.llm.generate(
            prompt=prompt,
            system_prompt=self.get_system_prompt()
        )

        response_text = result["content"]

        tracker.track_request(
            provider=result["provider"],
            model=self.llm.model_name,
            usage=result["usage"],
            latency_ms=result["latency_ms"]
        )

        logger.log_event("CHATBOT_OUTPUT", {
            "output": response_text,
            "latency_ms": result["latency_ms"],
            "tokens": result["usage"]
        })

        self.history.append({
            "user": user_input,
            "assistant": response_text
        })

        return response_text

    def reset(self):
        self.history = []
        logger.log_event("CHATBOT_RESET", {})