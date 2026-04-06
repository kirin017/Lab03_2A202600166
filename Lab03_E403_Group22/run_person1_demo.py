import os
import sys
from dotenv import load_dotenv

from src.core.openai_provider import OpenAIProvider
from src.telemetry.logger import logger
from chatbot_baseline import SimpleChatbot
from agent_v1 import ReActAgent as ReActAgentV1
from agent_v2 import ReActAgent as ReActAgentV2


def check_stock(args: str) -> str:
    product = args.strip().strip('"\'') or "unknown"
    catalog = {
        "gaming mouse": "In stock: 12 units",
        "mechanical keyboard": "Out of stock",
        "usb-c hub": "In stock: 7 units",
    }
    return catalog.get(product.lower(), f"No stock data found for {product}")


def calc_shipping(args: str) -> str:
    text = args.lower()
    if "hanoi" in text:
        return "Shipping to Hanoi: 30000 VND"
    if "hcm" in text or "ho chi minh" in text:
        return "Shipping to Ho Chi Minh City: 35000 VND"
    return "Shipping fee unavailable for that destination"


def get_tools():
    return [
        {
            "name": "check_stock",
            "description": "Check product inventory. Input should be a product name such as gaming mouse.",
            "func": check_stock,
        },
        {
            "name": "calc_shipping",
            "description": "Calculate shipping fee. Input should contain a destination city such as Hanoi or Ho Chi Minh City.",
            "func": calc_shipping,
        },
    ]


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("DEFAULT_MODEL", "gpt-4o")

    if not api_key:
        print("Missing OPENAI_API_KEY in .env")
        return 1

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = (
            "Check whether the gaming mouse is in stock and then estimate shipping to Hanoi. "
            "After using tools, give one final recommendation sentence."
        )

    llm = OpenAIProvider(model_name=model, api_key=api_key)
    tools = get_tools()

    baseline = SimpleChatbot(llm)
    v1 = ReActAgentV1(llm, tools, max_steps=5)
    v2 = ReActAgentV2(llm, tools, max_steps=5)

    print("\n=== QUESTION ===")
    print(question)

    print("\n=== BASELINE CHATBOT ===")
    baseline_answer = baseline.run(question)
    print(baseline_answer)

    print("\n=== AGENT V1 ===")
    v1_answer = v1.run(question)
    print(v1_answer)

    print("\n=== AGENT V2 ===")
    v2_answer = v2.run(question)
    print(v2_answer)

    logger.log_event(
        "PERSON1_DEMO_SUMMARY",
        {
            "question": question,
            "baseline_answer": baseline_answer,
            "v1_answer": v1_answer,
            "v2_answer": v2_answer,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
