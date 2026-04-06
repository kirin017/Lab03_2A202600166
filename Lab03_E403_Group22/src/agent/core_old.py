import sys
import os
import logging

logging.getLogger("AI-Lab-Agent").handlers = []
logging.getLogger("AI-Lab-Agent").addHandler(logging.NullHandler())

from dotenv import load_dotenv
from src.core.openai_provider import OpenAIProvider
from src.agent.tools import InternetSearch, WikiSearch, TimeSearch
from src.agent.agent import ReActAgent

load_dotenv()


def build_tools():
    return [
        {
            "name": "InternetSearch",
            "description": "Tìm kiếm tin tức thời tiết hiện tại, dự báo, sự kiện khí hậu gần đây trên Internet.",
            "func": lambda q: InternetSearch.invoke({"query": q, "k": 5}),
        },
        {
            "name": "TimeSearch",
            "description": "Lấy ngày giờ hiện tại theo múi giờ. Dùng khi cần biết hôm nay là ngày mấy, mấy giờ. Truyền tên timezone, mặc định là Asia/Ho_Chi_Minh.",
            "func": lambda q: TimeSearch.invoke({"timezone": q or "Asia/Ho_Chi_Minh"}),
        },
    ]


def run_interactive():
    print("=" * 60)
    print("  🌤️  WEATHER REACT AGENT")
    print("  (Gõ 'quit' hoặc 'exit' để thoát)")
    print("  (Gõ 'reset' để xóa lịch sử hội thoại)")
    print("=" * 60)

    llm = OpenAIProvider(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=5)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 Goodbye!")
                break
            if user_input.lower() == "reset":
                agent.history.clear()
                print("🔄 Conversation reset!")
                continue

            print("🤖 Agent: ", end="", flush=True)
            response = agent.run(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    run_interactive()