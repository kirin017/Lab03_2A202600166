# test_agent.py
import os
import sys
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.core.openai_provider import OpenAIProvider
from src.agent.tools import InternetSearch, TimeSearch, WeatherSearch
from src.agent.agent import ReActAgent

logging.getLogger("AI-Lab-Agent").handlers = []
logging.getLogger("AI-Lab-Agent").addHandler(logging.NullHandler())

def build_tools():
    return [
        {
            "name": "WeatherSearch",
            "description": (
                "Lấy thời tiết HIỆN TẠI và DỰ BÁO 5 ngày tại một địa điểm. "
                "Dùng khi người dùng hỏi thời tiết, nhiệt độ, mưa, gió, dự báo. "
                "Truyền tên thành phố bằng tiếng Anh, ví dụ: 'Hanoi', 'Da Nang'."
            ),
            "func": lambda q: WeatherSearch.invoke({"query": q}),
        },
        {
            "name": "InternetSearch",
            "description": (
                "Tìm kiếm tin tức thời tiết, cảnh báo thiên tai, "
                "bão lũ mới nhất trên Internet."
            ),
            "func": lambda q: InternetSearch.invoke({"query": q, "k": 5}),
        },
        {
            "name": "TimeSearch",
            "description": "Lấy ngày và giờ hiện tại tại Việt Nam.",
            "func": lambda q: TimeSearch.invoke({"query": q}),
        },
    ]


def main():
    agent = build_agent()

    print("=" * 60)
    print("🌤️  Weather ReAct Agent — Chat Terminal")
    print("Gõ 'reset' để xóa lịch sử | 'exit' để thoát")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n🧑 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("👋 Tạm biệt!")
            break

        if user_input.lower() == "reset":
            agent.history.clear()
            print("🔄 Đã xóa lịch sử hội thoại!")
            continue

        print("\n🤖 Agent đang suy luận", end="", flush=True)

        try:
            answer = agent.run(user_input)
            print(f"\r🤖 Agent: {answer}")
        except Exception as e:
            print(f"\r❌ Lỗi: {e}")


def build_agent():
    llm = OpenAIProvider(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return ReActAgent(llm=llm, tools=build_tools(), max_steps=5)


if __name__ == "__main__":
    main()