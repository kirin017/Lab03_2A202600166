import os
import logging
import chainlit as cl
import sys
import os


# Thêm root project vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.getLogger("AI-Lab-Agent").handlers = []
logging.getLogger("AI-Lab-Agent").addHandler(logging.NullHandler())

from src.agent.tools import InternetSearch, WikiSearch, TimeSearch, WeatherSearch
from dotenv import load_dotenv
from src.core.openai_provider import OpenAIProvider
from src.agent.tools import InternetSearch, WikiSearch, TimeSearch
from src.agent.agent import ReActAgent

load_dotenv()


def build_tools():
    return [
        {
            "name": "WeatherSearch",          # ← THÊM MỚI — ưu tiên cao nhất
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
                "Tìm kiếm tin tức thời tiết, cảnh báo thiên tai, sự kiện khí hậu "
                "cực đoan, bão lũ mới nhất trên Internet. Dùng khi cần tin tức "
                "hoặc WeatherSearch không đủ thông tin."
            ),
            "func": lambda q: InternetSearch.invoke({"query": q, "k": 5}),
        },
        {
            "name": "TimeSearch",
            "description": "Lấy ngày và giờ hiện tại tại Việt Nam.",
            "func": lambda q: TimeSearch.invoke({"query": q}),
        },
    ]


def build_agent():
    llm = OpenAIProvider(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return ReActAgent(llm=llm, tools=build_tools(), max_steps=5)


@cl.on_chat_start
async def on_chat_start():
    # Tạo agent mới cho mỗi session
    agent = build_agent()
    cl.user_session.set("agent", agent)

    await cl.Message(
        content=(
            "## 🌤️ Weather ReAct Agent\n"
            "Xin chào! Tôi có thể giúp bạn:\n"
            "- 🌦️ Tra cứu thời tiết & dự báo\n"
            "- 🌍 Tìm hiểu hiện tượng khí hậu\n"
            "- 🕐 Xem giờ hiện tại các múi giờ\n\n"
            "Gõ `reset` để xóa lịch sử hội thoại."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    agent: ReActAgent = cl.user_session.get("agent")
    user_input = message.content.strip()

    # Lệnh reset
    if user_input.lower() == "reset":
        agent.history.clear()
        await cl.Message(content="🔄 Đã xóa lịch sử hội thoại!").send()
        return

    # Hiển thị thinking indicator
    async with cl.Step(name="🤔 Agent đang suy luận...") as step:
        # Chạy agent (blocking) trong thread riêng để không block event loop
        import asyncio
        response = await asyncio.get_event_loop().run_in_executor(
            None, agent.run, user_input
        )
        step.output = "✅ Hoàn tất"

    await cl.Message(content=response).send()