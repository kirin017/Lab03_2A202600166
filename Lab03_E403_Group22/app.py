import os
import sys
import logging
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.getLogger("AI-Lab-Agent").handlers = []
logging.getLogger("AI-Lab-Agent").addHandler(logging.NullHandler())

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from src.core.openai_provider import OpenAIProvider
from src.agent.tools import InternetSearch, TimeSearch, WeatherSearch
from src.agent.agent import ReActAgent


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


def build_agent():
    llm = OpenAIProvider(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return ReActAgent(llm=llm, tools=build_tools(), max_steps=5)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", build_agent())
    await cl.Message(
        content=(
            "## 🌤️ Weather ReAct Agent\n"
            "Xin chào! Tôi có thể giúp bạn:\n"
            "- 🌦️ Thời tiết & dự báo theo thành phố\n"
            "- 🌍 Tin tức bão lũ, thiên tai mới nhất\n"
            "- 🕐 Ngày giờ hiện tại tại Việt Nam\n\n"
            "Gõ `reset` để xóa lịch sử hội thoại."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    agent: ReActAgent = cl.user_session.get("agent")
    user_input = message.content.strip()

    if user_input.lower() == "reset":
        agent.history.clear()
        await cl.Message(content="🔄 Đã xóa lịch sử hội thoại!").send()
        return

    async with cl.Step(name="🤔 Đang suy luận...") as step:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, agent.run, user_input
            )
            step.output = "✅ Hoàn tất"
        except Exception as e:
            step.output = f"❌ Lỗi: {e}"
            response = f"Xin lỗi, đã có lỗi xảy ra: {e}"

    await cl.Message(content=response).send()