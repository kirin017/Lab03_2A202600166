import sys
import os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.getLogger("AI-Lab-Agent").handlers = []
logging.getLogger("AI-Lab-Agent").addHandler(logging.NullHandler())

from dotenv import load_dotenv
from src.chatbot.chatbot import Chatbot  # ← dùng full path
from src.core.openai_provider import OpenAIProvider

load_dotenv()

def run_interactive():
    print("=" * 60)
    print("  🌤️  WEATHER CHATBOT BASELINE")
    print("  (Gõ 'quit' hoặc 'exit' để thoát)")
    print("  (Gõ 'reset' để xóa lịch sử hội thoại)")
    print("=" * 60)

    llm = OpenAIProvider(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    chatbot = Chatbot(llm=llm)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 Goodbye!")
                break
            if user_input.lower() == "reset":
                chatbot.reset()
                print("🔄 Conversation reset!")
                continue

            print("🤖 Chatbot: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    run_interactive()