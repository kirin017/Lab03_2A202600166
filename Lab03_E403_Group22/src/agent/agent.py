import os
import re
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger


class ReActAgent:
    """
    ReAct-style Agent for weather and climate Q&A.
    Follows the Thought → Action → Observation loop.
    """

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in self.tools]
        )
        return f"""Bạn là trợ lý chuyên gia về thời tiết và hiện tượng khí hậu.
Nhiệm vụ của bạn là trả lời câu hỏi về thời tiết, khí hậu, hiện tượng tự nhiên một cách chính xác.

Bạn có quyền truy cập các công cụ sau:
{tool_descriptions}

Luôn tuân thủ định dạng sau (bằng tiếng Việt):
Thought: <suy luận của bạn về bước tiếp theo>
Action: <tên_tool>("<câu truy vấn>")
Observation: <kết quả từ tool>
... (lặp lại Thought/Action/Observation nếu cần)
Final Answer: <câu trả lời cuối cùng cho người dùng>

Quy tắc:
- Khi câu hỏi liên quan đến thời tiết, nhiệt độ, dự báo, mưa, gió tại một địa điểm → 
  LUÔN dùng WeatherSearch TRƯỚC. Truyền tên thành phố bằng tiếng Anh.
- Dùng InternetSearch cho tin tức thiên tai, bão lũ, cảnh báo khẩn hoặc khi 
  WeatherSearch không đủ thông tin.
- Dùng WikiSearch cho kiến thức nền: định nghĩa hiện tượng khí hậu, lịch sử thời tiết.
- Dùng TimeSearch khi câu hỏi đề cập đến "hôm nay", "bây giờ", "hôm qua".
- Không bịa đặt dữ liệu. Nếu không tìm được, hãy nói thật.
- Final Answer phải bằng tiếng Việt, rõ ràng, dễ hiểu.
- Với thời tiết: luôn nêu rõ nhiệt độ, độ ẩm, mô tả bầu trời, gợi ý trang phục/hoạt động.

- QUAN TRỌNG: Sau khi viết Action, DỪNG LẠI NGAY. 
  KHÔNG tự viết Observation. Hệ thống sẽ tự động điền Observation.
- KHÔNG bao giờ bịa dữ liệu thời tiết hay ngày tháng từ bộ nhớ.
"""
    def _detect_long_forecast(self, text: str) -> bool:
        match = re.search(r'(\d+)\s*ngày', text)
        if match:
            days = int(match.group(1))
            return days > 5
        return False

    def run(self, user_input: str) -> str:
        is_long_forecast = self._detect_long_forecast(user_input)
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        assistant_scratchpad = ""
        steps = 0
        final_answer = None

        while steps < self.max_steps:
            current_prompt = user_input
            if assistant_scratchpad:
                current_prompt = f"{user_input}\n\n{assistant_scratchpad}"

            result = self.llm.generate(
                prompt=current_prompt,
                system_prompt=self.get_system_prompt(),
                stop=["Observation:"]  # Cắt ngay trước khi LLM bịa Observation
            )

            response_text = result["content"] if isinstance(result, dict) else result
            logger.log_event("LLM_RESPONSE", {"step": steps, "response": response_text})

            # ✅ Kiểm tra Action TRƯỚC
            action_match = re.search(
                r'Action:\s*(\w+)\s*\(\s*["\']([^"\']+)["\']\s*\)',
                response_text,
                re.DOTALL
            )

            if action_match:
                tool_name = action_match.group(1).strip()
                tool_args = action_match.group(2).strip()

                logger.log_event("TOOL_CALL", {"tool": tool_name, "args": tool_args})
                observation = self._execute_tool(tool_name, tool_args)

                assistant_scratchpad += f"\n{response_text}\nObservation: {observation}\n"
                logger.log_event("OBSERVATION", {"step": steps, "result": str(observation)[:500]})
                steps += 1
                continue  # ← Quay lại vòng lặp, gọi LLM tiếp

            # ✅ Chỉ kiểm tra Final Answer khi KHÔNG có Action
            final_match = re.search(r"Final Answer:\s*(.+)", response_text, re.DOTALL)
            if final_match:
                final_answer = final_match.group(1).strip()
                logger.log_event("FINAL_ANSWER_FOUND", {"answer": final_answer})
                break

            # Không có Action, không có Final Answer → trả lời trực tiếp
            final_answer = response_text.strip()
            logger.log_event("DIRECT_ANSWER", {"answer": final_answer})
            break

            # Parse Action: tool_name("query") hoặc tool_name('query')
            action_match = re.search(
                r'Action:\s*(\w+)\s*\(\s*["\']([^"\']+)["\']\s*\)',
                response_text,
                re.DOTALL
            )

            if action_match:
                tool_name = action_match.group(1).strip()
                tool_args = action_match.group(2).strip()

                logger.log_event("TOOL_CALL", {"tool": tool_name, "args": tool_args})

                observation = self._execute_tool(tool_name, tool_args)

                # Tích lũy scratchpad
                assistant_scratchpad += f"\n{response_text}\nObservation: {observation}\n"

                logger.log_event("OBSERVATION", {"step": steps, "result": str(observation)[:500]})
            else:
                # Không có Action → trả lời trực tiếp
                final_answer = response_text.strip()
                logger.log_event("DIRECT_ANSWER", {"answer": final_answer})
                break

            steps += 1

        if not final_answer:
            final_answer = "Đã đạt giới hạn số bước suy luận mà không có câu trả lời cuối cùng."
            logger.log_event("MAX_STEPS_REACHED", {"steps": steps})

        if is_long_forecast:
            final_answer += "\n\n⚠️ Lưu ý: Hiện tại hệ thống chỉ hỗ trợ dự báo tối đa 5 ngày."

        self.history.append({"user": user_input, "agent": final_answer})
        logger.log_event("AGENT_END", {"steps": steps, "answer": final_answer})
        return final_answer
    def _execute_tool(self, tool_name: str, args: str) -> str:
        """Tìm và gọi tool theo tên, truyền args dưới dạng chuỗi query."""
        for tool in self.tools:
            if tool["name"] == tool_name:
                try:
                    func = tool.get("func")
                    if func is None:
                        return f"Tool '{tool_name}' không có hàm thực thi."

                    # Gọi tool với query string
                    result = func(args)

                    # Nếu kết quả là list Document (LangChain), trích xuất text
                    if isinstance(result, list):
                        texts = []
                        for doc in result:
                            source = (
                                doc.metadata.get("source")
                                or doc.metadata.get("title")
                                or "N/A"
                            )
                            texts.append(f"[{source}]: {doc.page_content[:500]}")
                        return "\n\n".join(texts) if texts else "Không có kết quả."

                    return str(result)

                except Exception as e:
                    logger.log_event("TOOL_ERROR", {"tool": tool_name, "error": str(e)})
                    return f"Lỗi khi gọi tool '{tool_name}': {e}"

        return f"Không tìm thấy tool '{tool_name}'."
    
