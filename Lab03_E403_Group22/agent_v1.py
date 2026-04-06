import re
from typing import List, Dict, Any, Optional, Tuple
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    """
    Working v1 agent:
    - ReAct loop
    - single action parsing
    - tool execution
    - final answer stop condition

    Updated to support both:
    1) dict-based tools:
       {"name": "...", "description": "...", "func": callable}
    2) LangChain-style tool objects:
       tool.name, tool.description, tool.invoke(...)
    """

    ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_][\w]*)\((.*?)\)", re.DOTALL)
    FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)

    def __init__(self, llm: LLMProvider, tools: List[Any], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []

    def _tool_name(self, tool: Any) -> str:
        if isinstance(tool, dict):
            return str(tool.get("name", "")).strip()
        return str(getattr(tool, "name", "")).strip()

    def _tool_description(self, tool: Any) -> str:
        if isinstance(tool, dict):
            return str(tool.get("description", "")).strip()
        return str(getattr(tool, "description", "")).strip()

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            [f"- {self._tool_name(t)}: {self._tool_description(t)}" for t in self.tools]
        )
        return f"""
You are an intelligent assistant that follows the ReAct pattern.

Available tools:
{tool_descriptions}

Rules:
1. Think step by step.
2. If a tool is needed, output exactly one Action line using this format:
   Action: tool_name(arguments)
3. After seeing an Observation, continue reasoning.
4. When you have enough information, output:
   Final Answer: <your answer>
5. Do not invent tool names outside the tool list.
""".strip()

    def run(self, user_input: str) -> str:
        logger.log_event(
            "AGENT_START",
            {"input": user_input, "model": self.llm.model_name, "version": "v1"},
        )
        scratchpad = ""
        steps = 0
        final_answer: Optional[str] = None

        while steps < self.max_steps:
            prompt = self._build_prompt(user_input, scratchpad)
            result = self.llm.generate(prompt, system_prompt=self.get_system_prompt())
            content = (result.get("content") or "").strip()
            usage = result.get("usage", {}) or {}
            latency_ms = result.get("latency_ms", 0)
            provider = result.get("provider", getattr(self.llm, "provider_name", "unknown"))
            tracker.track_request(
                provider=provider,
                model=self.llm.model_name,
                usage=usage,
                latency_ms=latency_ms,
            )

            logger.log_event(
                "AGENT_STEP",
                {
                    "step": steps + 1,
                    "prompt": prompt,
                    "response": content,
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "version": "v1",
                },
            )

            final_answer = self._extract_final_answer(content)
            if final_answer:
                logger.log_event(
                    "AGENT_FINAL",
                    {"step": steps + 1, "answer": final_answer, "version": "v1"},
                )
                break

            action = self._extract_action(content)
            if not action:
                logger.log_event(
                    "AGENT_PARSE_ERROR",
                    {"step": steps + 1, "response": content, "version": "v1"},
                )
                final_answer = "I could not parse a valid action or final answer."
                break

            tool_name, args = action
            observation = self._execute_tool(tool_name, args)
            scratchpad += f"\n{content}\nObservation: {observation}\n"
            self.history.append(
                {
                    "step": steps + 1,
                    "response": content,
                    "tool_name": tool_name,
                    "args": args,
                    "observation": observation,
                }
            )

            logger.log_event(
                "TOOL_EXECUTION",
                {
                    "step": steps + 1,
                    "tool_name": tool_name,
                    "args": args,
                    "observation": observation,
                    "version": "v1",
                },
            )
            steps += 1

        if not final_answer:
            final_answer = "I reached the maximum number of steps before producing a final answer."

        logger.log_event(
            "AGENT_END",
            {"steps": steps, "final_answer": final_answer, "version": "v1"},
        )
        return final_answer

    def _build_prompt(self, user_input: str, scratchpad: str) -> str:
        if scratchpad.strip():
            return (
                f"User Question: {user_input}\n\n"
                f"Previous reasoning:\n{scratchpad}\n"
                "Continue from the latest observation."
            )
        return f"User Question: {user_input}"

    def _extract_final_answer(self, text: str) -> Optional[str]:
        match = self.FINAL_RE.search(text)
        return match.group(1).strip() if match else None

    def _extract_action(self, text: str) -> Optional[Tuple[str, str]]:
        match = self.ACTION_RE.search(text)
        if not match:
            return None
        return match.group(1).strip(), match.group(2).strip()

    def _call_tool_object(self, tool_obj: Any, args: str) -> str:
        # LangChain-style tool object
        if hasattr(tool_obj, "invoke"):
            # First try structured input: {"query": args}
            try:
                return str(tool_obj.invoke({"query": args}))
            except Exception:
                pass

            # Then try plain string input
            try:
                return str(tool_obj.invoke(args))
            except Exception as exc:
                return f"Tool execution error: {exc}"

        return "Tool object is not invokable."

    def _execute_tool(self, tool_name: str, args: str) -> str:
        for tool in self.tools:
            current_name = self._tool_name(tool)
            if current_name != tool_name:
                continue

            if isinstance(tool, dict):
                func = tool.get("func") or tool.get("function") or tool.get("fn")
                invoke_fn = tool.get("invoke")

                try:
                    if callable(func):
                        return str(func(args))
                    if callable(invoke_fn):
                        try:
                            return str(invoke_fn({"query": args}))
                        except Exception:
                            return str(invoke_fn(args))
                    return f"Result of {tool_name}: {args}"
                except Exception as exc:
                    return f"Tool execution error: {exc}"

            return self._call_tool_object(tool, args)

        return f"Tool {tool_name} not found."