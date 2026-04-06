import re
from typing import List, Dict, Any, Optional, Tuple
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    """
    Improved v2 agent:
    - stricter prompt
    - markdown fence cleanup
    - action validation against known tools
    - repeated-action guardrail
    - better failure logging
    - cleaner stop behavior

    Updated to support both:
    1) dict-based tools:
       {"name": "...", "description": "...", "func": callable}
    2) LangChain-style tool objects:
       tool.name, tool.description, tool.invoke(...)
    """

    ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_][\w]*)\((.*?)\)", re.DOTALL)
    FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
    THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?:\nAction:|\nFinal Answer:|$)", re.DOTALL)

    def __init__(self, llm: LLMProvider, tools: List[Any], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.tool_names = {self._tool_name(tool) for tool in tools if self._tool_name(tool)}

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

You must follow these rules exactly:
1. Output either:
   Thought: <reasoning>
   Action: tool_name(arguments)
   OR
   Thought: <reasoning>
   Final Answer: <answer>
2. Use only one Action per response.
3. Use only the tool names listed above.
4. Keep arguments concise and explicit.
5. After an Observation, reason using the new information instead of repeating the same action.
6. Do not wrap the answer in markdown code fences.
""".strip()

    def run(self, user_input: str) -> str:
        logger.log_event(
            "AGENT_START",
            {"input": user_input, "model": self.llm.model_name, "version": "v2"},
        )
        scratchpad = ""
        steps = 0
        final_answer: Optional[str] = None
        previous_action: Optional[Tuple[str, str]] = None

        while steps < self.max_steps:
            prompt = self._build_prompt(user_input, scratchpad)
            result = self.llm.generate(prompt, system_prompt=self.get_system_prompt())
            raw_content = (result.get("content") or "").strip()
            content = self._normalize_response(raw_content)
            usage = result.get("usage", {}) or {}
            latency_ms = result.get("latency_ms", 0)
            provider = result.get("provider", getattr(self.llm, "provider_name", "unknown"))
            tracker.track_request(
                provider=provider,
                model=self.llm.model_name,
                usage=usage,
                latency_ms=latency_ms,
            )

            thought = self._extract_thought(content)
            logger.log_event(
                "AGENT_STEP",
                {
                    "step": steps + 1,
                    "prompt": prompt,
                    "raw_response": raw_content,
                    "normalized_response": content,
                    "thought": thought,
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "version": "v2",
                },
            )

            final_answer = self._extract_final_answer(content)
            if final_answer:
                logger.log_event(
                    "AGENT_FINAL",
                    {"step": steps + 1, "answer": final_answer, "version": "v2"},
                )
                break

            action = self._extract_action(content)
            if not action:
                logger.log_event(
                    "AGENT_PARSE_ERROR",
                    {
                        "step": steps + 1,
                        "response": content,
                        "reason": "missing_action_and_final",
                        "version": "v2",
                    },
                )
                final_answer = "I could not parse a valid action or final answer."
                break

            tool_name, args = action
            if tool_name not in self.tool_names:
                logger.log_event(
                    "INVALID_TOOL",
                    {
                        "step": steps + 1,
                        "tool_name": tool_name,
                        "allowed_tools": sorted(self.tool_names),
                        "version": "v2",
                    },
                )
                final_answer = f"The model selected an unknown tool: {tool_name}."
                break

            if previous_action == action:
                logger.log_event(
                    "REPEATED_ACTION_STOP",
                    {
                        "step": steps + 1,
                        "tool_name": tool_name,
                        "args": args,
                        "version": "v2",
                    },
                )
                final_answer = "I stopped because the model repeated the same action without making progress."
                break

            observation = self._execute_tool(tool_name, args)
            scratchpad += (
                f"\nThought: {thought or 'No explicit thought provided.'}\n"
                f"Action: {tool_name}({args})\n"
                f"Observation: {observation}\n"
            )
            self.history.append(
                {
                    "step": steps + 1,
                    "thought": thought,
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
                    "version": "v2",
                },
            )
            previous_action = action
            steps += 1

        if not final_answer:
            final_answer = "I reached the maximum number of steps before producing a final answer."
            logger.log_event("MAX_STEPS_REACHED", {"steps": self.max_steps, "version": "v2"})

        logger.log_event(
            "AGENT_END",
            {"steps": steps, "final_answer": final_answer, "version": "v2"},
        )
        return final_answer

    def _build_prompt(self, user_input: str, scratchpad: str) -> str:
        if scratchpad.strip():
            return (
                f"User Question: {user_input}\n\n"
                f"Scratchpad so far:\n{scratchpad}\n"
                "Continue from the latest Observation. If enough information is available, produce Final Answer."
            )
        return f"User Question: {user_input}"

    def _normalize_response(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _extract_thought(self, text: str) -> Optional[str]:
        match = self.THOUGHT_RE.search(text)
        return match.group(1).strip() if match else None

    def _extract_final_answer(self, text: str) -> Optional[str]:
        match = self.FINAL_RE.search(text)
        return match.group(1).strip() if match else None

    def _extract_action(self, text: str) -> Optional[Tuple[str, str]]:
        match = self.ACTION_RE.search(text)
        if not match:
            return None
        return match.group(1).strip(), match.group(2).strip()

    def _call_tool_object(self, tool_obj: Any, args: str, tool_name: str) -> str:
        if hasattr(tool_obj, "invoke"):
            try:
                return str(tool_obj.invoke({"query": args}))
            except Exception:
                pass

            try:
                return str(tool_obj.invoke(args))
            except Exception as exc:
                logger.log_event(
                    "TOOL_ERROR",
                    {
                        "tool_name": tool_name,
                        "args": args,
                        "error": str(exc),
                        "version": "v2",
                    },
                )
                return f"Tool execution error: {exc}"

        logger.log_event(
            "TOOL_ERROR",
            {
                "tool_name": tool_name,
                "args": args,
                "error": "Tool object is not invokable.",
                "version": "v2",
            },
        )
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
                    logger.log_event(
                        "TOOL_ERROR",
                        {
                            "tool_name": tool_name,
                            "args": args,
                            "error": str(exc),
                            "version": "v2",
                        },
                    )
                    return f"Tool execution error: {exc}"

            return self._call_tool_object(tool, args, tool_name)

        return f"Tool {tool_name} not found."