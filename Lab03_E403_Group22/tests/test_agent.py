"""
Comprehensive test suite for ReActAgent weather queries.
Tests all 13 test cases covering basic, time-based, multi-step, and edge cases.
"""
import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm_provider import LLMProvider
from src.agent.agent import ReActAgent
from tests.conftest import create_agent_with_mock_llm, weather_tools


# ============================================================================
# I. BASIC WEATHER QUERIES (Test Cases 1-4)
# ============================================================================

class TestBasicWeatherQueries:
    """Test basic weather retrieval functionality"""
    
    def test_case_1_hanoi_current_weather(self, weather_tools, agent_responses):
        """
        Test Case 1: Check the current weather in Hanoi
        Expected: Temperature and weather condition (sunny, rainy...)
        Focus: Basic weather retrieval
        """
        # Setup mock LLM with expected response
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["hanoi_current"]["step_1"],
            "usage": {"prompt_tokens": 150, "completion_tokens": 80},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        # Create agent
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        
        # Run test
        result = agent.run("Check the current weather in Hanoi.")
        
        # Assertions
        assert result is not None, "Agent should return a result"
        assert isinstance(result, str), "Result should be a string"
        # Should contain temperature info
        assert any(keyword in result.lower() for keyword in ["28°c", "nhiệt độ", "temperature"]), \
            f"Response should contain temperature info. Got: {result}"
        # Should contain weather condition
        assert any(keyword in result.lower() for keyword in ["nắng", "sunny", "mưa", "rainy", "condition"]), \
            f"Response should contain weather condition. Got: {result}"
        
        # Verify LLM was called
        mock_llm.generate.assert_called_once()
    
    def test_case_2_tokyo_current_weather(self, weather_tools, agent_responses):
        """
        Test Case 2: What's the weather like in Tokyo right now?
        Expected: Real-time weather info
        Focus: Different location
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["tokyo_current"]["step_1"],
            "usage": {"prompt_tokens": 150, "completion_tokens": 80},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("What's the weather like in Tokyo right now?")
        
        assert result is not None, "Agent should return a result"
        assert "tokyo" in result.lower(), f"Response should mention Tokyo. Got: {result}"
        # Should contain weather information
        assert any(keyword in result.lower() for keyword in ["22°c", "nhiệt độ", "temperature", "cloudy", "nhiều mây"]), \
            f"Response should contain weather info. Got: {result}"
    
    def test_case_3_london_rain(self, weather_tools, agent_responses):
        """
        Test Case 3: Is it raining in London now?
        Expected: Yes/No + explanation
        Focus: Boolean reasoning from weather data
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["london_rain"]["step_1"],
            "usage": {"prompt_tokens": 150, "completion_tokens": 80},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Is it raining in London now?")
        
        assert result is not None, "Agent should return a result"
        # Should provide yes/no answer with explanation
        assert any(keyword in result.lower() for keyword in ["có", "yes", "không", "no", "đang mưa", "raining"]), \
            f"Response should contain yes/no about rain. Got: {result}"
        # Should mention London
        assert "london" in result.lower(), f"Response should mention London. Got: {result}"
    
    def test_case_4_newyork_temperature(self, weather_tools, agent_responses):
        """
        Test Case 4: Temperature in New York?
        Expected: Numeric temperature + unit
        Focus: Concise response
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["newyork_temp"]["step_1"],
            "usage": {"prompt_tokens": 150, "completion_tokens": 80},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Temperature in New York?")
        
        assert result is not None, "Agent should return a result"
        # Should contain numeric temperature with unit
        assert any(keyword in result for keyword in ["°c", "°f", "celsius", "fahrenheit", "độ"]), \
            f"Response should contain temperature unit. Got: {result}"
        # Should mention New York
        assert "new york" in result.lower(), f"Response should mention New York. Got: {result}"


# ============================================================================
# II. TIME-BASED WEATHER QUERIES (Test Cases 5-7)
# ============================================================================

class TestTimeBasedWeather:
    """Test time-based weather queries with historical and forecast data"""
    
    def test_case_5_hanoi_yesterday(self, weather_tools, agent_responses):
        """
        Test Case 5: What was the weather in Hanoi yesterday?
        Expected: Handle past data
        Focus: Time understanding, Historical data
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["hanoi_yesterday"]["step_1"],
            "usage": {"prompt_tokens": 160, "completion_tokens": 85},
            "latency_ms": 650,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("What was the weather in Hanoi yesterday?")
        
        assert result is not None, "Agent should return a result"
        # Should reference past time
        assert any(keyword in result.lower() for keyword in ["hôm qua", "yesterday", "quá khứ", "past"]), \
            f"Response should reference past time. Got: {result}"
        # Should contain historical weather data
        assert any(keyword in result.lower() for keyword in ["27°c", "nhiệt độ", "temperature"]), \
            f"Response should contain weather data. Got: {result}"
    
    def test_case_6_paris_tomorrow(self, weather_tools, agent_responses):
        """
        Test Case 6: What will the weather be like in Paris tomorrow?
        Expected: Forecast
        Focus: Future prediction
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["paris_tomorrow"]["step_1"],
            "usage": {"prompt_tokens": 160, "completion_tokens": 85},
            "latency_ms": 650,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("What will the weather be like in Paris tomorrow?")
        
        assert result is not None, "Agent should return a result"
        # Should reference future time
        assert any(keyword in result.lower() for keyword in ["ngày mai", "tomorrow", "dự báo", "forecast"]), \
            f"Response should reference future forecast. Got: {result}"
        # Should mention Paris
        assert "paris" in result.lower(), f"Response should mention Paris. Got: {result}"
    
    def test_case_7_sydney_3pm(self, weather_tools, agent_responses):
        """
        Test Case 7: Weather in Sydney at 3 PM today
        Expected: Parse time, return forecast at specific hour
        Focus: Time parsing, Structured query
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["sydney_3pm"]["step_1"],
            "usage": {"prompt_tokens": 170, "completion_tokens": 90},
            "latency_ms": 700,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Weather in Sydney at 3 PM today")
        
        assert result is not None, "Agent should return a result"
        # Should reference specific time
        assert any(keyword in result.lower() for keyword in ["3 pm", "3 giờ", "15:00", "chiều"]), \
            f"Response should reference specific time. Got: {result}"
        # Should mention Sydney
        assert "sydney" in result.lower(), f"Response should mention Sydney. Got: {result}"


# ============================================================================
# III. MULTI-STEP WEATHER SCENARIOS (Test Cases 8-10)
# ============================================================================

class TestMultiStepScenarios:
    """Test complex multi-step weather queries that require reasoning"""
    
    def test_case_8_umbrella_recommendation(self, weather_tools, agent_responses):
        """
        Test Case 8: Check the weather in Hanoi and tell me if I should bring an umbrella.
        Expected Steps: Get weather data, detect rain probability, give recommendation
        Focus: Reasoning + decision making
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["umbrella_recommendation"]["step_1"],
            "usage": {"prompt_tokens": 180, "completion_tokens": 100},
            "latency_ms": 750,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Check the weather in Hanoi and tell me if I should bring an umbrella.")
        
        assert result is not None, "Agent should return a result"
        # Should provide recommendation about umbrella
        assert any(keyword in result.lower() for keyword in ["ô", "dù", "umbrella", "mang", "bring"]), \
            f"Response should mention umbrella recommendation. Got: {result}"
        # Should reference rain or weather condition
        assert any(keyword in result.lower() for keyword in ["mưa", "rain", "nên", "should"]), \
            f"Response should provide reasoning. Got: {result}"
    
    def test_case_9_compare_cities(self, weather_tools, agent_responses):
        """
        Test Case 9: Compare today's weather in Hanoi and Ho Chi Minh City.
        Expected Steps: Fetch 2 locations, compare temperature/condition, summarize
        Focus: Multi-query handling
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        
        # Setup multi-step responses
        responses = [
            {
                "content": """Thought: I need to get weather for both cities.
Action: InternetSearch("current weather Hanoi Vietnam")
Observation: Hanoi weather: Temperature 28°C, Condition: Sunny, Humidity: 65%""",
                "usage": {"prompt_tokens": 150, "completion_tokens": 60},
                "latency_ms": 600,
                "provider": "mock"
            },
            {
                "content": """Thought: Now I need Ho Chi Minh City weather.
Action: InternetSearch("current weather Ho Chi Minh City Vietnam")
Observation: Ho Chi Minh City weather: Temperature 32°C, Condition: Hot and Humid, Humidity: 80%""",
                "usage": {"prompt_tokens": 200, "completion_tokens": 70},
                "latency_ms": 650,
                "provider": "mock"
            },
            {
                "content": """Final Answer: So sánh thời tiết hôm nay: Hà Nội 28°C, trời nắng, độ ẩm 65% trong khi TP.HCM 32°C, nóng ẩm, độ ẩm 80%. TP.HCM nóng hơn Hà Nội 4°C và ẩm hơn 15%.""",
                "usage": {"prompt_tokens": 250, "completion_tokens": 90},
                "latency_ms": 700,
                "provider": "mock"
            }
        ]
        
        # Setup mock to return responses sequentially
        mock_llm.generate.side_effect = responses
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Compare today's weather in Hanoi and Ho Chi Minh City.")
        
        assert result is not None, "Agent should return a result"
        # Should mention both cities
        assert any(keyword in result.lower() for keyword in ["hà nội", "hanoi"]), \
            f"Response should mention Hanoi. Got: {result}"
        assert any(keyword in result.lower() for keyword in ["hồ chí minh", "tp.hcm", "ho chi minh"]), \
            f"Response should mention Ho Chi Minh City. Got: {result}"
        # Should provide comparison
        assert any(keyword in result.lower() for keyword in ["so sánh", "compare", "hơn", "warmer", "cooler"]), \
            f"Response should provide comparison. Got: {result}"
        
        # Verify multiple LLM calls were made
        assert mock_llm.generate.call_count >= 2, \
            f"Should make multiple API calls for comparison. Got: {mock_llm.generate.call_count}"
    
    def test_case_10_warmest_city_vietnam(self, weather_tools, agent_responses):
        """
        Test Case 10: Find the warmest city in Vietnam right now.
        Expected Steps: Query multiple cities, compare temperatures, return best answer
        Focus: Aggregation, Search + ranking
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        
        # Setup multi-step responses for finding warmest city
        responses = [
            {
                "content": """Thought: I need to check multiple Vietnamese cities.
Action: InternetSearch("current weather Hanoi Vietnam")
Observation: Hanoi weather: Temperature 28°C""",
                "usage": {"prompt_tokens": 150, "completion_tokens": 50},
                "latency_ms": 600,
                "provider": "mock"
            },
            {
                "content": """Thought: Check Da Nang.
Action: InternetSearch("current weather Da Nang Vietnam")
Observation: Da Nang weather: Temperature 30°C""",
                "usage": {"prompt_tokens": 180, "completion_tokens": 50},
                "latency_ms": 620,
                "provider": "mock"
            },
            {
                "content": """Thought: Check Ho Chi Minh City.
Action: InternetSearch("current weather Ho Chi Minh City Vietnam")
Observation: Ho Chi Minh City weather: Temperature 32°C""",
                "usage": {"prompt_tokens": 210, "completion_tokens": 50},
                "latency_ms": 640,
                "provider": "mock"
            },
            {
                "content": """Thought: Check Hue.
Action: InternetSearch("current weather Hue Vietnam")
Observation: Hue weather: Temperature 26°C""",
                "usage": {"prompt_tokens": 240, "completion_tokens": 50},
                "latency_ms": 660,
                "provider": "mock"
            },
            {
                "content": """Final Answer: Trong các thành phố đã kiểm tra, TP.HCM ấm nhất với 32°C, tiếp theo là Đà Nẵng 30°C, Hà Nội 28°C, và Huế 26°C. Thành phố ấm nhất Việt Nam hiện tại là TP.HCM.""",
                "usage": {"prompt_tokens": 270, "completion_tokens": 100},
                "latency_ms": 750,
                "provider": "mock"
            }
        ]
        
        mock_llm.generate.side_effect = responses
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=10)
        result = agent.run("Find the warmest city in Vietnam right now.")
        
        assert result is not None, "Agent should return a result"
        # Should identify the warmest city
        assert any(keyword in result.lower() for keyword in ["32°c", "ấm nhất", "warmest", "tp.hcm", "hồ chí minh"]), \
            f"Response should identify warmest city. Got: {result}"
        # Should show temperature data or ranking
        assert any(keyword in result.lower() for keyword in ["nhiệt độ", "temperature", "°c", "ấm", "warm", "hot", "nóng"]), \
            f"Response should show temperature info. Got: {result}"
        
        # Verify multiple LLM calls
        assert mock_llm.generate.call_count >= 3, \
            f"Should query multiple cities. Got: {mock_llm.generate.call_count}"


# ============================================================================
# IV. EDGE CASES (Test Cases 11-13)
# ============================================================================

class TestEdgeCases:
    """Test edge cases for robustness and error handling"""
    
    def test_case_11_invalid_location(self, weather_tools, agent_responses):
        """
        Test Case 11: Weather in abcxyz123
        Expected: Error handling (invalid location)
        Focus: Robustness
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["invalid_location"]["step_1"],
            "usage": {"prompt_tokens": 150, "completion_tokens": 70},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Weather in abcxyz123")
        
        assert result is not None, "Agent should return a result"
        # Should indicate inability to find data
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "không tìm thấy", "not found", 
            "invalid", "không hợp lệ", "error"
        ]), f"Response should indicate error. Got: {result}"
    
    def test_case_12_missing_location(self, weather_tools, agent_responses):
        """
        Test Case 12: Weather?
        Expected: Ask for clarification (location missing)
        Focus: Conversation handling
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["missing_location"]["step_1"],
            "usage": {"prompt_tokens": 100, "completion_tokens": 60},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Weather?")
        
        assert result is not None, "Agent should return a result"
        # Should ask for clarification
        assert any(keyword in result.lower() for keyword in [
            "vui lòng", "please", "cho tôi", "which", "địa điểm",
            "location", "clarif"
        ]), f"Response should ask for clarification. Got: {result}"
    
    def test_case_13_future_year_query(self, weather_tools, agent_responses):
        """
        Test Case 13: Weather in Hanoi in 2050
        Expected: Cannot provide exact data / fallback
        Focus: Unrealistic query handling
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": agent_responses["future_year"]["step_1"],
            "usage": {"prompt_tokens": 150, "completion_tokens": 80},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Weather in Hanoi in 2050")
        
        assert result is not None, "Agent should return a result"
        # Should indicate inability to predict far future
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "không thể dự đoán", "cannot predict",
            "tương lai", "future", "2050"
        ]), f"Response should indicate limitation. Got: {result}"


# ============================================================================
# V. ADDITIONAL AGENT BEHAVIOR TESTS
# ============================================================================

class TestAgentBehavior:
    """Test agent behavior characteristics like max_steps, tool usage, etc."""
    
    def test_agent_respects_max_steps(self, weather_tools):
        """Test that agent stops after reaching max_steps"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        
        # Always return Action without Final Answer
        mock_llm.generate.return_value = {
            "content": """Thought: I need more information.
Action: InternetSearch("test query")
Observation: Test result""",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        max_steps = 3
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=max_steps)
        result = agent.run("Test query")
        
        # Should stop after max_steps
        assert mock_llm.generate.call_count <= max_steps, \
            f"Should not exceed max_steps. Called: {mock_llm.generate.call_count}"
        # Should return something even if no final answer found
        assert result is not None, "Agent should always return something"
    
    def test_agent_tool_executionution(self, weather_tools):
        """Test that agent properly executes tools"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        
        responses = [
            {
                "content": """Thought: I need to search for this.
Action: InternetSearch("test query")""",
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                "latency_ms": 500,
                "provider": "mock"
            },
            {
                "content": """Final Answer: Based on the search results, here is the answer.""",
                "usage": {"prompt_tokens": 150, "completion_tokens": 50},
                "latency_ms": 600,
                "provider": "mock"
            }
        ]
        
        mock_llm.generate.side_effect = responses
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Test query")
        
        # Should execute tool at least once
        assert mock_llm.generate.call_count >= 2, \
            "Should make multiple calls (action + final answer)"
        assert result is not None, "Should return final answer"
    
    def test_agent_direct_answer_without_tools(self, weather_tools):
        """Test that agent can answer without using tools"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Final Answer: This is a direct answer without using tools.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        result = agent.run("Simple question")
        
        assert result == "This is a direct answer without using tools.", \
            f"Should return direct answer. Got: {result}"
        mock_llm.generate.assert_called_once()
    
    def test_agent_history_tracking(self, weather_tools):
        """Test that agent tracks conversation history"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Final Answer: Test response",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        agent = ReActAgent(llm=mock_llm, tools=weather_tools, max_steps=5)
        
        # Make multiple queries
        agent.run("Query 1")
        agent.run("Query 2")
        agent.run("Query 3")
        
        # Should track history
        assert len(agent.history) == 3, \
            f"Should track all queries in history. Got: {len(agent.history)}"
        assert agent.history[0]["user"] == "Query 1", "First history entry should match"
        assert agent.history[2]["user"] == "Query 3", "Last history entry should match"


# ============================================================================
# VI. INTEGRATION TESTS (Requires actual API keys)
# ============================================================================

class TestIntegrationWithRealAPI:
    """Integration tests that use real API keys if available"""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set, skipping integration test"
    )
    def test_agent_with_real_openai(self, weather_tools):
        """Test agent with real OpenAI provider (uses actual API)"""
        from src.core.openai_provider import OpenAIProvider
        
        # This test will only run if OPENAI_API_KEY is set
        llm = OpenAIProvider(
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        agent = ReActAgent(llm=llm, tools=weather_tools, max_steps=3)
        result = agent.run("What is the capital of Vietnam?")
        
        assert result is not None, "Should return a result"
        assert len(result) > 0, "Result should not be empty"
        assert "hà nội" in result.lower() or "hanoi" in result.lower(), \
            f"Should mention Hanoi. Got: {result}"
    
    @pytest.mark.skipif(
        not os.getenv("TAVILY_API_KEY"),
        reason="TAVILY_API_KEY not set, skipping integration test"
    )
    def test_agent_with_real_search_tools(self):
        """Test agent with real search tools"""
        from src.core.openai_provider import OpenAIProvider
        from src.agent.tools import InternetSearch, WikiSearch
        
        # Skip if no API keys
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        llm = OpenAIProvider(
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        tools = [
            {
                "name": "InternetSearch",
                "description": "Search the internet",
                "func": lambda q: InternetSearch.invoke({"query": q, "k": 3})
            },
            {
                "name": "WikiSearch",
                "description": "Search Wikipedia",
                "func": lambda q: WikiSearch.invoke({"query": q, "k": 2})
            }
        ]
        
        agent = ReActAgent(llm=llm, tools=tools, max_steps=3)
        result = agent.run("What is the current weather in Hanoi?")
        
        assert result is not None, "Should return a result"
        assert len(result) > 0, "Result should not be empty"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run with: pytest tests/test_agent.py -v
    pytest.main([__file__, "-v"])
