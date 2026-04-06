"""
Shared pytest fixtures and utilities for weather agent tests.
"""
import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm_provider import LLMProvider
from src.core.openai_provider import OpenAIProvider
from src.agent.agent import ReActAgent
from src.chatbot.chatbot import Chatbot
from src.agent.tools import InternetSearch, WikiSearch


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file"""
    load_dotenv()


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for unit testing"""
    mock = Mock(spec=LLMProvider)
    mock.model_name = "test-model"
    mock.api_key = "test-key"
    return mock


@pytest.fixture
def mock_openai_provider():
    """Create a mock OpenAI provider with predefined responses"""
    mock = Mock(spec=OpenAIProvider)
    mock.model_name = "gpt-4"
    mock.api_key = "test-key"
    return mock


@pytest.fixture
def weather_tools():
    """Create weather tools for testing"""
    # Mock tool responses for different queries
    def mock_weather_search(query):
        """Mock weather search that returns structured weather data"""
        query_lower = query.lower()
        
        # Hanoi weather
        if "hanoi" in query_lower:
            return [
                Mock(
                    page_content="Hanoi weather: Temperature 28°C, Condition: Sunny, Humidity: 65%, Wind: 10 km/h",
                    metadata={"source": "weather-api.com", "title": "Hanoi Weather"}
                )
            ]
        # Tokyo weather
        elif "tokyo" in query_lower:
            return [
                Mock(
                    page_content="Tokyo weather: Temperature 22°C, Condition: Cloudy, Humidity: 70%, Wind: 15 km/h",
                    metadata={"source": "weather-api.com", "title": "Tokyo Weather"}
                )
            ]
        # London weather
        elif "london" in query_lower:
            return [
                Mock(
                    page_content="London weather: Temperature 15°C, Condition: Rainy, Humidity: 85%, Wind: 20 km/h, Precipitation: 80%",
                    metadata={"source": "weather-api.com", "title": "London Weather"}
                )
            ]
        # New York weather
        elif "new york" in query_lower:
            return [
                Mock(
                    page_content="New York weather: Temperature 18°C, Condition: Partly Cloudy, Humidity: 60%, Wind: 12 km/h",
                    metadata={"source": "weather-api.com", "title": "New York Weather"}
                )
            ]
        # Paris weather
        elif "paris" in query_lower:
            return [
                Mock(
                    page_content="Paris weather forecast tomorrow: Temperature 20°C, Condition: Sunny, Humidity: 55%, Wind: 8 km/h",
                    metadata={"source": "weather-api.com", "title": "Paris Weather Forecast"}
                )
            ]
        # Sydney weather
        elif "sydney" in query_lower:
            return [
                Mock(
                    page_content="Sydney weather at 3 PM: Temperature 25°C, Condition: Clear, Humidity: 50%, Wind: 18 km/h",
                    metadata={"source": "weather-api.com", "title": "Sydney Weather"}
                )
            ]
        # Ho Chi Minh City weather
        elif "ho chi minh" in query_lower or "saigon" in query_lower:
            return [
                Mock(
                    page_content="Ho Chi Minh City weather: Temperature 32°C, Condition: Hot and Humid, Humidity: 80%, Wind: 5 km/h",
                    metadata={"source": "weather-api.com", "title": "HCMC Weather"}
                )
            ]
        # Da Nang weather
        elif "da nang" in query_lower:
            return [
                Mock(
                    page_content="Da Nang weather: Temperature 30°C, Condition: Partly Sunny, Humidity: 75%, Wind: 10 km/h",
                    metadata={"source": "weather-api.com", "title": "Da Nang Weather"}
                )
            ]
        # Hue weather
        elif "hue" in query_lower:
            return [
                Mock(
                    page_content="Hue weather: Temperature 26°C, Condition: Overcast, Humidity: 85%, Wind: 8 km/h",
                    metadata={"source": "weather-api.com", "title": "Hue Weather"}
                )
            ]
        else:
            return [
                Mock(
                    page_content=f"Weather data for {query}: Temperature 20°C, Condition: Clear, Humidity: 60%",
                    metadata={"source": "weather-api.com", "title": f"{query} Weather"}
                )
            ]
    
    def mock_wiki_search(query):
        """Mock Wikipedia search"""
        return [
            Mock(
                page_content=f"Wikipedia article about {query}. This is general knowledge information.",
                metadata={"source": "wikipedia.org", "title": query}
            )
        ]
    
    return [
        {
            "name": "InternetSearch",
            "description": "Search for current weather information on the internet",
            "func": mock_weather_search
        },
        {
            "name": "WikiSearch",
            "description": "Search Wikipedia for weather and climate information",
            "func": mock_wiki_search
        }
    ]


@pytest.fixture
def mock_tools():
    """Create mock tools without actual API calls"""
    def mock_search(query):
        """Generic mock search"""
        return [
            Mock(
                page_content=f"Search results for: {query}",
                metadata={"source": "mock.com", "title": query}
            )
        ]
    
    return [
        {
            "name": "InternetSearch",
            "description": "Search for information on the internet",
            "func": mock_search
        },
        {
            "name": "WikiSearch",
            "description": "Search Wikipedia",
            "func": mock_search
        }
    ]


@pytest.fixture
def agent_responses():
    """Dictionary of predefined LLM responses for agent testing"""
    return {
        # Test Case 1: Hanoi current weather
        "hanoi_current": {
            "step_1": """Thought: I need to check the current weather in Hanoi.
Action: InternetSearch("current weather in Hanoi Vietnam")
Observation: Hanoi weather: Temperature 28°C, Condition: Sunny, Humidity: 65%, Wind: 10 km/h
Final Answer: Nhiệt độ hiện tại ở Hà Nội là 28°C, trời nắng, độ ẩm 65%, gió 10 km/h."""
        },
        # Test Case 2: Tokyo weather
        "tokyo_current": {
            "step_1": """Thought: I need to check the current weather in Tokyo.
Action: InternetSearch("current weather in Tokyo Japan right now")
Observation: Tokyo weather: Temperature 22°C, Condition: Cloudy, Humidity: 70%, Wind: 15 km/h
Final Answer: Thời tiết hiện tại ở Tokyo là 22°C, nhiều mây, độ ẩm 70%, gió 15 km/h."""
        },
        # Test Case 3: London rain
        "london_rain": {
            "step_1": """Thought: I need to check if it's currently raining in London.
Action: InternetSearch("is it raining in London now")
Observation: London weather: Temperature 15°C, Condition: Rainy, Humidity: 85%, Wind: 20 km/h, Precipitation: 80%
Final Answer: Có, hiện tại trời đang mưa ở London. Lượng mưa 80%, nhiệt độ 15°C và độ ẩm cao 85%."""
        },
        # Test Case 4: New York temperature
        "newyork_temp": {
            "step_1": """Thought: I need to find the temperature in New York.
Action: InternetSearch("temperature in New York")
Observation: New York weather: Temperature 18°C, Condition: Partly Cloudy, Humidity: 60%, Wind: 12 km/h
Final Answer: Nhiệt độ hiện tại ở New York là 18°C."""
        },
        # Test Case 5: Hanoi yesterday
        "hanoi_yesterday": {
            "step_1": """Thought: I need to find historical weather data for Hanoi yesterday.
Action: InternetSearch("weather in Hanoi Vietnam yesterday historical data")
Observation: Hanoi weather yesterday: Temperature 27°C, Condition: Partly Cloudy, Humidity: 70%
Final Answer: Hôm qua ở Hà Nội, nhiệt độ là 27°C, trời nhiều mây, độ ẩm 70%."""
        },
        # Test Case 6: Paris tomorrow
        "paris_tomorrow": {
            "step_1": """Thought: I need to check the weather forecast for Paris tomorrow.
Action: InternetSearch("weather forecast Paris France tomorrow")
Observation: Paris weather forecast tomorrow: Temperature 20°C, Condition: Sunny, Humidity: 55%, Wind: 8 km/h
Final Answer: Dự báo thời tiết Paris ngày mai là 20°C, trời nắng, độ ẩm 55%, gió 8 km/h."""
        },
        # Test Case 7: Sydney at 3 PM
        "sydney_3pm": {
            "step_1": """Thought: I need to find the weather in Sydney at 3 PM today.
Action: InternetSearch("Sydney Australia weather forecast 3 PM today")
Observation: Sydney weather at 3 PM: Temperature 25°C, Condition: Clear, Humidity: 50%, Wind: 18 km/h
Final Answer: Lúc 3 giờ chiều nay ở Sydney, nhiệt độ dự báo là 25°C, trời quang đãng, độ ẩm 50%, gió 18 km/h."""
        },
        # Test Case 8: Umbrella recommendation
        "umbrella_recommendation": {
            "step_1": """Thought: I need to check the weather in Hanoi and look for rain probability.
Action: InternetSearch("current weather Hanoi Vietnam rain forecast")
Observation: Hanoi weather: Temperature 25°C, Condition: Rainy, Humidity: 90%, Wind: 15 km/h, Precipitation: 85%
Final Answer: Hiện tại Hà Nội đang mưa với khả năng mưa 85%. Bạn nên mang theo ô (dù) khi ra ngoài."""
        },
        # Test Case 9: Compare Hanoi and HCMC
        "compare_cities": {
            "step_1": """Thought: I need to get weather for both Hanoi and Ho Chi Minh City.
Action: InternetSearch("current weather Hanoi Vietnam")
Observation: Hanoi weather: Temperature 28°C, Condition: Sunny, Humidity: 65%
Thought: Now I need to get weather for Ho Chi Minh City.
Action: InternetSearch("current weather Ho Chi Minh City Vietnam")
Observation: Ho Chi Minh City weather: Temperature 32°C, Condition: Hot and Humid, Humidity: 80%
Final Answer: So sánh thời tiết hôm nay: Hà Nội 28°C, trời nắng, độ ẩm 65% trong khi TP.HCM 32°C, nóng ẩm, độ ẩm 80%. TP.HCM nóng hơn Hà Nội 4°C."""
        },
        # Test Case 10: Warmest city in Vietnam
        "warmest_vietnam": {
            "step_1": """Thought: I need to check weather in multiple Vietnamese cities to find the warmest.
Action: InternetSearch("current weather Hanoi Vietnam")
Observation: Hanoi weather: Temperature 28°C
Thought: Check Da Nang weather.
Action: InternetSearch("current weather Da Nang Vietnam")
Observation: Da Nang weather: Temperature 30°C
Thought: Check Ho Chi Minh City weather.
Action: InternetSearch("current weather Ho Chi Minh City Vietnam")
Observation: Ho Chi Minh City weather: Temperature 32°C
Thought: Check Hue weather.
Action: InternetSearch("current weather Hue Vietnam")
Observation: Hue weather: Temperature 26°C
Final Answer: Trong các thành phố đã kiểm tra, TP.HCM ấm nhất với 32°C, tiếp theo là Đà Nẵng 30°C, Hà Nội 28°C, và Huế 26°C. Thành phố ấm nhất Việt Nam hiện tại là TP.HCM."""
        },
        # Test Case 11: Invalid location
        "invalid_location": {
            "step_1": """Thought: I need to search for this location but it seems invalid.
Action: InternetSearch("weather in abcxyz123")
Observation: No weather data found for abcxyz123
Final Answer: Xin lỗi, tôi không thể tìm thấy thông tin thời tiết cho địa điểm "abcxyz123". Vui lòng kiểm tra lại tên địa điểm."""
        },
        # Test Case 12: Missing location
        "missing_location": {
            "step_1": "Final Answer: Bạn vui lòng cho tôi biết bạn muốn kiểm tra thời tiết ở địa điểm nào không? Tôi cần biết thành phố hoặc khu vực để cung cấp thông tin chính xác."
        },
        # Test Case 13: Future year query
        "future_year": {
            "step_1": """Thought: This is asking about weather in 2050, which is in the future and cannot be predicted exactly.
Final Answer: Tôi không thể cung cấp thông tin thời tiết chính xác cho năm 2050. Các dự báo thời tiết chỉ chính xác trong vòng 7-10 ngày. Tuy nhiên, dựa trên các mô hình biến đổi khí hậu, nhiệt độ có thể tăng nhẹ trong tương lai, nhưng không thể đưa ra con số cụ thể."""
        }
    }


def create_agent_with_mock_llm(mock_responses, tools, max_steps=5):
    """
    Helper function to create a ReActAgent with a mock LLM.
    
    Args:
        mock_responses: List or dict of responses to return in sequence
        tools: List of tool dictionaries
        max_steps: Maximum steps for the agent
    
    Returns:
        ReActAgent instance with mocked LLM
    """
    mock_llm = Mock(spec=LLMProvider)
    mock_llm.model_name = "test-gpt-4"
    mock_llm.api_key = "test-key"
    
    # Setup mock to return responses sequentially
    if isinstance(mock_responses, list):
        mock_llm.generate.side_effect = mock_responses
    else:
        # If dict, use the first value
        mock_llm.generate.return_value = {
            "content": list(mock_responses.values())[0] if isinstance(mock_responses, dict) else mock_responses,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "latency_ms": 500,
            "provider": "mock"
        }
    
    return ReActAgent(llm=mock_llm, tools=tools, max_steps=max_steps), mock_llm


def create_chatbot_with_mock_llm():
    """
    Helper function to create a Chatbot with a mock LLM.
    
    Returns:
        Chatbot instance with mocked LLM
    """
    mock_llm = Mock(spec=LLMProvider)
    mock_llm.model_name = "test-gpt-4"
    mock_llm.api_key = "test-key"
    
    chatbot = Chatbot(llm=mock_llm)
    return chatbot, mock_llm
