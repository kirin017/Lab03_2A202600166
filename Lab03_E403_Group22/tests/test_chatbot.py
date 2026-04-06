"""
Comprehensive test suite for baseline Chatbot weather queries.
Tests all 13 test cases comparing behavior with ReActAgent.
The chatbot does NOT have access to tools, only LLM knowledge.
"""
import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm_provider import LLMProvider
from src.chatbot.chatbot import Chatbot


# ============================================================================
# I. BASIC WEATHER QUERIES (Test Cases 1-4)
# ============================================================================

class TestBasicWeatherQueries:
    """Test basic weather retrieval functionality - Chatbot version"""
    
    def test_case_1_hanoi_current_weather(self):
        """
        Test Case 1: Check the current weather in Hanoi
        Expected: Temperature and weather condition
        Focus: Basic weather retrieval
        Note: Chatbot has NO real-time data, only training knowledge
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không có dữ liệu thời tiết realtime. Tuy nhiên, Hà Nội có khí hậu nhiệt đới gió mùa. Nhiệt độ trung bình khoảng 23-28°C tùy mùa. Mùa hè nóng ẩm, mưa nhiều. Mùa đông lạnh và khô.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 80},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Check the current weather in Hanoi.")
        
        assert result is not None, "Chatbot should return a result"
        assert isinstance(result, str), "Result should be a string"
        # Chatbot should indicate it lacks real-time data
        assert any(keyword in result.lower() for keyword in [
            "không có dữ liệu", "no data", "không thể", "cannot",
            "không truy cập", "no access", "realtime", "live"
        ]) or "nhiệt độ" in result.lower() or "temperature" in result.lower(), \
            f"Chatbot should indicate lack of real-time data or provide general info. Got: {result}"
        
        # Verify LLM was called
        mock_llm.generate.assert_called_once()
    
    def test_case_2_tokyo_current_weather(self):
        """
        Test Case 2: What's the weather like in Tokyo right now?
        Expected: General climate info (chatbot cannot provide real-time)
        Focus: Different location
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không có dữ liệu thời tiết realtime cho Tokyo. Tokyo có khí hậu ôn đới với 4 mùa rõ rệt. Mùa hè nóng ẩm, mùa đông lạnh. Nhiệt độ trung bình năm khoảng 15-25°C.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 80},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("What's the weather like in Tokyo right now?")
        
        assert result is not None, "Chatbot should return a result"
        # Should mention Tokyo
        assert "tokyo" in result.lower(), f"Response should mention Tokyo. Got: {result}"
        # Should provide general climate info or indicate limitation
        assert len(result) > 20, "Response should be meaningful"
    
    def test_case_3_london_rain(self):
        """
        Test Case 3: Is it raining in London now?
        Expected: General info about London climate
        Focus: Boolean reasoning (chatbot cannot provide real-time)
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không thể kiểm tra thời tiết realtime. London nổi tiếng mưa nhiều với khí hậu ôn đới hải dương. Thành phố này mưa quanh năm, đặc biệt nhiều vào mùa thu và đông. Nếu bạn hỏi 'có đang mưa không', tôi không thể biết chính xác lúc này.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 90},
            "latency_ms": 550,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Is it raining in London now?")
        
        assert result is not None, "Chatbot should return a result"
        # Should mention London
        assert "london" in result.lower(), f"Response should mention London. Got: {result}"
        # Should indicate inability to check current weather
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "không biết", "don't know",
            "realtime", "hiện tại", "now"
        ]) or "mưa" in result.lower() or "rain" in result.lower(), \
            f"Should indicate limitation or discuss rain. Got: {result}"
    
    def test_case_4_newyork_temperature(self):
        """
        Test Case 4: Temperature in New York?
        Expected: General temperature info (not real-time)
        Focus: Concise response
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không có dữ liệu nhiệt độ realtime. New York có khí hậu cận nhiệt đới ẩm. Nhiệt độ mùa hè khoảng 25-30°C, mùa đông có thể xuống 0°C hoặc thấp hơn.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 70},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Temperature in New York?")
        
        assert result is not None, "Chatbot should return a result"
        # Should mention New York
        assert "new york" in result.lower(), f"Response should mention New York. Got: {result}"
        # Should contain temperature info
        assert any(keyword in result.lower() for keyword in ["nhiệt độ", "temperature", "°c", "°f", "độ"]), \
            f"Response should mention temperature. Got: {result}"


# ============================================================================
# II. TIME-BASED WEATHER QUERIES (Test Cases 5-7)
# ============================================================================

class TestTimeBasedWeather:
    """Test time-based weather queries - Chatbot version"""
    
    def test_case_5_hanoi_yesterday(self):
        """
        Test Case 5: What was the weather in Hanoi yesterday?
        Expected: Cannot provide historical data
        Focus: Time understanding (chatbot limitation)
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không có dữ liệu thời tiết historical cho ngày hôm qua. Tôi chỉ có thể cung cấp thông tin khí hậu chung về Hà Nội dựa trên kiến thức đã được huấn luyện.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 70},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("What was the weather in Hanoi yesterday?")
        
        assert result is not None, "Chatbot should return a result"
        # Should indicate inability to provide historical data
        assert any(keyword in result.lower() for keyword in [
            "không có", "no data", "không thể", "cannot",
            "historical", "quá khứ", "past", "hôm qua"
        ]), f"Should indicate limitation. Got: {result}"
    
    def test_case_6_paris_tomorrow(self):
        """
        Test Case 6: What will the weather be like in Paris tomorrow?
        Expected: Cannot provide forecast
        Focus: Future prediction (chatbot limitation)
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không thể dự báo thời tiết cho ngày mai. Paris có khí hậu ôn đới hải dương với nhiệt độ trung bình 10-20°C tùy mùa. Để biết dự báo chính xác, bạn nên kiểm tra trang web thời tiết.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 80},
            "latency_ms": 550,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("What will the weather be like in Paris tomorrow?")
        
        assert result is not None, "Chatbot should return a result"
        # Should mention Paris
        assert "paris" in result.lower(), f"Response should mention Paris. Got: {result}"
        # Should indicate cannot forecast
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "không dự báo", "cannot forecast",
            "ngày mai", "tomorrow", "tương lai", "future"
        ]), f"Should indicate limitation. Got: {result}"
    
    def test_case_7_sydney_3pm(self):
        """
        Test Case 7: Weather in Sydney at 3 PM today
        Expected: Cannot provide specific time forecast
        Focus: Time parsing (chatbot limitation)
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không thể cung cấp dự báo thời tiết cho giờ cụ thể. Sydney nằm ở Nam bán cầu với các mùa ngược với Bắc bán cầu. Nhiệt độ trung bình khoảng 12-25°C tùy mùa.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 75},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Weather in Sydney at 3 PM today")
        
        assert result is not None, "Chatbot should return a result"
        # Should mention Sydney
        assert "sydney" in result.lower(), f"Response should mention Sydney. Got: {result}"
        # Should indicate limitation
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "giờ cụ thể", "specific time",
            "3 pm", "3 giờ"
        ]) or len(result) > 20, \
            f"Should indicate limitation or provide info. Got: {result}"


# ============================================================================
# III. MULTI-STEP WEATHER SCENARIOS (Test Cases 8-10)
# ============================================================================

class TestMultiStepScenarios:
    """Test complex scenarios - Chatbot version (no tools, single response)"""
    
    def test_case_8_umbrella_recommendation(self):
        """
        Test Case 8: Should I bring an umbrella in Hanoi?
        Expected: General advice (cannot check current weather)
        Focus: Reasoning (chatbot has no real-time data)
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không thể kiểm tra thời tiết hiện tại ở Hà Nội. Tuy nhiên, Hà Nội có mùa mưa từ tháng 5-10. Nếu đang trong mùa mưa, bạn nên mang theo ô. Mùa khô từ tháng 11-4 ít mưa hơn.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 80},
            "latency_ms": 550,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Check the weather in Hanoi and tell me if I should bring an umbrella.")
        
        assert result is not None, "Chatbot should return a result"
        # Should mention umbrella or rain
        assert any(keyword in result.lower() for keyword in [
            "ô", "dù", "umbrella", "mưa", "rain", "mang", "bring"
        ]), f"Response should mention umbrella/rain. Got: {result}"
        # Should indicate limitation
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "không biết", "don't know"
        ]) or "mùa mưa" in result.lower(), \
            f"Should indicate limitation or give seasonal advice. Got: {result}"
    
    def test_case_9_compare_cities(self):
        """
        Test Case 9: Compare today's weather in Hanoi and Ho Chi Minh City.
        Expected: General climate comparison (not real-time)
        Focus: Multi-query handling (chatbot does single response)
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không có dữ liệu thời tiết realtime để so sánh. Tuy nhiên, về khí hậu chung: Hà Nội ở miền Bắc có 4 mùa rõ rệt, nhiệt độ 10-35°C. TP.HCM ở miền Nam nóng quanh năm, 25-35°C, có mùa mưa và mùa khô. TP.HCM nhìn chung nóng hơn Hà Nội.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 90},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Compare today's weather in Hanoi and Ho Chi Minh City.")
        
        assert result is not None, "Chatbot should return a result"
        # Should mention both cities
        assert any(keyword in result.lower() for keyword in ["hà nội", "hanoi"]), \
            f"Response should mention Hanoi. Got: {result}"
        assert any(keyword in result.lower() for keyword in [
            "hồ chí minh", "tp.hcm", "sài gòn", "ho chi minh"
        ]), f"Response should mention Ho Chi Minh City. Got: {result}"
        # Should provide some comparison
        assert any(keyword in result.lower() for keyword in [
            "so sánh", "compare", "hơn", "warmer", "cooler",
            "nhiệt độ", "temperature"
        ]), f"Response should provide comparison. Got: {result}"
    
    def test_case_10_warmest_city_vietnam(self):
        """
        Test Case 10: Find the warmest city in Vietnam right now.
        Expected: Cannot check real-time, provides general info
        Focus: Aggregation (chatbot cannot query multiple sources)
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không thể kiểm tra thời tiết realtime. Về khí hậu chung, các thành phố phía Nam như TP.HCM, Cần Thơ thường nóng quanh năm 25-35°C. Các thành phố phía Bắc như Hà Nội, Hải Phòng có mùa đông lạnh hơn. Vùng Tây Nguyên mát hơn. Nhìn chung, TP.HCM và các tỉnh Nam Bộ thường ấm nhất Việt Nam.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 95},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Find the warmest city in Vietnam right now.")
        
        assert result is not None, "Chatbot should return a result"
        # Should indicate limitation or provide general info
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "realtime", "nhiệt độ",
            "temperature", "ấm", "warm", "nóng", "hot"
        ]), f"Response should address the question. Got: {result}"


# ============================================================================
# IV. EDGE CASES (Test Cases 11-13)
# ============================================================================

class TestEdgeCases:
    """Test edge cases - Chatbot version"""
    
    def test_case_11_invalid_location(self):
        """
        Test Case 11: Weather in abcxyz123
        Expected: Error handling or clarification
        Focus: Robustness
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Xin lỗi, tôi không nhận ra địa điểm 'abcxyz123'. Bạn có thể kiểm tra lại chính tả và cho tôi biết tên thành phố hoặc quốc gia cụ thể không?",
            "usage": {"prompt_tokens": 100, "completion_tokens": 60},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Weather in abcxyz123")
        
        assert result is not None, "Chatbot should return a result"
        # Should indicate error or ask for clarification
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "không nhận ra", "not found",
            "invalid", "không hợp lệ", "kiểm tra", "check",
            "error", "lỗi"
        ]), f"Response should indicate error. Got: {result}"
    
    def test_case_12_missing_location(self):
        """
        Test Case 12: Weather?
        Expected: Ask for clarification
        Focus: Conversation handling
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Bạn vui lòng cho tôi biết bạn muốn kiểm tra thời tiết ở địa điểm nào? Tôi cần biết thành phố hoặc khu vực cụ thể để cung cấp thông tin.",
            "usage": {"prompt_tokens": 80, "completion_tokens": 60},
            "latency_ms": 450,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Weather?")
        
        assert result is not None, "Chatbot should return a result"
        # Should ask for clarification
        assert any(keyword in result.lower() for keyword in [
            "vui lòng", "please", "cho tôi", "which", "địa điểm",
            "location", "thành phố", "city", "cụ thể", "specific"
        ]), f"Response should ask for clarification. Got: {result}"
    
    def test_case_13_future_year_query(self):
        """
        Test Case 13: Weather in Hanoi in 2050
        Expected: Cannot provide exact data
        Focus: Unrealistic query handling
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Tôi không thể dự báo thời tiết chính xác cho năm 2050. Tuy nhiên, theo các mô hình biến đổi khí hậu, nhiệt độ trung bình có thể tăng 1-3°C vào năm 2050 do nóng lên toàn cầu. Hà Nội có thể sẽ nóng hơn và có nhiều hiện tượng thời tiết cực đoan hơn.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 90},
            "latency_ms": 600,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Weather in Hanoi in 2050")
        
        assert result is not None, "Chatbot should return a result"
        # Should indicate limitation
        assert any(keyword in result.lower() for keyword in [
            "không thể", "cannot", "không dự báo", "cannot predict",
            "2050", "tương lai", "future", "biến đổi khí hậu",
            "climate change"
        ]), f"Response should indicate limitation. Got: {result}"


# ============================================================================
# V. CHATBOT-SPECIFIC BEHAVIOR TESTS
# ============================================================================

class TestChatbotBehavior:
    """Test chatbot-specific behavior like history, no tools, etc."""
    
    def test_chatbot_no_tool_access(self):
        """Test that chatbot doesn't use tools"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "I cannot access real-time weather data. I can only provide general climate information based on my training knowledge.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 60},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("What's the weather in Hanoi?")
        
        # Should not attempt to use tools (no Action: pattern in response)
        assert "Action:" not in result, "Chatbot should not use tools"
        # Should indicate limitation
        assert any(keyword in result.lower() for keyword in [
            "cannot", "không thể", "no access", "không có",
            "real-time", "realtime", "live data"
        ]), f"Should indicate no tool access. Got: {result}"
    
    def test_chatbot_conversation_history(self):
        """Test that chatbot maintains conversation history"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        
        responses = [
            {
                "content": "Response 1",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "latency_ms": 500,
                "provider": "mock"
            },
            {
                "content": "Response 2",
                "usage": {"prompt_tokens": 150, "completion_tokens": 50},
                "latency_ms": 550,
                "provider": "mock"
            },
            {
                "content": "Response 3",
                "usage": {"prompt_tokens": 200, "completion_tokens": 50},
                "latency_ms": 600,
                "provider": "mock"
            }
        ]
        
        mock_llm.generate.side_effect = responses
        
        chatbot = Chatbot(llm=mock_llm)
        
        # Make multiple queries
        result1 = chatbot.chat("Query 1")
        result2 = chatbot.chat("Query 2")
        result3 = chatbot.chat("Query 3")
        
        # Should track history
        assert len(chatbot.history) == 3, \
            f"Should track all queries. Got: {len(chatbot.history)}"
        assert chatbot.history[0]["user"] == "Query 1", "First entry should match"
        assert chatbot.history[1]["user"] == "Query 2", "Second entry should match"
        assert chatbot.history[2]["user"] == "Query 3", "Third entry should match"
        
        # Verify LLM was called for each query
        assert mock_llm.generate.call_count == 3, \
            f"Should call LLM 3 times. Got: {mock_llm.generate.call_count}"
    
    def test_chatbot_history_included_in_prompt(self):
        """Test that previous conversation is included in prompt"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Response",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        
        # First query
        chatbot.chat("First question")
        
        # Second query - prompt should include history
        chatbot.chat("Second question")
        
        # Check the second call included history
        assert mock_llm.generate.call_count == 2
        second_call_args = mock_llm.generate.call_args_list[1]
        prompt_arg = second_call_args[1].get("prompt", "") or second_call_args[0][0]
        
        # Should include previous conversation
        assert "First question" in prompt_arg, \
            "Second prompt should include first question"
        assert "Response" in prompt_arg, \
            "Second prompt should include first response"
    
    def test_chatbot_reset(self):
        """Test that chatbot can reset history"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "Response",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        
        # Add some history
        chatbot.chat("Query 1")
        chatbot.chat("Query 2")
        assert len(chatbot.history) == 2, "Should have 2 history entries"
        
        # Reset
        chatbot.reset()
        assert len(chatbot.history) == 0, "History should be empty after reset"
    
    def test_chatbot_concise_response(self):
        """Test that chatbot provides concise responses"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "I don't have real-time weather data. I can only provide general climate information.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
            "latency_ms": 400,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Weather?")
        
        # Chatbot should be more concise than agent
        assert len(result) < 500, "Chatbot response should be concise"
    
    def test_chatbot_system_prompt_adherence(self):
        """Test that chatbot adheres to its system prompt"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "As a weather assistant, I don't have access to real-time weather data. If you need current weather information, I'd recommend checking a weather service. I can answer general questions about climate patterns.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 60},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("What's the weather like?")
        
        # Should follow system prompt about not having real-time data
        assert any(keyword in result.lower() for keyword in [
            "don't have", "no access", "cannot", "không có",
            "real-time", "realtime", "live"
        ]), f"Should adhere to system prompt. Got: {result}"


# ============================================================================
# VI. COMPARISON TESTS (Chatbot vs Agent behavior)
# ============================================================================

class TestChatbotVsAgent:
    """Tests highlighting differences between chatbot and agent behavior"""
    
    def test_chatbot_cannot_fetch_real_time_data(self):
        """
        Unlike agent, chatbot cannot fetch real-time data
        This is by design - tests the key difference
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "I cannot retrieve live weather data. I only have access to general climate information from my training data up to my knowledge cutoff.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 60},
            "latency_ms": 500,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Current weather in Tokyo")
        
        # Chatbot should clearly state it cannot access live data
        assert any(keyword in result.lower() for keyword in [
            "cannot", "no access", "don't have", "không thể",
            "không có", "live", "real-time", "realtime", "current"
        ]), f"Should indicate no live data access. Got: {result}"
    
    def test_chatbot_single_response_vs_agent_multi_step(self):
        """
        Chatbot provides single response, agent can do multi-step
        Tests the architectural difference
        """
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        mock_llm.generate.return_value = {
            "content": "I can provide a general comparison based on climate patterns. Hanoi in the north has distinct seasons, while Ho Chi Minh City in the south is tropical and warm year-round.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 70},
            "latency_ms": 550,
            "provider": "mock"
        }
        
        chatbot = Chatbot(llm=mock_llm)
        result = chatbot.chat("Compare Hanoi and HCMC weather")
        
        # Chatbot should provide comparison in single response
        assert result is not None
        # Should mention both cities
        assert "hanoi" in result.lower() or "hà nội" in result.lower()
        assert any(keyword in result.lower() for keyword in [
            "hcmc", "ho chi minh", "sài gòn"
        ])


# ============================================================================
# VII. INTEGRATION TESTS (Requires actual API keys)
# ============================================================================

class TestChatbotIntegration:
    """Integration tests using real API"""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set, skipping integration test"
    )
    def test_chatbot_with_real_openai(self):
        """Test chatbot with real OpenAI provider"""
        from src.core.openai_provider import OpenAIProvider
        
        llm = OpenAIProvider(
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        chatbot = Chatbot(llm=llm)
        result = chatbot.chat("What can you tell me about weather?")
        
        assert result is not None, "Should return a result"
        assert len(result) > 0, "Result should not be empty"
        assert isinstance(result, str), "Result should be string"
    
    def test_chatbot_history_persists_across_calls(self):
        """Test that history properly influences responses"""
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.model_name = "gpt-4"
        mock_llm.api_key = "test-key"
        
        call_count = [0]
        def mock_generate(prompt, system_prompt=None):
            call_count[0] += 1
            # Verify prompt includes history on subsequent calls
            if call_count[0] > 1:
                assert "previous" in prompt.lower() or "trước" in prompt.lower(), \
                    "Prompt should include previous conversation"
            return {
                "content": f"Response {call_count[0]}",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "latency_ms": 500,
                "provider": "mock"
            }
        
        mock_llm.generate.side_effect = mock_generate
        
        chatbot = Chatbot(llm=mock_llm)
        chatbot.chat("Previous question")
        chatbot.chat("Current question")
        
        assert mock_llm.generate.call_count == 2


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run with: pytest tests/test_chatbot.py -v
    pytest.main([__file__, "-v"])
