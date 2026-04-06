import os
from langchain_community.retrievers import WikipediaRetriever, TavilySearchAPIRetriever
from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import datetime
import pytz
import requests
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- CẤU HÌNH ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@tool
def InternetSearch(query: str, k: int = 5):
    """Tool tìm kiếm thông tin trên Internet sử dụng Tavily."""
    print(f"\n🔍 [InternetSearch] Đang tìm kiếm: '{query}'...")
    try:
        retriever = TavilySearchAPIRetriever(
            k=k,
            api_key=TAVILY_API_KEY
        )
        results = retriever.invoke(query)

        if not results:
            return "Không tìm thấy kết quả nào trên Internet."

        return results
    except Exception as e:
        return f"❌ InternetSearch failed: {e}"


@tool
def WikiSearch(query: str, k: int = 3):
    """Tool tìm kiếm trên Wikipedia."""
    print(f"\n📖 [WikiSearch] Đang tìm kiếm: '{query}'...")
    try:
        retriever = WikipediaRetriever(
            top_k_results=k,
            lang="vi",           # Tìm Wikipedia tiếng Việt; đổi thành "en" nếu muốn tiếng Anh
            doc_content_chars_max=2000  # Giới hạn độ dài mỗi bài để tránh quá dài
        )
        results = retriever.invoke(query)

        if not results:
            return "Không tìm thấy thông tin trên Wikipedia."

        return results
    except Exception as e:
        return f"❌ WikiSearch failed: {e}"

@tool
def TimeSearch(query: str = "") -> str:
    """Tool lấy ngày và giờ hiện tại tại Việt Nam."""
    now = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))
    return f"Hôm nay là {now.strftime('%d/%m/%Y')}, lúc {now.strftime('%H:%M:%S')}."


# --- HELPER: In kết quả đẹp ---
def print_results(label: str, results):
    print(f"\n{'='*60}")
    print(f"📌 {label}")
    print('='*60)

    if isinstance(results, str):
        # Trả về thông báo lỗi hoặc "không tìm thấy"
        print(results)
        return

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source") or doc.metadata.get("title") or "N/A"
        print(f"\n[{i}] Nguồn: {source}")
        print(f"    Nội dung: {doc.page_content[:300]}...")


# --- MAIN ---
def main():
    print("🚀 Bắt đầu test các search tools...\n")

    # --- Test WikiSearch ---
    wiki_query = "Hồ Chí Minh"
    wiki_results = WikiSearch.invoke({"query": wiki_query, "k": 2})
    print_results(f"WikiSearch: '{wiki_query}'", wiki_results)

    # --- Test InternetSearch (cần TAVILY_API_KEY) ---
    if TAVILY_API_KEY:
        internet_query = "Tin tức AI mới nhất 2025"
        internet_results = InternetSearch.invoke({"query": internet_query, "k": 3})
        print_results(f"InternetSearch: '{internet_query}'", internet_results)
    else:
        print("\n⚠️  Bỏ qua InternetSearch: TAVILY_API_KEY chưa được set.")
        print("   Chạy: export TAVILY_API_KEY='tvly-xxxx' trước khi test.")

    print("\n✅ Test hoàn tất!")


@tool
def WeatherSearch(query: str) -> str:
    """
    Tool lấy thời tiết HIỆN TẠI và DỰ BÁO 5 ngày tại một địa điểm.
    query: tên thành phố, ví dụ 'Hanoi', 'Ho Chi Minh City', 'Da Nang'
    """
    print(f"\n🌤️ [WeatherSearch] Đang lấy thời tiết: '{query}'...")
    
    if not OPENWEATHER_API_KEY:
        return "❌ Chưa cấu hình OPENWEATHER_API_KEY trong .env"
    
    try:
        # --- Thời tiết hiện tại ---
        current_url = "https://api.openweathermap.org/data/2.5/weather"
        current_res = requests.get(current_url, params={
            "q": query,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "vi"
        }, timeout=10)
        current_res.raise_for_status()
        c = current_res.json()

        current_info = (
            f"📍 {c['name']}, {c['sys']['country']}\n"
            f"🌡️ Nhiệt độ: {c['main']['temp']}°C "
            f"(cảm giác như {c['main']['feels_like']}°C)\n"
            f"💧 Độ ẩm: {c['main']['humidity']}%\n"
            f"💨 Gió: {c['wind']['speed']} m/s\n"
            f"☁️ Trời: {c['weather'][0]['description'].capitalize()}\n"
        )
        if 'rain' in c:
            current_info += f"🌧️ Lượng mưa 1h: {c['rain'].get('1h', 0)} mm\n"

        # --- Dự báo 5 ngày (lấy mỗi ngày 1 mốc 12h trưa) ---
        forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        forecast_res = requests.get(forecast_url, params={
            "q": query,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "vi",
            "cnt": 40  # 5 ngày x 8 mốc/ngày
        }, timeout=10)
        forecast_res.raise_for_status()
        f_data = forecast_res.json()

        # Lọc chỉ lấy mốc 12:00:00 mỗi ngày
        seen_dates = set()
        forecast_lines = []
        for item in f_data["list"]:
            date_str = item["dt_txt"].split(" ")[0]   # "2025-07-15"
            time_str = item["dt_txt"].split(" ")[1]   # "12:00:00"
            if time_str == "12:00:00" and date_str not in seen_dates:
                seen_dates.add(date_str)
                desc = item["weather"][0]["description"].capitalize()
                forecast_lines.append(
                    f"  {date_str}: {item['main']['temp']}°C, {desc}, "
                    f"độ ẩm {item['main']['humidity']}%"
                )

        forecast_info = "📅 Dự báo 5 ngày tới:\n" + "\n".join(forecast_lines)

        return current_info + "\n" + forecast_info

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"❌ Không tìm thấy địa điểm '{query}'. Thử tên tiếng Anh, ví dụ: 'Hanoi', 'Da Nang'."
        return f"❌ Lỗi HTTP: {e}"
    except Exception as e:
        return f"❌ WeatherSearch thất bại: {e}"

if __name__ == "__main__":
    main()