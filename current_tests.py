import finnhub
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("FINNHUB_API_KEY")

print(finnhub.Client(API_KEY).company_basic_financials("AAPL", "all").get("series").get("annual").get("eps"))
