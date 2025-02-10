import yfinance as yf
import requests
import pandas as pd
from dotenv import load_dotenv
import os

data_download = r"""
currencies = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF", "GBP/JPY", "GBP/CHF", "AUD/JPY", "AUD/CHF", "NZD/JPY", "NZD/CHF", "CAD/JPY", "USD/TRY", "EUR/TRY", "USD/ZAR", "EUR/ZAR", "USD/SGD", "USD/HKD", "USD/MXN", "GBP/TRY", "AUD/TRY", "USD/INR", "EUR/INR", "GBP/ZAR", "USD/BRL", "AUD/NZD", "EUR/AUD", "GBP/AUD", "CAD/CHF", "CHF/JPY", "EUR/SEK", "EUR/NOK", "GBP/NOK", "GBP/SEK", "USD/PLN", "USD/CNY", "USD/IDR", "USD/THB", "USD/PHP", "USD/CLP", "USD/RUB"]

for currency in currencies:

 currency = currency.replace("/", "")+"=X"

 print(currency)

 forex_data = yf.download(currency, end="2025-1-1")

 forex_data.to_csv(r"C:\Users\arnav\OneDrive\Documents\Merlin\forex\forex_data\{}.csv".format(currency.replace("/", "").replace("=X", "")))

 forex_data = pd.read_csv(r"C:\Users\arnav\OneDrive\Documents\Merlin\forex\forex_data\{}.csv".format(currency.replace("/", "").replace("=X", ""))).iloc[2:].reset_index(drop=True).rename({"Price":"Date"}, axis=1)

 forex_data.to_csv(r"C:\Users\arnav\OneDrive\Documents\Merlin\forex\forex_data\{}.csv".format(currency.replace("/", "").replace("=X", "")), index=False)

"""

df = pd.read_csv(r"C:\Users\arnav\OneDrive\Documents\Merlin\forex\forex_data\USDJPY.csv")

load_dotenv()

API_KEY = os.getenv("FRED_API_KEY")

series_id = "JPNNGDP"
start_date = "2000-01-01"
end_date = "2025-01-01"
url = "https://api.stlouisfed.org/fred/series/observations"

params = {
    "series_id": series_id,
    "api_key": API_KEY,
    "file_type": "json",
}

response = requests.get(url, params=params)

gdp_values = {}

if response.status_code == 200:
    data = response.json()
    
    for observation in data["observations"]:
        gdp_values[observation["date"]] = observation["value"]
else:
    print("Error:", response.status_code)
    print(response.text)


series_id = "DEXJPUS"
start_date = "2000-01-01"
end_date = "2025-01-01"
url = "https://api.stlouisfed.org/fred/series/observations"

params = {
    "series_id": series_id,
    "api_key": API_KEY,
    "file_type": "json",
}

response = requests.get(url, params=params)

exchange_rates = {}

if response.status_code == 200:
    data = response.json()
    
    for observation in data["observations"]:
        exchange_rates[observation["date"]] = observation["value"]
else:
    print("Error:", response.status_code)
    print(response.text)


for i in gdp_values:
    try:
        print(i, float(gdp_values[i])/float(exchange_rates[i]))
    except Exception as e:
        None