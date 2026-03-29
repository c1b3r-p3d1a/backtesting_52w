import yfinance as yf
import pandas as pd

ticker = "^GSPC"
start_date = "2000-01-01"
end_date = "2026-03-21"

data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print("Columnas disponibles:", data.columns.tolist())

required_cols = ["Open", "Adj Close", "High", "Low", "Volume"]

if "Adj Close" not in data.columns:
    print("⚠️ 'Adj Close' no encontrado, usando 'Close'")
    data["Adj Close"] = data["Close"]

data = data[required_cols].copy()

data.columns = ["OPEN", "ADJ_CLOSE", "HIGH", "LOW", "VOLUME"]

data.reset_index(inplace=True)
data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
data.rename(columns={"Date": "DATE"}, inplace=True)

data = data[["DATE", "OPEN", "ADJ_CLOSE", "HIGH", "LOW", "VOLUME"]]

data.to_csv("..\\db\\sp500.csv", index=False)

print("✅ CSV generado correctamente")
print(data.tail())