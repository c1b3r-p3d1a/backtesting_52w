import pandas as pd
import csv
import bisect

print("[+] Leyendo CSV")
valores_historicos = pd.read_csv("..\\db\\historico_completo.csv", sep=",", encoding='utf-8')
print("[+] CSV leído")

shares_df = pd.read_parquet("..\\db\\stock_shares_outstanding.parquet")
shares_df["report_date"] = shares_df["report_date"].astype(str)
shares_df = shares_df.sort_values(["symbol", "report_date"])
shares_lookup = {
    sym: grp[["report_date", "shares_outstanding"]].values.tolist()
    for sym, grp in shares_df.groupby("symbol")
}

def get_market_cap(ticker, fecha, close):
    rows = shares_lookup.get(ticker)
    if not rows:
        return None
    dates = [r[0] for r in rows]
    idx = bisect.bisect_right(dates, fecha) - 1
    if idx < 0:
        return None
    shares = rows[idx][1]
    if not shares or shares != shares:
        return None
    return round(shares * close)

current_ticker = None
window = []
with open("..\\db\\max.csv", "w", encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["TICKER", "FECHA", "OPEN", "CLOSE", "HIGH", "LOW", "VOLUME", "MARKET_CAP"])
    for row in valores_historicos.itertuples():
        ticker = row.TICKER
        fecha = row.DATE
        open = row.OPEN
        high = row.HIGH
        low = row.LOW
        close = row.CLOSE
        volume = row.VOLUME

        if ticker != current_ticker:
            current_ticker = ticker
            window = []
        if len(window) >= 252:
            if high > max(window):
                writer.writerow([ticker, fecha, open, close, high, low, volume, get_market_cap(ticker, fecha, close)])
        window.append(high)
        if len(window) > 252:
            window.pop(0)