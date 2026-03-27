import pandas as pd
import csv

print("[+] Leyendo CSV")
valores_historicos = pd.read_csv("historico_completo.csv", sep=",", encoding='utf-8')
print("[+] CSV leído")

current_ticker = None
window = []
with open("max.csv", "w", encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["TICKER", "FECHA", "HIGH"])
    for fila in valores_historicos.itertuples():
        ticker = fila.TICKER
        fecha = fila.DATE
        high = fila.HIGH
        if ticker != current_ticker:
            current_ticker = ticker
            window = []
        if window:
            if high > max(window):
                writer.writerow([ticker, fecha, high])
        window.append(high)
        if len(window) > 252:
            window.pop(0)