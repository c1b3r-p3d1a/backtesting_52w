import pandas as pd
import json
import requests
import os
import time

#Posibles variables .env?
CSV_PATH = ".\\max.csv"
STOCK_PATH = ".\\stock_profile.csv"
STOCK_PROFILE_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_profile.parquet"
STOCK_PRICES_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_prices.parquet"

def scrape_by_date(day: int, month: int, year: int):
    matches = []
    
    if len(str(day)) < 2:
        day = str(day).zfill(2)
    if len(str(month)) < 2:
        month = str(month).zfill(2)

    if ((len(str(day))) > 2) or ((len(str(month))) > 2) or ((len(str(year))) > 4):
        print("[-] Entrada inválida. Revise su fecha.")

    for row in PRICE_DATA.itertuples():
        if row.FECHA == str(year)+"-"+str(month)+"-"+str(day):
            matches.append([row.TICKER, row.FECHA, row.HIGH])

    return json.dumps(matches)

def scrape_by_ticker(ticker: str):
    matches = []

    ticker = ticker.upper()

    for row in PRICE_DATA.itertuples():
        if row.TICKER == ticker:
            matches.append([row.TICKER, row.FECHA, row.HIGH])

    return json.dumps(matches)

def update_parquet_files_and_transform_to_csv():
    r = requests.get("https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/raw/main/spec.json")
    last_data_updated = r.json()["update_time"]

    print(f"[*] Última actualización de la base de datos externa (HuggingFace): {last_data_updated}")

    if not int(input("[?] ¿Desea continuar con la actualización? (Sí:1 No:0): ")):
        main()
    
    if os.path.isfile("stock_prices.parquet"):
        os.rename("stock_prices.parquet", "stock_prices.parquet.bck1")

    if os.path.isfile("stock_profile.parquet"):
        os.rename("stock_profile.parquet", "stock_profile.parquet.bck1")

    if os.path.isfile("historico_completo.csv"):
        os.rename("historico_completo.csv", "historico_completo.csv.bck1")

    if os.path.isfile("stock_profile.csv"):
        os.rename("stock_profile.csv", "stock_profile.csv.bck1")

    with requests.get(STOCK_PRICES_URL, stream=True) as r:
        r.raise_for_status()

        with open("stock_prices.parquet", "wb") as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)
            file.close()

    with requests.get(STOCK_PROFILE_URL, stream=True) as r:
        r.raise_for_status()

        with open("stock_profile.parquet", "wb") as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)
            file.close()

    os.system("python parser_chatgpt.py")
    parquet_to_csv("stock_profile.parquet", "stock_profile")


def parquet_to_csv(file_path: str, file_name: str):
    parquet = pd.read_parquet(file_path)
    parquet.to_csv(f"{file_name}.csv", index=False)

def get_ticker_info(ticker: str):
    ticker = ticker.upper()

    for row in COMP_DATA.itertuples():
        if row.symbol == ticker:
            return json.dumps(row)

def clean_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    #ESTO ES PARA UN PRIMER PROTOTIPO. LA COSA SERÍA CREAR ENDPOINTS PARA CADA FUNCIÓN (UNA API VAMOS). LUEGO USAR NODEJS O SIMILAR PARA FRONTEND
    clean_screen()
    print(""" ____ ______        ______             _    _            _            
| ___|___ \ \      / / __ )  __ _  ___| | _| |_ ___  ___| |_ ___ _ __ 
|___ \ __) \ \ /\ / /|  _ \ / _` |/ __| |/ / __/ _ \/ __| __/ _ \ '__|
 ___) / __/ \ V  V / | |_) | (_| | (__|   <| ||  __/\__ \ ||  __/ |   
|____/_____| \_/\_/  |____/ \__,_|\___|_|\_\\__\___||___/\__\___|_|  """)
    print("\n[1] Buscar máximos en las últimas 52 semanas por fecha\n[2] Buscar máximos en las últimas 52 semanas por ticker\n[3] Buscar información empresa por ticker\n[4] Actualizar/descargar base de datos local")
    selec = input("\n")
    match selec:
        case "1":
            clean_screen()
            fecha = input("Introduzca fecha (DD/MM/YYYY): ")
            r = scrape_by_date(int(fecha.split("/")[0]), int(fecha.split("/")[1]), int(fecha.split("/")[2]))
            print(f"[*] JSON: {r}")
        case "2":
            clean_screen()
            ticker = input("Introduzca el ticker (ej. AMZN): ")
            r = scrape_by_ticker(ticker)
            print(f"[*] JSON: {r}")
        case "3":
            clean_screen()
            ticker = input("Introduzca el ticker (ej. AMZN): ")
            r = get_ticker_info(ticker)
            print(f"[*] JSON: {r}")
        case "4":
            update_parquet_files_and_transform_to_csv()



if __name__ == "__main__":
    clean_screen()
    try:
        print("[+] Leyendo fichero CSV")
        PRICE_DATA = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8")
        print("[+] Fichero leído correctamente")
        print("\n[+] Leyendo información de empresas")
        COMP_DATA = pd.read_csv(STOCK_PATH, sep=",", encoding="utf-8")
        print("[+] Fichero leído correctamente")
        time.sleep(2)
        main()
    except Exception as e:
        print(f"[-] Error desconocido: {e}")