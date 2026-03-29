import pandas as pd
import requests
import os
from price_parquet_to_csv import ParquetToCSV
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

load_dotenv()
app = FastAPI(docs_url=False, title="52W Backtester API", version="1.0.0", description="Consulta de máximos de 52 semanas sobre el mercado americano.")
security = HTTPBearer()

API_KEY = os.getenv("API_KEY")
CSV_PATH = ".\\db\\max.csv"
STOCK_PATH = ".\\db\\stock_profile.csv"
STOCK_PROFILE_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_profile.parquet"
STOCK_PRICES_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_prices.parquet"

print("[+] Leyendo fichero CSV")
PRICE_DATA = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8")
print("[+] Fichero leído correctamente")
print("\n[+] Leyendo información de empresas")
COMP_DATA = pd.read_csv(STOCK_PATH, sep=",", encoding="utf-8")
print("[+] Fichero leído correctamente\n\n")

if not API_KEY:
    raise RuntimeError("[-] API_KEY no encontrada en el fichero .env")

def verificar_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Token inválido o no autorizado")
    return credentials.credentials

@app.get("/date", tags=["Máximos 52W"], summary="Máximo 52W Fecha", description="De una fecha, obtiene todas las empresas (tickers) en las que tuvieron su mayor HIGH en esa sesión respecto a sus 252 sesiones anteriores.", responses={
        200: {
            "description": "Petición exitosa",
            "content": {
                "application/json": {
                    "example": [["BSRR","2003-03-03",14],["BXMT","2003-03-03",174.3],["CHN","2003-03-03",16.55],["CRMZ","2003-03-03",0.31],["DSS","2003-03-03",3565.46],["EBAY","2003-03-03",8.36],["FCCO","2003-03-03",17.09],["FNMAG","2003-03-03",53],["GGLXF","2003-03-03",7.25],["GRMN","2003-03-03",17.09],["HOFT","2003-03-03",11.1],["HOPE","2003-03-03",6.41],["IRM","2003-03-03",14.68],["IRS","2003-03-03",8.6],["JHS","2003-03-03",15.5],["LXU","2003-03-03",2.92],["MEOH","2003-03-03",9.55],["NEU","2003-03-03",9.99],["NRK","2003-03-03",15],["NRP","2003-03-03",116],["PWOD","2003-03-03",22.15],["TEI","2003-03-03",12.48],["TPR","2003-03-03",9.13],["TTC","2003-03-03",4.38],["VEON","2003-03-03",64.65],["VTRS","2003-03-03",19.17],["WEYS","2003-03-03",14.67]]
                }
            },
        },
        401: {
            "description": "Token inválido o no autorizado",
            "content": {
                "application/json": {
                    "example": {"detail": "Token inválido o no autorizado"}
                }
            }
        },
        422: {
            "description": "Error de validación en los parámetros",
            "content": {
                "application/json": {
                    "example": {"detail": [{"loc": ["query", "day"], "msg": "Input should be less than or equal to 31", "type": "less_than_equal"}]}
                }
            }
        }
    })
async def scrape_by_date(day: int = Query(..., ge=1, le=31, description="Día del mes (1-31)", openapi_examples={
                "normal": {
                    "description": "Ejemplo válido",
                    "value": 3
                }}),
    month: int = Query(..., ge=1, le=12, description="Mes del año (1-12)", openapi_examples={
                "normal": {
                    "description": "Ejemplo válido",
                    "value": 3
                }}),
    year: int = Query(..., ge=2000, description="Año en formato YYYY", openapi_examples={
                "normal": {
                    "description": "Ejemplo válido",
                    "value": 2003
                }}),
    token: str = Depends(verificar_api_key)):
    matches = []
    
    if len(str(day)) < 2:
        day = str(day).zfill(2)
    if len(str(month)) < 2:
        month = str(month).zfill(2)

    for row in PRICE_DATA.itertuples():
        if row.FECHA == str(year)+"-"+str(month)+"-"+str(day):
            matches.append([row.TICKER, row.FECHA, row.HIGH])

    return matches

@app.get("/ticker", tags=["Máximos 52W"], summary="Máximo 52W Ticker", description="De un ticker, obtiene todas las fechas en las que tuvo su mayor HIGH en esa sesión respecto de las 252 sesiones anteriores.", responses={
        200: {
            "description": "Petición exitosa",
            "content": {
                "application/json": {
                    "example": [["IBEX","2020-11-18",19.44],["IBEX","2020-11-19",19.5],["IBEX","2020-11-25",20.25],["IBEX","2020-11-30",20.37],["IBEX","2020-12-01",20.41],["IBEX","2020-12-03",20.5],["IBEX","2020-12-04",22],["IBEX","2020-12-08",22.45],["IBEX","2020-12-09",22.79],["IBEX","2021-02-19",23.57],["IBEX","2021-02-24",24.25],["IBEX","2021-04-05",24.68],["IBEX","2021-04-16",25],["IBEX","2021-04-23",25.5],["IBEX","2022-11-10",22.5],["IBEX","2022-11-11",23.36],["IBEX","2022-11-15",23.45],["IBEX","2022-11-16",24.98],["IBEX","2022-11-18",25.2],["IBEX","2022-11-21",26.93],["IBEX","2022-12-02",26.97],["IBEX","2022-12-06",27],["IBEX","2022-12-07",27.63],["IBEX","2022-12-13",27.77],["IBEX","2023-01-18",28.33],["IBEX","2023-01-19",28.47],["IBEX","2023-02-16",31.4],["IBEX","2024-09-13",20.02],["IBEX","2024-09-16",20.56],["IBEX","2024-11-20",20.95],["IBEX","2024-11-21",21.63],["IBEX","2024-12-30",21.96],["IBEX","2025-01-03",22.43],["IBEX","2025-01-16",22.45],["IBEX","2025-01-17",22.48],["IBEX","2025-01-21",22.52],["IBEX","2025-01-29",22.53],["IBEX","2025-01-30",22.67],["IBEX","2025-02-07",25.03],["IBEX","2025-02-11",26.22],["IBEX","2025-02-12",26.53],["IBEX","2025-02-13",27.34],["IBEX","2025-02-14",27.83],["IBEX","2025-05-09",32.08],["IBEX","2025-09-12",42.99]]
                }
            },
        },
        401: {
        "description": "Token inválido o no autorizado",
        "content": {
            "application/json": {
                "example": {"detail": "Token inválido o no autorizado"}
            }
        }},
        422: {
            "description": "Error de validación en los parámetros",
            "content": {
                "application/json": {
                    "example": {"detail": [{"loc": ["query", "ticker"], "msg": "Máximo 10 caracteres", "type": "string_too_long"}]}
                }
            }
        }
    })
async def scrape_by_ticker(ticker: str = Query(..., min_length=1, max_length=10, description="Símbolo bursátil del activo", openapi_examples={
                "normal": {
                    "summary": "Obtiene los días de la empresa en los cuales se cumple que su HIGH fue el máximo en las últimas 252 sesiones registradas",
                    "description": "",
                    "value": {
                        "ticker": "IBEX"
                    },
                }}),
    token: str = Depends(verificar_api_key)):
    matches = []

    ticker = ticker.upper()

    for row in PRICE_DATA.itertuples():
        if row.TICKER == ticker:
            matches.append([row.TICKER, row.FECHA, row.HIGH])

    return matches

def update_parquet_files_and_transform_to_csv():
    r = requests.get("https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/raw/main/spec.json")
    last_data_updated = r.json()["update_time"]

    print(f"[*] Última actualización de la base de datos externa (HuggingFace): {last_data_updated}")

    if not int(input("[?] ¿Desea continuar con la actualización? (Sí:1 No:0): ")):
        return
    
    if os.path.isfile(".\\db\\stock_prices.parquet"):
        os.rename(".\\db\\stock_prices.parquet", ".\\db\\stock_prices.parquet.bck1")

    if os.path.isfile(".\\db\\stock_profile.parquet"):
        os.rename(".\\db\\stock_profile.parquet", ".\\db\\stock_profile.parquet.bck1")

    if os.path.isfile(".\\db\\historico_completo.csv"):
        os.rename(".\\db\\historico_completo.csv", ".\\db\\historico_completo.csv.bck1")

    if os.path.isfile(".\\db\\stock_profile.csv"):
        os.rename(".\\db\\stock_profile.csv", ".\\db\\stock_profile.csv.bck1")

    with requests.get(STOCK_PRICES_URL, stream=True) as r:
        r.raise_for_status()

        with open("stock_prices.parquet", "wb") as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)
            file.close()

    with requests.get(STOCK_PROFILE_URL, stream=True) as r:
        r.raise_for_status()

        with open(".\\db\\stock_profile.parquet", "wb") as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)
            file.close()

    converter = ParquetToCSV(
        local_parquet=".\\db\\stock_prices.parquet",
        output_csv=".\\db\\historico_completo.csv",
        start_date="2000-01-01",
        end_date=None,
    )
    converter.convert()
    parquet_to_csv(".\\db\\stock_profile.parquet", ".\\db\\stock_profile")

def parquet_to_csv(file_path: str, file_name: str):
    parquet = pd.read_parquet(file_path)
    parquet.to_csv(f"{file_name}.csv", index=False)

@app.get("/info", tags=["Empresas"], summary="Info ticker", description="Obtiene información de una empresa mediante su ticker", responses={
        200: {
            "description": "Petición exitosa",
            "content": {
                "application/json": {
                    "example": [22,"AAPL","One Apple Park Way","Cupertino","United States","(408) 996-1010","95014","Consumer Electronics","Technology","Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising AirPods, Apple Vision Pro, Apple TV, Apple Watch, Beats products, and HomePod, as well as Apple branded and third-party accessories. It also provides AppleCare support and cloud services; and operates various platforms, including the App Store that allow customers to discover and download applications and digital content, such as books, music, video, games, and podcasts, as well as advertising services include third-party licensing arrangements and its own advertising platforms. In addition, the company offers various subscription-based services, such as Apple Arcade, a game subscription service; Apple Fitness+, a personalized fitness service; Apple Music, which offers users a curated listening experience with on-demand radio stations; Apple News+, a subscription news and magazine service; Apple TV, which offers exclusive original content and live sports; Apple Card, a co-branded credit card; and Apple Pay, a cashless payment service, as well as licenses its intellectual property. The company serves consumers, and small and mid-sized businesses; and the education, enterprise, and government markets. It distributes third-party applications for its products through the App Store. The company also sells its products through its retail and online stores, and direct sales force; and third-party cellular network carriers and resellers. The company was formerly known as Apple Computer, Inc. and changed its name to Apple Inc. in January 2007. Apple Inc. was founded in 1976 and is headquartered in Cupertino, California.",150000,"https://www.apple.com","2026-01-31"]
                }
            },
        },
        401: {
            "description": "Token inválido o no autorizado",
            "content": {
                "application/json": {
                    "example": {"detail": "Token inválido o no autorizado"}
                }
            }
        },
        422: {
            "description": "Error de validación en los parámetros",
            "content": {
                "application/json": {
                    "example": {"detail": [{"loc": ["query", "ticker"], "msg": "Máximo 10 caracteres", "type": "string_too_long"}]}
                }
            }
        },
        404: {
            "description": "Ticker no encontrado",
            "content": {
            "application/json": {
                "example": {"detail": "El ticker no se encuentra registrado en la base de datos"}
            }
        },
        }
    })
async def get_ticker_info(ticker: str = Query(..., min_length=1, max_length=10, description="Símbolo bursátil del activo", openapi_examples={
                "normal": {
                    "summary": "Obtiene información sobre Apple",
                    "description": "",
                    "value": {
                        "ticker": "AAPL"
                    },
                }}),
    token: str = Depends(verificar_api_key)):
    ticker = ticker.upper()

    result = COMP_DATA[COMP_DATA["symbol"] == ticker]
    if result.empty:
        raise HTTPException(status_code=404, detail="El ticker no se encuentra registrado en la base de datos")
    return result.iloc[0].to_dict()

def clean_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    clean_screen()
    print(""" ____ ______        ______             _    _            _            
| ___|___ \ \      / / __ )  __ _  ___| | _| |_ ___  ___| |_ ___ _ __ 
|___ \ __) \ \ /\ / /|  _ \ / _` |/ __| |/ / __/ _ \/ __| __/ _ \ '__|
 ___) / __/ \ V  V / | |_) | (_| | (__|   <| ||  __/\__ \ ||  __/ |   
|____/_____| \_/\_/  |____/ \__,_|\___|_|\_\\__\___||___/\__\___|_|  \n\n""")
    try:        
        uvicorn.run("backend:app", host="127.0.0.1", port=3333, reload=True)
    except Exception as e:
        print(f"[-] Error desconocido: {e}")