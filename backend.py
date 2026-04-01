import pandas as pd
import requests
import os
from tools.price_parquet_to_csv_claude import ParquetToCSV
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import math
import numpy as np

load_dotenv()
app = FastAPI(docs_url=False, title="52W Backtester API", version="1.0.0", description="Consulta de máximos de 52 semanas sobre el mercado americano.")
origins = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
    "https://52w-signal-insights.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET"],
    allow_headers=["Authorization"],
)
security = HTTPBearer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.getenv("API_KEY")
CSV_PATH = os.path.join(BASE_DIR, "db", "max.csv")
STOCK_PATH = os.path.join(BASE_DIR, "db", "stock_profile.csv")
SP500_PATH = os.path.join(BASE_DIR, "db", "sp500.csv")
STOCK_PROFILE_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_profile.parquet"
STOCK_PRICES_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_prices.parquet"

print("[+] Leyendo ficheros CSV")
PRICE_DATA = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8")
SP500 = pd.read_csv(SP500_PATH, sep=",", encoding="utf-8")
print("[+] Ficheros leído correctamente")
print("\n[+] Leyendo información de empresas")
COMP_DATA = pd.read_csv(STOCK_PATH, sep=",", encoding="utf-8")
print("[+] Fichero leído correctamente\n\n")

if not API_KEY:
    raise RuntimeError("[-] API_KEY no encontrada en el fichero .env")

def verificar_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Token inválido o no autorizado")
    return credentials.credentials

def limpiar_valores(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x

def clean_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

@app.get("/date", tags=["Máximos 52W"], summary="Máximo 52W Fecha", description="De una fecha, obtiene todas las empresas (tickers) en las que tuvieron su mayor HIGH en esa sesión respecto a sus 252 sesiones anteriores.", responses={
        200: {
            "description": "Petición exitosa",
            "content": {
                "application/json": {
                    "example": [["BSRR","2003-03-03",13.86,13.88,14,13.85,7900,128328928],["BXMT","2003-03-03",172.8,174.3,174.3,171,450,94575180],["CHN","2003-03-03",16.4,16.42,16.55,16.35,148400,"null"],["CRMZ","2003-03-03",0.31,0.31,0.31,0.31,6500,"null"],["DSS","2003-03-03",3231.2,3454.04,3565.46,3119.78,1,13125352],["EBAY","2003-03-03",8.26,8.16,8.36,8.15,46267373,10267107024],["FCCO","2003-03-03",17.09,17.09,17.09,17.09,500,27138920],["GGLXF","2003-03-03",7.25,7.25,7.25,6.5,360,16785200],["GRMN","2003-03-03",17.08,16.8,17.09,16.7,657000,3626103600],["HOPE","2003-03-03",6.12,6.29,6.41,6.12,67200,134590275],["IRM","2003-03-03",14.58,14.3,14.68,14.3,1146650,2738056750],["IRS","2003-03-03",8.6,8.43,8.6,8.43,106845,173233971],["JHS","2003-03-03",15.48,15.5,15.5,15.34,16300,"null"],["LXU","2003-03-03",2.88,2.92,2.92,2.88,4680,45440748],["MEOH","2003-03-03",9.3,9.08,9.55,9.08,329100,1140916528],["NEU","2003-03-03",8.96,9.9,9.99,8.96,85800,165221100],["PWOD","2003-03-03",22.15,21.31,22.15,21.31,3762,"null"],["TEI","2003-03-03",12.23,12.35,12.48,12.21,226100,"null"],["TPR","2003-03-03",9,8.86,9.13,8.79,5049200,3192856936],["TTC","2003-03-03",4.32,4.31,4.38,4.28,891200,843710946],["VEON","2003-03-03",63.17,64.07,64.65,63,261780,"null"],["VTRS","2003-03-03",18.93,18.79,19.17,18.67,2374350,"null"],["WEYS","2003-03-03",13.66,14.67,14.67,13.66,13800,166868316]]
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
            matches.append([row.TICKER, row.FECHA, limpiar_valores(row.OPEN), limpiar_valores(row.CLOSE), limpiar_valores(row.HIGH), limpiar_valores(row.LOW), limpiar_valores(row.VOLUME), limpiar_valores(row.MARKET_CAP)])

    return matches

@app.get("/sp500", tags=["S&P500"], summary="S&P500", description="Obtiene información sobre el S&P500 en una fecha concreta", responses={
        200: {
            "description": "Petición exitosa",
            "content": {
                "application/json": {
                    "example": [["2003-03-03",841.1500244140625,834.8099975585938,852.3400268554688,832.739990234375,1208900000]]
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
async def sp500_by_date(
    day: int = Query(..., ge=1, le=31, description="Día del mes (1-31)", openapi_examples={
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

    for row in SP500.itertuples():
        if row.DATE == str(year)+"-"+str(month)+"-"+str(day):
            matches.append([row.Index, row.DATE, limpiar_valores(row.OPEN), limpiar_valores(row.ADJ_CLOSE), limpiar_valores(row.HIGH), limpiar_valores(row.LOW), limpiar_valores(row.VOLUME)])

    return matches

@app.get("/ticker", tags=["Máximos 52W"], summary="Máximo 52W Ticker", description="De un ticker, obtiene todas las fechas en las que tuvo su mayor HIGH en esa sesión respecto de las 252 sesiones anteriores.", responses={
        200: {
            "description": "Petición exitosa",
            "content": {
                "application/json": {
                    "example": [["IBEX","2022-11-10",20.17,22.5,22.5,19.89,154800,410580000],["IBEX","2022-11-11",22.5,22.39,23.36,21.79,101800,408572720],["IBEX","2022-11-15",22.22,22.52,23.45,22.13,53900,410944960],["IBEX","2022-11-16",24,24.44,24.98,23.55,184800,445981120],["IBEX","2022-11-18",24.15,24.85,25.2,23.72,120300,453462800],["IBEX","2022-11-21",25.1,26.05,26.93,24.8,134800,475360400],["IBEX","2022-12-02",25.57,26.33,26.97,25.24,111200,480469840],["IBEX","2022-12-06",26.72,26.68,27,26.12,97100,486856640],["IBEX","2022-12-07",26.6,27.17,27.63,25.36,102100,495798160],["IBEX","2022-12-13",26.95,26.91,27.77,26.7,54600,491053680],["IBEX","2023-01-18",27.3,27.87,28.33,27.26,156500,508571760],["IBEX","2023-01-19",27.87,27.79,28.47,27.58,145700,507111920],["IBEX","2023-02-16",28.25,29.64,31.4,27.77,218600,540870720],["IBEX","2024-09-13",18,19.55,20.02,17.97,537200,328502560],["IBEX","2024-09-16",19.57,19.94,20.56,19.57,403000,335055808],["IBEX","2024-11-20",20.45,20.02,20.95,19.92,294200,264314050],["IBEX","2024-11-21",20,21.13,21.63,19.83,453900,278968825],["IBEX","2024-12-30",21.25,21.45,21.96,20.64,235900,283193625],["IBEX","2025-01-03",21.85,21.78,22.43,21.55,190600,287093070],["IBEX","2025-01-16",22.16,22.28,22.45,22.06,79400,293683820],["IBEX","2025-01-17",22.44,22.13,22.48,22.1,102400,291706595],["IBEX","2025-01-21",22.22,22.21,22.52,22,82000,292761115],["IBEX","2025-01-29",22.39,22.32,22.53,21.98,117500,294211080],["IBEX","2025-01-30",22.37,22.09,22.67,22.08,105600,291179335],["IBEX","2025-02-07",22.5,24.49,25.03,22.31,508100,323069631],["IBEX","2025-02-11",24.8,25.95,26.22,24.73,478700,342329805],["IBEX","2025-02-12",25.08,26.48,26.53,25.08,504100,349321512],["IBEX","2025-02-13",26.3,27.24,27.34,26.28,371400,359347356],["IBEX","2025-02-14",27.52,27.3,27.83,26.7,332000,360138870],["IBEX","2025-05-09",28.02,30.55,32.08,27.66,759000,408526820],["IBEX","2025-09-12",38.94,41.58,42.99,38.22,1473500,553641858]]
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
async def scrape_by_ticker(
    ticker: str = Query(..., min_length=1, max_length=10, description="Símbolo bursátil del activo", openapi_examples={
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
            matches.append([row.TICKER, row.FECHA, limpiar_valores(row.OPEN), limpiar_valores(row.CLOSE), limpiar_valores(row.HIGH), limpiar_valores(row.LOW), limpiar_valores(row.VOLUME), limpiar_valores(row.MARKET_CAP)])

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
async def get_ticker_info(
    ticker: str = Query(..., min_length=1, max_length=10, description="Símbolo bursátil del activo", openapi_examples={
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
    
    row = result.iloc[0].replace([np.nan, np.inf, -np.inf], None)

    return row.to_dict()

@app.get("/rend", tags=["Rendimiento"], summary="Rendiento ticker", description="Obtiene el rendimienot de un ticker a lo largo del tiempo frente al S&P500")
async def get_performance_ticker(
    ticker: str = Query(..., min_length=1, max_length=10, description="Símbolo bursátil del activo", openapi_examples={
                "normal": {
                    "description": "Símbolo bursátil",
                    "value": {
                        "ticker": "AMZN"
                    },
                }}), 
    day: int = Query(..., ge=1, le=31, description="Día del mes (1-31)", openapi_examples={
                "normal": {
                    "description": "Ejemplo válido",
                    "value": 17
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
    
    date = f"{year}-{month}-{day}"

    if not ((PRICE_DATA["TICKER"] == ticker) & (pd.to_datetime(PRICE_DATA["FECHA"]) == pd.to_datetime(date))).any():
        raise HTTPException(status_code=400, detail=f"El ticker no cumple la condición de 52W en el {day} del {month} del {year}")
    
    if len(str(ticker)) < 2:
        db = pd.read_parquet(os.path.join(BASE_DIR, "db", "fragmented", f"{str(ticker)[0].lower()}", f"_.parquet"))
    else:
        db = pd.read_parquet(os.path.join(BASE_DIR, "db", "fragmented", f"{str(ticker)[0].lower()}", f"{str(ticker)[1].lower()}.parquet"))

    mask = (db['symbol'] == ticker) & (pd.to_datetime(db['report_date']) == pd.to_datetime(date))
    index = db.index[mask]
    
    offsets = list(range(8, 253, 7))
    resultados = []

    if not index.empty:
        index = index[0]
        pos = db.index.get_loc(index)
        sp500_index = await sp500_by_date(day, month, year)
        sp500_index = sp500_index[0][0]
        precio_inicial = db.iloc[pos+1]["open"]
        sp500_inicial = SP500.iloc[sp500_index+1]["ADJ_CLOSE"]
        for offset in offsets:
            if (pos+offset < len(db)) and (sp500_index+offset < len(SP500)):
                precio_final = db.iloc[pos+offset]["close"]
                rend_ticker = ((precio_final-precio_inicial)/precio_inicial)
                sp500_final = SP500.iloc[sp500_index+offset]["ADJ_CLOSE"]
                rend_sp500 = ((sp500_final-sp500_inicial)/sp500_inicial)
                alpha = float(rend_ticker) - rend_sp500
                resultados.append([offset-1, rend_ticker, rend_sp500, alpha])
    
    return resultados

    

if __name__ == "__main__":
    clean_screen()
    print(r""" ____ ______        ______             _    _            _            
| ___|___ \ \      / / __ )  __ _  ___| | _| |_ ___  ___| |_ ___ _ __ 
|___ \ __) \ \ /\ / /|  _ \ / _` |/ __| |/ / __/ _ \/ __| __/ _ \ '__|
 ___) / __/ \ V  V / | |_) | (_| | (__|   <| ||  __/\__ \ ||  __/ |   
|____/_____| \_/\_/  |____/ \__,_|\___|_|\_\\__\___||___/\__\___|_|  \n\n""")
    try:        
        uvicorn.run("backend:app", host="127.0.0.1", port=3333, reload=True)
    except Exception as e:
        print(f"[-] Error desconocido: {e}")