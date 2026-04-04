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
import json
import duckdb
import time

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
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)
security = HTTPBearer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.getenv("API_KEY")
CSV_PATH = os.path.join(BASE_DIR, "db", "max.csv")
STOCK_PATH = os.path.join(BASE_DIR, "db", "stock_profile.csv")
SP500_PATH = os.path.join(BASE_DIR, "db", "sp500.csv")
STOCK_PROFILE_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_profile.parquet"
STOCK_PRICES_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_prices.parquet"
REND_DIR = os.path.join(BASE_DIR, "db", "rend_year")

TRADING_DAYS = 252.0

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

def _winsorize(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Winsoriza en percentiles [lo, hi]. Seguro con arrays pequeños."""
    if len(arr) < 10:
        return arr
    lo_val, hi_val = np.percentile(arr, [lo, hi])
    return np.clip(arr, lo_val, hi_val)

def _safe_annualize(r_raw: float, period: int) -> float:
    """
    Anualiza un retorno acumulado bruto con tres capas de protección:
      1. Clamp del retorno bruto a [-99.99%, +500%] antes de exponenciar.
         (500% raw en N días sigue siendo un outlier extremo; por encima
         la anualización es artefacto, no señal.)
      2. Usa log/exp en vez de potencia directa → sin OverflowError.
      3. Clamp del resultado anualizado a [-100%, +10 000%].
    """
    r_clipped = max(-0.9999, min(r_raw, 5.0))          # capa 1: cap raw ±
    log_ann   = np.log1p(r_clipped) * (TRADING_DAYS / float(period))  # capa 2: log-space
    result    = np.expm1(np.clip(log_ann, -20.0, 20.0))               # capa 3: exp clamp
    return float(result)

def _sharpe(returns: np.ndarray, period: int, annualize: bool, rf_annual: float = 0.0) -> float:
    """Sharpe del portfolio sobre la distribución de retornos individuales."""
    rf_period = (1.0 + rf_annual) ** (float(period) / TRADING_DAYS) - 1.0
    excess = returns - rf_period
    std = float(np.std(excess))
    if std < 1e-9:
        return 0.0
    s = float(np.mean(excess)) / std
    return s * float(np.sqrt(TRADING_DAYS / period)) if annualize else s

def _geo_mean_return(returns: np.ndarray) -> float:
    """Media geométrica de retornos: exp(mean(log(1+r))) - 1.
    Más precisa que np.mean para anualizar retornos compuestos."""
    clipped = np.clip(returns, -0.9999, None)
    return float(np.expm1(np.mean(np.log1p(clipped))))

def _load_granular(year: int, rend_dir: str) -> pd.DataFrame:
    path = os.path.join(rend_dir, f"granular_alpha_{year}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, f"No hay datos granulares para {year}")
    return pd.read_parquet(path, columns=["ticker", "fecha", "day", "alpha"])

def _build_matrix(df: pd.DataFrame):
    """
    Convierte el DataFrame en una matriz (n_señales × 252) de alphas.
    Devuelve (mat, cum_max, n_signals).
    """
    df["_sig"]            = df["ticker"] + "|" + df["fecha"]
    sig_codes, sig_uniq  = pd.factorize(df["_sig"])
    n_signals             = len(sig_uniq)

    mat      = np.full((n_signals, 252), np.nan, dtype=np.float32)
    days_idx = df["day"].to_numpy(dtype=np.int32) - 1
    alpha_v  = df["alpha"].to_numpy(dtype=np.float32)
    valid    = (days_idx >= 0) & (days_idx < 252)
    mat[sig_codes[valid], days_idx[valid]] = alpha_v[valid]

    mat_filled = np.where(np.isnan(mat), np.float32(-np.inf), mat)
    cum_max    = np.maximum.accumulate(mat_filled, axis=1)
    return mat, cum_max, n_signals

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

@app.get("/rend", tags=["Rendimiento"], summary="Rendimiento ticker", description="Obtiene el rendimiento de un ticker a lo largo del tiempo frente al S&P500", responses={
        200: {
            "description": "Petición exitosa",
            "content": {
                "application/json": {
                    "example": []
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

@app.get("/max_year", tags=["Máximos 52W"], summary="Conteo máximos", description="Cuenta el número de empresas que hacen máximo en cada día de un año entero", responses={
        200: {
            "description": "Petición exitosa",
            "content": {
                "application/json": {
                    "example": {"2023-01-03":82,"2023-01-04":70,"2023-01-05":68,"2023-01-06":111,"2023-01-09":128,"2023-01-10":91,"2023-01-11":118,"2023-01-12":166,"2023-01-13":164,"2023-01-17":164,"2023-01-18":140,"2023-01-19":36,"2023-01-20":74,"2023-01-23":127,"2023-01-24":131,"2023-01-25":105,"2023-01-26":172,"2023-01-27":172,"2023-01-30":82,"2023-01-31":163,"2023-02-01":285,"2023-02-02":334,"2023-02-03":210,"2023-02-06":101,"2023-02-07":130,"2023-02-08":115,"2023-02-09":127,"2023-02-10":71,"2023-02-13":122,"2023-02-14":134,"2023-02-15":134,"2023-02-16":141,"2023-02-17":121,"2023-02-21":61,"2023-02-22":54,"2023-02-23":103,"2023-02-24":75,"2023-02-27":124,"2023-02-28":140,"2023-03-01":134,"2023-03-02":132,"2023-03-03":192,"2023-03-06":164,"2023-03-07":75,"2023-03-08":83,"2023-03-09":95,"2023-03-10":36,"2023-03-13":25,"2023-03-14":32,"2023-03-15":24,"2023-03-16":46,"2023-03-17":31,"2023-03-20":40,"2023-03-21":69,"2023-03-22":70,"2023-03-23":89,"2023-03-24":37,"2023-03-27":69,"2023-03-28":64,"2023-03-29":99,"2023-03-30":117,"2023-03-31":153,"2023-04-03":134,"2023-04-04":133,"2023-04-05":88,"2023-04-06":81,"2023-04-10":78,"2023-04-11":120,"2023-04-12":130,"2023-04-13":154,"2023-04-14":98,"2023-04-17":107,"2023-04-18":132,"2023-04-19":117,"2023-04-20":124,"2023-04-21":111,"2023-04-24":118,"2023-04-25":80,"2023-04-26":67,"2023-04-27":90,"2023-04-28":120,"2023-05-01":160,"2023-05-02":90,"2023-05-03":145,"2023-05-04":107,"2023-05-05":128,"2023-05-08":128,"2023-05-09":115,"2023-05-10":143,"2023-05-11":103,"2023-05-12":106,"2023-05-15":93,"2023-05-16":84,"2023-05-17":120,"2023-05-18":166,"2023-05-19":152,"2023-05-22":128,"2023-05-23":87,"2023-05-24":29,"2023-05-25":86,"2023-05-26":124,"2023-05-30":131,"2023-05-31":56,"2023-06-01":74,"2023-06-02":166,"2023-06-05":144,"2023-06-06":202,"2023-06-07":268,"2023-06-08":155,"2023-06-09":155,"2023-06-12":198,"2023-06-13":297,"2023-06-14":221,"2023-06-15":203,"2023-06-16":266,"2023-06-20":134,"2023-06-21":168,"2023-06-22":97,"2023-06-23":84,"2023-06-26":104,"2023-06-27":169,"2023-06-28":185,"2023-06-29":238,"2023-06-30":336,"2023-07-03":150,"2023-07-05":98,"2023-07-06":50,"2023-07-07":87,"2023-07-10":124,"2023-07-11":229,"2023-07-12":341,"2023-07-13":338,"2023-07-14":221,"2023-07-17":289,"2023-07-18":371,"2023-07-19":312,"2023-07-20":172,"2023-07-21":166,"2023-07-24":167,"2023-07-25":215,"2023-07-26":200,"2023-07-27":266,"2023-07-28":207,"2023-07-31":249,"2023-08-01":186,"2023-08-02":102,"2023-08-03":129,"2023-08-04":162,"2023-08-07":164,"2023-08-08":113,"2023-08-09":140,"2023-08-10":126,"2023-08-11":91,"2023-08-14":90,"2023-08-15":73,"2023-08-16":58,"2023-08-17":32,"2023-08-18":29,"2023-08-21":60,"2023-08-22":62,"2023-08-23":89,"2023-08-24":65,"2023-08-25":47,"2023-08-28":93,"2023-08-29":115,"2023-08-30":162,"2023-08-31":147,"2023-09-01":197,"2023-09-05":119,"2023-09-06":80,"2023-09-07":65,"2023-09-08":91,"2023-09-11":115,"2023-09-12":121,"2023-09-13":78,"2023-09-14":114,"2023-09-15":112,"2023-09-18":83,"2023-09-19":89,"2023-09-20":94,"2023-09-21":39,"2023-09-22":50,"2023-09-25":74,"2023-09-26":51,"2023-09-27":64,"2023-09-28":75,"2023-09-29":59,"2023-10-02":35,"2023-10-03":19,"2023-10-04":24,"2023-10-05":30,"2023-10-06":53,"2023-10-09":59,"2023-10-10":99,"2023-10-11":81,"2023-10-12":71,"2023-10-13":52,"2023-10-16":71,"2023-10-17":91,"2023-10-18":50,"2023-10-19":27,"2023-10-20":17,"2023-10-23":20,"2023-10-24":26,"2023-10-25":35,"2023-10-26":28,"2023-10-27":34,"2023-10-30":23,"2023-10-31":31,"2023-11-01":62,"2023-11-02":114,"2023-11-03":141,"2023-11-06":77,"2023-11-07":81,"2023-11-08":78,"2023-11-09":84,"2023-11-10":110,"2023-11-13":123,"2023-11-14":235,"2023-11-15":201,"2023-11-16":87,"2023-11-17":129,"2023-11-20":180,"2023-11-21":133,"2023-11-22":175,"2023-11-24":171,"2023-11-27":191,"2023-11-28":136,"2023-11-29":170,"2023-11-30":191,"2023-12-01":305,"2023-12-04":292,"2023-12-05":176,"2023-12-06":252,"2023-12-07":147,"2023-12-08":216,"2023-12-11":256,"2023-12-12":279,"2023-12-13":471,"2023-12-14":652,"2023-12-15":364,"2023-12-18":275,"2023-12-19":460,"2023-12-20":463,"2023-12-21":210,"2023-12-22":399,"2023-12-26":493,"2023-12-27":503,"2023-12-28":288,"2023-12-29":191}
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
                    "example": {"detail": [{"loc": ["query", "day"], "msg": "Input should be less than or equal to 2025", "type": "less_than_equal"}]}
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
async def get_max_year(
    year: int = Query(..., ge=2000, description="Año en formato YYYY", openapi_examples={
                "normal": {
                    "description": "Ejemplo válido",
                    "value": 2023
                }}),
    token: str = Depends(verificar_api_key)):   
    conteo = {}

    for row in PRICE_DATA.itertuples():
        if row.FECHA.startswith(str(year)):
            if row.FECHA not in conteo:
                conteo[row.FECHA] = 0
            conteo[row.FECHA] += 1

    return dict(sorted(conteo.items()))

@app.get("/rend_year", tags=["Rendimiento"], summary="Rendimiento agregado año 52W", description=("Para un año X, toma todas las señales de máximo 52 semanas entre el 01/01/X y el 31/12/X y calcula la curva media de rendimiento durante las 252 sesiones posteriores. "), responses={
             200: {
                 "description": "Petición exitosa",
                 "content": {
                     "application/json": {
                        "example": {"año":2023,"señales_procesadas":34133,"all":[[1,-0.00094421,0.00055077,-0.00149498],[2,-0.00102409,0.00112964,-0.00215373],[3,-0.00156816,0.00139668,-0.00296484],[4,-0.00227426,0.00181599,-0.00409025],[5,-0.00231059,0.00242764,-0.00473823],[6,-0.00214035,0.00324494,-0.00538529],[7,-0.00193483,0.00401089,-0.00594573],[8,-0.00222608,0.0047078,-0.00693388],[9,-0.00270282,0.00508152,-0.00778434],[10,-0.00332477,0.00557237,-0.00889714],[11,-0.00336738,0.00590082,-0.00926819],[12,-0.00387266,0.00612241,-0.00999507],[13,-0.00412606,0.00657875,-0.01070481],[14,-0.00438126,0.00724812,-0.01162938],[15,-0.00421174,0.00813314,-0.01234488],[16,-0.00372176,0.00897829,-0.01270005],[17,-0.00367099,0.00996764,-0.01363863],[18,-0.00328435,0.01113602,-0.01442037],[19,-0.00311699,0.01211157,-0.01522856],[20,-0.00333726,0.01269938,-0.01603664],[21,-0.0040783,0.01317901,-0.01725731],[22,-0.00357521,0.01395766,-0.01753287],[23,-0.00341307,0.01484923,-0.0182623],[24,-0.0035457,0.01563499,-0.01918069],[25,-0.00369978,0.01651926,-0.02021904],[26,-0.00416614,0.01737992,-0.02154606],[27,-0.00442258,0.01844137,-0.02286395],[28,-0.00459754,0.01945422,-0.02405176],[29,-0.00499092,0.02027712,-0.02526804],[30,-0.00481975,0.02102667,-0.02584642],[31,-0.00472865,0.02180205,-0.0265307],[32,-0.00479977,0.02245721,-0.02725698],[33,-0.00455468,0.02319479,-0.02774947],[34,-0.00512681,0.02394164,-0.02906845],[35,-0.00545258,0.02453989,-0.02999246],[36,-0.00575194,0.02527261,-0.03102455],[37,-0.00573387,0.0262154,-0.03194927],[38,-0.00559232,0.02702981,-0.03262213],[39,-0.00616345,0.02773137,-0.03389481],[40,-0.00650266,0.02871182,-0.03521449],[41,-0.00688303,0.02990558,-0.03678861],[42,-0.00644198,0.03121566,-0.03765764],[43,-0.00592756,0.03227983,-0.0382074],[44,-0.00526645,0.03303627,-0.03830273],[45,-0.00503258,0.03392974,-0.03896232],[46,-0.00442554,0.03481859,-0.03924413],[47,-0.00441533,0.03574355,-0.04015888],[48,-0.00393534,0.03677252,-0.04070786],[49,-0.00334352,0.03769829,-0.04104181],[50,-0.0028059,0.03872832,-0.04153422],[51,-0.00229418,0.03993973,-0.04223391],[52,-0.0019831,0.04078904,-0.04277214],[53,-0.00192224,0.04130147,-0.04322371],[54,-0.00184417,0.04221634,-0.04406051],[55,-0.00205609,0.04318372,-0.04523981],[56,-0.00151154,0.04383935,-0.04535089],[57,-0.00102095,0.04471806,-0.04573901],[58,-0.00051851,0.04594816,-0.04646666],[59,-0.00051651,0.04666636,-0.04718287],[60,0.00056166,0.04734658,-0.04678492],[61,0.00106463,0.04822239,-0.04715776],[62,0.00171663,0.0492143,-0.04749768],[63,0.00117004,0.04996017,-0.04879012],[64,0.00151971,0.05034697,-0.04882726],[65,0.00133047,0.05085883,-0.04952836],[66,0.00321359,0.05141158,-0.04819799],[67,0.00317483,0.0521333,-0.04895847],[68,0.00287227,0.05276572,-0.04989345],[69,0.00331699,0.05349029,-0.0501733],[70,0.0035555,0.05401471,-0.05045921],[71,0.00378567,0.0545265,-0.05074083],[72,0.00173018,0.05485828,-0.0531281],[73,0.00167827,0.05503376,-0.05335549],[74,0.00127079,0.05514334,-0.05387255],[75,0.00072754,0.05538841,-0.05466087],[76,0.00101922,0.05586141,-0.05484219],[77,0.001879,0.05643322,-0.05455423],[78,0.00225401,0.05736246,-0.05510844],[79,0.00278722,0.05828617,-0.05549895],[80,0.00250439,0.05878523,-0.05628084],[81,0.00193414,0.05901823,-0.05708409],[82,0.0017971,0.0594421,-0.057645],[83,0.00169979,0.06008004,-0.05838025],[84,0.00202203,0.06060601,-0.05858398],[85,0.00225855,0.06124672,-0.05898817],[86,0.0036643,0.0625141,-0.0588498],[87,0.00446761,0.06387631,-0.0594087],[88,0.00543706,0.0650358,-0.05959874],[89,0.0064393,0.06589244,-0.05945314],[90,0.00755665,0.0671336,-0.05957695],[91,0.00816602,0.06805753,-0.05989151],[92,0.00898409,0.06899855,-0.06001447],[93,0.00981689,0.06993641,-0.06011953],[94,0.01089649,0.07115421,-0.06025772],[95,0.01294447,0.07297404,-0.06002957],[96,0.01520053,0.07470963,-0.0595091],[97,0.01664448,0.07616009,-0.05951561],[98,0.01816587,0.07752535,-0.05935948],[99,0.0197079,0.07900309,-0.05929519],[100,0.02116697,0.08039818,-0.05923121],[101,0.02251116,0.08155848,-0.05904732],[102,0.0243336,0.083089,-0.05875539],[103,0.02599288,0.08464024,-0.05864737],[104,0.02665037,0.08588924,-0.05923888],[105,0.02722796,0.08691622,-0.05968826],[106,0.02867267,0.08805956,-0.05938689],[107,0.03002351,0.08930329,-0.05927978],[108,0.03044403,0.09039686,-0.05995282],[109,0.03113826,0.09134954,-0.06021129],[110,0.03244956,0.09256315,-0.06011359],[111,0.03314101,0.09380234,-0.06066133],[112,0.03370193,0.09497679,-0.06127487],[113,0.0342845,0.09606111,-0.06177661],[114,0.03513968,0.09734663,-0.06220695],[115,0.03471463,0.098551,-0.06383637],[116,0.03491009,0.09994145,-0.06503137],[117,0.03533818,0.10137059,-0.06603241],[118,0.03590778,0.10257177,-0.06666398],[119,0.036591,0.10367432,-0.06708332],[120,0.03715912,0.10493123,-0.06777211],[121,0.03828531,0.10641037,-0.06812505],[122,0.03977546,0.10802845,-0.06825299],[123,0.04150121,0.10957166,-0.06807046],[124,0.04239442,0.11091014,-0.06851572],[125,0.04352033,0.11207839,-0.06855807],[126,0.04459195,0.11339415,-0.0688022],[127,0.04550292,0.11464618,-0.06914326],[128,0.04616279,0.1157558,-0.06959301],[129,0.04732251,0.11709302,-0.06977051],[130,0.04836004,0.11873042,-0.07037038],[131,0.04930799,0.12034663,-0.07103864],[132,0.0505626,0.12190803,-0.07134543],[133,0.05220981,0.12321474,-0.07100493],[134,0.05340561,0.12429655,-0.07089094],[135,0.05444058,0.12521901,-0.07077843],[136,0.05582635,0.12645336,-0.07062701],[137,0.05678004,0.1273418,-0.07056176],[138,0.05736452,0.12797168,-0.07060715],[139,0.0584824,0.12875757,-0.07027517],[140,0.06054798,0.12977351,-0.06922553],[141,0.06213724,0.13055081,-0.06841357],[142,0.06346855,0.13126121,-0.06779265],[143,0.0645698,0.13198612,-0.06741631],[144,0.06578399,0.13262354,-0.06683955],[145,0.06669897,0.13333396,-0.06663499],[146,0.0669981,0.13366717,-0.06666908],[147,0.06852954,0.1338976,-0.06536806],[148,0.06892258,0.13435138,-0.0654288],[149,0.06911034,0.13485935,-0.06574901],[150,0.06886397,0.13476098,-0.06589701],[151,0.069124,0.13510632,-0.06598232],[152,0.06879823,0.13568706,-0.06688883],[153,0.0695716,0.1363581,-0.0667865],[154,0.0709676,0.13700952,-0.06604193],[155,0.07206579,0.1378178,-0.06575201],[156,0.0730105,0.13862904,-0.06561854],[157,0.07314903,0.13923273,-0.0660837],[158,0.07307636,0.13963962,-0.06656327],[159,0.07375209,0.14033166,-0.06657957],[160,0.07476975,0.14180761,-0.06703786],[161,0.07613471,0.14328403,-0.06714932],[162,0.07740737,0.14453748,-0.06713011],[163,0.07834515,0.14578887,-0.06744372],[164,0.07937722,0.14722985,-0.06785263],[165,0.08025072,0.14827219,-0.06802146],[166,0.08108312,0.14922524,-0.06814212],[167,0.08217326,0.15041778,-0.06824452],[168,0.08282944,0.15162776,-0.06879833],[169,0.0839353,0.15264418,-0.06870888],[170,0.08471895,0.15357079,-0.06885185],[171,0.08498116,0.15390005,-0.06891889],[172,0.08543208,0.15439817,-0.0689661],[173,0.08622388,0.15507119,-0.0688473],[174,0.08670249,0.15582512,-0.06912264],[175,0.08795377,0.15651314,-0.06855937],[176,0.08925848,0.15764821,-0.06838973],[177,0.08968192,0.15846655,-0.06878462],[178,0.09026003,0.15886634,-0.06860631],[179,0.09129858,0.15946181,-0.06816323],[180,0.09291447,0.16061912,-0.06770465],[181,0.0938717,0.16154616,-0.06767446],[182,0.09549549,0.16283139,-0.0673359],[183,0.09654232,0.16441256,-0.06787024],[184,0.09776384,0.1657776,-0.06801375],[185,0.09981285,0.16723835,-0.0674255],[186,0.10143354,0.16871881,-0.06728527],[187,0.10290252,0.17010742,-0.0672049],[188,0.10301871,0.17133592,-0.06831722],[189,0.10475277,0.17267253,-0.06791976],[190,0.10576763,0.17402224,-0.06825461],[191,0.10659993,0.17543011,-0.06883018],[192,0.10703768,0.17694416,-0.06990648],[193,0.1089685,0.17869191,-0.06972341],[194,0.10925413,0.1803288,-0.07107467],[195,0.1104663,0.18197183,-0.07150553],[196,0.11139184,0.18368463,-0.0722928],[197,0.11240229,0.18511241,-0.07271012],[198,0.11389388,0.18647695,-0.07258308],[199,0.11507347,0.18788806,-0.07281459],[200,0.11673618,0.18950344,-0.07276727],[201,0.11803827,0.1911014,-0.07306313],[202,0.11939856,0.19270432,-0.07330577],[203,0.12148415,0.19479167,-0.07330752],[204,0.12240907,0.19641889,-0.07400982],[205,0.1236765,0.19794935,-0.07427285],[206,0.12415986,0.19927299,-0.07511313],[207,0.12622176,0.20087683,-0.07465507],[208,0.12745421,0.20213893,-0.07468472],[209,0.12921252,0.20334038,-0.07412787],[210,0.12990728,0.20437282,-0.07446554],[211,0.13067524,0.2054905,-0.07481526],[212,0.13090066,0.20664698,-0.07574632],[213,0.13166811,0.20769363,-0.07602552],[214,0.13332752,0.20894096,-0.07561344],[215,0.13576991,0.21069665,-0.07492674],[216,0.13848951,0.2122413,-0.07375179],[217,0.14151146,0.21356891,-0.07205745],[218,0.14436537,0.215088,-0.07072263],[219,0.14722224,0.21647447,-0.06925224],[220,0.15025828,0.21770014,-0.06744187],[221,0.15213055,0.21907259,-0.06694204],[222,0.15418621,0.22034496,-0.06615875],[223,0.15561836,0.22170444,-0.06608608],[224,0.1574875,0.22326923,-0.06578173],[225,0.15966596,0.22487704,-0.06521108],[226,0.16118574,0.22651615,-0.06533041],[227,0.16347554,0.22801732,-0.06454178],[228,0.16524841,0.22923293,-0.06398452],[229,0.16651886,0.23060098,-0.06408213],[230,0.16821044,0.23201389,-0.06380345],[231,0.17005667,0.23344206,-0.06338539],[232,0.17085478,0.23518284,-0.06432806],[233,0.18014471,0.23719566,-0.05705094],[234,0.18152242,0.23894383,-0.05742141],[235,0.18291301,0.24064495,-0.05773193],[236,0.18396565,0.24203639,-0.05807074],[237,0.18515431,0.24307283,-0.05791852],[238,0.18641628,0.24432563,-0.05790936],[239,0.18828346,0.24572575,-0.05744229],[240,0.18984983,0.24724569,-0.05739586],[241,0.19134756,0.24847682,-0.05712926],[242,0.19888898,0.24965957,-0.05077059],[243,0.19984591,0.2504529,-0.05060699],[244,0.2000985,0.25145492,-0.05135642],[245,0.1994594,0.25217162,-0.05271222],[246,0.2007238,0.25275313,-0.05202932],[247,0.20067911,0.25353976,-0.05286065],[248,0.20193946,0.25461904,-0.05267958],[249,0.20405676,0.25540018,-0.05134342],[250,0.20839073,0.25585651,-0.04746577],[251,0.21299748,0.25652708,-0.04352961],[252,0.2099901,0.25704031,-0.04705021]],"cap_1b":[[1,-0.00065682,0.00054695,-0.00120377],[2,-0.00101753,0.00108544,-0.00210297],[3,-0.00157085,0.00119551,-0.00276636],[4,-0.00183882,0.0015095,-0.00334832],[5,-0.00166761,0.00209019,-0.0037578],[6,-0.00112714,0.00294547,-0.00407261],[7,-0.00038793,0.00380839,-0.00419633],[8,-0.00021323,0.00454133,-0.00475456],[9,-0.00032936,0.00484215,-0.00517151],[10,-0.0002897,0.00534473,-0.00563443],[11,-0.00017243,0.00551522,-0.00568764],[12,-0.00071225,0.00557215,-0.0062844],[13,-0.00095489,0.00593811,-0.006893],[14,-0.00093662,0.00647627,-0.0074129],[15,-0.00070126,0.00733954,-0.0080408],[16,-0.00047372,0.00822312,-0.00869683],[17,0.00017951,0.00931634,-0.00913683],[18,0.00073839,0.010611,-0.00987261],[19,0.00116928,0.01162584,-0.01045656],[20,0.00094305,0.01218908,-0.01124603],[21,0.0005308,0.01256836,-0.01203756],[22,0.00070774,0.01333644,-0.0126287],[23,0.00117403,0.0141756,-0.01300158],[24,0.00091558,0.01486113,-0.01394555],[25,0.00094295,0.0157735,-0.01483055],[26,0.00108535,0.01666142,-0.01557607],[27,0.00159824,0.01778279,-0.01618455],[28,0.00202037,0.01887633,-0.01685596],[29,0.0025214,0.01975775,-0.01723635],[30,0.00298804,0.02048127,-0.01749323],[31,0.00342222,0.02118609,-0.01776387],[32,0.00357879,0.02170282,-0.01812403],[33,0.00372802,0.02227164,-0.01854362],[34,0.00420758,0.02295295,-0.01874537],[35,0.00434197,0.02349683,-0.01915486],[36,0.00451144,0.02413024,-0.0196188],[37,0.00511732,0.02507311,-0.01995579],[38,0.00519478,0.02582829,-0.0206335],[39,0.00495385,0.02644635,-0.0214925],[40,0.00506927,0.02732953,-0.02226026],[41,0.00544681,0.02846662,-0.02301981],[42,0.0062317,0.02975217,-0.02352047],[43,0.00715147,0.03077995,-0.02362848],[44,0.0079159,0.03147917,-0.02356327],[45,0.00849066,0.03232827,-0.02383761],[46,0.0092913,0.033165,-0.0238737],[47,0.00976689,0.03406664,-0.02429976],[48,0.01099528,0.03511832,-0.02412304],[49,0.01193183,0.03598299,-0.02405116],[50,0.0125046,0.03693367,-0.02442907],[51,0.01332118,0.03814121,-0.02482003],[52,0.01407502,0.03894449,-0.02486948],[53,0.01401611,0.0393394,-0.02532328],[54,0.01400244,0.04007063,-0.02606819],[55,0.01386313,0.0408525,-0.02698937],[56,0.01383951,0.04134186,-0.02750235],[57,0.01423118,0.04217395,-0.02794278],[58,0.01498926,0.04337699,-0.02838773],[59,0.01464464,0.04394331,-0.02929868],[60,0.01489672,0.04448896,-0.02959224],[61,0.015617,0.0452734,-0.0296564],[62,0.01625092,0.0462801,-0.03002919],[63,0.01622342,0.04694631,-0.0307229],[64,0.01672894,0.04720502,-0.03047608],[65,0.01729774,0.04759575,-0.03029801],[66,0.01753457,0.04812532,-0.03059075],[67,0.0179044,0.04883883,-0.03093443],[68,0.01836553,0.04940355,-0.03103802],[69,0.01898657,0.05009194,-0.03110536],[70,0.01937356,0.05057051,-0.03119695],[71,0.01933196,0.05111966,-0.0317877],[72,0.01921383,0.0515178,-0.03230397],[73,0.01920782,0.05176255,-0.03255473],[74,0.01877071,0.05181762,-0.03304691],[75,0.01843494,0.05202793,-0.03359299],[76,0.01881424,0.05261097,-0.03379673],[77,0.01931246,0.0532968,-0.03398434],[78,0.01938191,0.05436366,-0.03498175],[79,0.01995303,0.05546281,-0.03550978],[80,0.01976412,0.05591636,-0.03615224],[81,0.01964327,0.05608862,-0.03644536],[82,0.01960473,0.05651595,-0.03691122],[83,0.01980685,0.05704801,-0.03724116],[84,0.02023663,0.05743692,-0.03720029],[85,0.02110225,0.05801556,-0.03691331],[86,0.02302586,0.05937084,-0.03634499],[87,0.02415769,0.06082448,-0.0366668],[88,0.02550741,0.06201977,-0.03651236],[89,0.02667596,0.06286391,-0.03618795],[90,0.02811278,0.06414909,-0.03603631],[91,0.02889224,0.06501784,-0.0361256],[92,0.02941986,0.0658055,-0.03638563],[93,0.03090062,0.06654215,-0.03564153],[94,0.03272631,0.06766161,-0.03493531],[95,0.0351312,0.06954232,-0.03441111],[96,0.03744731,0.07143419,-0.03398688],[97,0.03942726,0.07296577,-0.03353851],[98,0.0410483,0.07440325,-0.03335495],[99,0.04286536,0.0759672,-0.03310183],[100,0.04438544,0.07744295,-0.03305751],[101,0.04572231,0.07859035,-0.03286804],[102,0.04775091,0.08011068,-0.03235978],[103,0.04947226,0.08171111,-0.03223885],[104,0.05070054,0.08300875,-0.03230821],[105,0.05165854,0.08407963,-0.0324211],[106,0.05348326,0.08536069,-0.03187743],[107,0.05518264,0.08666809,-0.03148545],[108,0.0561546,0.08780227,-0.03164768],[109,0.05727072,0.08883133,-0.03156061],[110,0.05894004,0.09011479,-0.03117475],[111,0.05998358,0.0914673,-0.03148372],[112,0.06096358,0.09261819,-0.03165461],[113,0.06191882,0.09365359,-0.03173477],[114,0.06332055,0.09497886,-0.03165832],[115,0.06365814,0.09619694,-0.0325388],[116,0.06432027,0.09775768,-0.0334374],[117,0.06526116,0.09924997,-0.03398881],[118,0.06645614,0.1004376,-0.03398147],[119,0.06739837,0.10146527,-0.0340669],[120,0.06802715,0.10273951,-0.03471236],[121,0.06952739,0.10418122,-0.03465383],[122,0.07155039,0.10582189,-0.0342715],[123,0.07319811,0.10746012,-0.034262],[124,0.07382425,0.10878245,-0.0349582],[125,0.07477074,0.10997733,-0.03520659],[126,0.07598756,0.11136393,-0.03537637],[127,0.07669089,0.11264359,-0.0359527],[128,0.07739282,0.11366679,-0.03627397],[129,0.0784715,0.11504323,-0.03657173],[130,0.07962941,0.11681623,-0.03718683],[131,0.0808514,0.11859227,-0.03774087],[132,0.08259682,0.1202889,-0.03769209],[133,0.08421672,0.12164504,-0.03742832],[134,0.08570308,0.12284428,-0.0371412],[135,0.08716458,0.12386365,-0.03669907],[136,0.08859389,0.12524004,-0.03664614],[137,0.09007524,0.12630516,-0.03622992],[138,0.09121872,0.12708437,-0.03586565],[139,0.09267265,0.12795582,-0.03528317],[140,0.09516691,0.1292057,-0.0340388],[141,0.09770339,0.13016815,-0.03246476],[142,0.09994833,0.13108516,-0.03113683],[143,0.10204237,0.13205364,-0.03001127],[144,0.10394998,0.13284476,-0.02889478],[145,0.10554141,0.13360986,-0.02806844],[146,0.1060655,0.13389079,-0.02782529],[147,0.10684229,0.13408471,-0.02724242],[148,0.10777916,0.13462324,-0.02684408],[149,0.10848178,0.13528094,-0.02679916],[150,0.10872673,0.13526111,-0.02653438],[151,0.10933154,0.13562298,-0.02629144],[152,0.11044338,0.1362036,-0.02576021],[153,0.11135529,0.13693318,-0.02557789],[154,0.11200682,0.1376462,-0.02563939],[155,0.11308649,0.13849745,-0.02541096],[156,0.11394898,0.13926383,-0.02531485],[157,0.1141321,0.13962633,-0.02549422],[158,0.11437377,0.13975535,-0.02538159],[159,0.11520157,0.14032991,-0.02512834],[160,0.11659137,0.14182262,-0.02523125],[161,0.11844851,0.14340871,-0.0249602],[162,0.12005657,0.14466998,-0.02461341],[163,0.12174135,0.14609981,-0.02435846],[164,0.1235835,0.14767899,-0.02409549],[165,0.12442975,0.14873449,-0.02430474],[166,0.12519969,0.14966687,-0.02446718],[167,0.12665994,0.15094177,-0.02428183],[168,0.12826421,0.15225207,-0.02398787],[169,0.1299882,0.15324884,-0.02326064],[170,0.13133834,0.15421143,-0.02287309],[171,0.13172562,0.15440967,-0.02268405],[172,0.13296608,0.15490628,-0.0219402],[173,0.1339553,0.15561715,-0.02166186],[174,0.13450807,0.15632601,-0.02181794],[175,0.13531186,0.15697862,-0.02166676],[176,0.13651467,0.15815516,-0.02164049],[177,0.13674057,0.1588393,-0.02209874],[178,0.13701913,0.15898078,-0.02196165],[179,0.13822185,0.15942378,-0.02120193],[180,0.13939925,0.16051363,-0.02111438],[181,0.14015259,0.16131652,-0.02116393],[182,0.1418362,0.16244185,-0.02060565],[183,0.14337923,0.16392413,-0.0205449],[184,0.14463616,0.16515951,-0.02052335],[185,0.1464888,0.16668482,-0.02019603],[186,0.14824722,0.16816212,-0.01991489],[187,0.1498373,0.16946363,-0.01962633],[188,0.1508038,0.17059533,-0.01979153],[189,0.15203172,0.17180808,-0.01977637],[190,0.15293854,0.1730538,-0.02011526],[191,0.15394942,0.17437641,-0.02042699],[192,0.15538207,0.17585497,-0.0204729],[193,0.15701421,0.17767375,-0.02065953],[194,0.15873769,0.1793487,-0.02061102],[195,0.16006333,0.18100691,-0.02094358],[196,0.16188766,0.18273995,-0.02085229],[197,0.16347368,0.18408579,-0.02061211],[198,0.16502578,0.18534403,-0.02031825],[199,0.16623924,0.18659351,-0.02035427],[200,0.16808104,0.18813124,-0.0200502],[201,0.16948147,0.18960397,-0.0201225],[202,0.17072777,0.19114999,-0.02042222],[203,0.17280823,0.19337323,-0.02056499],[204,0.17404195,0.19510528,-0.02106333],[205,0.1753683,0.19674932,-0.02138102],[206,0.17638483,0.19809551,-0.02171068],[207,0.17786987,0.19977788,-0.02190801],[208,0.17929477,0.20099257,-0.0216978],[209,0.18047989,0.20219856,-0.02171867],[210,0.18133484,0.20324315,-0.02190832],[211,0.18197181,0.20442205,-0.02245024],[212,0.18293373,0.20579783,-0.0228641],[213,0.18399319,0.20694739,-0.0229542],[214,0.18569978,0.20828344,-0.02258366],[215,0.18780189,0.21008903,-0.02228715],[216,0.1896362,0.21160695,-0.02197075],[217,0.19120052,0.21281511,-0.02161459],[218,0.19249192,0.21423238,-0.02174046],[219,0.19382705,0.21550279,-0.02167574],[220,0.19481039,0.21660135,-0.02179096],[221,0.19567459,0.21782723,-0.02215265],[222,0.19636312,0.21902448,-0.02266136],[223,0.19700495,0.22041981,-0.02341487],[224,0.19837641,0.22211713,-0.02374072],[225,0.1996509,0.2237957,-0.0241448],[226,0.20099903,0.22552126,-0.02452224],[227,0.20306691,0.22707156,-0.02400466],[228,0.20427238,0.22819779,-0.02392541],[229,0.20529823,0.22951085,-0.02421261],[230,0.20658533,0.23088438,-0.02429906],[231,0.20775425,0.23230516,-0.02455091],[232,0.20911807,0.23409372,-0.02497565],[233,0.21138806,0.2362238,-0.02483574],[234,0.21333355,0.23804871,-0.02471515],[235,0.21578899,0.23990324,-0.02411424],[236,0.21762068,0.24131414,-0.02369346],[237,0.21865631,0.24229513,-0.02363881],[238,0.220311,0.2435385,-0.02322749],[239,0.22187204,0.24493294,-0.02306089],[240,0.22303504,0.24650004,-0.023465],[241,0.22428889,0.24773778,-0.02344889],[242,0.22517775,0.2490177,-0.02383995],[243,0.2254085,0.24983043,-0.02442193],[244,0.22619791,0.25097223,-0.02477432],[245,0.22714088,0.25170313,-0.02456225],[246,0.22747035,0.25230243,-0.02483208],[247,0.22743961,0.25301472,-0.02557511],[248,0.22793374,0.25397504,-0.0260413],[249,0.22824835,0.25467312,-0.02642477],[250,0.22780844,0.25504279,-0.02723435],[251,0.22766508,0.25562804,-0.02796296],[252,0.22721654,0.25596459,-0.02874804]],"cap_5b":[[1,-0.00043199,0.00054222,-0.00097422],[2,-0.00089076,0.00104161,-0.00193237],[3,-0.00148701,0.00121934,-0.00270636],[4,-0.00171129,0.00149442,-0.00320571],[5,-0.00152241,0.00202501,-0.00354741],[6,-0.00089696,0.00291587,-0.00381283],[7,-0.0002538,0.00391171,-0.0041655],[8,-2.065e-05,0.00469191,-0.00471256],[9,-7.2e-06,0.00512048,-0.00512767],[10,0.00020829,0.0057036,-0.00549531],[11,0.0002133,0.00599132,-0.00577802],[12,-0.00042134,0.00603631,-0.00645764],[13,-0.00064271,0.0064403,-0.00708301],[14,-0.00055147,0.00694227,-0.00749374],[15,-0.00034959,0.00785881,-0.0082084],[16,-5.839e-05,0.00886472,-0.00892311],[17,0.00068445,0.00999513,-0.00931067],[18,0.00154599,0.0112818,-0.0097358],[19,0.00214223,0.01228,-0.01013777],[20,0.00190576,0.01280695,-0.01090119],[21,0.00163406,0.01319588,-0.01156182],[22,0.00172498,0.01398537,-0.0122604],[23,0.00208966,0.01485414,-0.01276448],[24,0.00197635,0.01555227,-0.01357592],[25,0.00219168,0.01644033,-0.01424866],[26,0.0025702,0.01735614,-0.01478594],[27,0.00315596,0.01843154,-0.01527558],[28,0.00353268,0.01938035,-0.01584767],[29,0.00411641,0.02023787,-0.01612146],[30,0.00447793,0.02091812,-0.01644019],[31,0.00504254,0.02161639,-0.01657385],[32,0.00538883,0.02214704,-0.01675821],[33,0.00570734,0.02271581,-0.01700847],[34,0.00626996,0.02346615,-0.01719619],[35,0.00672982,0.02417545,-0.01744564],[36,0.00722625,0.02490715,-0.01768089],[37,0.00805184,0.02583324,-0.0177814],[38,0.00840277,0.02668344,-0.01828067],[39,0.0086856,0.02736308,-0.01867748],[40,0.00879612,0.02824,-0.01944388],[41,0.00928306,0.02934075,-0.02005769],[42,0.01014707,0.03064627,-0.0204992],[43,0.01105624,0.03172545,-0.02066921],[44,0.01199648,0.03249545,-0.02049897],[45,0.01278664,0.03343522,-0.02064858],[46,0.01375767,0.03427138,-0.02051371],[47,0.01421745,0.03516837,-0.02095093],[48,0.01576471,0.03623358,-0.02046887],[49,0.0167897,0.03716749,-0.02037779],[50,0.01770612,0.03812597,-0.02041985],[51,0.0188264,0.03931598,-0.02048957],[52,0.01988134,0.04009048,-0.02020913],[53,0.02010433,0.04057316,-0.02046883],[54,0.02028489,0.04126522,-0.02098033],[55,0.02030732,0.04197024,-0.02166292],[56,0.02046286,0.04237611,-0.02191326],[57,0.02069456,0.04317086,-0.0224763],[58,0.02149889,0.04432127,-0.02282238],[59,0.02132782,0.04488743,-0.02355962],[60,0.0217895,0.04546774,-0.02367824],[61,0.02253528,0.04624429,-0.02370901],[62,0.02296257,0.04710272,-0.02414015],[63,0.02278067,0.04760858,-0.02482791],[64,0.02340595,0.04779189,-0.02438594],[65,0.02361098,0.04811396,-0.02450299],[66,0.02371687,0.04864957,-0.0249327],[67,0.02435833,0.04939757,-0.02503924],[68,0.02511862,0.04993316,-0.02481454],[69,0.02571215,0.05058064,-0.02486849],[70,0.02626195,0.05104029,-0.02477834],[71,0.02624786,0.05146193,-0.02521407],[72,0.0265872,0.0518344,-0.02524719],[73,0.02656598,0.05208834,-0.02552236],[74,0.0259212,0.0521193,-0.0261981],[75,0.02586491,0.05232248,-0.02645757],[76,0.02647278,0.05301868,-0.0265459],[77,0.02705945,0.05380267,-0.02674323],[78,0.02731863,0.05495864,-0.02764001],[79,0.02820898,0.0560838,-0.02787482],[80,0.02826573,0.05653276,-0.02826702],[81,0.0281737,0.05662923,-0.02845552],[82,0.02774009,0.05694679,-0.0292067],[83,0.02773625,0.05725947,-0.02952322],[84,0.02801786,0.05755892,-0.02954106],[85,0.02896585,0.05813646,-0.02917061],[86,0.0305588,0.05942367,-0.02886487],[87,0.03166756,0.06083049,-0.02916293],[88,0.03313533,0.0621154,-0.02898007],[89,0.03413865,0.06289264,-0.028754],[90,0.03538087,0.06394231,-0.02856145],[91,0.03588916,0.06460036,-0.0287112],[92,0.03624853,0.06532351,-0.02907498],[93,0.0374167,0.06598452,-0.02856781],[94,0.03906784,0.06711235,-0.02804452],[95,0.04111849,0.06889009,-0.0277716],[96,0.04314537,0.07083237,-0.027687],[97,0.0451149,0.07240256,-0.02728766],[98,0.04669741,0.07384226,-0.02714485],[99,0.04822592,0.07526129,-0.02703537],[100,0.04956532,0.07662704,-0.02706172],[101,0.05098054,0.07767633,-0.02669578],[102,0.05258371,0.07903267,-0.02644896],[103,0.05387468,0.08052358,-0.0266489],[104,0.0551144,0.08190714,-0.02679275],[105,0.05605708,0.0831402,-0.02708312],[106,0.05774188,0.08444958,-0.0267077],[107,0.05916018,0.08573076,-0.02657058],[108,0.06001302,0.08688452,-0.0268715],[109,0.06088042,0.08785414,-0.02697371],[110,0.06210773,0.08899383,-0.0268861],[111,0.06292203,0.09025938,-0.02733735],[112,0.06383452,0.09141826,-0.02758374],[113,0.06490108,0.09243838,-0.02753729],[114,0.06646722,0.09378537,-0.02731815],[115,0.06690447,0.09504997,-0.0281455],[116,0.06805375,0.09669114,-0.02863739],[117,0.06920684,0.09814023,-0.02893339],[118,0.07036666,0.09923539,-0.02886873],[119,0.07123098,0.10019806,-0.02896707],[120,0.07218913,0.10150024,-0.02931111],[121,0.07380434,0.10291108,-0.02910674],[122,0.07573549,0.1044876,-0.02875211],[123,0.07732039,0.10617484,-0.02885445],[124,0.07810882,0.10750621,-0.02939739],[125,0.07907675,0.1087673,-0.02969055],[126,0.08050986,0.1101464,-0.02963654],[127,0.08134705,0.11142544,-0.03007839],[128,0.08202094,0.11235845,-0.03033751],[129,0.08321024,0.11370386,-0.03049362],[130,0.08439882,0.11542446,-0.03102564],[131,0.08564411,0.11721172,-0.0315676],[132,0.08738801,0.118976,-0.03158799],[133,0.08881026,0.12039973,-0.03158947],[134,0.09023687,0.1216602,-0.03142333],[135,0.09154453,0.12272989,-0.03118537],[136,0.09268296,0.1240226,-0.03133964],[137,0.09386046,0.1251169,-0.03125644],[138,0.09506691,0.12605747,-0.03099055],[139,0.09647172,0.12709095,-0.03061923],[140,0.09870995,0.12839377,-0.02968382],[141,0.10112814,0.12951247,-0.02838433],[142,0.10311917,0.13063283,-0.02751366],[143,0.10499046,0.13185856,-0.0268681],[144,0.10653289,0.1327654,-0.0262325],[145,0.10803185,0.13366619,-0.02563434],[146,0.1085131,0.13418861,-0.02567551],[147,0.10919762,0.13464906,-0.02545144],[148,0.10993826,0.13518984,-0.02525158],[149,0.11043216,0.13589417,-0.02546201],[150,0.11064073,0.1361211,-0.02548037],[151,0.11111229,0.13661026,-0.02549797],[152,0.11216689,0.13731104,-0.02514415],[153,0.11319984,0.13819013,-0.02499029],[154,0.11414234,0.13917494,-0.0250326],[155,0.11523584,0.14003895,-0.02480311],[156,0.11618402,0.14086719,-0.02468317],[157,0.11680744,0.14135242,-0.02454498],[158,0.11722092,0.14149867,-0.02427774],[159,0.11805855,0.14199188,-0.02393333],[160,0.11967913,0.14332243,-0.0236433],[161,0.12174978,0.14486594,-0.02311616],[162,0.12370851,0.14616817,-0.02245967],[163,0.12591299,0.14778428,-0.02187129],[164,0.12778896,0.14947069,-0.02168172],[165,0.12882279,0.15071733,-0.02189454],[166,0.12971776,0.15175953,-0.02204176],[167,0.13144433,0.15304873,-0.02160439],[168,0.13299842,0.15424787,-0.02124945],[169,0.13450257,0.15510012,-0.02059755],[170,0.13597175,0.15609453,-0.02012278],[171,0.13647294,0.15635669,-0.01988376],[172,0.13767817,0.15686484,-0.01918668],[173,0.13880655,0.15759827,-0.01879172],[174,0.13972471,0.15847806,-0.01875335],[175,0.14074947,0.15929495,-0.01854548],[176,0.14231245,0.16056965,-0.01825719],[177,0.14318271,0.16133863,-0.01815592],[178,0.14361585,0.16142808,-0.01781222],[179,0.14477142,0.16182555,-0.01705413],[180,0.14581523,0.16284115,-0.01702592],[181,0.14651369,0.16366759,-0.01715389],[182,0.14769228,0.16466021,-0.01696793],[183,0.14900932,0.1660665,-0.01705718],[184,0.15039711,0.16736172,-0.01696461],[185,0.15227815,0.16898412,-0.01670597],[186,0.15392223,0.17037805,-0.01645582],[187,0.15556692,0.17156034,-0.01599342],[188,0.15642575,0.17262653,-0.01620078],[189,0.15739482,0.17381213,-0.01641731],[190,0.15823355,0.17491347,-0.01667992],[191,0.15927986,0.17618512,-0.01690527],[192,0.16104195,0.17762534,-0.01658339],[193,0.1626037,0.17943029,-0.01682658],[194,0.16449457,0.18115884,-0.01666427],[195,0.16605663,0.18277633,-0.0167197],[196,0.1674984,0.18433841,-0.01684001],[197,0.16916286,0.18553279,-0.01636993],[198,0.17090566,0.18674966,-0.015844],[199,0.17234267,0.18781362,-0.01547095],[200,0.174036,0.18914928,-0.01511328],[201,0.17539116,0.19048983,-0.01509868],[202,0.17664568,0.19198283,-0.01533714],[203,0.17878878,0.19420019,-0.0154114],[204,0.18024433,0.19604805,-0.01580371],[205,0.18172589,0.19773124,-0.01600535],[206,0.18294306,0.19909723,-0.01615417],[207,0.18476872,0.20078988,-0.01602115],[208,0.18646962,0.20200986,-0.01554023],[209,0.18769269,0.2032111,-0.01551842],[210,0.18875225,0.20426058,-0.01550833],[211,0.1896891,0.20546714,-0.01577804],[212,0.1912526,0.20708783,-0.01583523],[213,0.19239142,0.20843203,-0.01604062],[214,0.19353511,0.2097943,-0.01625919],[215,0.19512963,0.2115743,-0.01644467],[216,0.19656744,0.21316539,-0.01659795],[217,0.19750226,0.21417682,-0.01667457],[218,0.19830291,0.21535991,-0.017057],[219,0.19929517,0.21646005,-0.01716488],[220,0.19993808,0.21752833,-0.01759025],[221,0.2005255,0.21867067,-0.01814517],[222,0.20108843,0.21983459,-0.01874616],[223,0.20127244,0.22109054,-0.01981809],[224,0.20246918,0.22280628,-0.0203371],[225,0.20377113,0.22455555,-0.02078442],[226,0.20532901,0.2264887,-0.02115969],[227,0.2071572,0.22813885,-0.02098165],[228,0.20816785,0.22942613,-0.02125828],[229,0.20874875,0.23069609,-0.02194734],[230,0.20959777,0.23201776,-0.02241998],[231,0.21039568,0.23331284,-0.02291716],[232,0.21121683,0.23503911,-0.02382228],[233,0.21312147,0.23718932,-0.02406785],[234,0.21513406,0.23899545,-0.02386139],[235,0.2174392,0.24082894,-0.02338974],[236,0.21885708,0.24221646,-0.02335938],[237,0.21939616,0.24323837,-0.02384221],[238,0.22063825,0.24448988,-0.02385163],[239,0.22236044,0.24587233,-0.02351189],[240,0.2237837,0.24742842,-0.02364473],[241,0.22518088,0.24862731,-0.02344643],[242,0.22601956,0.24989888,-0.02387932],[243,0.22616149,0.25070385,-0.02454236],[244,0.2270491,0.2518199,-0.0247708],[245,0.22771265,0.25265952,-0.02494687],[246,0.22797634,0.25341012,-0.02543378],[247,0.2279596,0.25420902,-0.02624942],[248,0.22817013,0.25508386,-0.02691372],[249,0.22851178,0.25584101,-0.02732923],[250,0.22784518,0.2563224,-0.02847723],[251,0.22804511,0.25696303,-0.02891793],[252,0.22758914,0.2573011,-0.02971196]],"cap_10b":[[1,-0.00038495,0.00060065,-0.0009856],[2,-0.00093846,0.0010695,-0.00200796],[3,-0.00159233,0.00119934,-0.00279167],[4,-0.00184608,0.0014683,-0.00331438],[5,-0.00166092,0.00207367,-0.0037346],[6,-0.00124126,0.00302225,-0.00426352],[7,-0.00037649,0.0040765,-0.00445298],[8,3.474e-05,0.00493354,-0.00489881],[9,0.00035242,0.00547286,-0.00512044],[10,0.00057563,0.00612601,-0.00555038],[11,0.00036686,0.00646926,-0.0061024],[12,-0.00029146,0.00650326,-0.00679471],[13,-0.00048487,0.00702425,-0.00750912],[14,-0.00040594,0.00761715,-0.0080231],[15,-0.00025013,0.00865999,-0.00891012],[16,0.00015426,0.00979908,-0.00964482],[17,0.00125643,0.01109457,-0.00983814],[18,0.00222551,0.01244027,-0.01021476],[19,0.00278633,0.01348557,-0.01069924],[20,0.00286772,0.01401586,-0.01114813],[21,0.00273779,0.0144038,-0.01166601],[22,0.00298149,0.01524813,-0.01226665],[23,0.00340564,0.01623609,-0.01283045],[24,0.00359268,0.0171187,-0.01352602],[25,0.00395878,0.01807877,-0.01411999],[26,0.00444647,0.01908975,-0.01464328],[27,0.00506048,0.02016061,-0.01510013],[28,0.00539181,0.02110851,-0.01571671],[29,0.0061194,0.02194615,-0.01582675],[30,0.00651054,0.0226006,-0.01609006],[31,0.00720174,0.0233272,-0.01612545],[32,0.00772688,0.02397398,-0.0162471],[33,0.00829957,0.02465616,-0.01635659],[34,0.00874929,0.02544207,-0.01669278],[35,0.00908994,0.02618583,-0.01709589],[36,0.00984415,0.02702503,-0.01718088],[37,0.01084496,0.02807817,-0.01723321],[38,0.01154616,0.02907569,-0.01752954],[39,0.01187871,0.02977033,-0.01789162],[40,0.01207502,0.03052642,-0.0184514],[41,0.01291116,0.03168126,-0.0187701],[42,0.01402297,0.03316335,-0.01914039],[43,0.01496047,0.03427685,-0.01931637],[44,0.01583232,0.03509327,-0.01926095],[45,0.01674415,0.03614016,-0.01939601],[46,0.01763669,0.03709997,-0.01946328],[47,0.01826466,0.03811787,-0.0198532],[48,0.01977507,0.03929035,-0.01951528],[49,0.02067574,0.04026249,-0.01958674],[50,0.02151136,0.04129111,-0.01977976],[51,0.02278149,0.04268931,-0.01990782],[52,0.02377198,0.04359093,-0.01981895],[53,0.02446489,0.04420596,-0.01974107],[54,0.02490343,0.04486338,-0.01995994],[55,0.02517302,0.04563574,-0.02046273],[56,0.02574879,0.0461117,-0.02036291],[57,0.02623785,0.04696275,-0.0207249],[58,0.02722296,0.04814516,-0.0209222],[59,0.0271727,0.04872315,-0.02155045],[60,0.02782906,0.04944446,-0.0216154],[61,0.02839276,0.05028029,-0.02188753],[62,0.0287169,0.05101203,-0.02229514],[63,0.02839369,0.05139848,-0.02300479],[64,0.02878939,0.05161555,-0.02282616],[65,0.02931175,0.05205899,-0.02274724],[66,0.02978449,0.05262117,-0.02283669],[67,0.03050973,0.05341623,-0.0229065],[68,0.03137828,0.054029,-0.02265072],[69,0.03203349,0.05473821,-0.02270472],[70,0.03272528,0.05527008,-0.0225448],[71,0.03272694,0.0556759,-0.02294896],[72,0.03271896,0.05599043,-0.02327148],[73,0.03274291,0.05623884,-0.02349593],[74,0.03233592,0.05620294,-0.02386702],[75,0.03213749,0.05637328,-0.02423579],[76,0.0326545,0.05709189,-0.0244374],[77,0.0332063,0.05791382,-0.02470752],[78,0.03402718,0.05907099,-0.0250438],[79,0.03486472,0.06011739,-0.02525267],[80,0.03466022,0.06047826,-0.02581805],[81,0.03444635,0.06033496,-0.02588862],[82,0.03419177,0.06052876,-0.02633699],[83,0.03387166,0.06060919,-0.02673753],[84,0.03409098,0.0607641,-0.02667312],[85,0.03484335,0.06123281,-0.02638945],[86,0.03626348,0.06242704,-0.02616356],[87,0.03718095,0.06371431,-0.02653336],[88,0.03859055,0.06493191,-0.02634136],[89,0.0396464,0.06563976,-0.02599336],[90,0.04083323,0.0664862,-0.02565298],[91,0.04127565,0.06698808,-0.02571244],[92,0.04187068,0.06767196,-0.02580128],[93,0.0427667,0.06814887,-0.02538218],[94,0.04438827,0.06914906,-0.02476079],[95,0.04638339,0.07084541,-0.02446202],[96,0.04854558,0.07280162,-0.02425604],[97,0.05051631,0.07439317,-0.02387686],[98,0.05171491,0.07574887,-0.02403395],[99,0.05319546,0.07696501,-0.02376954],[100,0.05438519,0.0781481,-0.02376291],[101,0.05563316,0.07904112,-0.02340796],[102,0.0568554,0.08026749,-0.02341208],[103,0.05799882,0.08156405,-0.02356523],[104,0.05935123,0.08289006,-0.02353883],[105,0.06012374,0.08410047,-0.02397672],[106,0.06179321,0.08538446,-0.02359124],[107,0.06282584,0.08646086,-0.02363502],[108,0.06343821,0.08741168,-0.02397347],[109,0.06410369,0.08824417,-0.02414048],[110,0.06524412,0.0892301,-0.02398598],[111,0.06595101,0.09038903,-0.02443802],[112,0.0669368,0.09148499,-0.02454818],[113,0.06793374,0.09243504,-0.02450131],[114,0.06929129,0.09374734,-0.02445605],[115,0.0698478,0.09507936,-0.02523156],[116,0.07114037,0.09673312,-0.02559275],[117,0.07236679,0.09809585,-0.02572906],[118,0.07379446,0.09913043,-0.02533597],[119,0.07494776,0.10012227,-0.0251745],[120,0.0764052,0.10151379,-0.0251086],[121,0.07817728,0.10292158,-0.0247443],[122,0.07988445,0.1044969,-0.02461245],[123,0.08185062,0.10625357,-0.02440295],[124,0.08282104,0.10757465,-0.02475361],[125,0.08391785,0.10882386,-0.02490601],[126,0.08545369,0.11011817,-0.02466447],[127,0.08643431,0.11138642,-0.02495211],[128,0.08718782,0.11232733,-0.02513951],[129,0.08847324,0.11373253,-0.02525928],[130,0.08970658,0.11538836,-0.02568178],[131,0.09101983,0.11713489,-0.02611507],[132,0.09293702,0.11890661,-0.02596959],[133,0.09453065,0.12038815,-0.02585751],[134,0.09610589,0.12166793,-0.02556205],[135,0.09741644,0.12279855,-0.0253821],[136,0.09870261,0.12398366,-0.02528106],[137,0.09977577,0.12495269,-0.02517691],[138,0.10101341,0.12594334,-0.02492993],[139,0.10244902,0.1270058,-0.02455678],[140,0.10456044,0.12834563,-0.02378519],[141,0.10695137,0.12943656,-0.02248519],[142,0.108994,0.13059741,-0.0216034],[143,0.11067332,0.13183458,-0.02116127],[144,0.11199569,0.13271227,-0.02071658],[145,0.1132399,0.13359553,-0.02035563],[146,0.11371066,0.13417704,-0.02046638],[147,0.11447324,0.13477808,-0.02030484],[148,0.11539112,0.1353298,-0.01993868],[149,0.11576242,0.13599236,-0.02022994],[150,0.11594756,0.1363164,-0.02036885],[151,0.11628293,0.13691934,-0.02063641],[152,0.11727081,0.13771746,-0.02044665],[153,0.11814052,0.13866119,-0.02052067],[154,0.11920986,0.13976987,-0.02056001],[155,0.12028804,0.14066029,-0.02037225],[156,0.1213448,0.14158228,-0.02023748],[157,0.12209695,0.14217556,-0.02007861],[158,0.12273165,0.14243937,-0.01970772],[159,0.12383526,0.14294255,-0.01910729],[160,0.12542535,0.14425011,-0.01882476],[161,0.12749883,0.14579832,-0.01829949],[162,0.12926307,0.14714968,-0.01788661],[163,0.13135649,0.14880213,-0.01744564],[164,0.13325604,0.15063983,-0.01738379],[165,0.13431939,0.15199212,-0.01767273],[166,0.13535257,0.15318575,-0.01783319],[167,0.13705627,0.15446787,-0.01741159],[168,0.13844265,0.15561042,-0.01716777],[169,0.13970878,0.15643311,-0.01672433],[170,0.14110662,0.15749175,-0.01638513],[171,0.14153464,0.15782731,-0.01629267],[172,0.14275141,0.15824421,-0.01549279],[173,0.14377865,0.15888733,-0.01510868],[174,0.14471707,0.15983223,-0.01511516],[175,0.14576859,0.16074901,-0.01498043],[176,0.14749028,0.16213192,-0.01464164],[177,0.14814599,0.162942,-0.01479601],[178,0.1484248,0.16297603,-0.01455122],[179,0.14952234,0.16343814,-0.0139158],[180,0.15031918,0.16445727,-0.01413809],[181,0.15102822,0.1653069,-0.01427868],[182,0.1523796,0.1662593,-0.0138797],[183,0.15421052,0.1678612,-0.01365068],[184,0.15558606,0.1693792,-0.01379314],[185,0.15756575,0.17112564,-0.01355989],[186,0.15933636,0.17252119,-0.01318483],[187,0.16121204,0.17368668,-0.01247463],[188,0.1625617,0.1747343,-0.0121726],[189,0.16375302,0.17588539,-0.01213237],[190,0.16484038,0.17697047,-0.01213009],[191,0.16617371,0.17828524,-0.01211153],[192,0.16795506,0.17983324,-0.01187818],[193,0.16963497,0.18173055,-0.01209558],[194,0.17173804,0.18350868,-0.01177064],[195,0.17338593,0.18516184,-0.01177591],[196,0.17482776,0.18669291,-0.01186515],[197,0.17635428,0.18794117,-0.0115869],[198,0.17799145,0.18914638,-0.01115493],[199,0.17942214,0.19015002,-0.01072788],[200,0.18101638,0.19144887,-0.01043249],[201,0.18256578,0.19282708,-0.0102613],[202,0.1841978,0.19444923,-0.01025144],[203,0.18664889,0.19678654,-0.01013765],[204,0.18810056,0.19866762,-0.01056705],[205,0.18959448,0.20038775,-0.01079326],[206,0.19072011,0.20171954,-0.01099943],[207,0.19241273,0.20341677,-0.01100404],[208,0.19391531,0.20465596,-0.01074065],[209,0.19529712,0.2058598,-0.01056268],[210,0.19647318,0.20694797,-0.01047479],[211,0.19743124,0.20811421,-0.01068297],[212,0.1987884,0.20968657,-0.01089817],[213,0.2000433,0.21107252,-0.01102922],[214,0.20109398,0.21241688,-0.0113229],[215,0.20267212,0.21414494,-0.01147282],[216,0.20412249,0.2157376,-0.01161511],[217,0.20489268,0.21664953,-0.01175685],[218,0.20537858,0.21771686,-0.01233828],[219,0.20593703,0.21877166,-0.01283462],[220,0.20638415,0.21986612,-0.01348197],[221,0.20700705,0.22102524,-0.01401819],[222,0.20758946,0.2222457,-0.01465625],[223,0.20765105,0.22343704,-0.01578598],[224,0.20890834,0.22518725,-0.01627891],[225,0.2102213,0.22699225,-0.01677095],[226,0.21198033,0.22896139,-0.01698105],[227,0.2136533,0.23057132,-0.01691802],[228,0.21499945,0.23194893,-0.01694948],[229,0.21598296,0.23327282,-0.01728986],[230,0.21669919,0.23457084,-0.01787165],[231,0.21712483,0.23570961,-0.01858479],[232,0.21799102,0.2372658,-0.01927477],[233,0.22017793,0.2393525,-0.01917457],[234,0.22225986,0.24108165,-0.01882178],[235,0.22426262,0.24281724,-0.01855462],[236,0.22564964,0.24418228,-0.01853264],[237,0.22636145,0.24527992,-0.01891848],[238,0.22778078,0.24647737,-0.01869659],[239,0.22944006,0.24786251,-0.01842245],[240,0.23125925,0.24933231,-0.01807306],[241,0.23282439,0.25052139,-0.017697],[242,0.23367412,0.25177596,-0.01810183],[243,0.23388697,0.25258956,-0.01870259],[244,0.23483354,0.25357681,-0.01874327],[245,0.23575394,0.25432906,-0.01857512],[246,0.23598789,0.25505485,-0.01906695],[247,0.23599175,0.25582873,-0.01983698],[248,0.23627198,0.25671098,-0.020439],[249,0.23658401,0.25750507,-0.02092106],[250,0.23644343,0.25800728,-0.02156384],[251,0.23670469,0.25859653,-0.02189183],[252,0.2364323,0.25882137,-0.02238908]],"cap_20b":[[1,-0.0005483,0.00062552,-0.00117381],[2,-0.00115216,0.0010985,-0.00225066],[3,-0.00180328,0.00125323,-0.00305651],[4,-0.00218583,0.00154258,-0.00372841],[5,-0.00160493,0.00213443,-0.00373935],[6,-0.00117336,0.00307695,-0.0042503],[7,-8.091e-05,0.00409357,-0.00417448],[8,0.00021687,0.00489122,-0.00467435],[9,0.00039773,0.00547982,-0.00508208],[10,0.00070616,0.00617499,-0.00546883],[11,0.00020379,0.00646175,-0.00625796],[12,-0.00070655,0.0064465,-0.00715305],[13,-0.00121188,0.00690914,-0.00812102],[14,-0.00116817,0.0075503,-0.00871847],[15,-0.00103113,0.00867504,-0.00970617],[16,-0.00045324,0.00990355,-0.01035679],[17,0.00070694,0.01118172,-0.01047477],[18,0.00151382,0.01246554,-0.01095172],[19,0.0021598,0.01356002,-0.01140022],[20,0.00236078,0.01414313,-0.01178235],[21,0.00238178,0.01456238,-0.0121806],[22,0.00258339,0.01538485,-0.01280145],[23,0.00286999,0.01634654,-0.01347655],[24,0.00311666,0.01732154,-0.01420488],[25,0.00346883,0.01828263,-0.0148138],[26,0.00410734,0.019263,-0.01515566],[27,0.00451929,0.02028894,-0.01576965],[28,0.00475668,0.02120217,-0.01644549],[29,0.0053841,0.02204568,-0.01666159],[30,0.00561914,0.02267028,-0.01705114],[31,0.00620013,0.02332055,-0.01712042],[32,0.00663149,0.02394503,-0.01731354],[33,0.00715762,0.02472666,-0.01756904],[34,0.00781339,0.0255914,-0.01777801],[35,0.00818737,0.02641452,-0.01822715],[36,0.00937915,0.02734778,-0.01796863],[37,0.01060284,0.02844807,-0.01784524],[38,0.01152202,0.02957196,-0.01804994],[39,0.01213159,0.03039537,-0.01826378],[40,0.01257904,0.03119759,-0.01861855],[41,0.013641,0.03238018,-0.01873918],[42,0.0148977,0.03391696,-0.01901927],[43,0.01586602,0.03516465,-0.01929862],[44,0.01664668,0.0360082,-0.01936152],[45,0.01720691,0.0370337,-0.01982679],[46,0.01824699,0.03813946,-0.01989248],[47,0.01883174,0.03924058,-0.02040884],[48,0.02050132,0.0404345,-0.01993318],[49,0.02119112,0.04138964,-0.02019852],[50,0.021954,0.04239085,-0.02043685],[51,0.02309141,0.04372412,-0.02063271],[52,0.02405114,0.04456653,-0.02051539],[53,0.02475873,0.04518967,-0.02043094],[54,0.02519216,0.04580373,-0.02061157],[55,0.02552534,0.04652028,-0.02099494],[56,0.0260691,0.04702114,-0.02095205],[57,0.02648051,0.04792388,-0.02144337],[58,0.02726279,0.04906967,-0.02180688],[59,0.02714748,0.0496056,-0.02245812],[60,0.02794304,0.05037691,-0.02243388],[61,0.0284482,0.05120444,-0.02275623],[62,0.02846251,0.05186732,-0.02340481],[63,0.02782265,0.05220473,-0.02438208],[64,0.02827778,0.05247382,-0.02419605],[65,0.02897435,0.05305664,-0.02408229],[66,0.02952114,0.05360478,-0.02408364],[67,0.03057582,0.05443453,-0.02385871],[68,0.03158124,0.05514329,-0.02356205],[69,0.03221021,0.05587984,-0.02366963],[70,0.03299638,0.05641612,-0.02341975],[71,0.03310629,0.05679722,-0.02369093],[72,0.03316212,0.05707439,-0.02391227],[73,0.03336806,0.05727707,-0.02390902],[74,0.03289723,0.05722001,-0.02432278],[75,0.03279639,0.05742362,-0.02462722],[76,0.03316812,0.05813972,-0.0249716],[77,0.03365077,0.0589338,-0.02528303],[78,0.03429351,0.06005781,-0.02576431],[79,0.03512871,0.06102042,-0.02589171],[80,0.03484671,0.06126466,-0.02641795],[81,0.03442812,0.06096421,-0.02653609],[82,0.03392624,0.06101603,-0.02708979],[83,0.03346596,0.06094272,-0.02747676],[84,0.03357661,0.06101467,-0.02743806],[85,0.03437722,0.06147905,-0.02710183],[86,0.03585193,0.06264125,-0.02678932],[87,0.0366096,0.06388321,-0.02727362],[88,0.0381015,0.06512991,-0.02702841],[89,0.03921712,0.06592505,-0.02670793],[90,0.04052671,0.06669208,-0.02616536],[91,0.04092165,0.06708026,-0.02615861],[92,0.04162081,0.06777269,-0.02615188],[93,0.04254977,0.06824321,-0.02569344],[94,0.04403979,0.0691642,-0.02512441],[95,0.04569641,0.07070195,-0.02500554],[96,0.04753652,0.07261656,-0.02508005],[97,0.04954479,0.07415708,-0.02461229],[98,0.05079831,0.0754612,-0.02466288],[99,0.05225102,0.07656842,-0.02431741],[100,0.05339843,0.07766672,-0.02426829],[101,0.05457362,0.07852292,-0.02394931],[102,0.05586739,0.0796912,-0.02382381],[103,0.05694155,0.08088509,-0.02394353],[104,0.05817369,0.08209796,-0.02392427],[105,0.05884285,0.08330179,-0.02445894],[106,0.06023879,0.08454714,-0.02430835],[107,0.06114266,0.08554242,-0.02439976],[108,0.0617792,0.08652216,-0.02474295],[109,0.06212944,0.08739738,-0.02526794],[110,0.063141,0.08830609,-0.0251651],[111,0.06366165,0.08942863,-0.02576698],[112,0.06458714,0.09055467,-0.02596753],[113,0.06585909,0.09151499,-0.02565589],[114,0.06714676,0.09277837,-0.02563161],[115,0.06796224,0.09414966,-0.02618742],[116,0.06953017,0.09582476,-0.0262946],[117,0.07050373,0.09713392,-0.02663019],[118,0.07181043,0.0981084,-0.02629797],[119,0.07309739,0.09910018,-0.02600278],[120,0.07448671,0.10055055,-0.02606384],[121,0.0762167,0.10197072,-0.02575403],[122,0.07780876,0.10348499,-0.02567623],[123,0.07968225,0.10521179,-0.02552954],[124,0.08072636,0.10647641,-0.02575005],[125,0.08182724,0.10763894,-0.0258117],[126,0.08316848,0.10884284,-0.02567437],[127,0.08415616,0.11006146,-0.02590529],[128,0.0848823,0.11099737,-0.02611507],[129,0.08625925,0.11244737,-0.02618812],[130,0.08765148,0.11412104,-0.02646956],[131,0.08912075,0.11582919,-0.02670844],[132,0.09139078,0.11764396,-0.02625318],[133,0.09323654,0.11917016,-0.02593362],[134,0.09501746,0.12049793,-0.02548047],[135,0.09640059,0.12162348,-0.02522288],[136,0.09762741,0.12275407,-0.02512666],[137,0.09875539,0.12379972,-0.02504433],[138,0.10009838,0.12495198,-0.0248536],[139,0.10183127,0.12616332,-0.02433205],[140,0.10391024,0.12758835,-0.02367811],[141,0.10652229,0.12877384,-0.02225155],[142,0.1084598,0.13006843,-0.02160862],[143,0.11016337,0.13138233,-0.02121896],[144,0.11127343,0.13228365,-0.02101022],[145,0.11247974,0.13312313,-0.02064339],[146,0.11298344,0.13376772,-0.02078428],[147,0.11372477,0.1344442,-0.02071943],[148,0.11419566,0.13494745,-0.02075179],[149,0.11409416,0.13553878,-0.02144462],[150,0.11426804,0.13579372,-0.02152567],[151,0.11489407,0.13638749,-0.02149341],[152,0.11599288,0.13722181,-0.02122893],[153,0.11703941,0.13838022,-0.02134081],[154,0.11809988,0.13961082,-0.02151094],[155,0.11889717,0.14047555,-0.02157839],[156,0.11955821,0.1413426,-0.02178439],[157,0.12047123,0.14198566,-0.02151443],[158,0.1211719,0.14235521,-0.02118332],[159,0.12220708,0.14287953,-0.02067245],[160,0.12386782,0.14418804,-0.02032022],[161,0.12600429,0.14582099,-0.0198167],[162,0.12775285,0.14730188,-0.01954903],[163,0.12972657,0.14895307,-0.0192265],[164,0.13143683,0.15076501,-0.01932819],[165,0.13289172,0.15217976,-0.01928803],[166,0.13451136,0.15351704,-0.01900568],[167,0.13623785,0.15483552,-0.01859767],[168,0.13772516,0.15590242,-0.01817726],[169,0.13871897,0.15666117,-0.0179422],[170,0.14000117,0.15775014,-0.01774897],[171,0.14037919,0.1582234,-0.0178442],[172,0.14173101,0.15867125,-0.01694024],[173,0.14295105,0.15930782,-0.01635678],[174,0.14397254,0.16030401,-0.01633147],[175,0.14503228,0.16136415,-0.01633187],[176,0.14678144,0.16278869,-0.01600725],[177,0.14768756,0.16352942,-0.01584186],[178,0.14801476,0.16365822,-0.01564346],[179,0.14906217,0.16409873,-0.01503656],[180,0.14966971,0.16517425,-0.01550454],[181,0.15031887,0.1660188,-0.01569993],[182,0.15155035,0.1669778,-0.01542745],[183,0.15340196,0.16861828,-0.01521633],[184,0.15507575,0.17027949,-0.01520374],[185,0.15732099,0.17225212,-0.01493113],[186,0.15914812,0.17370665,-0.01455853],[187,0.16109938,0.17488123,-0.01378185],[188,0.16232233,0.17605171,-0.01372938],[189,0.16344517,0.17723434,-0.01378918],[190,0.16432922,0.1782195,-0.01389028],[191,0.16556734,0.17957448,-0.01400715],[192,0.16737869,0.18113899,-0.0137603],[193,0.16903667,0.18301618,-0.0139795],[194,0.17072798,0.18471562,-0.01398764],[195,0.17205808,0.18630834,-0.01425026],[196,0.17335183,0.18776486,-0.01441304],[197,0.17454043,0.18899171,-0.01445128],[198,0.17611219,0.19018952,-0.01407733],[199,0.17733176,0.19116947,-0.01383771],[200,0.17886505,0.19230907,-0.01344402],[201,0.1802558,0.19361452,-0.01335872],[202,0.18169825,0.19527838,-0.01358013],[203,0.18407441,0.19753552,-0.01346111],[204,0.18564406,0.19945737,-0.01381331],[205,0.18731815,0.20131631,-0.01399816],[206,0.18840045,0.20279841,-0.01439796],[207,0.19022306,0.2046673,-0.01444425],[208,0.19191671,0.20599662,-0.01407991],[209,0.19327525,0.20726103,-0.01398578],[210,0.19436098,0.20840976,-0.01404878],[211,0.19527181,0.20963981,-0.014368],[212,0.19662372,0.21121353,-0.01458981],[213,0.19790486,0.2125304,-0.01462554],[214,0.19898281,0.21378376,-0.01480095],[215,0.20047414,0.21542314,-0.014949],[216,0.20204137,0.2169465,-0.01490513],[217,0.20279262,0.2178077,-0.01501509],[218,0.20311572,0.21878911,-0.01567339],[219,0.20351082,0.21969663,-0.01618581],[220,0.20387386,0.22070983,-0.01683597],[221,0.20433796,0.22175211,-0.01741416],[222,0.20468721,0.22282197,-0.01813476],[223,0.20442578,0.22390283,-0.01947705],[224,0.20552598,0.22566512,-0.02013914],[225,0.20661008,0.22748883,-0.02087875],[226,0.20803941,0.22944329,-0.02140388],[227,0.20964335,0.23107182,-0.02142847],[228,0.2109384,0.23247641,-0.02153801],[229,0.21158144,0.23375226,-0.02217082],[230,0.21209141,0.23500318,-0.02291177],[231,0.21239139,0.23603523,-0.02364385],[232,0.21287727,0.23743463,-0.02455737],[233,0.21505234,0.23951977,-0.02446742],[234,0.21701895,0.24119691,-0.02417796],[235,0.21871383,0.24291359,-0.02419976],[236,0.22027623,0.24427568,-0.02399945],[237,0.22104437,0.24541955,-0.02437518],[238,0.22292136,0.24677457,-0.02385321],[239,0.22444806,0.24826281,-0.02381475],[240,0.22633792,0.24971319,-0.02337526],[241,0.22793369,0.25100388,-0.02307019],[242,0.22869136,0.25217896,-0.0234876],[243,0.22920928,0.25298022,-0.02377095],[244,0.23009465,0.25391154,-0.02381688],[245,0.23128073,0.25461352,-0.02333279],[246,0.23116523,0.25542435,-0.02425911],[247,0.23097995,0.25620914,-0.02522919],[248,0.2317272,0.25704932,-0.02532211],[249,0.23197319,0.25788467,-0.02591148],[250,0.23198212,0.25848991,-0.02650779],[251,0.23199976,0.25901735,-0.02701758],[252,0.2312633,0.25910108,-0.02783778]],"cap_50b":[[1,-0.00115252,0.00049044,-0.00164296],[2,-0.0022777,0.00095011,-0.00322781],[3,-0.00309954,0.00094835,-0.00404789],[4,-0.00399925,0.00097702,-0.00497628],[5,-0.00367624,0.00142473,-0.00510097],[6,-0.00335252,0.0021073,-0.00545982],[7,-0.00208164,0.00311664,-0.00519828],[8,-0.00246694,0.00374897,-0.00621592],[9,-0.00216825,0.00424799,-0.00641624],[10,-0.00178238,0.00480637,-0.00658875],[11,-0.0023806,0.00503317,-0.00741378],[12,-0.00347193,0.00481247,-0.0082844],[13,-0.00428335,0.00520926,-0.00949261],[14,-0.00429654,0.00583356,-0.0101301],[15,-0.00426887,0.00696676,-0.01123563],[16,-0.00332258,0.00824734,-0.01156992],[17,-0.00162507,0.00950482,-0.01112989],[18,-0.00062761,0.01067992,-0.01130753],[19,8.788e-05,0.0116466,-0.01155872],[20,0.0004304,0.01223403,-0.01180362],[21,0.00076643,0.01282193,-0.0120555],[22,0.00091522,0.01380284,-0.01288762],[23,0.00138683,0.01489795,-0.01351112],[24,0.00228347,0.01593367,-0.01365019],[25,0.00262419,0.01684614,-0.01422195],[26,0.00322496,0.01766488,-0.01443991],[27,0.00364894,0.01868708,-0.01503814],[28,0.00411324,0.01957337,-0.01546013],[29,0.00506598,0.02034524,-0.01527926],[30,0.00531649,0.02106099,-0.0157445],[31,0.00638667,0.02186882,-0.01548216],[32,0.00716064,0.02262637,-0.01546573],[33,0.00820362,0.02355163,-0.01534801],[34,0.00888947,0.02443213,-0.01554266],[35,0.00912876,0.02522786,-0.0160991],[36,0.01047513,0.02622485,-0.01574972],[37,0.011915,0.0273895,-0.0154745],[38,0.01250889,0.02854325,-0.01603436],[39,0.01317158,0.02945375,-0.01628217],[40,0.01390154,0.03029745,-0.0163959],[41,0.01503834,0.03140937,-0.01637103],[42,0.01650501,0.0329826,-0.01647759],[43,0.01737689,0.03429199,-0.0169151],[44,0.01837863,0.03513551,-0.01675688],[45,0.01927989,0.03626012,-0.01698023],[46,0.02082482,0.03745871,-0.01663389],[47,0.02075783,0.03850203,-0.0177442],[48,0.02262515,0.03963917,-0.01701402],[49,0.02324652,0.04063779,-0.01739127],[50,0.02358168,0.04167028,-0.0180886],[51,0.02492087,0.04320953,-0.01828866],[52,0.02615138,0.0441622,-0.01801082],[53,0.02706425,0.04474044,-0.01767619],[54,0.02733061,0.04536461,-0.01803401],[55,0.02746465,0.04606192,-0.01859727],[56,0.02804975,0.0465918,-0.01854205],[57,0.02867523,0.04762595,-0.01895072],[58,0.02964297,0.0489075,-0.01926453],[59,0.02904251,0.04959523,-0.02055272],[60,0.0299016,0.05036329,-0.02046169],[61,0.02999946,0.0512897,-0.02129024],[62,0.02958008,0.05205635,-0.02247626],[63,0.02858877,0.05251607,-0.0239273],[64,0.02906617,0.05287674,-0.02381057],[65,0.02950217,0.05360342,-0.02410125],[66,0.02976028,0.05427465,-0.02451436],[67,0.03085105,0.05501422,-0.02416317],[68,0.03224437,0.05572817,-0.02348381],[69,0.03322335,0.056503,-0.02327965],[70,0.0342326,0.05706124,-0.02282864],[71,0.0349413,0.05752947,-0.02258818],[72,0.03488383,0.05771036,-0.02282653],[73,0.03509318,0.05777029,-0.02267711],[74,0.03402749,0.05769388,-0.02366639],[75,0.03389878,0.05789652,-0.02399775],[76,0.03390702,0.05844917,-0.02454215],[77,0.03465924,0.0593111,-0.02465186],[78,0.03515851,0.06039339,-0.02523488],[79,0.03603684,0.06137672,-0.02533988],[80,0.0357549,0.06160402,-0.02584912],[81,0.03510493,0.06135205,-0.02624713],[82,0.0347454,0.06144384,-0.02669844],[83,0.03449609,0.0613328,-0.0268367],[84,0.03451913,0.06144727,-0.02692814],[85,0.03503245,0.06168493,-0.02665249],[86,0.03610848,0.06286279,-0.02675431],[87,0.03698811,0.06410386,-0.02711576],[88,0.03837568,0.06541801,-0.02704233],[89,0.03940735,0.06617197,-0.02676462],[90,0.04071758,0.06692276,-0.02620517],[91,0.04101733,0.0671732,-0.02615587],[92,0.04176708,0.06780318,-0.02603611],[93,0.0426423,0.06821521,-0.02557291],[94,0.04443417,0.06924569,-0.02481152],[95,0.04611779,0.070886,-0.02476821],[96,0.04778952,0.07287698,-0.02508746],[97,0.04973127,0.07431303,-0.02458177],[98,0.05073391,0.07555086,-0.02481695],[99,0.05186241,0.07657146,-0.02470905],[100,0.05282349,0.07754358,-0.02472009],[101,0.05406294,0.07844048,-0.02437754],[102,0.05524461,0.07966471,-0.0244201],[103,0.05623541,0.0808764,-0.02464099],[104,0.05729341,0.08209833,-0.02480492],[105,0.05803382,0.08325512,-0.0252213],[106,0.05978234,0.08461736,-0.02483503],[107,0.0611124,0.08553095,-0.02441855],[108,0.06199758,0.08662625,-0.02462866],[109,0.06254867,0.08763063,-0.02508196],[110,0.06360713,0.08870688,-0.02509976],[111,0.06451893,0.08981702,-0.02529809],[112,0.06581372,0.09095617,-0.02514245],[113,0.06726686,0.09206466,-0.0247978],[114,0.06878785,0.09333272,-0.02454487],[115,0.06978067,0.09469297,-0.0249123],[116,0.07146407,0.0963194,-0.02485533],[117,0.07229445,0.09749636,-0.02520192],[118,0.07371608,0.0983747,-0.02465863],[119,0.07501611,0.09926071,-0.0242446],[120,0.07707871,0.10071605,-0.02363734],[121,0.07876274,0.10215545,-0.02339271],[122,0.0805122,0.1037131,-0.0232009],[123,0.08245284,0.10536081,-0.02290796],[124,0.08376522,0.10654855,-0.02278332],[125,0.08505153,0.10763288,-0.02258134],[126,0.08661769,0.10865267,-0.02203498],[127,0.08777122,0.10983505,-0.02206383],[128,0.08863756,0.1108137,-0.02217614],[129,0.09045663,0.11243484,-0.02197821],[130,0.09182521,0.1141133,-0.02228809],[131,0.09310499,0.11571926,-0.02261427],[132,0.09595874,0.1176229,-0.02166416],[133,0.09841407,0.11934614,-0.02093207],[134,0.10054292,0.12083266,-0.02028975],[135,0.10287754,0.12208275,-0.01920521],[136,0.10438522,0.12321727,-0.01883204],[137,0.10563534,0.12409866,-0.01846331],[138,0.1065853,0.12524242,-0.01865712],[139,0.10821483,0.12645379,-0.01823896],[140,0.11050283,0.12797685,-0.01747402],[141,0.11297776,0.12909544,-0.01611768],[142,0.11459246,0.13030333,-0.01571087],[143,0.11633385,0.13149616,-0.01516231],[144,0.11742738,0.13236126,-0.01493387],[145,0.11870646,0.13313383,-0.01442737],[146,0.11932135,0.13386668,-0.01454533],[147,0.1204726,0.13457542,-0.01410283],[148,0.12090397,0.13510235,-0.01419838],[149,0.12053693,0.1354579,-0.01492097],[150,0.12048486,0.13534004,-0.01485518],[151,0.12127266,0.13588814,-0.01461547],[152,0.12321141,0.13687148,-0.01366007],[153,0.12501547,0.13824479,-0.01322932],[154,0.12613543,0.13959783,-0.0134624],[155,0.12721304,0.14053799,-0.01332495],[156,0.12807789,0.14130135,-0.01322347],[157,0.12952994,0.14191476,-0.01238482],[158,0.13045964,0.14230996,-0.01185032],[159,0.13205133,0.14310969,-0.01105836],[160,0.13445136,0.14454551,-0.01009416],[161,0.13693137,0.14632585,-0.00939447],[162,0.13908279,0.14793713,-0.00885433],[163,0.14078656,0.14959649,-0.00880993],[164,0.14240417,0.15144645,-0.00904228],[165,0.14433579,0.15302807,-0.00869228],[166,0.14608817,0.15460257,-0.0085144],[167,0.14746493,0.15577523,-0.0083103],[168,0.14880457,0.15682947,-0.0080249],[169,0.14961526,0.15750329,-0.00788803],[170,0.15070879,0.15858922,-0.00788043],[171,0.15104332,0.15921507,-0.00817175],[172,0.15215161,0.15978784,-0.00763622],[173,0.15332111,0.16029297,-0.00697185],[174,0.15413353,0.16127139,-0.00713786],[175,0.15532549,0.16238718,-0.0070617],[176,0.15762238,0.16391724,-0.00629485],[177,0.15879565,0.16476501,-0.00596936],[178,0.15933481,0.16501209,-0.00567727],[179,0.16029692,0.16547113,-0.00517422],[180,0.16094501,0.1663406,-0.00539559],[181,0.16189559,0.16713561,-0.00524001],[182,0.16326016,0.1681374,-0.00487724],[183,0.16566165,0.17006623,-0.00440458],[184,0.16727799,0.17205306,-0.00477507],[185,0.16984368,0.17432661,-0.00448293],[186,0.17157916,0.17572897,-0.00414981],[187,0.17298002,0.17666603,-0.00368601],[188,0.17378031,0.17773891,-0.0039586],[189,0.17472585,0.17886859,-0.00414274],[190,0.17542955,0.17999698,-0.00456744],[191,0.17651747,0.18132033,-0.00480286],[192,0.17867791,0.18271905,-0.00404114],[193,0.18021541,0.18448869,-0.00427328],[194,0.18190624,0.18611403,-0.00420779],[195,0.18287685,0.18766779,-0.00479094],[196,0.18374925,0.18908054,-0.00533128],[197,0.18480573,0.19037952,-0.00557379],[198,0.18587773,0.19152003,-0.0056423],[199,0.18698476,0.19278118,-0.00579642],[200,0.18862,0.19378817,-0.00516816],[201,0.19051059,0.19514593,-0.00463534],[202,0.19190365,0.19697454,-0.00507089],[203,0.19467688,0.19945556,-0.00477868],[204,0.19672962,0.20143923,-0.00470961],[205,0.19881424,0.20322065,-0.00440641],[206,0.19961689,0.20461142,-0.00499453],[207,0.20164166,0.20651508,-0.00487342],[208,0.20363706,0.2080223,-0.00438525],[209,0.2050607,0.20938007,-0.00431936],[210,0.20622224,0.21044285,-0.0042206],[211,0.20673859,0.21160394,-0.00486535],[212,0.20784218,0.21300557,-0.00516339],[213,0.20897937,0.21413004,-0.00515066],[214,0.20983603,0.21530088,-0.00546485],[215,0.21135737,0.21686438,-0.00550702],[216,0.21285956,0.21839085,-0.00553129],[217,0.21349372,0.21926032,-0.0057666],[218,0.21317449,0.21996608,-0.0067916],[219,0.21284739,0.22054111,-0.00769372],[220,0.21241399,0.22134308,-0.00892909],[221,0.2122894,0.22218623,-0.00989683],[222,0.21237925,0.22306475,-0.0106855],[223,0.21188581,0.22390015,-0.01201434],[224,0.21317566,0.22560072,-0.01242507],[225,0.21408031,0.22713865,-0.01305834],[226,0.21487684,0.2292041,-0.01432726],[227,0.21613199,0.23079743,-0.01466545],[228,0.21774515,0.23237183,-0.01462667],[229,0.2187304,0.23363083,-0.01490043],[230,0.21954894,0.23500177,-0.01545283],[231,0.2197069,0.23584473,-0.01613783],[232,0.21973145,0.237348,-0.01761656],[233,0.2220868,0.23953197,-0.01744517],[234,0.22430658,0.24142511,-0.01711853],[235,0.22593264,0.24326574,-0.0173331],[236,0.22776832,0.24474784,-0.01697952],[237,0.22875244,0.24592675,-0.01717431],[238,0.23043506,0.2472261,-0.01679104],[239,0.23177402,0.24873488,-0.01696086],[240,0.23389481,0.25020175,-0.01630694],[241,0.23592968,0.25166307,-0.01573339],[242,0.23697119,0.25275189,-0.0157807],[243,0.23779584,0.25343744,-0.0156416],[244,0.23865382,0.25417679,-0.01552297],[245,0.23937831,0.25445225,-0.01507395],[246,0.23900023,0.25496638,-0.01596615],[247,0.23844829,0.25564916,-0.01720086],[248,0.23925494,0.2565839,-0.01732896],[249,0.23998558,0.25752263,-0.01753705],[250,0.24066299,0.25814871,-0.01748572],[251,0.24034994,0.25850107,-0.01815113],[252,0.23964065,0.25839872,-0.01875807]],"cap_100b":[[1,-0.00097354,0.00054054,-0.00151408],[2,-0.00185265,0.00116432,-0.00301697],[3,-0.00171583,0.00163109,-0.00334693],[4,-0.00268323,0.00182288,-0.00450612],[5,-0.00192011,0.00221876,-0.00413887],[6,-0.00110115,0.00306739,-0.00416854],[7,0.00086282,0.00415301,-0.00329019],[8,0.00052073,0.00465451,-0.00413378],[9,0.00091653,0.00516317,-0.00424663],[10,0.0017203,0.00582636,-0.00410605],[11,0.00140461,0.00589085,-0.00448624],[12,0.00161065,0.0057365,-0.00412585],[13,0.00150308,0.00633112,-0.00482803],[14,0.00223464,0.00697706,-0.00474241],[15,0.00195408,0.00827698,-0.0063229],[16,0.00354221,0.00952464,-0.00598243],[17,0.00575889,0.01081457,-0.00505568],[18,0.00674883,0.01192469,-0.00517586],[19,0.00770744,0.01288098,-0.00517353],[20,0.00849957,0.01354788,-0.00504831],[21,0.00935586,0.01401736,-0.0046615],[22,0.01038962,0.01502828,-0.00463866],[23,0.01072118,0.01604953,-0.00532834],[24,0.01154081,0.01729996,-0.00575916],[25,0.01232637,0.01824597,-0.00591959],[26,0.01256949,0.01896475,-0.00639526],[27,0.01305353,0.02009225,-0.00703872],[28,0.01336424,0.02100288,-0.00763865],[29,0.01511288,0.02162587,-0.006513],[30,0.01528208,0.02215534,-0.00687326],[31,0.01642151,0.02327933,-0.00685782],[32,0.01725274,0.02420289,-0.00695015],[33,0.01893949,0.02515199,-0.0062125],[34,0.02026051,0.02611221,-0.0058517],[35,0.02011143,0.02718386,-0.00707244],[36,0.02210306,0.02822277,-0.00611971],[37,0.02326525,0.02929442,-0.00602917],[38,0.02381046,0.03034121,-0.00653075],[39,0.02455193,0.0313596,-0.00680767],[40,0.02521414,0.03230064,-0.00708649],[41,0.02651842,0.03333779,-0.00681936],[42,0.02781523,0.03507003,-0.0072548],[43,0.02807075,0.03621212,-0.00814137],[44,0.02938032,0.03700657,-0.00762625],[45,0.03065715,0.03803814,-0.00738099],[46,0.03236533,0.03930379,-0.00693846],[47,0.03150366,0.04039484,-0.00889118],[48,0.03379276,0.04152997,-0.0077372],[49,0.03380264,0.04249207,-0.00868943],[50,0.03280875,0.04340312,-0.01059437],[51,0.03424814,0.04516534,-0.01091719],[52,0.0361677,0.0462641,-0.0100964],[53,0.03767478,0.04690831,-0.00923352],[54,0.0374873,0.04755421,-0.01006691],[55,0.03728128,0.04823172,-0.01095044],[56,0.03822952,0.04879546,-0.01056595],[57,0.0389629,0.05007853,-0.01111563],[58,0.03954045,0.05154177,-0.01200132],[59,0.03849315,0.05217593,-0.01368278],[60,0.0385823,0.05303536,-0.01445306],[61,0.03779706,0.05400412,-0.01620706],[62,0.03655887,0.05478043,-0.01822156],[63,0.03588917,0.05513956,-0.01925039],[64,0.03604165,0.05563304,-0.0195914],[65,0.03649486,0.05630021,-0.01980535],[66,0.03599781,0.05705226,-0.02105445],[67,0.03766562,0.0576614,-0.01999578],[68,0.03928492,0.05834827,-0.01906335],[69,0.03939461,0.05896033,-0.01956571],[70,0.04084234,0.05954808,-0.01870574],[71,0.04156459,0.06008528,-0.01852069],[72,0.04156082,0.05994454,-0.01838373],[73,0.04138343,0.05972671,-0.01834328],[74,0.03960513,0.05931602,-0.01971089],[75,0.03987214,0.05973188,-0.01985974],[76,0.03903006,0.05995441,-0.02092435],[77,0.03927301,0.0607621,-0.02148908],[78,0.03933867,0.06179308,-0.02245441],[79,0.04046562,0.06258058,-0.02211496],[80,0.03967513,0.06253307,-0.02285794],[81,0.03815138,0.06212956,-0.02397818],[82,0.03773668,0.06226335,-0.02452667],[83,0.03748019,0.06228888,-0.02480869],[84,0.03717556,0.06249001,-0.02531445],[85,0.03788879,0.06279659,-0.0249078],[86,0.03830998,0.06390692,-0.02559694],[87,0.03917704,0.06498349,-0.02580645],[88,0.04116259,0.06587577,-0.02471317],[89,0.0415401,0.0665473,-0.0250072],[90,0.04258342,0.06725969,-0.02467627],[91,0.04287529,0.06741245,-0.02453716],[92,0.04354985,0.06784514,-0.0242953],[93,0.04451311,0.06817703,-0.02366392],[94,0.04664943,0.0691085,-0.02245907],[95,0.04819586,0.0705615,-0.02236564],[96,0.04956319,0.07232842,-0.02276524],[97,0.0516221,0.07335253,-0.02173044],[98,0.0523651,0.0742134,-0.0218483],[99,0.05385418,0.07500445,-0.02115027],[100,0.05478084,0.07547214,-0.0206913],[101,0.05635131,0.07610005,-0.01974874],[102,0.05765397,0.07732543,-0.01967146],[103,0.05869594,0.07827872,-0.01958278],[104,0.05965367,0.07909768,-0.01944401],[105,0.06022818,0.08012456,-0.01989637],[106,0.06203256,0.08145896,-0.0194264],[107,0.06285185,0.0822889,-0.01943705],[108,0.06373155,0.08335241,-0.01962086],[109,0.06449266,0.08454654,-0.02005388],[110,0.06504256,0.08557317,-0.02053061],[111,0.06616605,0.08646564,-0.02029958],[112,0.06692885,0.08759113,-0.02066228],[113,0.06862607,0.08903693,-0.02041086],[114,0.07079624,0.09050822,-0.01971198],[115,0.07249425,0.09187846,-0.01938421],[116,0.07364159,0.09384491,-0.02020332],[117,0.07430153,0.09492996,-0.02062843],[118,0.07528261,0.09561155,-0.02032893],[119,0.07636954,0.09648763,-0.02011809],[120,0.07880009,0.09786983,-0.01906974],[121,0.08015879,0.0992939,-0.01913511],[122,0.08195287,0.10074184,-0.01878897],[123,0.0840493,0.10221206,-0.01816275],[124,0.08565869,0.10312167,-0.01746298],[125,0.08687287,0.10385305,-0.01698018],[126,0.08838867,0.10464679,-0.01625812],[127,0.08940806,0.10578503,-0.01637697],[128,0.08988368,0.10677942,-0.01689574],[129,0.09157321,0.10825907,-0.01668586],[130,0.09274915,0.11009459,-0.01734544],[131,0.09361878,0.11171026,-0.01809149],[132,0.09650116,0.11336111,-0.01685994],[133,0.09858459,0.11497736,-0.01639277],[134,0.10085636,0.1165978,-0.01574144],[135,0.1040687,0.11801622,-0.01394752],[136,0.10614449,0.11934366,-0.01319917],[137,0.10705369,0.12020683,-0.01315314],[138,0.10848145,0.12164682,-0.01316538],[139,0.1103196,0.12287624,-0.01255664],[140,0.11312046,0.1247037,-0.01158324],[141,0.11521498,0.12580349,-0.01058851],[142,0.11662061,0.12722488,-0.01060426],[143,0.11784829,0.12842498,-0.01057669],[144,0.11919013,0.12935976,-0.01016963],[145,0.12046684,0.12999886,-0.00953202],[146,0.12090109,0.13074238,-0.00984128],[147,0.12235331,0.13156606,-0.00921275],[148,0.12271641,0.13225837,-0.00954196],[149,0.12255556,0.13253389,-0.00997833],[150,0.12295002,0.13279158,-0.00984156],[151,0.12396696,0.13361427,-0.00964731],[152,0.12658692,0.134949,-0.00836208],[153,0.12929263,0.13633901,-0.00704638],[154,0.13138701,0.13782408,-0.00643707],[155,0.13290874,0.13897974,-0.006071],[156,0.1334579,0.13980406,-0.00634617],[157,0.13536208,0.1410356,-0.00567352],[158,0.13733728,0.14190102,-0.00456374],[159,0.14065523,0.14339686,-0.00274163],[160,0.14406451,0.14511983,-0.00105532],[161,0.14680568,0.14698345,-0.00017777],[162,0.14862963,0.14848309,0.00014653],[163,0.14985435,0.15018761,-0.00033326],[164,0.15116669,0.15198976,-0.00082307],[165,0.1529231,0.15352386,-0.00060076],[166,0.15533202,0.15513821,0.00019381],[167,0.15726683,0.15606158,0.00120524],[168,0.15874475,0.15724286,0.00150188],[169,0.15941817,0.157941,0.00147717],[170,0.16113416,0.15897925,0.00215491],[171,0.16122672,0.15963586,0.00159086],[172,0.16241189,0.16036673,0.00204516],[173,0.16292087,0.16065485,0.00226602],[174,0.16343584,0.16162435,0.00181148],[175,0.16413371,0.16250401,0.0016297],[176,0.16634318,0.16384158,0.0025016],[177,0.16729095,0.16427115,0.0030198],[178,0.16797036,0.16440576,0.0035646],[179,0.16995036,0.16508775,0.00486261],[180,0.17175108,0.16634111,0.00540996],[181,0.17342962,0.16757844,0.00585118],[182,0.17511318,0.16885986,0.00625332],[183,0.17758465,0.17080267,0.00678198],[184,0.17943777,0.17309965,0.00633812],[185,0.18174928,0.17523653,0.00651275],[186,0.1831227,0.17655841,0.00656429],[187,0.18428643,0.17748859,0.00679784],[188,0.18574934,0.17887406,0.00687529],[189,0.1877577,0.18011052,0.00764718],[190,0.18880275,0.18136102,0.00744173],[191,0.1901301,0.18247764,0.00765246],[192,0.19297934,0.18382251,0.00915683],[193,0.19524473,0.18534004,0.00990469],[194,0.19761045,0.18660621,0.01100424],[195,0.19893297,0.18797814,0.01095483],[196,0.19976101,0.18932238,0.01043863],[197,0.20149076,0.19075547,0.01073529],[198,0.20290696,0.19170958,0.01119738],[199,0.20387362,0.19299283,0.0108808],[200,0.20559878,0.19418815,0.01141063],[201,0.20737579,0.19564955,0.01172624],[202,0.20879961,0.19765091,0.0111487],[203,0.21219911,0.20042507,0.01177404],[204,0.21487612,0.20239279,0.01248333],[205,0.2172229,0.20431862,0.01290428],[206,0.21881825,0.20571096,0.01310728],[207,0.22087814,0.20770413,0.01317401],[208,0.22342759,0.20941477,0.01401282],[209,0.22587011,0.21108105,0.01478905],[210,0.2271831,0.21220263,0.01498047],[211,0.22875448,0.21313171,0.01562277],[212,0.22951999,0.21446166,0.01505833],[213,0.2305825,0.2154596,0.0151229],[214,0.23070486,0.21627783,0.01442703],[215,0.23193233,0.21778252,0.01414981],[216,0.23258885,0.21930194,0.01328691],[217,0.23221801,0.22011926,0.01209875],[218,0.23148731,0.22043269,0.01105463],[219,0.23013593,0.2208068,0.00932913],[220,0.22943898,0.22194608,0.00749289],[221,0.22908732,0.22303555,0.00605177],[222,0.2292742,0.22368879,0.0055854],[223,0.2288364,0.22445664,0.00437976],[224,0.23070072,0.2261525,0.00454822],[225,0.23089467,0.22737693,0.00351773],[226,0.23136026,0.2293302,0.00203007],[227,0.23178571,0.2307856,0.00100011],[228,0.23313995,0.23221616,0.00092379],[229,0.23402602,0.23357843,0.00044758],[230,0.23504544,0.23482542,0.00022002],[231,0.23603448,0.23564918,0.0003853],[232,0.23619522,0.23729623,-0.00110101],[233,0.2392525,0.23945055,-0.00019806],[234,0.24157642,0.24137672,0.0001997],[235,0.24399976,0.24304877,0.00095099],[236,0.24612133,0.24438925,0.00173208],[237,0.24754797,0.24571016,0.00183781],[238,0.24863878,0.24704941,0.00158937],[239,0.2506609,0.24878852,0.00187239],[240,0.25339092,0.25024588,0.00314504],[241,0.256175,0.25185897,0.00431603],[242,0.25793991,0.25256723,0.00537268],[243,0.25915263,0.2533524,0.00580024],[244,0.26050981,0.25402244,0.00648738],[245,0.26163491,0.25420439,0.00743052],[246,0.26226523,0.25466682,0.00759841],[247,0.26219046,0.25542765,0.00676281],[248,0.26365417,0.2561004,0.00755376],[249,0.26538888,0.2568012,0.00858768],[250,0.26656376,0.25745824,0.00910553],[251,0.26617073,0.25766233,0.00850841],[252,0.26544156,0.25742371,0.00801785]]}
                     }
                 }
             },
             401: {
                 "description": "Token inválido o no autorizado",
                 "content": {"application/json": {"example": {"detail": "Token inválido o no autorizado"}}}
             },
             404: {
                 "description": "Sin datos para el año indicado",
                 "content": {"application/json": {"example": {"detail": "No hay señales de 52W para el año 2026"}}}
             },
             422: {
                 "description": "Error de validación",
                 "content": {
                     "application/json": {
                         "example": {"detail": [{"loc": ["query", "year"], "msg": "Input should be less than or equal to 2025", "type": "less_than_equal"}]}
                     }
                 }
             }
         })
async def get_performance_year(
    year: int = Query(..., ge=2000, le=2025, description="Año a analizar (2000-2025)"),
    token: str = Depends(verificar_api_key),
):
    csv_path = os.path.join(REND_DIR, f"rend_{year}.csv")
    meta_path = os.path.join(REND_DIR, f"rend_{year}_meta.json")
    
    df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)
        
    result = {
        "año": meta["año"],
        "señales_procesadas": meta["señales_procesadas"],
        "all": [],
        "cap_1b": [],
        "cap_5b": [],
        "cap_10b": [],
        "cap_20b": [],
        "cap_50b": [],
        "cap_100b": []
    }

    for cap_key, group in df.groupby("CAP"):
        group = group.sort_values("DAY")
            
        curve = []
        for _, row in group.iterrows():
            curve.append([
                int(row["DAY"]),
                None if pd.isna(row["STOCK_RET"]) else float(row["STOCK_RET"]),
                None if pd.isna(row["SP500_RET"]) else float(row["SP500_RET"]),
                None if pd.isna(row["ALPHA"]) else float(row["ALPHA"])
            ])
            
        if cap_key == "all":
            result["all"] = curve
        else:
            result[cap_key] = curve

    return result

@app.get("/analyze", tags=["Analizar"], summary="Analizar en base a alpha",
         description="Permite analizar tendencias en base a un porcentaje X% de alpha",
         responses={
             200: {"description": "Petición exitosa", "content": {"application/json": {
                 "example": {
                     "muestras_que_superan": 1240,
                     "muestras_que_no_superan": 8760,
                     "total_muestras": 10000,
                     "pct_superan_Xpct": 12.4,
                     "dias_promedio_hasta_Xpct": 47.3,
                     "alpha_neg_promedio": -0.0312,
                     "n_muestras_alpha_negativo": 4102,
                 }}}},
             401: {"description": "Token inválido o no autorizado"},
             404: {"description": "Sin datos para el año solicitado"},
             422: {"description": "Error de validación"},
         })
async def analyze_alpha(
    year:   int   = Query(..., ge=2000, description="Año YYYY"),
    alpha:  float = Query(..., description="Umbral alpha (%)"),
    period: int   = Query(..., ge=1, le=253, description="Periodo en días de mercado"),
    token:  str   = Depends(verificar_api_key),
):
    if alpha == 0:
        raise HTTPException(422, "alpha no puede ser 0")

    alpha_p      = alpha / 100.0
    parquet_path = os.path.join(REND_DIR, f"granular_alpha_{year}.parquet")
    if not os.path.exists(parquet_path):
        raise HTTPException(404, f"No hay datos para el año {year}")

    query = f"""
        WITH base AS (
            SELECT ticker, fecha, day, alpha
            FROM read_parquet('{parquet_path}')
            WHERE day <= {period}
              AND alpha IS NOT NULL
        ),
        universe AS (
            SELECT COUNT(DISTINCT (ticker, fecha)) AS total
            FROM read_parquet('{parquet_path}')
        ),
        signal_stats AS (
            SELECT
                ticker,
                fecha,
                LAST(alpha ORDER BY day)                          AS last_alpha,
                MIN(CASE WHEN alpha > {alpha_p} THEN day END)     AS first_day_above
            FROM base
            GROUP BY ticker, fecha
        )
        SELECT
            COUNT(CASE WHEN first_day_above IS NOT NULL THEN 1 END)
                                                                    AS muestras_que_superan,
            u.total                                                 AS total_muestras,
            ROUND(
                100.0 * COUNT(CASE WHEN first_day_above IS NOT NULL THEN 1 END)
                / NULLIF(u.total, 0), 2)                            AS pct_superan_Xpct,
            ROUND(AVG(
                CASE WHEN first_day_above IS NOT NULL
                     THEN first_day_above END), 1)                  AS dias_promedio_hasta_Xpct,
            ROUND(AVG(
                CASE WHEN first_day_above IS NULL AND last_alpha < 0
                     THEN last_alpha END), 4)                       AS alpha_neg_promedio,
            COUNT(CASE WHEN first_day_above IS NULL AND last_alpha < 0
                       THEN 1 END)                                  AS n_muestras_alpha_negativo,
            COUNT(CASE WHEN first_day_above IS NULL THEN 1 END)     AS n_muestras_no_superan
        FROM signal_stats
        CROSS JOIN universe u
        GROUP BY u.total
    """

    row = duckdb.query(query).fetchone()
    if row is None:
        raise HTTPException(404, "La query no devolvió resultados")

    (muestras_que_superan, total_muestras, pct_superan_Xpct,
     dias_promedio_hasta_Xpct, alpha_neg_promedio,
     n_muestras_alpha_negativo, muestras_que_no_superan) = row

    return {
        "año":                       year,
        "alpha_pct":                 alpha,
        "period_dias":               period,
        "muestras_que_superan":      muestras_que_superan,
        "muestras_que_no_superan":   muestras_que_no_superan,
        "total_muestras":            total_muestras,
        "pct_superan_Xpct":          pct_superan_Xpct,
        "dias_promedio_hasta_Xpct":  dias_promedio_hasta_Xpct,
        "alpha_neg_promedio":        alpha_neg_promedio,
        "n_muestras_alpha_negativo": n_muestras_alpha_negativo,
    }

@app.get("/analyze/multi", tags=["Analizar"],
         summary="Analizar alpha en rango de años",
         description=(
             "Ejecuta `/analyze` para cada año en [year_from, year_to] y devuelve "
             "resultados por año + resumen agregado. Permite ver consistencia histórica."
         ))
async def analyze_alpha_multi(
    year_from: int   = Query(..., ge=2000, le=2025, description="Año inicio"),
    year_to:   int   = Query(..., ge=2000, le=2025, description="Año fin (inclusive)"),
    alpha:     float = Query(..., description="Umbral alpha (%)"),
    period:    int   = Query(..., ge=1, le=253, description="Periodo en días de mercado"),
    token:     str   = Depends(verificar_api_key),
):
    if year_from > year_to:
        raise HTTPException(422, "year_from debe ser <= year_to")
    if alpha == 0:
        raise HTTPException(422, "alpha no puede ser 0")

    alpha_p    = alpha / 100.0
    años       = range(year_from, year_to + 1)
    por_año    = {}
    pcts_lista = []
    dias_lista = []

    for year in años:
        parquet_path = os.path.join(REND_DIR, f"granular_alpha_{year}.parquet")
        if not os.path.exists(parquet_path):
            por_año[year] = {"error": f"Sin datos para {year}"}
            continue

        query = f"""
            WITH base AS (
                SELECT ticker, fecha, day, alpha
                FROM read_parquet('{parquet_path}')
                WHERE day <= {period} AND alpha IS NOT NULL
            ),
            universe AS (
                SELECT COUNT(DISTINCT (ticker, fecha)) AS total
                FROM read_parquet('{parquet_path}')
            ),
            signal_stats AS (
                SELECT
                    ticker, fecha,
                    LAST(alpha ORDER BY day)                      AS last_alpha,
                    MIN(CASE WHEN alpha > {alpha_p} THEN day END) AS first_day_above
                FROM base
                GROUP BY ticker, fecha
            )
            SELECT
                COUNT(CASE WHEN first_day_above IS NOT NULL THEN 1 END),
                u.total,
                ROUND(100.0 * COUNT(CASE WHEN first_day_above IS NOT NULL THEN 1 END)
                      / NULLIF(u.total, 0), 2),
                ROUND(AVG(CASE WHEN first_day_above IS NOT NULL THEN first_day_above END), 1),
                ROUND(AVG(CASE WHEN first_day_above IS NULL AND last_alpha < 0
                               THEN last_alpha END), 4),
                COUNT(CASE WHEN first_day_above IS NULL AND last_alpha < 0 THEN 1 END),
                COUNT(CASE WHEN first_day_above IS NULL THEN 1 END)
            FROM signal_stats CROSS JOIN universe u
            GROUP BY u.total
        """
        row = duckdb.query(query).fetchone()
        if row is None:
            por_año[year] = {"error": "Sin resultados"}
            continue

        (mq, total, pct, dias, ang, nmng, mnq) = row
        por_año[year] = {
            "muestras_que_superan":      mq,
            "muestras_que_no_superan":   mnq,
            "total_muestras":            total,
            "pct_superan_Xpct":          pct,
            "dias_promedio_hasta_Xpct":  dias,
            "alpha_neg_promedio":        ang,
            "n_muestras_alpha_negativo": nmng,
        }
        if pct is not None:
            pcts_lista.append(pct)
        if dias is not None:
            dias_lista.append(dias)

    años_con_datos = [y for y in años if "error" not in por_año.get(y, {})]
    resumen = {
        "años_con_datos":           len(años_con_datos),
        "años_sin_datos":           len(años) - len(años_con_datos),
        "pct_superan_promedio":     round(float(np.mean(pcts_lista)), 2)  if pcts_lista else None,
        "pct_superan_mediana":      round(float(np.median(pcts_lista)), 2) if pcts_lista else None,
        "pct_superan_std":          round(float(np.std(pcts_lista)), 2)   if pcts_lista else None,
        "dias_promedio_global":     round(float(np.mean(dias_lista)), 1)  if dias_lista else None,
        "consistencia_pct_positivo": (
            round(100.0 * sum(1 for p in pcts_lista if p > 0) / len(pcts_lista), 1)
            if pcts_lista else None
        ),
    }

    return {
        "alpha_pct":    alpha,
        "period_dias":  period,
        "resultados":   por_año,
        "resumen":      resumen,
    }

@app.get("/optimize", tags=["Optimizar"],
         summary="Optimizar umbral alpha y periodo de tenencia",
         description=(
             "Grid search sobre (alpha%, periodo) que maximiza la TIR ponderada L/S.\n\n"
             "**Novedades v2:**\n"
             "- Winsorización p1-p99 antes de promediar → elimina outliers que causaban TIR ×10¹⁸⁸\n"
             "- `_safe_annualize` con log/exp y triple capa de clamp\n"
             "- Sharpe ratio del portfolio long-short\n"
             "- Win rate por grupo\n"
             "- Modelado de costes de transacción (`cost_bps`)"
             "- **Grupo negativo**: señales que nunca lo superaron → CORTO vía futuro S&P 500 "
             "(muy líquido, coste mínimo incluido en cost_bps)\n"
         )
)
async def optimize_alpha_period(
    year:        int   = Query(..., ge=2000, le=2025),
    min_samples: int   = Query(default=30, ge=5, le=10_000),
    alpha_min:   float = Query(default=0.5,  ge=0.1, le=99.0),
    alpha_max:   float = Query(default=25.0, ge=0.5, le=200.0),
    alpha_step:  float = Query(default=0.5,  ge=0.1, le=10.0),
    period_min:  int   = Query(default=10,   ge=1,   le=252),
    period_max:  int   = Query(default=252,  ge=1,   le=252),
    period_step: int   = Query(default=5,    ge=1,   le=50),
    top_n:       int   = Query(default=10,   ge=1,   le=100),
    annualize:   bool  = Query(default=True),
    cost_bps: float = Query(default=10.0, ge=0.0, le=50.0,
        description=(
            "Coste por operación en basis points (bps). "
            "Se aplica round-trip (entrada + salida) a cada señal: "
            "cost_rt = 2 × cost_bps / 10 000. "
            "Descuenta del retorno bruto antes de promediar y anualizar."
        )
    ),
    tax_rate: float = Query(
        default=0.01, ge=0.0, le=0.35,
        description="Tipo impositivo sobre beneficios (vehículo regulado = 1 % = 0.01)"
    ),
    slippage_bps: float = Query(
        default=5.0, ge=0.0, le=50.0,
        description="Slippage estimado por operación en bps (round-trip). "
                    "Líquidas ~3-5 bps, ilíquidas 10-20 bps."
    ),
    rf_annual: float = Query(
        default=0.0, ge=0.0, le=0.20,
        description="Tasa libre de riesgo anual para el Sharpe (e.g. 0.045 = 4,5 %)"
    ),
    token: str = Depends(verificar_api_key),
):
    t0 = time.time()

    if alpha_min >= alpha_max:
        raise HTTPException(422, "alpha_min debe ser < alpha_max")
    if period_min > period_max:
        raise HTTPException(422, "period_min debe ser <= period_max")

    df = _load_granular(year, REND_DIR)
    mat, cum_max, n_signals = _build_matrix(df)
    del df

    n_alphas    = max(1, round((alpha_max - alpha_min) / alpha_step) + 1)
    alphas_grid = np.linspace(alpha_min, alpha_min + alpha_step * (n_alphas - 1), n_alphas)
    periods_lst = list(range(period_min, period_max + 1, period_step))
    if not periods_lst or periods_lst[-1] != period_max:
        periods_lst.append(period_max)
    periods_grid = np.array(periods_lst, dtype=np.int32)

    total_combos = len(alphas_grid) * len(periods_grid)
    if total_combos > 100_000:
        raise HTTPException(422,
            f"Grid demasiado grande ({total_combos:,} combinaciones, máx 100 000). "
            "Reduce rango o aumenta paso.")

    cost_rt     = 2.0 * float(cost_bps)     / 10_000.0
    slippage_rt = 2.0 * float(slippage_bps) / 10_000.0
    total_cost  = cost_rt + slippage_rt

    results: list[dict] = []

    for period in periods_grid:
        Y = int(period) - 1
        if Y <= 0 or Y >= 252:    
            continue

        max_window = cum_max[:, Y - 1]
        alpha_end  = mat[:, Y]

        finite_mask = np.isfinite(alpha_end)
        mw_f = max_window[finite_mask].astype(np.float64)
        ae_f = alpha_end[finite_mask].astype(np.float64)
        if len(ae_f) < 2 * min_samples:
            continue

        for alpha_pct in alphas_grid:
            alpha_val = float(alpha_pct) / 100.0
            q_mask = mw_f > alpha_val

            pos_raw = ae_f[q_mask]
            neg_raw = ae_f[~q_mask]
            n_pos, n_neg = len(pos_raw), len(neg_raw)
            if n_pos < min_samples or n_neg < min_samples:
                continue

            pos_w = _winsorize(pos_raw)
            neg_w = _winsorize(neg_raw)

            pos_net = pos_w - total_cost
            neg_net = neg_w - total_cost

            avg_pos_raw_val = _geo_mean_return(pos_net)
            avg_neg_raw_val = _geo_mean_return(neg_net)

            if annualize:
                avg_pos = _safe_annualize(avg_pos_raw_val, period)
                avg_neg = _safe_annualize(avg_neg_raw_val, period)
            else:
                avg_pos = avg_pos_raw_val
                avg_neg = avg_neg_raw_val

            n_total = n_pos + n_neg
            w_pos   = n_pos / n_total
            w_neg   = n_neg / n_total
            score   = w_pos * avg_pos - w_neg * avg_neg

            ls_returns = np.concatenate([pos_net, -neg_net])
            sharpe_val = _sharpe(ls_returns, period, annualize, rf_annual)

            win_rate_pos = float(np.mean(pos_raw > 0)) * 100.0
            win_rate_neg = float(np.mean(neg_raw > 0)) * 100.0

            score_net = score * (1.0 - tax_rate) if score > 0 else score

            results.append({
                "alpha_pct":               round(float(alpha_pct), 2),
                "period_dias":             int(period),
                "score":                   round(score_net, 6),
                "tir_pos_pct":             round(avg_pos * 100.0, 4),
                "tir_neg_pct":             round(avg_neg * 100.0, 4),
                "tir_cartera_pct":         round(score_net * 100.0, 4),
                "tir_cartera_bruta_pct":   round(score * 100.0, 4),
                "tir_pos_crudo_pct":       round(float(np.mean(pos_raw)) * 100.0, 4),
                "tir_neg_crudo_pct":       round(float(np.mean(neg_raw)) * 100.0, 4),
                "sharpe":                  round(sharpe_val, 4),
                "win_rate_pos_pct":        round(win_rate_pos, 2),
                "win_rate_neg_pct":        round(win_rate_neg, 2),
                "cost_rt_pct":             round(total_cost * 100.0, 4),
                "n_pos":                   n_pos,
                "n_neg":                   n_neg,
                "pct_seniales_positivas":  round(100.0 * w_pos, 2),
                "n_total_valido":          n_total,
            })

    if not results:
        raise HTTPException(404,
            "Sin combinaciones con muestras suficientes. "
            "Prueba a reducir min_samples o ampliar rangos.")

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "año":               year,
        "mejor_combinacion": results[0],
        "top":               results[:top_n],
        "stats": {
            "combinaciones_evaluadas": len(results),
            "n_seniales_totales":      n_signals,
            "tiempo_ejecucion_s":      round(time.time() - t0, 3),
            "annualize":               annualize,
            "cost_bps":                cost_bps,
            "tax_rate_pct":       round(tax_rate * 100.0, 1),
            "slippage_bps":            slippage_bps,
            "rf_annual_pct":           round(rf_annual * 100.0, 2),
        },
        "parametros_busqueda": {
            "alpha_min_pct":  round(float(alpha_min), 2),
            "alpha_max_pct":  round(float(alpha_max), 2),
            "alpha_step_pct": round(float(alpha_step), 2),
            "period_min":     int(period_min),
            "period_max":     int(period_max),
            "period_step":    int(period_step),
            "min_samples":    min_samples,
        },
    }

@app.get("/optimize/multi", tags=["Optimizar"],
         summary="Optimizar en rango de años (consistencia histórica)",
         description=(
             "Ejecuta el grid search de `/optimize` en cada año de [year_from, year_to] "
             "con los mismos parámetros. Devuelve:\n\n"
             "- Mejor combinación por año\n"
             "- Ranking agregado por frecuencia de aparición en el top-N\n"
             "- Métricas de consistencia (% años en positivo, Sharpe medio, etc.)"
         ))
async def optimize_alpha_period_multi(
    year_from:   int   = Query(..., ge=2000, le=2025),
    year_to:     int   = Query(..., ge=2000, le=2025),
    min_samples: int   = Query(default=30,   ge=5,   le=10_000),
    alpha_min:   float = Query(default=0.5,  ge=0.1, le=99.0),
    alpha_max:   float = Query(default=25.0, ge=0.5, le=200.0),
    alpha_step:  float = Query(default=0.5,  ge=0.1, le=10.0),
    period_min:  int   = Query(default=10,   ge=1,   le=252),
    period_max:  int   = Query(default=252,  ge=1,   le=252),
    period_step: int   = Query(default=5,    ge=1,   le=50),
    top_n:       int   = Query(default=5,    ge=1,   le=20,
                                description="Top-N por año para el ranking agregado"),
    annualize:   bool  = Query(default=True),
    cost_bps:    float = Query(default=7.5,  ge=0.0, le=50.0),
    tax_rate: float = Query(
        default=0.01, ge=0.0, le=0.35,
        description="Tipo impositivo sobre beneficios (vehículo regulado = 1 % = 0.01)"
    ),
    slippage_bps: float = Query(
        default=5.0, ge=0.0, le=50.0,
        description="Slippage estimado por operación en bps (round-trip)."
    ),
    rf_annual: float = Query(
        default=0.0, ge=0.0, le=0.20,
        description="Tasa libre de riesgo anual para el Sharpe (e.g. 0.045 = 4,5 %)"
    ),
    token:       str   = Depends(verificar_api_key),
):
    if year_from > year_to:
        raise HTTPException(422, "year_from debe ser <= year_to")

    t0           = time.time()
    por_año      = {}
    from collections import defaultdict
    freq: dict[tuple, list] = defaultdict(list)

    n_alphas    = max(1, round((alpha_max - alpha_min) / alpha_step) + 1)
    alphas_grid = np.linspace(alpha_min, alpha_min + alpha_step * (n_alphas - 1), n_alphas)
    periods_lst = list(range(period_min, period_max + 1, period_step))
    if not periods_lst or periods_lst[-1] != period_max:
        periods_lst.append(period_max)
    periods_grid = np.array(periods_lst, dtype=np.int32)

    total_combos = len(alphas_grid) * len(periods_grid)
    if total_combos > 100_000:
        raise HTTPException(422, f"Grid demasiado grande ({total_combos:,}). Máx 100 000.")

    cost_rt     = 2.0 * float(cost_bps)     / 10_000.0
    slippage_rt = 2.0 * float(slippage_bps) / 10_000.0
    total_cost  = cost_rt + slippage_rt

    for year in range(year_from, year_to + 1):
        try:
            df = _load_granular(year, REND_DIR)
        except HTTPException:
            por_año[year] = {"error": f"Sin datos para {year}"}
            continue

        mat, cum_max, n_sig = _build_matrix(df)
        del df

        year_results = []

        for period in periods_grid:
            Y = int(period) - 1
            if Y <= 0 or Y >= 252:
                continue

            max_window  = cum_max[:, Y - 1]
            alpha_end   = mat[:, Y]
            finite_mask = np.isfinite(alpha_end)
            mw_f = max_window[finite_mask].astype(np.float64)
            ae_f = alpha_end[finite_mask].astype(np.float64)
            if len(ae_f) < 2 * min_samples:
                continue

            for alpha_pct in alphas_grid:
                alpha_val = float(alpha_pct) / 100.0
                q_mask    = mw_f > alpha_val
                pos_raw   = ae_f[q_mask]
                neg_raw   = ae_f[~q_mask]
                if len(pos_raw) < min_samples or len(neg_raw) < min_samples:
                    continue

                pos_w = _winsorize(pos_raw) - total_cost
                neg_w = _winsorize(neg_raw) - total_cost

                avg_p = _geo_mean_return(pos_w)
                avg_n = _geo_mean_return(neg_w)

                if annualize:
                    avg_p = _safe_annualize(avg_p, period)
                    avg_n = _safe_annualize(avg_n, period)

                n_t   = len(pos_raw) + len(neg_raw)
                w_pos = len(pos_raw) / n_t
                w_neg = len(neg_raw) / n_t
                score = w_pos * avg_p - w_neg * avg_n

                score_net = score * (1.0 - tax_rate) if score > 0 else score

                ls     = np.concatenate([pos_w, -neg_w])
                sharpe = _sharpe(ls, period, annualize, rf_annual)

                year_results.append({
                    "alpha_pct":             round(float(alpha_pct), 2),
                    "period_dias":           int(period),
                    "score":                 round(score_net, 6),
                    "tir_cartera_pct":       round(score_net * 100.0, 4),
                    "tir_cartera_bruta_pct": round(score * 100.0, 4),
                    "sharpe":                round(sharpe, 4),
                    "n_pos":                 len(pos_raw),
                    "n_neg":                 len(neg_raw),
                })

        if not year_results:
            por_año[year] = {"error": "Sin combinaciones válidas"}
            continue

        year_results.sort(key=lambda x: x["score"], reverse=True)
        best      = year_results[0]
        top_local = year_results[:top_n]

        for r in top_local:
            key = (r["alpha_pct"], r["period_dias"])
            freq[key].append({
                "year":  year,
                "score": r["score"],
                "sharpe": r["sharpe"],
            })

        por_año[year] = {
            "mejor":           best,
            "top":             top_local,
            "n_seniales":      n_sig,
            "combinaciones":   len(year_results),
        }

    ranking_agg = []
    años_total  = year_to - year_from + 1
    for (a_pct, p_dias), entries in freq.items():
        scores_list  = [e["score"]  for e in entries]
        sharpe_list  = [e["sharpe"] for e in entries]
        años_pos     = sum(1 for s in scores_list if s > 0)
        ranking_agg.append({
            "alpha_pct":            a_pct,
            "period_dias":          p_dias,
            "n_años_en_top":        len(entries),
            "pct_años_en_top":      round(100.0 * len(entries) / años_total, 1),
            "pct_años_positivo":    round(100.0 * años_pos / len(entries), 1),
            "score_medio":          round(float(np.mean(scores_list)), 6),
            "score_mediana":        round(float(np.median(scores_list)), 6),
            "score_std":            round(float(np.std(scores_list)), 6),
            "sharpe_medio":         round(float(np.mean(sharpe_list)), 4),
        })

    ranking_agg.sort(key=lambda x: (x["n_años_en_top"], x["score_medio"]), reverse=True)

    return {
        "rango_años":        f"{year_from}-{year_to}",
        "años_con_datos":    sum(1 for v in por_año.values() if "error" not in v),
        "resultados_por_año": por_año,
        "ranking_agregado":  ranking_agg[:20],
        "stats": {
            "tiempo_ejecucion_s": round(time.time() - t0, 3),
            "annualize":          annualize,
            "cost_bps":           cost_bps,
            "tax_rate_pct":       round(tax_rate * 100.0, 1),
            "slippage_bps":            slippage_bps,
            "rf_annual_pct":           round(rf_annual * 100.0, 2),
        },
    }

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