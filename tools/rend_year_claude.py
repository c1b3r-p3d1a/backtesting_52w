from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ====================== CONFIGURACIÓN ======================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

BASE_DIR = SCRIPT_DIR.parent

PRECOMPUTED_DIR = BASE_DIR / "db" / "rend_year"
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

SP500_PATH = BASE_DIR / "db" / "sp500.csv"
CSV_PATH = BASE_DIR / "db" / "max.csv"

SP500 = pd.read_csv(SP500_PATH, sep=",", encoding="utf-8")
PRICE_DATA = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8")

# Normalizar fechas para que el mapping sea fiable
if "DATE" not in SP500.columns:
    raise ValueError("SP500.csv debe tener columna 'DATE'")
if "ADJ_CLOSE" not in SP500.columns:
    raise ValueError("SP500.csv debe tener columna 'ADJ_CLOSE'")
if "FECHA" not in PRICE_DATA.columns:
    raise ValueError("max.csv debe tener columna 'FECHA'")
if "TICKER" not in PRICE_DATA.columns:
    raise ValueError("max.csv debe tener columna 'TICKER'")
if "MARKET_CAP" not in PRICE_DATA.columns:
    raise ValueError("max.csv debe tener columna 'MARKET_CAP'")

SP500["DATE"] = pd.to_datetime(SP500["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
SP500 = SP500.dropna(subset=["DATE"]).reset_index(drop=True)

PRICE_DATA["FECHA"] = PRICE_DATA["FECHA"].astype(str)

PARQUET_CACHE: dict = {}

SP500_DATE_TO_IDX: dict = {date: i for i, date in enumerate(SP500["DATE"])}
SP500_ADJ_CLOSE: np.ndarray = SP500["ADJ_CLOSE"].astype(float).to_numpy()

UMBRALES_CAP = [
    ("1b", 1_000_000_000),
    ("5b", 5_000_000_000),
    ("10b", 10_000_000_000),
    ("20b", 20_000_000_000),
    ("50b", 50_000_000_000),
    ("100b", 100_000_000_000),
]


def limpiar_valores(x):
    if pd.isna(x):
        return None
    if isinstance(x, (float, np.floating)) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def get_parquet_db(ticker: str):
    ticker = str(ticker)
    if not ticker:
        return None

    first = ticker[0].lower()
    if len(ticker) < 2:
        path = BASE_DIR / "db" / "fragmented" / first / "_.parquet"
    else:
        second = ticker[1].lower()
        path = BASE_DIR / "db" / "fragmented" / first / f"{second}.parquet"

    path_str = str(path)
    if path_str not in PARQUET_CACHE:
        if not path.is_file():
            PARQUET_CACHE[path_str] = None
        else:
            db = pd.read_parquet(path).reset_index(drop=True)
            if "report_date" not in db.columns:
                PARQUET_CACHE[path_str] = None
            else:
                db["_fecha_str"] = pd.to_datetime(db["report_date"], errors="coerce").dt.strftime("%Y-%m-%d")
                PARQUET_CACHE[path_str] = db

    return PARQUET_CACHE[path_str]


def calcular_curvas_señal(ticker: str, fecha: str):
    db = get_parquet_db(ticker)
    if db is None:
        return None, None

    if "symbol" not in db.columns or "open" not in db.columns or "close" not in db.columns:
        return None, None

    mask = (db["symbol"] == ticker) & (db["_fecha_str"] == fecha)
    posiciones = np.where(mask.values)[0]
    if len(posiciones) == 0:
        return None, None

    pos_ticker = int(posiciones[0])
    pos_sp = SP500_DATE_TO_IDX.get(fecha)
    if pos_sp is None:
        return None, None

    if pos_ticker + 1 >= len(db) or pos_sp + 1 >= len(SP500_ADJ_CLOSE):
        return None, None

    precio_entrada = float(db["open"].iat[pos_ticker + 1])
    sp500_entrada = float(SP500_ADJ_CLOSE[pos_sp + 1])

    if not np.isfinite(precio_entrada) or precio_entrada == 0:
        return None, None
    if not np.isfinite(sp500_entrada) or sp500_entrada == 0:
        return None, None

    fin_t = min(pos_ticker + 254, len(db))
    fin_s = min(pos_sp + 254, len(SP500_ADJ_CLOSE))

    precios_slice = pd.to_numeric(db["close"].iloc[pos_ticker + 2:fin_t], errors="coerce").to_numpy(dtype=float)
    sp500_slice = SP500_ADJ_CLOSE[pos_sp + 2:fin_s].astype(float)

    rets_ticker = np.full(252, np.nan, dtype=float)
    rets_sp500 = np.full(252, np.nan, dtype=float)

    if len(precios_slice):
        rets_ticker[:len(precios_slice)] = (precios_slice - precio_entrada) / precio_entrada
    if len(sp500_slice):
        rets_sp500[:len(sp500_slice)] = (sp500_slice - sp500_entrada) / sp500_entrada

    return rets_ticker, rets_sp500


def _nanmean_axis0(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return np.full(252, np.nan, dtype=float)

    valid = ~np.isnan(mat)
    count = valid.sum(axis=0)
    total = np.nansum(mat, axis=0)
    out = np.full(mat.shape[1], np.nan, dtype=float)
    np.divide(total, count, out=out, where=count != 0)
    return out


def agregar_curvas(curvas: list) -> list:
    if not curvas:
        return []

    mat_ticker = np.vstack([np.asarray(c["rets_ticker"], dtype=float) for c in curvas])
    mat_sp500 = np.vstack([np.asarray(c["rets_sp500"], dtype=float) for c in curvas])

    avg_ticker = _nanmean_axis0(mat_ticker)
    avg_sp500 = _nanmean_axis0(mat_sp500)
    alpha = avg_ticker - avg_sp500

    return [
        [
            k + 1,
            None if np.isnan(avg_ticker[k]) else float(avg_ticker[k]),
            None if np.isnan(avg_sp500[k]) else float(avg_sp500[k]),
            None if np.isnan(alpha[k]) else float(alpha[k]),
        ]
        for k in range(252)
    ]


def calcular_year_performance(year: int, progress_bar=None):
    señales_raw = [
        (row.TICKER, row.FECHA, limpiar_valores(row.MARKET_CAP))
        for row in PRICE_DATA.itertuples(index=False)
        if str(row.FECHA).startswith(str(year))
    ]

    if not señales_raw:
        return None

    cap_inicio: dict = {}
    for ticker, fecha, cap in señales_raw:
        if ticker not in cap_inicio:
            cap_inicio[ticker] = cap

    curvas = []
    for ticker, fecha, _ in señales_raw:
        rets_ticker, rets_sp500 = calcular_curvas_señal(ticker, fecha)
        if rets_ticker is not None:
            curvas.append(
                {
                    "ticker": ticker,
                    "cap": cap_inicio.get(ticker),
                    "rets_ticker": rets_ticker,
                    "rets_sp500": rets_sp500,
                }
            )
        if progress_bar is not None:
            progress_bar.update(1)

    if not curvas:
        return {}

    respuesta = {
        "año": year,
        "señales_procesadas": len(curvas),
        "all": agregar_curvas(curvas),
    }

    for etiqueta, umbral in UMBRALES_CAP:
        subset = [c for c in curvas if c["cap"] is not None and c["cap"] >= umbral]
        respuesta[f"cap_{etiqueta}"] = agregar_curvas(subset) if subset else []

    return respuesta


def precompute_all_years():
    print("Iniciando precomputación de rendimientos por año (2000-2025)...")

    years = list(range(2000, 2026))
    year_counts = (
        PRICE_DATA["FECHA"]
        .astype(str)
        .str[:4]
        .value_counts()
        .to_dict()
    )

    with tqdm(total=len(years), desc="Años", unit="año") as years_bar:
        for year in years:
            csv_path = PRECOMPUTED_DIR / f"rend_{year}.csv"
            meta_path = PRECOMPUTED_DIR / f"rend_{year}_meta.json"

            if csv_path.exists() and meta_path.exists():
                tqdm.write(f"Año {year} ya precomputado; saltando.")
                years_bar.update(1)
                continue

            total_señales = int(year_counts.get(str(year), 0))

            tqdm.write(f"Calculando año {year}...")
            start_time = time.time()

            with tqdm(total=total_señales, desc=f"{year}", unit="señal", leave=False) as signal_bar:
                resultado = calcular_year_performance(year, signal_bar)

            if resultado is None:
                tqdm.write(f"Año {year}: no hay señales.")
                years_bar.update(1)
                continue

            if not resultado or not resultado.get("all"):
                tqdm.write(f"Año {year}: no se generaron curvas.")
                years_bar.update(1)
                continue

            records = []
            for key, curves in resultado.items():
                if key in ["año", "señales_procesadas"]:
                    continue
                if not curves:
                    continue

                for row in curves:
                    records.append(
                        {
                            "CAP": key,
                            "DAY": int(row[0]),
                            "STOCK_RET": row[1],
                            "SP500_RET": row[2],
                            "ALPHA": row[3],
                        }
                    )

            df = pd.DataFrame(records)
            df.to_csv(csv_path, index=False, float_format="%.8f")

            meta = {
                "año": resultado["año"],
                "señales_procesadas": resultado["señales_procesadas"],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            elapsed = time.time() - start_time
            tqdm.write(
                f"Año {year} guardado correctamente: "
                f"{resultado['señales_procesadas']} señales en {elapsed:.1f}s"
            )

            years_bar.update(1)

    print("Precomputación finalizada.")


if __name__ == "__main__":
    precompute_all_years()