# ===================================================================
# SCRIPT: precompute_granular_alpha.py
# ===================================================================
# GUÁRDALO en la raíz de tu proyecto (al lado de tu archivo backend.py)
# Ejecuta: python precompute_granular_alpha.py
#
# Esto genera un archivo por año:
#    db/rend_year/granular_alpha_{year}.parquet
#
# Columnas:
#    ticker, fecha, market_cap, day, alpha
#
# Con esto tu API ya puede leerlo y hacer los cálculos de X, Y en milisegundos.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import math

# ====================== MISMA CONFIGURACIÓN QUE TU API ======================
SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
BASE_DIR = SCRIPT_DIR.parent
REND_DIR = os.path.join(BASE_DIR, "db", "rend_year")
os.makedirs(REND_DIR, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "db", "max.csv")
SP500_PATH = os.path.join(BASE_DIR, "db", "sp500.csv")

print("[+] Cargando PRICE_DATA y SP500...")
PRICE_DATA = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8")
SP500 = pd.read_csv(SP500_PATH, sep=",", encoding="utf-8")

# Normalizar fechas
SP500["DATE"] = pd.to_datetime(SP500["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
SP500 = SP500.dropna(subset=["DATE"]).reset_index(drop=True)

SP500_DATE_TO_IDX = {date: i for i, date in enumerate(SP500["DATE"])}
SP500_ADJ_CLOSE = SP500["ADJ_CLOSE"].astype(float).to_numpy()

# ====================== FUNCIONES QUE YA USAS EN TU PROYECTO ======================
def get_parquet_db(ticker: str):
    ticker = str(ticker).upper()
    if not ticker or len(ticker) < 1:
        return None

    first = ticker[0].lower()
    if len(ticker) < 2:
        path = Path(BASE_DIR) / "db" / "fragmented" / first / "_.parquet"
    else:
        second = ticker[1].lower()
        path = Path(BASE_DIR) / "db" / "fragmented" / first / f"{second}.parquet"

    if not path.is_file():
        return None

    db = pd.read_parquet(path).reset_index(drop=True)
    if "report_date" not in db.columns:
        return None

    db["_fecha_str"] = pd.to_datetime(db["report_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return db


def calcular_curvas_señal(ticker: str, fecha: str):
    db = get_parquet_db(ticker)
    if db is None:
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


def limpiar_valores(x):
    if pd.isna(x):
        return None
    if isinstance(x, (float, np.floating)) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


# ====================== PRECÁLCULO GRANULAR ======================
def precompute_granular_alpha(year: int):
    """Genera el archivo granular_alpha_{year}.parquet"""
    print(f"\n🔄 Procesando año {year}...")

    señales_raw = [
        (row.TICKER, row.FECHA, limpiar_valores(row.MARKET_CAP))
        for row in PRICE_DATA.itertuples(index=False)
        if str(row.FECHA).startswith(str(year))
    ]

    if not señales_raw:
        print(f"❌ No hay señales para el año {year}")
        return

    records = []
    for ticker, fecha, cap in tqdm(señales_raw, desc=f"Año {year}", unit="señal"):
        rets_ticker, rets_sp500 = calcular_curvas_señal(ticker, fecha)
        if rets_ticker is None:
            continue

        alpha = rets_ticker - rets_sp500

        for d in range(252):
            val = alpha[d]
            records.append({
                "ticker": ticker,
                "fecha": fecha,
                "market_cap": cap,
                "day": d + 1,
                "alpha": float(val) if np.isfinite(val) else None,
            })

    df = pd.DataFrame(records)
    output_path = os.path.join(REND_DIR, f"granular_alpha_{year}.parquet")

    df.to_parquet(output_path, index=False, compression="zstd")

    print(f"✅ Guardado: {len(df):,} registros • {df['ticker'].nunique():,} señales • {output_path}")


def precompute_all_years_granular(years=None):
    """Ejecuta para todos los años (o los que le pases)"""
    if years is None:
        years = list(range(2000, 2026))

    for year in tqdm(years, desc="Años completos"):
        path = os.path.join(REND_DIR, f"granular_alpha_{year}.parquet")
        if os.path.exists(path):
            tqdm.write(f"⏭️  {year} ya existe → saltando")
            continue
        precompute_granular_alpha(year)

    print("\n🎉 ¡Precomputación granular FINALIZADA!")
    print("   Ahora ya puedes crear el endpoint /analyze en tu API")


# ====================== EJECUCIÓN ======================
if __name__ == "__main__":
    # Opción 1: Todos los años (recomendado la primera vez)
    precompute_all_years_granular()

    # Opción 2: Solo un año (para pruebas rápidas)
    # precompute_granular_alpha(2020)

    # Opción 3: Varios años concretos
    # precompute_all_years_granular([2020, 2021, 2022, 2023, 2024])