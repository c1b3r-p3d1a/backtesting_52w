"""
split_parquet.py
----------------
Lee un archivo .parquet y lo fragmenta en múltiples archivos según
la primera letra/número del símbolo (ticker).

Uso:
    python split_parquet.py <input.parquet> [output_dir]

Argumentos:
    input.parquet  → ruta al archivo parquet de entrada
    output_dir     → carpeta de salida (por defecto: misma carpeta que el input)
"""

import sys
import os
import pandas as pd


# ── Configuración ────────────────────────────────────────────────────────────

COLUMN_RENAME = {
    "symbol":      "TICKER",
    "report_date": "FECHA",
    "open":        "OPEN",
    "high":        "HIGH",
    "low":         "LOW",
    "close":       "CLOSE",
    "volume":      "VOLUME",
}

# Caracteres válidos para particionar (a-z + 0-9)
VALID_CHARS = [c for c in "abcdefghijklmnopqrstuvwxyz0123456789"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    if len(sys.argv) < 2:
        print("Uso: python split_parquet.py <input.parquet> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"❌ No se encontró el archivo: {input_path}")
        sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    return input_path, output_dir


def load_and_prepare(input_path: str) -> pd.DataFrame:
    print(f"📂 Leyendo: {input_path}")
    df = pd.read_parquet(input_path)

    # Validar columnas requeridas
    missing = [c for c in COLUMN_RENAME if c not in df.columns]
    if missing:
        print(f"❌ Columnas no encontradas en el parquet: {missing}")
        sys.exit(1)

    # Renombrar columnas
    df = df.rename(columns=COLUMN_RENAME)

    # Asegurar tipo correcto en VOLUME
    df["VOLUME"] = df["VOLUME"].astype("Int64")

    # Columna auxiliar con la primera letra en minúscula
    df["_first_char"] = df["TICKER"].astype(str).str[0].str.lower()

    print(f"✅ Filas cargadas: {len(df):,}")
    return df


def write_partitions(df: pd.DataFrame, output_dir: str):
    created = []

    for char in VALID_CHARS:
        subset = df[df["_first_char"] == char].drop(columns=["_first_char"])

        if subset.empty:
            continue  # No crear archivo si no hay datos

        out_path = os.path.join(output_dir, f"{char}.parquet")
        subset.to_parquet(out_path, index=False, engine="pyarrow")
        created.append((char, len(subset), out_path))
        print(f"  ✔ {char}.parquet  →  {len(subset):>10,} filas")

    return created


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    input_path, output_dir = parse_args()
    df = load_and_prepare(input_path)

    print(f"\n🔀 Fragmentando en: {output_dir}\n")
    created = write_partitions(df, output_dir)

    print(f"\n{'─'*50}")
    print(f"✅ Archivos generados: {len(created)}")
    print(f"   Letras/números sin datos: "
          f"{len(VALID_CHARS) - len(created)} (no se crearon)")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()