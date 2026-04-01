"""
split_parquet_nested.py
-----------------------
Lee un archivo .parquet y lo fragmenta en múltiples carpetas y archivos según
la primera y segunda letra del símbolo (ticker), usando chunks para memoria eficiente.
Tickers de un solo carácter se guardan en <primera_letra>/_.parquet

Uso:
    python split_parquet_nested.py <input.parquet> [output_dir]
"""

import sys
import os
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq

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

VALID_CHARS = [c for c in "abcdefghijklmnopqrstuvwxyz0123456789"] + ["_"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    if len(sys.argv) < 2:
        print("Uso: python split_parquet_nested.py <input.parquet> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"❌ No se encontró el archivo: {input_path}")
        sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)
    return input_path, output_dir

def load_and_prepare_chunks(input_path: str, batch_size=100_000):
    """Lee el parquet en chunks y agrega columnas auxiliares para fragmentar."""
    print(f"📂 Leyendo: {input_path}")
    pq_file = pq.ParquetFile(input_path)
    total_rows = pq_file.metadata.num_rows
    total_batches = (total_rows + batch_size - 1) // batch_size

    for batch in tqdm(pq_file.iter_batches(batch_size=batch_size),
                      total=total_batches,
                      desc="Leyendo parquet"):
        df = batch.to_pandas()
        df["_first_char"] = df["symbol"].str[0].str.lower()
        # Tickers de un solo carácter usan "_" como segunda letra
        df["_second_char"] = df["symbol"].apply(
            lambda s: s[1].lower() if len(str(s)) > 1 else "_"
        )
        yield df

def write_nested_partitions(df: pd.DataFrame, output_dir: str):
    """Fragmenta un chunk de DataFrame por primera y segunda letra y guarda los parquet."""
    created = []

    for first in VALID_CHARS:
        df_first = df[df["_first_char"] == first]
        if df_first.empty:
            continue

        first_dir = os.path.join(output_dir, first)
        os.makedirs(first_dir, exist_ok=True)

        for second in VALID_CHARS:
            df_second = df_first[df_first["_second_char"] == second].drop(
                columns=["_first_char", "_second_char"]
            )
            if df_second.empty:
                continue

            out_path = os.path.join(first_dir, f"{second}.parquet")
            if os.path.exists(out_path):
                df_existing = pd.read_parquet(out_path)
                df_second = pd.concat([df_existing, df_second], ignore_index=True)

            df_second.to_parquet(out_path, index=False, engine="pyarrow")
            created.append((first, second, len(df_second), out_path))
            print(f"  ✔ {first}/{second}.parquet  →  {len(df_second):>10,} filas")

    return created

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    input_path, output_dir = parse_args()

    print(f"\n🔀 Fragmentando en carpetas por primera y segunda letra: {output_dir}\n")
    total_created = []

    for df_chunk in load_and_prepare_chunks(input_path):
        created = write_nested_partitions(df_chunk, output_dir)
        total_created.extend(created)

    print(f"\n{'─'*50}")
    print(f"✅ Archivos generados: {len(total_created)}")
    print(f"{'─'*50}\n")

if __name__ == "__main__":
    main()