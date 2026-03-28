"""
descargar_historico.py
======================
Descarga el histórico completo de NYSE + NASDAQ usando defeatbeta-api
(datos en Parquet sobre Hugging Face, sin rate limits, sin API key).

Estrategia:
  1. Descarga el archivo stock_prices.parquet de Hugging Face con barra de progreso.
  2. Lo guarda en disco como caché local (no se vuelve a descargar si ya existe).
  3. Convierte al CSV con el esquema:  TICKER, DATE, OPEN, HIGH, LOW, CLOSE, VOLUME

Requisitos:
    pip install duckdb requests rich pandas

Uso:
    python descargar_historico.py            # descarga + convierte
    python descargar_historico.py --solo-csv # salta la descarga (usa caché local)
    python descargar_historico.py --forzar   # re-descarga aunque ya exista el parquet
    python descargar_historico.py --consultar 2024-03-15   # bonus: 52-week high query
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import duckdb
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.table import Table

# ── Configuración ─────────────────────────────────────────────────────────────

LOCAL_PARQUET = "stock_prices.parquet"   # caché local
OUTPUT_CSV    = "historico_completo.csv"

# Filtro de fechas (ajusta si quieres otro rango)
START_DATE = "2000-01-01"
END_DATE   = None   # None = hasta el último dato disponible

CHUNK_SIZE = 8 * 1024 * 1024   # 8 MB por chunk de descarga

console = Console()

def _fmt_bytes(n: int) -> str:
    if n < 1024:      return f"{n} B"
    if n < 1024**2:   return f"{n/1024:.1f} KB"
    if n < 1024**3:   return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERSIÓN PARQUET → CSV CON BARRA DE PROGRESO
# ─────────────────────────────────────────────────────────────────────────────

def _where_fecha(alias: str = "") -> str:
    """Genera la cláusula WHERE de fechas (vacía si no hay filtro)."""
    col = f"{alias}.report_date" if alias else "report_date"
    partes = []
    if START_DATE:
        partes.append(f"{col} >= '{START_DATE}'")
    if END_DATE:
        partes.append(f"{col} <= '{END_DATE}'")
    return ("WHERE " + " AND ".join(partes)) if partes else ""


def convertir_a_csv(parquet: str, salida: str) -> None:
    """Convierte el Parquet local a CSV con el esquema del proyecto."""

    console.print("[bold cyan]→ Convirtiendo Parquet → CSV...[/]")

    con = duckdb.connect()
    where = _where_fecha()

    # ── Conteo previo ────────────────────────────────────────────────────────
    console.print("  [dim]Analizando Parquet (puede tardar unos segundos)...[/]", end="")
    total_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquet}') {where}"
    ).fetchone()[0]
    n_tickers = con.execute(
        f"SELECT COUNT(DISTINCT symbol) FROM read_parquet('{parquet}') {where}"
    ).fetchone()[0]
    console.print(
        f"\r  Registros: [bold white]{total_rows:,}[/] · "
        f"Tickers: [bold white]{n_tickers:,}[/]\n"
    )

    # ── Lista de tickers ─────────────────────────────────────────────────────
    tickers_list: list[str] = [
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT symbol FROM read_parquet('{parquet}') {where} ORDER BY symbol"
        ).fetchall()
    ]

    # ── Progress + panel en vivo ─────────────────────────────────────────────
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=38),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )
    task = progress.add_task("[cyan]Exportando a CSV[/]", total=total_rows)
    stats = {"filas": 0, "tickers_ok": 0, "t0": time.time()}

    def _panel() -> Panel:
        elapsed = time.time() - stats["t0"]
        vel = stats["filas"] / elapsed if elapsed > 0 else 0
        tam = os.path.getsize(salida) if os.path.exists(salida) else 0
        g = Table.grid(padding=(0, 1))
        g.add_column(style="bold cyan", justify="right")
        g.add_column()
        g.add_row("Filas escritas", f"[white]{stats['filas']:,}[/]")
        g.add_row("Tickers OK",     f"[green]{stats['tickers_ok']:,}[/]")
        g.add_row("Tamaño CSV",     f"[white]{_fmt_bytes(tam)}[/]")
        g.add_row("Velocidad",      f"[white]{vel:,.0f} filas/s[/]")
        g.add_row("Tiempo",         f"[white]{elapsed:.0f}s[/]")
        return Panel(g, title="[bold]Exportando[/]", border_style="cyan", expand=True)

    # ── Escritura incremental ─────────────────────────────────────────────────
    escribir_cabecera = not os.path.exists(salida) or os.path.getsize(salida) == 0

    with open(salida, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if escribir_cabecera:
            writer.writerow(["TICKER", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

        with Live(console=console, refresh_per_second=8) as live:
            for ticker in tickers_list:

                # Condición de fechas por ticker (más eficiente que WHERE global en
                # DuckDB cuando se repite la misma tabla muchas veces)
                cond = f"symbol = '{ticker}'"
                if START_DATE:
                    cond += f" AND report_date >= '{START_DATE}'"
                if END_DATE:
                    cond += f" AND report_date <= '{END_DATE}'"

                rows = con.execute(f"""
                    SELECT
                        symbol      AS TICKER,
                        report_date AS DATE,
                        open        AS OPEN,
                        high        AS HIGH,
                        low         AS LOW,
                        close       AS CLOSE,
                        CAST(volume AS BIGINT) AS VOLUME
                    FROM read_parquet('{parquet}')
                    WHERE {cond}
                    ORDER BY report_date
                """).fetchall()

                writer.writerows(rows)

                stats["filas"]     += len(rows)
                stats["tickers_ok"] += 1
                progress.advance(task, len(rows))

                layout = Table.grid(padding=(0, 2))
                layout.add_column(ratio=2)
                layout.add_column(min_width=30)
                layout.add_row(progress, _panel())
                live.update(layout)

    con.close()

    elapsed = time.time() - stats["t0"]
    tam_final = os.path.getsize(salida)
    console.print(
        f"\n[bold green]✓ CSV generado:[/] [bold cyan]{os.path.abspath(salida)}[/]"
    )
    console.print(f"  Filas:   [bold white]{stats['filas']:,}[/]")
    console.print(f"  Tickers: [bold white]{stats['tickers_ok']:,}[/]")
    console.print(f"  Tamaño:  [bold white]{_fmt_bytes(tam_final)}[/]")
    console.print(f"  Tiempo:  [bold white]{elapsed:.0f}s[/]")

if __name__ == "__main__":
    convertir_a_csv(LOCAL_PARQUET, OUTPUT_CSV)