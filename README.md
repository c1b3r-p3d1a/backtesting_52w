# 52-Week High Signal System

Sistema de detección automática de señales 52-Week High sobre el mercado americano. Procesa el historial completo de precios OHLCV, detecta cuándo un valor supera el máximo de sus 252 sesiones anteriores, calcula el market cap en el día de la señal, y expone los resultados a través de una API REST local autenticada con cinco endpoints.

---

## Estructura del repositorio

```
backtesting_52w/
│
├── 52w.py                         # Motor de señales → genera db/max.csv
├── backend.py                     # API REST (FastAPI, puerto 3333)
├── price_parquet_to_csv.py        # Conversor parquet → db/historico_completo.csv
│
├── tools/
│   └── price_parquet_to_csv_claude.py   # Clase ParquetToCSV usada por backend.py
│
├── db/                            # Datos — no commiteados salvo los parquets base
│   ├── stock_prices.parquet       # Historial de precios OHLCV (HuggingFace)
│   ├── stock_profile.parquet      # Datos descriptivos de empresas (HuggingFace)
│   ├── stock_profile.csv          # stock_profile convertido a CSV
│   ├── stock_shares_outstanding.parquet  # Acciones en circulación (HuggingFace)
│   ├── sp500.csv                  # Historial del S&P 500 (DATE, OPEN, ADJ_CLOSE, HIGH, LOW, VOLUME)
│   ├── historico_completo.csv     # Precios OHLCV convertidos (~1,4 GB, no commiteado)
│   ├── max.csv                    # Señales 52W generadas por 52w.py
│   └── fragmented/                # Parquets por letra para /rend (a.parquet, b.parquet…)
│
├── .gitignore
├── .gitattributes
└── LICENSE.md
```

> Los ficheros grandes (`historico_completo.csv`, `sp500.csv`, `db/fragmented/`) no están en el repositorio. Deben generarse o descargarse localmente siguiendo los pasos de instalación.

---

## Flujo de datos

```
HuggingFace (parquets)
        │
        ▼
price_parquet_to_csv.py  →  db/historico_completo.csv
        │
        ▼
      52w.py             →  db/max.csv  (señales + market cap)
        │
        ▼
    backend.py
```

---

## Dataset

| Fichero | Fuente | Contenido |
|---|---|---|
| `db/stock_prices.parquet` | [HuggingFace / defeatbeta](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data) | Precios OHLCV diarios — 10.907 tickers desde 1994 |
| `db/stock_profile.parquet` | HuggingFace / defeatbeta | Sector, descripción, empleados, web |
| `db/stock_shares_outstanding.parquet` | [ycharts](https://ycharts.com/companies/${symbol}/shares_outstanding) | Acciones en circulación por fecha (para calcular market cap) |
| `db/sp500.csv` | Yahoo Finance | Histórico del S&P 500 — OHLCV + ADJ_CLOSE |
| `db/fragmented/` | Generado desde `historico_completo.csv` | Un parquet por inicial de ticker (a.parquet…z.parquet) para consultas de rendimiento |

---

## Instalación

```bash
git clone https://github.com/c1b3r-p3d1a/backtesting_52w.git
cd backtesting_52w
pip install pandas pyarrow duckdb fastapi uvicorn python-dotenv numpy bisect
```

Crea un fichero `.env` en la raíz:

```
API_KEY=tu_clave_aqui
```

Crea la carpeta `db/` y coloca en ella los parquets de HuggingFace (`stock_prices.parquet`, `stock_profile.parquet`, `stock_shares_outstanding.parquet`) y el fichero `sp500.csv`.

---

## Uso

### 1. Convertir el parquet a CSV

```bash
python price_parquet_to_csv.py
```

Genera `db/historico_completo.csv` ordenado por ticker y fecha. Solo necesario la primera vez o tras actualizar el parquet.

### 2. Generar las señales

```bash
python 52w.py
```

Genera `db/max.csv`. Columnas: `TICKER, FECHA, OPEN, CLOSE, HIGH, LOW, VOLUME, MARKET_CAP`.

El market cap se calcula como `shares_outstanding × CLOSE` usando la entrada más reciente disponible en `stock_shares_outstanding.parquet` para esa fecha.

### 3. Arrancar la API

```bash
python backend.py
```

Disponible en `http://127.0.0.1:3333`.  
Documentación: `http://127.0.0.1:3333/redoc`

---

## API

Todas las rutas requieren autenticación Bearer.

```
Authorization: Bearer <API_KEY>
```

---

### `GET /ticker` — Historial de señales de un ticker

Devuelve todas las sesiones en las que el ticker marcó máximo de 52 semanas.

**Parámetros**

| Parámetro | Tipo | Restricciones |
|---|---|---|
| `ticker` | string | 1–10 caracteres |

**Ejemplo**

```
GET http://127.0.0.1:3333/ticker?ticker=NVDA
Authorization: Bearer <API_KEY>
```

**Respuesta** — array de arrays: `[TICKER, FECHA, OPEN, CLOSE, HIGH, LOW, VOLUME, MARKET_CAP]`

```json
[
  ["NVDA", "2024-02-22", 785.0, 788.17, 794.35, 780.0, 53000000, 1950000000000],
  ["NVDA", "2025-01-07", 150.0, 153.13, 153.98, 149.5, 47000000, 3740000000000]
]
```

---

### `GET /date` — Señales por fecha

Devuelve todos los tickers que marcaron máximo de 52 semanas en una fecha concreta.

**Parámetros**

| Parámetro | Tipo | Restricciones |
|---|---|---|
| `day` | int | 1–31 |
| `month` | int | 1–12 |
| `year` | int | ≥ 2000 |

**Ejemplo**

```
GET http://127.0.0.1:3333/date?day=3&month=3&year=2003
Authorization: Bearer <API_KEY>
```

**Respuesta** — mismo formato que `/ticker`: `[TICKER, FECHA, OPEN, CLOSE, HIGH, LOW, VOLUME, MARKET_CAP]`

```json
[
  ["EBAY", "2003-03-03", 8.26, 8.16, 8.36, 8.15, 46267373, 10267107024],
  ["GRMN", "2003-03-03", 17.08, 16.8, 17.09, 16.7, 657000, 3626103600]
]
```

---

### `GET /sp500` — S&P 500 por fecha

Devuelve los datos del índice S&P 500 para una fecha concreta.

**Parámetros** — idénticos a `/date`: `day`, `month`, `year`

**Ejemplo**

```
GET http://127.0.0.1:3333/sp500?day=3&month=3&year=2003
Authorization: Bearer <API_KEY>
```

**Respuesta** — `[[rowIndex, DATE, OPEN, ADJ_CLOSE, HIGH, LOW, VOLUME]]`

```json
[
  [0, "2003-03-03", 841.15, 834.81, 852.34, 832.74, 1208900000]
]
```

Devuelve array vacío si la fecha es fin de semana o festivo.

---

### `GET /info` — Información de una empresa

Devuelve los datos descriptivos de una empresa a partir de su ticker.

**Parámetros**

| Parámetro | Tipo | Restricciones |
|---|---|---|
| `ticker` | string | 1–10 caracteres |

**Ejemplo**

```
GET http://127.0.0.1:3333/info?ticker=AAPL
Authorization: Bearer <API_KEY>
```

**Respuesta** — objeto JSON con los campos del `stock_profile`

```json
{
  "symbol": "AAPL",
  "city": "Cupertino",
  "country": "United States",
  "industry": "Consumer Electronics",
  "sector": "Technology",
  "longBusinessSummary": "Apple Inc. designs, manufactures...",
  "fullTimeEmployees": 150000,
  "website": "https://www.apple.com"
}
```

**Errores**

| Código | Descripción |
|---|---|
| `401` | Token inválido o no autorizado |
| `404` | Ticker no encontrado en la base de datos |
| `422` | Error de validación en los parámetros |

---

### `GET /rend` — Rendimiento vs S&P 500

Compara el rendimiento acumulado de un ticker frente al S&P 500 a partir de una fecha de señal concreta. Toma como precio de entrada el OPEN de la sesión siguiente a la señal y calcula el rendimiento cada 7 sesiones durante el año siguiente (~35 puntos).

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `ticker` | string | Símbolo bursátil (1–10 chars) |
| `day` | int (1–31) | Día de la señal |
| `month` | int (1–12) | Mes de la señal |
| `year` | int (≥ 2000) | Año de la señal |

La fecha debe corresponder a una sesión en la que el ticker efectivamente marcó máximo de 52 semanas (aparece en `max.csv`). Si no es así, devuelve `400`.

Requiere que existan los parquets fragmentados en `db/fragmented/` para el ticker consultado.

**Ejemplo**

```
GET http://127.0.0.1:3333/rend?ticker=NVDA&day=22&month=2&year=2024
Authorization: Bearer <API_KEY>
```

**Respuesta** — array de ~35 puntos: `[days_elapsed, rend_ticker, rend_sp500, alpha]`

```json
[
  [7,   0,0231,  0.0084,  0.0147],
  [14,  0.0510,  0.0123,  0.0387],
  [21,  0.0875,  0.0201,  0.0674],
  ...
  [252, 1.423, 0.284, 1.139]
]
```

| Campo | Tipo | Descripción |
|---|---|---|
| `days_elapsed` | int | Sesiones de trading transcurridas desde la señal |
| `rend_ticker` | float | Rentabilidad acumulada del ticker (u.) |
| `rend_sp500` | float | Rentabilidad acumulada del S&P 500 (u.) |
| `alpha` | float | `rend_ticker − rend_sp500` (u.) |

**Errores**

| Código | Descripción |
|---|---|
| `400` | El ticker no marcó 52W High en esa fecha |
| `401` | Token inválido o no autorizado |

---

## Lógica de la señal

Una señal se emite en la sesión `X` cuando:

```
HIGH(X) > max( HIGH(X-1), HIGH(X-2), ..., HIGH(X-252) )
```

Se requieren **al menos 252 sesiones previas** antes de que un ticker pueda emitir su primera señal válida. El precio de referencia para seguimiento posterior (Día 0) es el **CLOSE** de la sesión de señal. El rendimiento en `/rend` se mide desde el **OPEN** de la sesión siguiente (D+1).

---

## Actualizar el dataset

Para descargar una versión nueva del parquet y regenerar las señales, ejecutar manualmente desde `backend.py`:

```python
update_parquet_files_and_transform_to_csv()
```

Descarga los parquets de HuggingFace, guarda copia de seguridad de los anteriores (`.bck1`) y regenera `db/historico_completo.csv`.

---

## Requisitos

- Python 3.10+
- `pandas`, `pyarrow`, `duckdb`, `fastapi`, `uvicorn`, `python-dotenv`, `numpy`

---

## Licencia

GPL-3.0 — ver [LICENSE.md](LICENSE.md)
