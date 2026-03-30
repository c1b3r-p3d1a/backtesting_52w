# 52-Week High Signal System

Sistema de detección automática de señales 52-Week High sobre el mercado americano. Procesa el historial completo de precios OHLCV, detecta cuándo un valor supera su máximo de las 252 sesiones anteriores, y expone los resultados a través de una API REST local autenticada.

---

## Dataset

Los parquets provienen de [HuggingFace / defeatbeta](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data).

| Fichero | Contenido |
|---|---|
| `stock_prices.parquet` | Precios diarios OHLCV — 10.907 tickers, 34M+ filas, desde 1994 |
| `stock_profile.parquet` | Sector, descripción, empleados, web de cada empresa |

---

## Instalación

```bash
git clone https://github.com/c1b3r-p3d1a/backtesting_52w.git
cd backtesting_52w
pip install pandas pyarrow duckdb fastapi uvicorn python-dotenv
```

Crea un fichero `.env` en la raíz:

```
API_KEY=tu_clave_aqui
```

---

## Uso

### 1. Convertir el parquet a CSV

```bash
python price_parquet_to_csv.py
```

Genera `historico_completo.csv` ordenado por ticker y fecha. Solo necesario la primera vez o tras actualizar el parquet.

### 2. Generar las señales

```bash
python 52w.py
```

Genera `max.csv`. Columnas: `TICKER, FECHA, OPEN, CLOSE, HIGH, LOW, VOLUME, MARKET_CAP`.

### 3. Arrancar la API

```bash
python backend.py
```

Disponible en `http://127.0.0.1:3333`.  
Documentación en: `http://127.0.0.1:3333/redoc`

---

## API

Todas las rutas requieren autenticación por Bearer token.

```
Authorization: Bearer <API_KEY>
```

---

### `GET /date` — Señales por fecha

Devuelve todos los tickers que marcaron máximo de 52 semanas en una fecha concreta.

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `day` | int (1–31) | Día del mes |
| `month` | int (1–12) | Mes |
| `year` | int (≥2000) | Año |

**Ejemplo**

```
GET http://127.0.0.1:3333/date?day=3&month=3&year=2003
Authorization: Bearer <API_KEY>
```

**Respuesta**

```json
[
  ["BSRR","2003-03-03",13.86,13.88,14.0,13.85,7900,128328928.0],
  ["BXMT","2003-03-03",172.8,174.3,174.3,171.0,450,94575180.0],
  ["CHN","2003-03-03",16.4,16.42,16.55,16.35,148400,null],
  ...
]
```

---

### `GET /ticker` — Historial de señales de un ticker

Devuelve todas las fechas en las que un ticker ha marcado máximo de 52 semanas.

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `ticker` | string (1–10 chars) | Símbolo bursátil |

**Ejemplo**

```
GET http://127.0.0.1:3333/ticker?ticker=NVDA
Authorization: Bearer <API_KEY>
```

**Respuesta**

```json
[
  ["NVDA","2001-05-01",0.35,0.38,0.38,0.34,1867368000,6395485462.0],
  ["NVDA","2001-05-02",0.39,0.37,0.39,0.37,1547784000,6227183213.0],
  ["NVDA","2001-05-08",0.4,0.36,0.4,0.36,1620144000,6058880964.0],
  ...
]
```

---

### `GET /info` — Información de una empresa

Devuelve los datos descriptivos de una empresa a partir de su ticker.

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `ticker` | string (1–10 chars) | Símbolo bursátil |

**Ejemplo**

```
GET http://127.0.0.1:3333/info?ticker=AAPL
Authorization: Bearer <API_KEY>
```

**Respuesta**

```json
{
  "symbol":"AAPL",
  "address":"One Apple Park Way",
  "city":"Cupertino",
  "country":"United States",
  "phone":"(408) 996-1010",
  "zip":"95014",
  "industry":"Consumer Electronics",
  "sector":"Technology",
  "long_business_summary":"Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising AirPods, Apple Vision Pro, Apple TV, Apple Watch, Beats products, and HomePod, as well as Apple branded and third-party accessories. It also provides AppleCare support and cloud services; and operates various platforms, including the App Store that allow customers to discover and download applications and digital content, such as books, music, video, games, and podcasts, as well as advertising services include third-party licensing arrangements and its own advertising platforms. In addition, the company offers various subscription-based services, such as Apple Arcade, a game subscription service; Apple Fitness+, a personalized fitness service; Apple Music, which offers users a curated listening experience with on-demand radio stations; Apple News+, a subscription news and magazine service; Apple TV, which offers exclusive original content and live sports; Apple Card, a co-branded credit card; and Apple Pay, a cashless payment service, as well as licenses its intellectual property. The company serves consumers, and small and mid-sized businesses; and the education, enterprise, and government markets. It distributes third-party applications for its products through the App Store. The company also sells its products through its retail and online stores, and direct sales force; and third-party cellular network carriers and resellers. The company was formerly known as Apple Computer, Inc. and changed its name to Apple Inc. in January 2007. Apple Inc. was founded in 1976 and is headquartered in Cupertino, California.",
  "full_time_employees":150000.0,
  "web_site":"https://www.apple.com",
  "report_date":"2026-01-31"}
```

**Errores**

| Código | Descripción |
|---|---|
| `401` | Token inválido o no autorizado |
| `404` | Ticker no encontrado en la base de datos |
| `422` | Error de validación en los parámetros |

---

## Lógica de la señal

Una señal se emite en la sesión `X` cuando:

```
HIGH(X) > max( HIGH(X-1), HIGH(X-2), ..., HIGH(X-252) )
```

Se requieren **al menos 252 sesiones previas** antes de que un ticker pueda emitir su primera señal. El precio de referencia para seguimiento posterior es el **OPEN** de la siguiente sesión de la señal.

---

## Actualizar el dataset (no incluye todos los datasets la func)

Para descargar una versión nueva del parquet y regenerar las señales, ejecutar manualmente desde `backend.py`:

```python
update_parquet_files_and_transform_to_csv()
```

Descarga los parquets de HuggingFace, guarda copia de seguridad de los anteriores (`.bck1`) y regenera `historico_completo.csv`.

---

## Licencia

GPL-3.0 — ver [LICENSE.md](LICENSE.md)
