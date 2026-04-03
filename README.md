# 52-Week High Signal System

Sistema de detección automática de señales **52-Week High** sobre el mercado americano. Procesa el historial completo de precios OHLCV, detecta cuándo un valor supera el máximo de sus 252 sesiones anteriores, calcula el market cap en el día de la señal y expone los resultados a través de una **API REST local** autenticada.

**Novedades principales (actualización 2026)**:
- Nuevos endpoints de análisis agregado: `/max_year`, `/rend_year`, `/analyze` y `/optimize`.
- Precomputación masiva de rendimientos y alphas granulares (`db/rend_year/`).
- Optimización automática de umbrales de alpha + periodo de tenencia mediante grid search.

---

## Estructura del repositorio
```
backtesting_52w/
│
├── 52w.py                          # Motor de señales → genera db/max.csv
├── backend.py                      # API REST (FastAPI, puerto 3333)
├── price_parquet_to_csv.py         # Conversor parquet → db/historico_completo.csv
│
├── tools/
│   ├── price_parquet_to_csv_claude.py
│   ├── rend_year_granulated_grok.py   # ← Precomputación de rendimientos y alpha granular
│   ├── rend_year_claude.py
│   └── fragmentator_chatgpt.py
│
├── db/                             # Datos (no commiteados salvo parquets base)
│   ├── stock_prices.parquet
│   ├── stock_profile.parquet
│   ├── stock_shares_outstanding.parquet
│   ├── sp500.csv
│   ├── historico_completo.csv      # ~1,4 GB (no commiteado)
│   ├── max.csv                     # Señales 52W + market cap
│   ├── fragmented/                 # Parquets por letra (a.parquet … z.parquet + _.parquet)
│   └── rend_year/                  # ← NUEVO: datos precomputados
│       ├── rend_YYYY.csv
│       ├── rend_YYYY_meta.json
│       └── granular_alpha_YYYY.parquet   # (2001-2025)
│
├── .gitignore
├── .gitattributes
├── requirements.txt
└── LICENSE.md
```

> Los ficheros grandes (`historico_completo.csv`, `sp500.csv`, `db/fragmented/`, `db/rend_year/`) **no están en el repositorio**. Se generan localmente.

---

## Flujo de datos
```
HuggingFace (parquets)
        │
        ▼
price_parquet_to_csv.py → db/historico_completo.csv
        │
        ▼
      52w.py → db/max.csv
        │
        ▼
rend_year_granulated_grok.py → db/rend_year/ (precomputación)
        │
        ▼
    backend.py (API completa)
```

---

## Dataset
| Fichero                        | Fuente                          | Contenido |
|--------------------------------|---------------------------------|-----------|
| `db/stock_prices.parquet`      | HuggingFace / defeatbeta        | Precios OHLCV diarios (10.907 tickers) |
| `db/stock_profile.parquet`     | HuggingFace / defeatbeta        | Datos descriptivos de empresas |
| `db/stock_shares_outstanding.parquet` | ycharts                  | Acciones en circulación (market cap) |
| `db/sp500.csv`                 | Yahoo Finance                   | Histórico S&P 500 |
| `db/fragmented/`               | Generado                        | Parquets por inicial de ticker |
| `db/rend_year/`                | Generado por `tools/rend_year_granulated_grok.py` | Rendimientos diarios + alpha granular (2001-2025) |

---

## Instalación
```bash
git clone https://github.com/c1b3r-p3d1a/backtesting_52w.git
cd backtesting_52w

# Recomendado (incluye todas las dependencias)
pip install -r requirements.txt
```

Crea `.env` en la raíz:
```env
API_KEY=tu_clave_aqui
```

Crea la carpeta `db/` y coloca los parquets de HuggingFace + `sp500.csv`.

---

## Uso

### 1. Convertir parquet a CSV
```bash
python price_parquet_to_csv.py
```

### 2. Generar señales 52-Week High
```bash
python 52w.py
```

### 3. Precomputar datos de rendimiento (obligatorio para `/rend_year`, `/analyze` y `/optimize`)
```bash
python tools/rend_year_granulated_grok.py
```
Esto genera toda la carpeta `db/rend_year/` (puede tardar varios minutos la primera vez).

### 4. Arrancar la API
```bash
python backend.py
```
Disponible en `http://127.0.0.1:3333`.

---

## API
Todas las rutas requieren **autenticación Bearer**:
```
Authorization: Bearer <API_KEY>
```

### `GET /ticker` — Historial de señales de un ticker
### `GET /date` — Señales por fecha
### `GET /sp500` — Datos del S&P 500 por fecha
### `GET /info` — Información de empresa
### `GET /rend` — Rendimiento individual vs S&P 500 (desde señal)

*(documentación sin cambios respecto a la versión anterior)*

---

### `GET /max_year` — Conteo diario de señales en un año
Devuelve el número de empresas que marcaron máximo de 52 semanas **cada día** del año.

**Parámetros**
- `year` (int ≥ 2000)

**Ejemplo**
```http
GET /max_year?year=2023
```
**Respuesta**
```json
{
  "2023-01-03": 82,
  "2023-01-04": 70,
  ...
  "2023-12-29": 191
}
```

---

### `GET /rend_year` — Rendimiento agregado por año
Toma **todas** las señales de un año completo y calcula la curva media de rendimiento (vs S&P 500) durante los 252 días posteriores, desglosada por capitalización.

**Parámetros**
- `year` (2000-2025)

**Respuesta**
```json
{
  "año": 2023,
  "señales_procesadas": 34133,
  "all": [[1, -0.00094, 0.00055, -0.00149], ...],
  "cap_1b": [...],
  "cap_5b": [...],
  "cap_10b": [...],
  "cap_20b": [...],
  "cap_50b": [...],
  "cap_100b": [...]
}
```

---

### `GET /analyze` — Análisis de alpha
Porcentaje de señales que superan un determinado alpha en un periodo concreto + estadísticas.

**Parámetros**
- `year`
- `alpha` (porcentaje, ej. 1.33)
- `period` (días de mercado, 1-253)

**Respuesta** (array)
```json
[
  muestras_que_superan,
  muestras_que_no_superan,
  total_muestras,
  pct_superan_Xpct,
  dias_promedio_hasta_Xpct,
  alpha_neg_promedio,
  n_muestras_alpha_negativo
]
```

---

### `GET /optimize` — Optimización automática de estrategia
Grid search que encuentra la mejor combinación **umbral alpha + periodo de tenencia** que maximiza la TIR ponderada de cartera (LONG señales positivas + SHORT señales negativas).

**Parámetros** (todos opcionales con defaults inteligentes)
- `year`
- `min_samples` (mínimo señales por grupo)
- `alpha_min` / `alpha_max` / `alpha_step`
- `period_min` / `period_max` / `period_step`
- `top_n` (cuántas mejores combinaciones devolver)
- `annualize` (True por defecto)

**Respuesta** (ejemplo)
```json
{
  "año": 2023,
  "mejor_combinacion": {
    "alpha_pct": 4.5,
    "period_dias": 85,
    "score": 0.312,
    "tir_cartera_pct": 18.7,
    ...
  },
  "top": [ ... ],
  "stats": { ... }
}
```

---

## Lógica de la señal
Sin cambios:
```python
HIGH(X) > max(HIGH(X-1) … HIGH(X-252))
```
Requiere al menos 252 sesiones previas. El `/rend` y `/rend_year` usan **OPEN del día siguiente** como precio de entrada.

---

## Actualizar el dataset
```python
# Dentro de backend.py (o ejecuta manualmente)
update_parquet_files_and_transform_to_csv()
```
Después de actualizar los parquets, **vuelve a ejecutar**:
```bash
python 52w.py
python tools/rend_year_granulated_grok.py
```

---

## Requisitos
- Python 3.10+
- Ver `requirements.txt`

---

## Licencia
GPL-3.0 — ver [LICENSE.md](LICENSE.md)