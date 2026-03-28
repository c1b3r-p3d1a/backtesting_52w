# parquet_to_csv.py

import csv
import duckdb


class ParquetToCSV:
    def __init__(
        self,
        local_parquet: str,
        output_csv: str,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
    ):
        self.local_parquet = local_parquet
        self.output_csv = output_csv
        self.start_date = start_date
        self.end_date = end_date

    def _build_where(self) -> str:
        conditions = []
        if self.start_date:
            conditions.append(f"report_date >= '{self.start_date}'")
        if self.end_date:
            conditions.append(f"report_date <= '{self.end_date}'")
        return ("WHERE " + " AND ".join(conditions)) if conditions else ""

    def convert(self) -> None:
        con = duckdb.connect()
        where = self._build_where()

        rows = con.execute(f"""
            SELECT
                symbol      AS TICKER,
                report_date AS DATE,
                open        AS OPEN,
                high        AS HIGH,
                low         AS LOW,
                close       AS CLOSE,
                CAST(volume AS BIGINT) AS VOLUME
            FROM read_parquet('{self.local_parquet}')
            {where}
            ORDER BY symbol, report_date
        """).fetchall()

        con.close()

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["TICKER", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
            writer.writerows(rows)

        print(f"Done. {len(rows):,} rows written to {self.output_csv}")