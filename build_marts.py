"""
Пересобирает аналитический слой (analytics.sql). Запускать после каждого etl_load.py.

    python build_marts.py

Если рядом лежит cpi.csv (month,cpi; например 2024-01-01,100.0), индекс цен загружается в ref_cpi.
"""
import csv
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")


def main(conn=None, commit=True, verbose=True):
    own = conn is None
    conn = conn or psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS ref_cpi (month DATE PRIMARY KEY, cpi NUMERIC NOT NULL)")

    cpi = Path("cpi.csv")
    if cpi.exists():
        # строки-комментарии (# ...) в начале файла: источник и способ расчёта
        rows = [(r["month"], float(r["cpi"])) for r in csv.DictReader(l for l in cpi.open() if not l.startswith("#"))
                if r.get("cpi", "").strip()]
        cur.execute("TRUNCATE ref_cpi")
        cur.executemany("INSERT INTO ref_cpi (month, cpi) VALUES (%s, %s)", rows)
        if verbose:
            print(f"ИПЦ: загружено месяцев {len(rows)}")
    cur.execute("SELECT count(*) FROM ref_cpi")
    n_cpi = cur.fetchone()[0]
    if verbose:
        print(f"ИПЦ в базе: {n_cpi} мес." if n_cpi else "ИПЦ не загружен: инфляция не учитывается")

    cur.execute(Path("analytics.sql").read_text(encoding="utf-8"))
    if commit:
        conn.commit()

    stats = {}
    for t in ["mv_price_base", "mv_enstru_stats", "mv_enstru_city_stats", "mv_season", "mv_volumes"]:
        cur.execute(f"SELECT count(*) FROM {t}")
        stats[t] = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM mv_enstru_stats WHERE n >= 5")
    stats["enstru_n_ge_5"] = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM v_price_anomalies WHERE n >= 5 AND dev_from_wavg > 0.3")
    stats["anomalies_gt_30"] = cur.fetchone()[0]
    if verbose:
        for k, v in stats.items():
            print(f"{k:22} {v}")
    if own:
        conn.close()
    return stats


if __name__ == "__main__":
    main()
