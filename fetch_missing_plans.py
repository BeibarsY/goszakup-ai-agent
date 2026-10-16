"""
Догружает пункты плана, на которые ссылаются позиции договоров, но которых нет в plan_points
(обычно это планы 2023 года для договоров 2024 года или планы другого организатора).
Без них у позиции договора нет единицы измерения и места поставки.

Запуск (после etl_load.py):
    python fetch_missing_plans.py
    python etl_load.py        # повторно, чтобы загрузить догруженное
"""
import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from fetch_raw import PLANS_QUERY, fetch_all

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
OUT = Path("data/raw/Plans/_missing.jsonl")
CHUNK = 100


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ci.plan_point_id FROM contract_items ci
        WHERE ci.plan_point_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM plan_points p WHERE p.id = ci.plan_point_id)
    """)
    ids = [r[0] for r in cur.fetchall()]
    conn.close()
    print(f"не хватает пунктов плана: {len(ids)}")

    got = 0
    with OUT.open("a", encoding="utf-8") as f:
        for i in range(0, len(ids), CHUNK):
            for item in fetch_all("Plans", PLANS_QUERY, {"id": ids[i:i + CHUNK]}):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                got += 1
            print(f"  {min(i + CHUNK, len(ids))}/{len(ids)}, получено {got}")
    print(f"готово, записано в {OUT}")


if __name__ == "__main__":
    main()
