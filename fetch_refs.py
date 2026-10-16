"""
Справочники (названия RU/KZ) из OWS v3: способы закупки и статусы договоров, статусы объявлений и пунктов плана.

Отдельных запросов-справочников в API нет, названия приходят во вложенных полях (Contract.FaktTradeMethods,
Contract.RefContractStatus, TrdBuy.RefBuyStatus, PlnPoint.RefPlnPointStatus). Для каждого id, который
встречается в базе, запрашиваем одну запись с этим id (фильтр по БИН ускоряет ответ) и берём из неё название.
Статусы объявлений и планов в основном приходят при обычной выгрузке (ETL сохраняет их сам); здесь
дозапрашиваются только те id, для которых названия ещё нет.

Запуск после etl_load.py:  python fetch_refs.py   (update.py вызывает его сам с only_missing=True)
"""
import psycopg2

from fetch_raw import gql
from tools import DB_URL

DDL = """
CREATE TABLE IF NOT EXISTS ref_trade_methods (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);
CREATE TABLE IF NOT EXISTS ref_contract_status (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);
CREATE TABLE IF NOT EXISTS ref_buy_status (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);
CREATE TABLE IF NOT EXISTS ref_plan_status (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);
"""

# (справочник, сущность API, тип фильтра, поле id в фильтре, поле БИН в фильтре, вложенное поле,
#  SQL: id → любой БИН, у которого он встречается)
JOBS = [
    ("ref_trade_methods", "Contract", "ContractFiltersInput", "faktTradeMethodsId", "customerBin", "FaktTradeMethods",
     "SELECT trade_method_id, min(customer_bin) FROM contracts WHERE trade_method_id IS NOT NULL {missing} GROUP BY 1",
     "trade_method_id"),
    ("ref_contract_status", "Contract", "ContractFiltersInput", "refContractStatusId", "customerBin", "RefContractStatus",
     "SELECT status_id, min(customer_bin) FROM contracts WHERE status_id IS NOT NULL {missing} GROUP BY 1",
     "status_id"),
    ("ref_buy_status", "TrdBuy", "TrdBuyFiltersInput", "refBuyStatusId", "orgBin", "RefBuyStatus",
     "SELECT status_id, min(org_bin) FROM announcements WHERE status_id IS NOT NULL {missing} GROUP BY 1",
     "status_id"),
    ("ref_plan_status", "Plans", "PlansFiltersInput", "refPlnPointStatusId", "subjectBiin", "RefPlnPointStatus",
     "SELECT status_id, min(customer_bin) FROM plan_points WHERE status_id IS NOT NULL {missing} GROUP BY 1",
     "status_id"),
]


def fetch(entity, filter_type, id_key, bin_key, field, ids, bins_by_id):
    query = f"""
    query($filter: {filter_type}) {{
      {entity}(filter: $filter, limit: 1) {{ {field} {{ id nameRu nameKz }} }}
    }}"""
    out = {}
    for i in ids:
        filt = {id_key: [i] if entity in ("TrdBuy", "Plans") else i, bin_key: bins_by_id[i]}
        res = gql(query, {"filter": filt})
        rows = res["data"][entity] or []
        ref = rows[0].get(field) if rows else None
        if ref:
            out[ref["id"]] = (ref["nameRu"], ref["nameKz"])
        else:
            print(f"  {field} {i}: не найден")
    return out


def main(only_missing=False):
    """only_missing: запрашивать только id, которых ещё нет в справочнике (регулярное обновление)."""
    stats = {}
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        for table, entity, filter_type, id_key, bin_key, field, sql, col in JOBS:
            missing = f"AND {col} NOT IN (SELECT id FROM {table})" if only_missing else ""
            cur.execute(sql.format(missing=missing))
            bins_by_id = dict(cur.fetchall())
            names = fetch(entity, filter_type, id_key, bin_key, field, sorted(bins_by_id), bins_by_id) \
                if bins_by_id else {}
            for i, (ru, kz) in names.items():
                cur.execute(f"INSERT INTO {table} VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE "
                            f"SET name_ru = EXCLUDED.name_ru, name_kz = EXCLUDED.name_kz", (i, ru, kz))
            stats[table] = {"requested": len(bins_by_id), "loaded": len(names)}
            if not only_missing:
                print(f"{table}: {len(names)} из {len(bins_by_id)}")
    return stats


if __name__ == "__main__":
    main()
