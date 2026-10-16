"""
ETL: сырые JSONL из data/raw -> очищенные таблицы PostgreSQL (schema.sql).

Запуск:
    pip install psycopg2-binary python-dotenv
    # .env: DATABASE_URL=postgresql://postgres:<пароль>@127.0.0.1:5433/goszakup  (см. .env.example)
    python etl_load.py

Скрипт идемпотентный: повторный запуск обновляет записи (UPSERT), а не дублирует их.
База приводится в соответствие сырым данным: лоты, договоры и позиции, которых больше нет в data/raw
(или они помечены удалёнными), удаляются. Пункты плана помечаются removed_at только по флагу isDeleted:
индекс планов OWS теряет записи без удаления (14.10.2026 пропало 8 412 пунктов плана 2026 года, из них
6 284 уже закуплены), поэтому update.py хранит последнюю известную версию пункта, а ETL отмечает
такие пункты в журнале качества (missing_from_api), но не исключает из аналитики.
Если удалить пришлось бы слишком много (MAX_DELETE_SHARE), загрузка прерывается: скорее всего,
выгрузка неполная, а не данные действительно удалены.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")   # из .env; пароль в коде не храним
RAW = Path("data/raw")
YEARS = {2024, 2025, 2026}
ANNOUNCE_URL = "https://goszakup.gov.kz/ru/announce/index/{}"
MAX_DELETE_SHARE = float(os.getenv("ETL_MAX_DELETE_SHARE", "0.10"))


class MassDeleteError(RuntimeError):
    pass


def sync_deletions(cur, table, keep_ids, stats, soft_column=None):
    """Удаляет (или помечает soft_column) строки table, id которых нет в keep_ids."""
    cur.execute(f"CREATE TEMP TABLE IF NOT EXISTS keep_{table} (id BIGINT PRIMARY KEY) ON COMMIT DROP")
    cur.execute(f"TRUNCATE keep_{table}")
    execute_values(cur, f"INSERT INTO keep_{table} (id) VALUES %s", [(i,) for i in keep_ids], page_size=5000)
    active = f" AND t.{soft_column} IS NULL" if soft_column else ""
    cur.execute(f"SELECT count(*) FROM {table} t WHERE NOT EXISTS (SELECT 1 FROM keep_{table} k WHERE k.id = t.id){active}")
    n_gone = cur.fetchone()[0]
    cur.execute(f"SELECT count(*) FROM {table} t WHERE TRUE{active}")
    n_all = cur.fetchone()[0]
    if n_all and n_gone / n_all > MAX_DELETE_SHARE:
        raise MassDeleteError(f"{table}: из источника пропало бы {n_gone} из {n_all} строк "
                              f"(> {MAX_DELETE_SHARE:.0%}); вероятно, выгрузка неполная. "
                              f"Проверьте data/raw или задайте ETL_MAX_DELETE_SHARE.")
    if soft_column:
        cur.execute(f"UPDATE {table} t SET {soft_column} = now() WHERE {soft_column} IS NULL "
                    f"AND NOT EXISTS (SELECT 1 FROM keep_{table} k WHERE k.id = t.id)")
        cur.execute(f"UPDATE {table} t SET {soft_column} = NULL WHERE {soft_column} IS NOT NULL "
                    f"AND EXISTS (SELECT 1 FROM keep_{table} k WHERE k.id = t.id)")
    else:
        cur.execute(f"DELETE FROM {table} t WHERE NOT EXISTS (SELECT 1 FROM keep_{table} k WHERE k.id = t.id)")
    stats[table] = n_gone

# ---------- нормализация единиц измерения ----------
# код из API -> (код нормализованной единицы, название, множитель для количества)
UNIT_OVERRIDES = {
    "163": ("166", "Килограмм", 0.001),   # грамм -> кг
    "168": ("166", "Килограмм", 1000),    # тонна -> кг
    "111": ("112", "Литр", 0.001),        # мл (куб. см) -> литр
    "113": ("112", "Литр", 1000),         # куб. м -> литр
    "018": ("006", "Метр", 1),            # метр погонный -> метр
    "5156": ("006", "Метр", 1),           # пог.м -> метр
}
# запасной вариант, если код неизвестен, а название понятное
UNIT_NAME_RULES = [
    (r"^(грамм|г\.?)$", ("166", "Килограмм", 0.001)),
    (r"^тонн", ("166", "Килограмм", 1000)),
    (r"^(миллилитр|мл\.?)$", ("112", "Литр", 0.001)),
    (r"^(пог\.?\s*м|метр погонный)", ("006", "Метр", 1)),
]


def norm_unit(code, name):
    if code in UNIT_OVERRIDES:
        return UNIT_OVERRIDES[code]
    n = (name or "").strip().lower()
    for pattern, res in UNIT_NAME_RULES:
        if re.search(pattern, n):
            return res
    return (code, name, 1)


# ---------- очистка названий ----------
# служебные вставки в названиях лотов: «Лот №5», «лот 3», «4-Лота.», «согласно технической спецификации».
# Только отдельным словом: раньше «лот\S+» вырезал и внутри слов («кислота» → «кис», «полотенце» → «по»)
JUNK_PATTERNS = [
    r"(?<![а-яa-z0-9])\d+\s*-\s*лот[а-я]*\.?",                     # «4-Лота.»
    r"(?<![а-яa-z])лот[а-я]*\s*(?:№\s*|n\s*|#\s*)?\d+[\w-]*\.?",  # «Лот №5», «лот 3»
    r"(?<![а-яa-z])лот[а-я]*\s*№",                                  # «Лот №» без номера
    r"согласно\s+(?:технической\s+)?(?:спецификации|тех\.?\s*спецификации)",
]


def clean_name(*parts):
    """Нормализованное название для поиска: нижний регистр, ё→е, без служебных вставок, кавычек и лишних
    пробелов. Описание добавляется, только если не повторяет название (раньше «X X» при одинаковых полях)."""
    texts = []
    for p in parts:
        t = re.sub(r"\s+", " ", (p or "").lower().replace("ё", "е")).strip()
        if t and not any(t in prev or prev in t for prev in texts):
            texts.append(t)
        elif t and any(prev in t for prev in texts):   # описание длиннее и включает название
            texts = [x for x in texts if x not in t] + [t]
    s = " ".join(texts)
    for pat in JUNK_PATTERNS:
        s = re.sub(pat, " ", s)
    s = re.sub(r"[\"«»“”„'`]", " ", s)                  # кавычки
    s = re.sub(r"\s+([,;:.)])", r"\1", s)               # пробел перед знаком
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[\s,;:.\-–—]+|[\s,;:.\-–—]+$", "", s)   # пунктуация по краям
    return s or None


# ---------- КАТО -> город ----------
# Города республиканского значения: Астана 71, Алматы 75, Шымкент 79 -> город = 2 первые цифры.
# Остальные: 4 первые цифры (область + город областного значения/район). Считается в SQL (пост-обработка).


def NULLIF_EMPTY(v):  # noqa: N802 — по аналогии с SQL NULLIF
    return v if v else None


def ts(v):
    if not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("T", " ")[:19])
    except ValueError:
        return None


def read_jsonl(entity):
    for f in sorted((RAW / entity).glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def upsert(cur, table, cols, rows, key="id"):
    """INSERT ... ON CONFLICT DO UPDATE только для действительно изменившихся строк: ETL перечитывает
    весь data/raw при каждом обновлении, и без условия каждая строка переписывалась бы (мёртвые версии
    строк копятся, запросы внутри транзакции обновления деградируют)."""
    if not rows:
        return
    keys = [key] if isinstance(key, str) else key
    upd_cols = [c for c in cols if c not in keys]
    if upd_cols:
        changed = (f"({', '.join('t.' + c for c in upd_cols)}) IS DISTINCT FROM "
                   f"({', '.join('EXCLUDED.' + c for c in upd_cols)})")
        action = f"DO UPDATE SET {', '.join(f'{c}=EXCLUDED.{c}' for c in upd_cols)} WHERE {changed}"
    else:
        action = "DO NOTHING"
    sql = (f"INSERT INTO {table} AS t ({', '.join(cols)}) VALUES %s "
           f"ON CONFLICT ({', '.join(keys)}) {action}")
    execute_values(cur, sql, rows, page_size=1000)


class DQ:
    """Собирает проблемы качества данных в таблицу dq_issues."""
    def __init__(self):
        self.rows = {}

    def add(self, entity, eid, issue, detail=""):
        self.rows[(entity, eid, issue)] = (entity, eid, issue, str(detail)[:500])


def main(conn=None, commit=True, verbose=True):
    """conn и commit=False позволяют выполнить ETL и пересборку витрин одной транзакцией (update.py)."""
    own = conn is None
    conn = conn or psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(Path("schema.sql").read_text(encoding="utf-8"))
    cur.execute("TRUNCATE dq_issues")   # журнал качества пересчитывается заново при каждом запуске
    dq = DQ()

    # ---------- subjects ----------
    subjects = {}
    for s in read_jsonl("Subjects"):
        subjects[s["bin"]] = (s["bin"], s["pid"], s["nameRu"], s["nameKz"], s["fullNameRu"],
                              s.get("katoList") or [], bool(s.get("customer")),
                              bool(s.get("supplier")), bool(s.get("qvazi")), ts(s.get("lastUpdateDate")))

    # ---------- plans + справочники единиц и ЕНС ТРУ ----------
    units, enstru, plans = {}, {}, {}
    subject_types, plan_statuses, buy_statuses = {}, {}, {}   # справочники из вложенных полей API
    deleted_plans = set()
    for p in read_jsonl("Plans"):
        if p.get("isDeleted"):
            dq.add("plan_point", p["id"], "deleted_in_source")
            deleted_plans.add(p["id"])
            continue
        if p.get("_missing_from_api_since"):
            dq.add("plan_point", p["id"], "missing_from_api", p["_missing_from_api_since"])
        ru = p.get("RefUnits") or {}
        code = p.get("refUnitsCode")
        n_code, n_name, factor = norm_unit(code, ru.get("nameRu"))
        units[code] = (code, ru.get("nameRu"), ru.get("nameKz"), n_code, n_name, factor)
        if p.get("refEnstruCode"):
            re_ = p.get("RefEnstru") or {}
            enstru[p["refEnstruCode"]] = (p["refEnstruCode"], re_.get("nameRu"), re_.get("nameKz"))
        else:
            dq.add("plan_point", p["id"], "no_enstru")

        kato = (p.get("PlansKato") or [{}])[0] if p.get("PlansKato") else {}
        price = p.get("price")
        if not price or price <= 0:
            dq.add("plan_point", p["id"], "bad_price", price)
        count = p.get("count")
        plans[p["id"]] = (
            p["id"], p.get("rootrecordId"), p["subjectBiin"], p["plnPointYear"],
            p.get("nameRu"), p.get("nameKz"), p.get("descRu"), p.get("extraDescRu"),
            clean_name(p.get("nameRu"), p.get("extraDescRu")),
            p.get("refEnstruCode") or None, code, n_code,
            count, count * factor if count is not None else None,
            price, price / factor if price and factor else None,
            p.get("amount"), p.get("refMonthsId"), p.get("refTradeMethodsId"),
            p.get("refPlnPointStatusId"),
            NULLIF_EMPTY(kato.get("refKatoCode")), kato.get("fullDeliveryPlaceNameRu"),
            ts(p.get("dateCreate")), ts(p.get("timestamp")), p.get("refSubjectTypeId"),
        )
        for ref, store in ((p.get("RefSubjectType"), subject_types), (p.get("RefPlnPointStatus"), plan_statuses)):
            if ref and ref.get("id") is not None:
                store[ref["id"]] = (ref["id"], ref.get("nameRu"), ref.get("nameKz"))

    upsert(cur, "ref_units", ["code", "name_ru", "name_kz", "norm_code", "norm_name", "factor"],
           list(units.values()), key="code")
    upsert(cur, "ref_enstru", ["code", "name_ru", "name_kz"], list(enstru.values()), key="code")
    upsert(cur, "subjects", ["bin", "pid", "name_ru", "name_kz", "full_name_ru", "kato_list",
                             "is_customer", "is_supplier", "is_qvazi", "updated_at"],
           list(subjects.values()), key="bin")
    upsert(cur, "plan_points", [
        "id", "rootrecord_id", "customer_bin", "plan_year", "name_ru", "name_kz", "desc_ru",
        "extra_desc_ru", "name_clean", "enstru_code", "unit_code", "unit_norm", "count_raw",
        "count_norm", "price_raw", "price_norm", "amount", "month_id", "trade_method_id",
        "status_id", "kato_plan", "delivery_place", "date_create",
        "updated_at", "subject_type_id"], list(plans.values()))

    # ---------- lots + announcements ----------
    annos, lots, links = {}, {}, []
    skipped_old = 0
    for l in read_jsonl("Lots"):
        tb = l.get("TrdBuy") or {}
        pub = ts(tb.get("publishDate"))
        if not pub or pub.year not in YEARS:
            skipped_old += 1
            continue
        if l.get("isDeleted"):
            dq.add("lot", l["id"], "deleted_in_source")
            continue
        annos[tb["id"]] = (
            tb["id"], tb.get("numberAnno"), tb.get("nameRu"), tb.get("orgBin"), pub,
            ts(tb.get("endDate")), ts(tb.get("itogiDatePublic")), tb.get("totalSum"),
            tb.get("countLots"), tb.get("refBuyStatusId"), tb.get("refTradeMethodsId"),
            (tb.get("finYear") or [None])[0], ANNOUNCE_URL.format(tb["id"]),
            ts(tb.get("lastUpdateDate")), tb.get("refSubjectTypeId"),
        )
        if (bs := tb.get("RefBuyStatus")) and bs.get("id") is not None:
            buy_statuses[bs["id"]] = (bs["id"], bs.get("nameRu"), bs.get("nameKz"))
        if not (l.get("nameRu") or "").strip():
            dq.add("lot", l["id"], "empty_name")
        lots[l["id"]] = (
            l["id"], tb["id"], l.get("lotNumber"), l.get("customerBin"), l.get("nameRu"),
            l.get("descriptionRu"), clean_name(l.get("nameRu"), l.get("descriptionRu")),
            l.get("count"), l.get("amount"), l.get("refLotStatusId"),
            (l.get("RefLotsStatus") or {}).get("nameRu"), l.get("refTradeMethodsId"),
            l.get("plnPointKatoList") or [], bool(l.get("dumping")), ts(l.get("lastUpdateDate")),
        )
        for pid in l.get("pointList") or []:
            links.append((l["id"], pid))

    upsert(cur, "announcements", [
        "id", "number_anno", "name_ru", "org_bin", "publish_date", "end_date", "itogi_date",
        "total_sum", "count_lots", "status_id", "trade_method_id", "fin_year", "url",
        "updated_at", "subject_type_id"], list(annos.values()))
    for table, store in (("ref_subject_type", subject_types), ("ref_plan_status", plan_statuses),
                         ("ref_buy_status", buy_statuses)):
        upsert(cur, table, ["id", "name_ru", "name_kz"], list(store.values()))
    upsert(cur, "lots", [
        "id", "announcement_id", "lot_number", "customer_bin", "name_ru", "description_ru",
        "name_clean", "count", "amount", "status_id", "status_name", "trade_method_id",
        "kato_codes", "dumping", "updated_at"], list(lots.values()))
    # связи лот–пункт плана: добавляем новые, удаляем исчезнувшие (а не перезаписываем все)
    links = list(set(links))
    upsert(cur, "lot_plan_points", ["lot_id", "plan_point_id"], links, key=["lot_id", "plan_point_id"])
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS keep_links (lot_id BIGINT, plan_point_id BIGINT, "
                "PRIMARY KEY (lot_id, plan_point_id)) ON COMMIT DROP")
    cur.execute("TRUNCATE keep_links")
    execute_values(cur, "INSERT INTO keep_links VALUES %s", links, page_size=5000)
    cur.execute("""DELETE FROM lot_plan_points lp WHERE NOT EXISTS (
                       SELECT 1 FROM keep_links k WHERE k.lot_id = lp.lot_id AND k.plan_point_id = lp.plan_point_id)""")

    # ---------- contracts + items ----------
    contracts, items = {}, []
    for c in read_jsonl("Contract"):
        if c.get("deleted"):
            dq.add("contract", c["id"], "deleted_in_source")
            continue
        if c.get("refCurrencyCode") not in (None, "KZT"):
            dq.add("contract", c["id"], "foreign_currency", c.get("refCurrencyCode"))
        contracts[c["id"]] = (
            c["id"], c.get("rootId") or c["id"], c.get("parentId"), c.get("trdBuyId"),
            c.get("trdBuyNumberAnno"), c.get("contractNumber"), c.get("customerBin"), c.get("supplierBiin"),
            (c.get("Supplier") or {}).get("nameRu"), c.get("refContractStatusId"),
            c.get("refContractTypeId"), c.get("faktTradeMethodsId"), c.get("finYear"),
            ts(c.get("signDate")), c.get("contractSum"), c.get("contractSumWnds"),
            c.get("faktSum"), c.get("refCurrencyCode"), ts(c.get("lastUpdateDate")),
        )
        for u in c.get("ContractUnits") or []:
            if u.get("deleted"):
                continue
            price, qty, total = u.get("itemPrice"), u.get("quantity"), u.get("totalSum")
            if not price or price <= 0 or not qty or qty <= 0:
                dq.add("contract_item", u["id"], "bad_price_or_qty", f"{price} x {qty}")
            elif total and abs(price * qty - total) / total > 0.01:
                dq.add("contract_item", u["id"], "sum_mismatch", f"{price} x {qty} != {total}")
            items.append((
                u["id"], c["id"], u.get("lotId") or None, u.get("plnPointId") or None,
                (u.get("refEnstruCode") or [None])[0], qty, price, u.get("itemPriceWnds"),
                total, u.get("totalSumWnds"),
            ))

    upsert(cur, "contracts", [
        "id", "root_id", "parent_id", "announcement_id", "number_anno", "contract_number", "customer_bin",
        "supplier_bin", "supplier_name", "status_id", "type_id", "trade_method_id", "fin_year",
        "sign_date", "contract_sum", "contract_sum_wnds", "fakt_sum", "currency", "updated_at"],
        list(contracts.values()))
    upsert(cur, "contract_items", [
        "id", "contract_id", "lot_id", "plan_point_id", "enstru_code", "quantity", "unit_price",
        "unit_price_wnds", "total_sum", "total_sum_wnds"], items)

    # ---------- удалённое в источнике ----------
    # порядок важен: позиции и связи раньше родительских строк
    removed = {}
    item_ids = {it[0] for it in items}
    sync_deletions(cur, "contract_items", item_ids, removed)
    sync_deletions(cur, "contracts", set(contracts), removed)
    sync_deletions(cur, "lots", set(lots), removed)
    cur.execute("DELETE FROM announcements a WHERE NOT EXISTS (SELECT 1 FROM lots l WHERE l.announcement_id = a.id)")
    removed["announcements"] = cur.rowcount
    # пункты плана: removed_at только по флагу isDeleted (пропажа из API — не удаление, см. docstring)
    cur.execute("UPDATE plan_points SET removed_at = now() WHERE removed_at IS NULL AND id = ANY(%s)",
                (list(deleted_plans),))
    removed["plan_points"] = cur.rowcount
    cur.execute("UPDATE plan_points SET removed_at = NULL WHERE removed_at IS NOT NULL AND NOT (id = ANY(%s))",
                (list(deleted_plans),))
    removed["plan_points_restored"] = cur.rowcount
    cur.execute("ANALYZE contracts; ANALYZE contract_items")

    # ---------- пост-обработка в SQL ----------
    # 1) последняя редакция договора (доп.соглашения создают новые записи с тем же root_id)
    cur.execute("""
        UPDATE contracts c SET is_latest = (c.id = m.max_id)
        FROM (SELECT root_id, MAX(id) AS max_id FROM contracts GROUP BY root_id) m
        WHERE c.root_id = m.root_id AND c.is_latest IS DISTINCT FROM (c.id = m.max_id)
    """)
    # 2) место поставки: КАТО из плана, иначе из лота, иначе из адреса заказчика.
    #    В лотах КАТО бывает пустой строкой: такой код не город, идём дальше к адресу заказчика.
    #    Пишем только строки, где итог изменился.
    cur.execute("ANALYZE plan_points; ANALYZE lot_plan_points; ANALYZE lots")
    cur.execute("""
        WITH lot_kato AS (
            SELECT lp.plan_point_id, MIN(NULLIF(l.kato_codes[1], '')) AS kato
            FROM lot_plan_points lp JOIN lots l ON l.id = lp.lot_id
            WHERE cardinality(l.kato_codes) > 0 GROUP BY lp.plan_point_id),
        resolved AS (
            SELECT p.id,
                   CASE WHEN p.kato_plan IS NOT NULL THEN p.kato_plan
                        WHEN lk.kato IS NOT NULL THEN lk.kato
                        ELSE NULLIF(s.kato_list[1], '') END AS kato,
                   CASE WHEN p.kato_plan IS NOT NULL THEN 'plan'
                        WHEN lk.kato IS NOT NULL THEN 'lot'
                        WHEN NULLIF(s.kato_list[1], '') IS NOT NULL THEN 'customer' END AS source
            FROM plan_points p
            LEFT JOIN lot_kato lk ON lk.plan_point_id = p.id
            LEFT JOIN subjects s ON s.bin = p.customer_bin)
        UPDATE plan_points p
        SET kato_code = r.kato, kato_source = r.source,
            city_code = CASE WHEN left(r.kato, 2) IN ('71','75','79') THEN left(r.kato, 2) ELSE left(r.kato, 4) END
        FROM resolved r
        WHERE p.id = r.id AND (p.kato_code, p.kato_source) IS DISTINCT FROM (r.kato, r.source)
    """)
    cur.execute("""
        INSERT INTO dq_issues (entity, entity_id, issue, detail)
        SELECT 'plan_point', id, 'kato_fallback', kato_source FROM plan_points
        WHERE kato_source IN ('lot', 'customer')
        ON CONFLICT DO NOTHING
    """)

    # позиции действующих договоров, для которых не нашёлся пункт плана ни напрямую, ни через лот
    cur.execute("""
        INSERT INTO dq_issues (entity, entity_id, issue, detail)
        SELECT 'contract_item', item_id, 'no_plan_point_unit_unknown', contract_id::TEXT
        FROM v_price_facts WHERE unit_norm IS NULL
        ON CONFLICT DO NOTHING
    """)
    cur.execute("""
        INSERT INTO dq_issues (entity, entity_id, issue, detail)
        SELECT 'contract_item', item_id, 'plan_point_via_lot', contract_id::TEXT
        FROM v_price_facts WHERE plan_source = 'via_lot'
        ON CONFLICT DO NOTHING
    """)
    upsert(cur, "dq_issues", ["entity", "entity_id", "issue", "detail"], list(dq.rows.values()),
           key=["entity", "entity_id", "issue"])
    if commit:
        conn.commit()

    # ---------- отчёт ----------
    stats = {"removed": removed, "skipped_old_lots": skipped_old, "rows": {}}
    for t in ["subjects", "plan_points", "announcements", "lots", "lot_plan_points",
              "contracts", "contract_items", "ref_units", "ref_enstru", "v_price_facts"]:
        cur.execute(f"SELECT count(*) FROM {t}")
        stats["rows"][t] = cur.fetchone()[0]
    cur.execute("SELECT entity, issue, count(*) FROM dq_issues GROUP BY 1, 2 ORDER BY 3 DESC")
    dq_counts = cur.fetchall()
    if verbose:
        print(f"лотов старше 2024 пропущено: {skipped_old}")
        for t, n in stats["rows"].items():
            print(f"{t:18} {n}")
        print("удалено / помечено удалёнными: " + ", ".join(f"{k} {v}" for k, v in removed.items()))
        print("\nпроблемы качества данных:")
        for e_, i, n in dq_counts:
            print(f"  {e_:15} {i:25} {n}")
    if own:
        conn.close()
    return stats


if __name__ == "__main__":
    main()
