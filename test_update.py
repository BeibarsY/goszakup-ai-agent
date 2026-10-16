"""
Тесты инкрементального обновления. ETL-тесты идут в транзакции с откатом: база не меняется.
Нельзя запускать одновременно с update.py (ETL берёт эксклюзивные блокировки витрин).

    python -m pytest test_update.py -q      или      python test_update.py
"""
import json
import tempfile
from pathlib import Path

import psycopg2

import etl_load
import update
from tools import DB_URL


def test_merge_delta_replaces_by_id_and_adds_new():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        update.write_jsonl(p, [{"id": 1, "v": "old"}, {"id": 2, "v": "keep"}])
        new, upd = update.merge_delta(p, [{"id": 1, "v": "new"}, {"id": 3, "v": "added"}])
        rows = {r["id"]: r["v"] for r in update.read_jsonl(p)}
        assert (new, upd) == (1, 1)
        assert rows == {1: "new", 2: "keep", 3: "added"}
        assert not p.with_suffix(".jsonl.tmp").exists()


def _run_etl_without(drop, flag_deleted=None):
    """ETL в транзакции, где из сырых данных «исчезли» записи drop = {entity: set(id)}, а записи
    flag_deleted = {entity: set(id)} помечены удалёнными в источнике. Возвращает (conn, stats)."""
    orig = etl_load.read_jsonl
    flag_deleted = flag_deleted or {}

    def patched(entity):
        for r in orig(entity):
            if r.get("id") in drop.get(entity, ()):
                continue
            if r.get("id") in flag_deleted.get(entity, ()):
                r = {**r, "isDeleted": 1, "deleted": True}
            yield r

    etl_load.read_jsonl = patched
    conn = psycopg2.connect(DB_URL)
    try:
        return conn, etl_load.main(conn, commit=False, verbose=False)
    except Exception:
        conn.rollback()
        conn.close()
        raise
    finally:
        etl_load.read_jsonl = orig


def _one(cur, sql, p=None):
    cur.execute(sql, p)
    return cur.fetchone()


def test_etl_removes_what_disappeared_from_source():
    with psycopg2.connect(DB_URL) as c0, c0.cursor() as cur:
        contract_id, n_items = _one(cur, """SELECT c.id, count(ci.id) FROM contracts c JOIN contract_items ci
                                             ON ci.contract_id = c.id GROUP BY 1 ORDER BY 1 LIMIT 1""")
        lot_id = _one(cur, "SELECT lot_id FROM lot_plan_points ORDER BY 1 LIMIT 1")[0]
        cur.execute("SELECT id FROM plan_points WHERE removed_at IS NULL ORDER BY id LIMIT 2")
        plan_gone, plan_deleted = [r[0] for r in cur.fetchall()]
    conn, stats = _run_etl_without({"Contract": {contract_id}, "Lots": {lot_id}, "Plans": {plan_gone}},
                                   flag_deleted={"Plans": {plan_deleted}})
    try:
        cur = conn.cursor()
        assert _one(cur, "SELECT count(*) FROM contracts WHERE id = %s", (contract_id,))[0] == 0
        assert _one(cur, "SELECT count(*) FROM contract_items WHERE contract_id = %s", (contract_id,))[0] == 0
        assert _one(cur, "SELECT count(*) FROM lots WHERE id = %s", (lot_id,))[0] == 0
        assert _one(cur, "SELECT count(*) FROM lot_plan_points WHERE lot_id = %s", (lot_id,))[0] == 0
        # пункт плана, пропавший из API, остаётся в аналитике: индекс планов OWS теряет записи без удаления
        assert _one(cur, "SELECT removed_at IS NULL FROM plan_points WHERE id = %s", (plan_gone,))[0]
        # с флагом isDeleted — помечается, но не удаляется (на него могут ссылаться позиции договоров)
        assert _one(cur, "SELECT removed_at IS NOT NULL FROM plan_points WHERE id = %s", (plan_deleted,))[0]
        assert stats["removed"]["contracts"] == 1 and stats["removed"]["contract_items"] == n_items
        assert stats["removed"]["plan_points"] == 1
    finally:
        conn.rollback()
        conn.close()


def test_plans_snapshot_keeps_missing_and_flags_them():
    from datetime import datetime
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "plans.jsonl"
        update.write_jsonl(p, [{"id": 1, "v": "old"}, {"id": 2, "v": "keep"}])
        new, gone = update.merge_plans_snapshot(p, [{"id": 1, "v": "new"}, {"id": 3, "v": "added"}],
                                                datetime(2026, 10, 14))
        rows = {r["id"]: r for r in update.read_jsonl(p)}
        assert (new, gone) == (1, 1)
        assert rows[1]["v"] == "new" and rows[3]["v"] == "added"
        assert rows[2]["_missing_from_api_since"] == "2026-10-14"   # не удалён, а помечен
        # вернулся в API — заменяется свежей версией без отметки
        update.merge_plans_snapshot(p, [{"id": 2, "v": "back"}], datetime(2026, 10, 15))
        rows = {r["id"]: r for r in update.read_jsonl(p)}
        assert rows[2] == {"id": 2, "v": "back"}


def test_mass_delete_guard():
    with psycopg2.connect(DB_URL) as c0, c0.cursor() as cur:
        cur.execute("SELECT id FROM contracts ORDER BY id")
        ids = [r[0] for r in cur.fetchall()]
    try:
        _run_etl_without({"Contract": set(ids[: len(ids) // 2])})
    except etl_load.MassDeleteError as e:
        assert "contract" in str(e)
    else:
        raise AssertionError("ETL должен был остановиться: из источника пропала половина договоров")


def test_unchanged_rerun_writes_nothing():
    """Повторный ETL на тех же данных не переписывает строки (UPSERT ... WHERE IS DISTINCT FROM)."""
    conn = psycopg2.connect(DB_URL)
    try:
        cur = conn.cursor()
        cur.execute("SELECT txid_current()")
        etl_load.main(conn, commit=False, verbose=False)
        # строки, изменённые этой транзакцией, имеют xmin = её id
        cur.execute("SELECT txid_current()")
        xid = cur.fetchone()[0]
        changed = {t: _one(cur, f"SELECT count(*) FROM {t} WHERE xmin::TEXT::BIGINT = %s", (xid % 2**32,))[0]
                   for t in ["contracts", "contract_items", "lots", "lot_plan_points", "plan_points", "announcements"]}
        assert all(v == 0 for v in changed.values()), changed
    finally:
        conn.rollback()
        conn.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print(json.dumps({"status": "all passed"}))
