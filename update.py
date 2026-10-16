"""
Инкрементальное обновление хранилища: API → data/raw → ETL → витрины. Запускается по расписанию
(cron, Kubernetes CronJob) каждые 6 часов: задержка данных относительно портала ≤ 24 ч с запасом.

Что и как забирается из OWS v3 (проверено на API 14.10.2026):
  Lots, Contract  по фильтру indexDate (дата попадания в индекс OWS) с последнего успешного запуска
                  минус OVERLAP_HOURS; изменения вливаются в data/raw/<Entity>/<bin>.jsonl по id.
                  Для одного БИН: 8 изменённых лотов из 5 427 за 0,2 с вместо 38 с.
                  Раз в FULL_EVERY_DAYS — полная выгрузка: удаление записи в источнике без флага
                  инкрементом не увидеть.
  Plans           всегда целиком: индекс планов пересобирается ежедневно (indexDate у всех записей
                  свежий, фильтр бесполезен), а удалённые пункты пропадают из API без isDeleted
                  (у одного БИН за 2 дня исчезло 182 пункта 2026 года).
  Subjects        целиком (27 записей).
Затем догружаются пункты плана, на которые ссылаются договоры (fetch_missing_plans), и одной транзакцией
выполняются etl_load + build_marts: агент во время пересборки ждёт, а не получает пустые витрины.
Водяные знаки (etl_state) двигаются только после успешного commit, поэтому сбой не теряет изменений:
следующий запуск заберёт то же окно заново. Журнал запусков в etl_runs.

Запуск:
  python update.py                  инкремент (полная выгрузка, если водяного знака нет или он старше FULL_EVERY_DAYS)
  python update.py --full           полная выгрузка всего
  python update.py --bins 990740002243,000740001307   только указанные БИН (отладка)
  python update.py --skip-fetch     только ETL + витрины по текущим data/raw
Код выхода 0 — успех, 1 — ошибка (для мониторинга планировщика).
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

import build_marts
import etl_load
import fetch_refs
from fetch_raw import (CONTRACT_QUERY, DATE_RANGE, LOTS_QUERY, PLANS_QUERY, SUBJECT_QUERY, YEARS,
                       fetch_all)
from tools import DB_URL

RAW = Path("data/raw")
# все отметки времени — местное время процесса (как indexDate в API), а не now() в контейнере БД (UTC)
INC = Path("data/raw_inc")             # копии дельт каждого запуска (аудит, повторная загрузка)
OVERLAP_HOURS = int(os.getenv("UPDATE_OVERLAP_HOURS", "24"))
FULL_EVERY_DAYS = int(os.getenv("UPDATE_FULL_EVERY_DAYS", "7"))
LOCK_KEY = 74_201_026                   # pg_advisory_lock: два обновления одновременно не идут


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


# ---------------- файлы ----------------

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Атомарная запись: при сбое посередине старый файл остаётся целым."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def merge_delta(path, delta):
    """Вливает изменённые записи в снимок по id (новая версия заменяет старую). Возвращает (новых, обновлённых)."""
    base = {r["id"]: r for r in read_jsonl(path)}
    new = sum(1 for r in delta if r["id"] not in base)
    for r in delta:
        base[r["id"]] = r
    write_jsonl(path, base.values())
    return new, len(delta) - new


def merge_plans_snapshot(path, rows, started):
    """Снимок планов вливается по id, а не перезаписывает файл: индекс планов OWS теряет записи без флага
    isDeleted (14.10.2026 — 8 412 пунктов плана 2026, из них 6 284 уже закуплены). Пропавшим ставится
    отметка _missing_from_api_since (ETL пишет её в журнал качества), вернувшиеся заменяются свежей версией.
    Возвращает (новых, впервые пропавших)."""
    base = {r["id"]: r for r in read_jsonl(path)}
    fresh = {r["id"]: r for r in rows}
    new = len(fresh.keys() - base.keys())
    gone = 0
    for i, r in base.items():
        if i not in fresh and not r.get("_missing_from_api_since") and not r.get("isDeleted"):
            r["_missing_from_api_since"] = started.strftime("%Y-%m-%d")
            gone += 1
    base.update(fresh)
    write_jsonl(path, base.values())
    return new, gone


# ---------------- водяные знаки ----------------

def get_marks(cur):
    cur.execute("SELECT entity, bin, last_update FROM etl_state")
    return {(e, b): t for e, b, t in cur.fetchall()}


def set_marks(cur, marks):
    for (entity, bin_), t in marks.items():
        cur.execute("""INSERT INTO etl_state (entity, bin, last_update) VALUES (%s, %s, %s)
                       ON CONFLICT (entity, bin) DO UPDATE SET last_update = EXCLUDED.last_update""",
                    (entity, bin_, t))


# ---------------- выгрузка ----------------

INCREMENTAL = {
    # сущность: (запрос, базовый фильтр для БИН)
    "Lots": (LOTS_QUERY, lambda b: {"customerBin": b}),
    "Contract": (CONTRACT_QUERY, lambda b: {"customerBin": b, "signDate": DATE_RANGE}),
}
SNAPSHOT = {
    "Subjects": (SUBJECT_QUERY, lambda b: {"bin": b}),
    "Plans": (PLANS_QUERY, lambda b: {"subjectBiin": b, "plnPointYear": YEARS}),
}


def fetch(bins, marks, started, full, run_dir):
    """Забирает данные из API. Возвращает (статистику, новые водяные знаки)."""
    stats, new_marks = {}, {}
    for b in bins:
        timing = {}
        for entity, (query, filt) in SNAPSHOT.items():
            t = time.time()
            rows = list(fetch_all(entity, query, filt(b)))
            path = RAW / entity / f"{b}.jsonl"
            st = stats.setdefault(entity, {"rows": 0, "new": 0, "gone": 0})
            if entity == "Plans":
                n_new, n_gone = merge_plans_snapshot(path, rows, started)
            else:
                old = {r.get("bin") for r in read_jsonl(path)}
                write_jsonl(path, rows)
                n_new, n_gone = len({r.get("bin") for r in rows} - old), len(old - {r.get("bin") for r in rows})
            st["rows"] += len(rows)
            st["new"] += n_new
            st["gone"] += n_gone
            st["seconds"] = st.get("seconds", 0) + round(time.time() - t)
            timing[entity] = round(time.time() - t)
            new_marks[(entity, b)] = started
        for entity, (query, filt) in INCREMENTAL.items():
            t = time.time()
            mark = marks.get((entity, b))
            path = RAW / entity / f"{b}.jsonl"
            st = stats.setdefault(entity, {"full": 0, "incremental": 0, "fetched": 0, "new": 0, "updated": 0})
            if full or mark is None or started - mark > timedelta(days=FULL_EVERY_DAYS) or not path.exists():
                rows = list(fetch_all(entity, query, filt(b)))
                old = {r["id"] for r in read_jsonl(path)}
                write_jsonl(path, rows)
                st["full"] += 1
                st["fetched"] += len(rows)
                st["new"] += len({r["id"] for r in rows} - old)
                # для полной выгрузки знак ставим на её начало: всё, что изменилось позже, заберёт инкремент
                new_marks[(entity, b)] = started
            else:
                since = mark - timedelta(hours=OVERLAP_HOURS)
                window = [since.strftime("%Y-%m-%d %H:%M:%S"), (started + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")]
                delta = list(fetch_all(entity, query, {**filt(b), "indexDate": window}))
                if delta:
                    write_jsonl(run_dir / entity / f"{b}.jsonl", delta)
                    n_new, n_upd = merge_delta(path, delta)
                    st["new"] += n_new
                    st["updated"] += n_upd
                st["incremental"] += 1
                st["fetched"] += len(delta)
                new_marks[(entity, b)] = started
            st["seconds"] = st.get("seconds", 0) + round(time.time() - t)
            timing[entity] = round(time.time() - t)
        log(f"  БИН {b}: " + ", ".join(f"{k} {v} с" for k, v in timing.items()))
    return stats, new_marks


UNAVAILABLE = RAW / "Plans" / "_unavailable.json"   # id, которые API не отдаёт (удалены или чужие)


def fetch_missing_plans(retry_unavailable=False):
    """Пункты плана, на которые ссылаются позиции договоров в data/raw, но которых нет среди планов.
    Считается по сырым файлам, чтобы ETL выполнялся один раз. Id, которые API уже не вернул,
    запоминаются и повторно запрашиваются только при полной выгрузке."""
    unavailable = set() if retry_unavailable or not UNAVAILABLE.exists() else set(json.loads(UNAVAILABLE.read_text()))
    have = set()
    for f in (RAW / "Plans").glob("*.jsonl"):
        have |= {r["id"] for r in read_jsonl(f)}
    need = set()
    for f in (RAW / "Contract").glob("*.jsonl"):
        for c in read_jsonl(f):
            need |= {u["plnPointId"] for u in c.get("ContractUnits") or [] if u.get("plnPointId")}
    missing = sorted(need - have - unavailable)
    got = []
    for i in range(0, len(missing), 100):
        got += list(fetch_all("Plans", PLANS_QUERY, {"id": missing[i:i + 100]}))
    if got:
        path = RAW / "Plans" / "_missing.jsonl"
        merge_delta(path, got)
    unavailable |= set(missing) - {r["id"] for r in got}
    UNAVAILABLE.write_text(json.dumps(sorted(unavailable)))
    return {"requested": len(missing), "fetched": len(got), "unavailable_known": len(unavailable)}


# ---------------- запуск ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="полная выгрузка всех сущностей")
    ap.add_argument("--bins", help="БИН через запятую (по умолчанию bins.txt)")
    ap.add_argument("--skip-fetch", action="store_true", help="без обращения к API: только ETL и витрины")
    ap.add_argument("--marks-at", help="с --skip-fetch: данные в data/raw выгружены полностью начиная с этого "
                                        "момента (ГГГГ-ММ-ДД ЧЧ:ММ:СС) — проставить водяные знаки для всех БИН. "
                                        "Нужно, если выгрузка прошла, а загрузка в БД упала")
    a = ap.parse_args()
    bins = a.bins.split(",") if a.bins else [b.strip() for b in Path("bins.txt").read_text().split() if b.strip()]

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(Path("schema.sql").read_text(encoding="utf-8").split("-- DROP ... CASCADE")[0])  # таблицы, без пересоздания view
    cur.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
    if not cur.fetchone()[0]:
        log("другое обновление ещё выполняется, выходим")
        return 0
    started = datetime.now().replace(microsecond=0)
    mode = "skip-fetch" if a.skip_fetch else "full" if a.full else "incremental"
    if a.marks_at and not a.skip_fetch:
        log("--marks-at используется только с --skip-fetch")
        return 1
    marks_at = datetime.strptime(a.marks_at, "%Y-%m-%d %H:%M:%S") if a.marks_at else None
    if marks_at:
        mode = "load-fetched"
    # data_as_of: на какой момент данные из API. Для выгрузки — её начало (записи, изменённые во время
    # выгрузки, могли не попасть); для --marks-at — начало той выгрузки; для --skip-fetch — не меняется
    data_as_of = marks_at or (None if a.skip_fetch else started)
    cur.execute("INSERT INTO etl_runs (started_at, mode, status, data_as_of) VALUES (%s, %s, 'running', %s) RETURNING id",
                (started, mode, data_as_of))
    run_id = cur.fetchone()[0]
    stats, t0 = {"bins": len(bins)}, time.time()
    try:
        new_marks = {}
        if marks_at:
            new_marks = {(e, b): marks_at for b in bins for e in list(SNAPSHOT) + list(INCREMENTAL)}
        if not a.skip_fetch:
            log(f"выгрузка из API ({mode}), БИН: {len(bins)}")
            stats["fetch"], new_marks = fetch(bins, get_marks(cur), started, a.full,
                                              INC / started.strftime("%Y%m%dT%H%M%S"))
            stats["missing_plans"] = fetch_missing_plans(retry_unavailable=a.full)
            stats["fetch_seconds"] = round(time.time() - t0)
            log(f"выгрузка: {json.dumps(stats['fetch'], ensure_ascii=False)}; пункты плана для договоров: {stats['missing_plans']}")

        # ETL + витрины + водяные знаки — одной транзакцией
        t1 = time.time()
        conn.autocommit = False
        stats["etl"] = etl_load.main(conn, commit=False, verbose=False)
        stats["marts"] = build_marts.main(conn, commit=False, verbose=False)
        set_marks(conn.cursor(), new_marks)
        conn.commit()
        conn.autocommit = True
        stats["etl_seconds"] = round(time.time() - t1)
        log(f"ETL и витрины: {stats['etl_seconds']} с; удалено/помечено: {stats['etl']['removed']}")

        if not a.skip_fetch:
            stats["refs"] = fetch_refs.main(only_missing=True)
        stats["total_seconds"] = round(time.time() - t0)
        cur.execute("UPDATE etl_runs SET finished_at = %s, status = 'ok', stats = %s WHERE id = %s",
                    (datetime.now(), Json(stats), run_id))
        log(f"готово за {stats['total_seconds']} с")
        return 0
    except Exception as e:  # noqa: BLE001 — любая ошибка фиксируется в журнале, водяные знаки не двигаются
        conn.rollback()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("UPDATE etl_runs SET finished_at = %s, status = 'failed', stats = %s, error = %s WHERE id = %s",
                    (datetime.now(), Json(stats), f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}", run_id))
        log(f"ОШИБКА: {type(e).__name__}: {e}")
        return 1
    finally:
        cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
