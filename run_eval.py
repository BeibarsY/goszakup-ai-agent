"""
Eval агента: вопросы RU/KZ из eval_cases.json, эталоны считаются здесь же независимо от tools.py.

Эталон (gold) пересчитывается на Python по определениям метрик из docs/analytics.md прямо из таблиц
(mv_price_base, plan_points, lots), без витрин v_price_anomalies / v_volume_growth / mv_enstru_stats
и без функций tools.py. Поэтому расхождение ловит ошибки и агента, и SQL-витрин.
Эталон хранится как SQL-логика, а не как числа: после обновления данных eval остаётся валидным.

Наборы: dev (40 вопросов, по ним агент дорабатывался) и holdout (вопросы, добавленные после
доработки и не использованные для правок): честная оценка — по holdout.

Метрики:
  route_acc       доля вопросов, где выбран верный инструмент и верные ключевые параметры
  fact_recall     доля эталонных фактов (числа, ID, БИН, вердикты), найденных в ответе агента:
                  основная метрика «точность извлечения фактов», цель ≥ 85%
  fact_precision  доля чисел/дат/БИН в ответе, подтверждённых данными инструмента (agent.verify)

Запуск:
  python run_eval.py                    все вопросы, отчёт eval_report.md и eval_results.json
  python run_eval.py --only 3,7 -v      отдельные вопросы с подробностями
"""
import argparse
import json
import re
import statistics
import time
from collections import defaultdict

import psycopg2
from psycopg2.extras import RealDictCursor

import agent
from tools import DB_URL, MIN_N, REGION_SHRINK, SEARCH_STOP, _norm_text, _stem

CASES_FILE = "eval_cases.json"


# ---------------- загрузка данных ----------------

def q(sql, p=None):
    with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, p or {})
        return [{k: (float(v) if type(v).__name__ == "Decimal" else v) for k, v in r.items()} for r in cur.fetchall()]


class Data:
    def __init__(self):
        t = time.time()
        self.prices = q("""SELECT item_id, contract_id, contract_number, lot_id, customer_bin, sign_date, enstru_code,
                                  unit_norm, is_service, quantity_norm, price_real, city_code, month_num, category
                           FROM mv_price_base""")
        self.items = q("SELECT id, contract_id, plan_point_id, enstru_code, total_sum_wnds, lot_id FROM contract_items")
        self.item_plan = {r["id"]: r["plan_point_id"] for r in self.items}
        self.contracts = q("""SELECT id, root_id, is_latest, contract_number, number_anno, customer_bin, supplier_bin,
                                     trade_method_id, status_id, sign_date, contract_sum, contract_sum_wnds, currency
                              FROM contracts""")
        self.methods = {r["id"]: r for r in q("SELECT * FROM ref_trade_methods")}
        self.cpi = sorted((r["month"], r["cpi"]) for r in q("SELECT month, cpi FROM ref_cpi"))
        self.statuses = {r["id"]: r for r in q("SELECT * FROM ref_contract_status")}
        self.season = {(r["category"], r["month_num"]): r["coef"] for r in q("SELECT category, month_num, coef FROM mv_season")}
        self.plans = q("""SELECT p.id, p.customer_bin, p.plan_year, p.enstru_code, p.unit_norm, p.count_norm, p.amount,
                                 COALESCE(p.subject_type_id IN (2, 3), u.norm_name ILIKE '%%услуг%%' OR u.norm_name ILIKE '%%работ%%') AS is_service
                          FROM plan_points p LEFT JOIN ref_units u ON u.code = p.unit_code
                          WHERE p.plan_year IN (2024, 2025, 2026) AND p.count_norm > 0""")
        self.lots = q("""SELECT l.id, l.lot_number, l.customer_bin, l.name_clean, l.count, l.amount,
                                a.number_anno, a.publish_date
                         FROM lots l JOIN announcements a ON a.id = l.announcement_id""")
        self.by_code = defaultdict(list)
        for r in self.prices:
            self.by_code[(r["enstru_code"], r["unit_norm"])].append(r)
        print(f"данные загружены за {time.time() - t:.1f} с: цен {len(self.prices)}, пунктов плана {len(self.plans)}, лотов {len(self.lots)}")


def median(xs):
    return statistics.median(xs) if xs else None


def pct(xs, p):
    """percentile_cont: линейная интерполяция, как в PostgreSQL."""
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


# ---------------- эталоны по типам вопросов ----------------

def gold_anomalies(d, customer_bin=None, year=None, threshold=0.30, enstru_code=None, top=3):
    stats = {}
    for key, rows in d.by_code.items():
        qty = sum(r["quantity_norm"] for r in rows)
        stats[key] = (len(rows), sum(r["price_real"] * r["quantity_norm"] for r in rows) / qty if qty else None)
    checked, anom = [], []
    for r in d.prices:
        n, wavg = stats[(r["enstru_code"], r["unit_norm"])]
        if n < MIN_N or wavg is None:
            continue
        if enstru_code and r["enstru_code"] != enstru_code:
            continue
        if not enstru_code and r["is_service"]:
            continue
        if customer_bin and r["customer_bin"] != customer_bin:
            continue
        if year and r["sign_date"].year != year:
            continue
        checked.append(r)
        dev = (r["price_real"] - wavg) / wavg
        if dev > threshold:
            anom.append(((r["price_real"] - wavg) * r["quantity_norm"], r, wavg, dev))
    anom.sort(key=lambda x: -x[0])
    facts = [("проверено позиций", len(checked), "num"), ("аномалий", len(anom), "num"),
             ("суммарная переплата", sum(a[0] for a in anom), "num")]
    if anom:
        over, r, wavg, dev = anom[0]
        facts += [("топ-1 договор ID", r["contract_id"], "id"), ("топ-1 цена", r["price_real"], "num"),
                  ("топ-1 средневзвешенная", wavg, "num"), ("топ-1 отклонение", dev, "pct"),
                  ("топ-1 переплата", over, "num")]
        seen = []
        for a in anom:  # следующие договоры топа (могут повторяться позициями)
            if a[1]["contract_id"] not in seen:
                seen.append(a[1]["contract_id"])
        facts += [(f"топ-{i + 1} договор ID", c, "id") for i, c in enumerate(seen[1:top], 1)]
    return facts


def cpi_factor(d, dt):
    """Приведение цены месяца dt к ценам последнего месяца ИПЦ (как price_real): индекс ближайшего месяца ≤ dt."""
    if not d.cpi:
        return 1.0
    prior = [c for m, c in d.cpi if m <= dt.date().replace(day=1)]
    return d.cpi[-1][1] / prior[-1] if prior else 1.0


def gold_lot(d, lot_ref):
    """Спецификация: каждый пункт плана лота оценивается отдельно; цена пункта из позиции договора
    (по plan_point_id, иначе по коду и единице), иначе сумма пункта плана / количество."""
    lots = [l for l in d.lots if lot_ref in (l["lot_number"], str(l["id"]), l["number_anno"])]
    if len(lots) != 1:
        raise ValueError(f"эталон рассчитан на один лот, найдено {len(lots)}")
    lot = lots[0]
    points = q("""SELECT pp.id, pp.name_ru, pp.enstru_code, pp.unit_norm, pp.city_code, pp.count_norm, pp.amount
                  FROM lot_plan_points lp JOIN plan_points pp ON pp.id = lp.plan_point_id
                  WHERE lp.lot_id = %(l)s""", {"l": lot["id"]})
    facts = sorted((r for r in d.prices if r["lot_id"] == lot["id"]), key=lambda r: r["sign_date"], reverse=True)
    items = []
    for pp in points:
        if not pp["enstru_code"] or not pp["unit_norm"]:
            continue
        f = (next((r for r in facts if d.item_plan.get(r["item_id"]) == pp["id"]), None)
             or next((r for r in facts if (r["enstru_code"], r["unit_norm"]) == (pp["enstru_code"], pp["unit_norm"])), None))
        if f:
            price, month = f["price_real"], f["month_num"]
        elif pp["count_norm"] and pp["amount"]:
            price, month = pp["amount"] / pp["count_norm"] * cpi_factor(d, lot["publish_date"]), lot["publish_date"].month
        else:
            continue
        others = [r for r in d.by_code[(pp["enstru_code"], pp["unit_norm"])] if r["customer_bin"] != lot["customer_bin"]]
        if not others:
            continue
        base = median([r["price_real"] for r in others])
        city = [r["price_real"] for r in others if pp["city_code"] and r["city_code"] == pp["city_code"]]
        raw_k = median(city) / base if city else 1.0
        k_region = 1 + len(city) / (len(city) + REGION_SHRINK) * (raw_k - 1)
        fair = base * k_region * d.season.get((pp["enstru_code"][:2], month), 1.0)
        items.append({"name": pp["name_ru"], "price": price, "base": base, "n": len(others), "n_city": len(city),
                      "fair": fair, "dev": (price - fair) / fair, "qty": pp["count_norm"] or 0})
    facts_out = [("БИН заказчика", lot["customer_bin"], "id"), ("номер объявления", lot["number_anno"], "id")]
    if len(points) == 1 and items:
        i = items[0]
        return facts_out + [("цена за единицу", i["price"], "num"), ("P_base", i["base"], "num"),
                            ("N аналогов", i["n"], "num"), ("N в городе", i["n_city"], "num_small"),
                            ("Fair Price", i["fair"], "num"), ("отклонение", i["dev"], "pct"),
                            ("вердикт", verdict(i["dev"]), "verdict")]
    cost = sum(i["price"] * i["qty"] for i in items)
    fair_cost = sum(i["fair"] * i["qty"] for i in items)
    dev = (cost - fair_cost) / fair_cost
    w = max(items, key=lambda i: abs(i["dev"]))
    return facts_out + [("пунктов плана", len(points), "num_small"), ("оценено пунктов", len(items), "num_small"),
                        ("стоимость оценённых пунктов", cost, "num"), ("стоимость по Fair Price", fair_cost, "num"),
                        ("отклонение лота", dev, "pct"), ("вердикт лота", verdict(dev), "verdict"),
                        ("пункт с макс. отклонением: цена", w["price"], "num"),
                        ("пункт с макс. отклонением: Fair Price", w["fair"], "num"),
                        ("пункт с макс. отклонением: отклонение", w["dev"], "pct")]


def verdict(dev):
    return ("в пределах рынка" if abs(dev) <= 0.15 else "требует внимания" if abs(dev) <= 0.30 else
            "существенно выше рынка" if dev > 0 else "существенно ниже рынка")


def gold_volumes(d, customer_bin=None, year=2026, min_ratio=2.0, min_amount=1_000_000, enstru_code=None):
    agg = defaultdict(lambda: {"qty": 0.0, "amount": 0.0})
    svc = {}
    for p in d.plans:
        k = (p["customer_bin"], p["enstru_code"], p["unit_norm"], p["plan_year"])
        agg[k]["qty"] += p["count_norm"]
        agg[k]["amount"] += p["amount"] or 0
        svc[k] = svc.get(k, False) or bool(p["is_service"])   # как bool_or в mv_volumes
    checked, flagged = 0, []
    for (b, e, u, y), v in agg.items():
        if y != year or (customer_bin and b != customer_bin) or (enstru_code and e != enstru_code):
            continue
        if not enstru_code and svc[(b, e, u, y)]:
            continue
        prev = [agg[(b, e, u, py)]["qty"] for py in range(2024, year) if (b, e, u, py) in agg]
        if not prev:
            continue
        checked += 1
        med = median(prev)
        ratio = v["qty"] / med if med else None
        if ratio and ratio >= min_ratio and v["amount"] >= min_amount:
            flagged.append((v["amount"] * (1 - med / v["qty"]), b, e, v["qty"], med, ratio))
    flagged.sort(key=lambda x: -x[0])
    facts = [("проверено пар", checked, "num"), ("отмечено", len(flagged), "num")]
    if flagged:
        extra, b, e, qty, med, ratio = flagged[0]
        facts += [("топ-1 БИН", b, "id"), ("топ-1 код", e, "id"), ("топ-1 количество", qty, "num"),
                  ("топ-1 медиана прошлых лет", med, "num"), ("топ-1 рост", ratio, "num"),
                  ("топ-1 сумма на прирост", extra, "num")]
        facts += [(f"топ-{i} код", f[2], "id") for i, f in enumerate(flagged[1:3], 2)]
    return facts


def gold_search(d, text, customer_bin=None, year=None):
    words = [w for w in re.findall(r"\w+", _norm_text(text)) if len(w) >= 3 and w not in SEARCH_STOP] or [text.lower()]
    stems = [_stem(w) for w in words]
    latest = {}
    for l in d.lots:
        if not all(s in (l["name_clean"] or "") for s in stems):
            continue
        if customer_bin and l["customer_bin"] != customer_bin:
            continue
        if year and l["publish_date"].year != year:
            continue
        k = (l["lot_number"].split("-")[0], l["customer_bin"])
        if k not in latest or l["publish_date"] > latest[k]["publish_date"]:
            latest[k] = l
    rows = sorted(latest.values(), key=lambda l: -(l["amount"] or 0))
    facts = [("найдено лотов", len(rows), "num"), ("общая сумма", sum(l["amount"] or 0 for l in rows), "num")]
    if rows:
        facts += [("топ-1 лот", rows[0]["lot_number"], "id"), ("топ-1 сумма", rows[0]["amount"], "num")]
    return facts


def gold_compare(d, enstru_code):
    units = defaultdict(int)
    for (e, u), rows in d.by_code.items():
        if e == enstru_code:
            units[u] = len(rows)
    unit = max(units, key=units.get)
    rows = d.by_code[(enstru_code, unit)]
    qty = sum(r["quantity_norm"] for r in rows)
    per = defaultdict(list)
    for r in rows:
        per[r["customer_bin"]].append(r["price_real"])
    meds = {b: (median(v), len(v)) for b, v in per.items()}
    reliable = {b: m for b, m in meds.items() if m[1] >= agent.COMPARE_MIN_N} or meds
    hi = max(reliable, key=lambda b: reliable[b][0])
    lo = min(reliable, key=lambda b: reliable[b][0])
    return [("N позиций", len(rows), "num"), ("медиана", median([r["price_real"] for r in rows]), "num"),
            ("средневзвешенная", sum(r["price_real"] * r["quantity_norm"] for r in rows) / qty, "num"),
            ("заказчиков", len(meds), "num_small"), ("БИН с макс. медианой", hi, "id"),
            ("макс. медиана", reliable[hi][0], "num"), ("БИН с мин. медианой", lo, "id"),
            ("мин. медиана", reliable[lo][0], "num")]


def gold_contract(d, ref, customer_bin=None):
    lot_ids = {l["id"] for l in d.lots if l["lot_number"] == ref}
    by_lot = {i["contract_id"] for i in d.items if i["lot_id"] in lot_ids}
    roots = {c["root_id"] for c in d.contracts
             if (str(c["id"]) == ref or c["contract_number"] == ref or c["number_anno"] == ref or c["id"] in by_lot)
             and (not customer_bin or c["customer_bin"] == customer_bin)}
    latest = [c for c in d.contracts if c["is_latest"] and c["root_id"] in roots]
    if len(latest) > 1:
        return [("кандидатов", len(latest), "num_small")] + [(f"кандидат ID", c["id"], "id") for c in latest[:3]]
    c = latest[0]
    versions = sorted((x for x in d.contracts if x["root_id"] == c["root_id"]), key=lambda x: (x["sign_date"], x["id"]))
    m, st = d.methods.get(c["trade_method_id"], {}), d.statuses.get(c["status_id"], {})
    facts = [("ID договора", c["id"], "id"), ("дата подписания", str(c["sign_date"])[:10], "id"),
             ("БИН поставщика", c["supplier_bin"], "id"), ("сумма с НДС", c["contract_sum_wnds"], "num"),
             ("сумма без НДС", c["contract_sum"], "num"), ("редакций", len(versions), "num_small"),
             ("способ закупки", (m.get("name_ru"), m.get("name_kz")), "text"),
             ("статус", (st.get("name_ru"), st.get("name_kz")), "text")]
    if len(versions) > 1:
        facts.append(("первоначальная сумма", versions[0]["contract_sum_wnds"], "num"))
    return facts


def gold_analytics(d, customer_bin=None, year=None, enstru_code=None, supplier_bin=None):
    ss = {6, 23, 131}
    cs = {c["id"]: c for c in d.contracts if c["is_latest"] and c["currency"] == "KZT"
          and (not customer_bin or c["customer_bin"] == customer_bin)
          and (not supplier_bin or c["supplier_bin"] == supplier_bin)
          and (not year or c["sign_date"].year == year)}
    amount = defaultdict(float)
    if enstru_code:
        for i in d.items:
            if i["contract_id"] in cs and i["enstru_code"] == enstru_code:
                amount[i["contract_id"]] += i["total_sum_wnds"] or 0
    else:
        for cid, c in cs.items():
            amount[cid] = c["contract_sum_wnds"] or 0
    total = sum(amount.values())
    single = sum(v for k, v in amount.items() if cs[k]["trade_method_id"] in ss)
    sup = defaultdict(float)
    for k, v in amount.items():
        sup[cs[k]["supplier_bin"]] += v
    top_sup = max(sup, key=sup.get)
    top_c = max(amount, key=amount.get)
    return [("договоров", len(amount), "num"), ("сумма с НДС", total, "num"),
            ("из одного источника, сумма", single, "num"), ("доля из одного источника", single / total, "pct"),
            ("поставщиков", len(sup), "num"), ("топ-1 поставщик БИН", top_sup, "id"),
            ("топ-1 поставщик сумма", sup[top_sup], "num"), ("крупнейший договор ID", top_c, "id")]


GOLD = {"find_price_anomalies": gold_anomalies, "assess_lot_price": gold_lot, "find_volume_spikes": gold_volumes,
        "search_purchases": gold_search, "compare_customers": gold_compare,
        "contract_lookup": gold_contract, "purchase_analytics": gold_analytics}


# ---------------- сверка фактов с ответом ----------------

RE_NUM = re.compile(r"[-+−]?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?!\d)|[-+−]?\d+(?:[.,]\d+)?")
VERDICT_KZ = agent.VERDICT_KZ


def answer_numbers(text):
    # ID, БИН, коды и даты не должны склеиваться с соседними числами
    text = re.sub(r"\d{4}-\d{2}-\d{2}|\d{6}\.\d{3}\.\d{6}|\(ID \d+|БИН \d{12}", " ", text)
    out = []
    for m in RE_NUM.finditer(text):
        s = m.group(0).replace(" ", "").replace(" ", "").replace(" ", "").replace("−", "-").replace(",", ".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def num_found(v, nums, kind):
    v = float(v)
    if kind == "pct":
        v *= 100
        return any(abs(abs(x) - abs(v)) <= max(0.051, abs(v) * 0.001) for x in nums)
    if kind == "num_small":
        return any(x == round(v) for x in nums)
    tol = max(0.006, abs(v) * 0.0005)  # округление до целых для сумм, до копеек для цен
    return any(abs(x - v) <= max(tol, 0.5 if abs(v) >= 1000 else tol) for x in nums)


def check_facts(answer, facts, lang):
    nums = answer_numbers(answer)
    res = []
    for label, val, kind in facts:
        if kind == "id":
            ok = str(val) in answer
        elif kind == "text":  # название из справочника: RU или KZ
            ok = any(v and v in answer for v in val)
            val = val[0]
        elif kind == "verdict":
            ok = (VERDICT_KZ.get(val, val) if lang == "kz" else val) in answer
        else:
            ok = num_found(val, nums, kind)
        res.append({"fact": label, "gold": val if not isinstance(val, float) else round(val, 4), "kind": kind, "found": ok})
    return res


def check_route(case, out):
    exp = case["expect"]
    if out.get("tool") != exp["tool"]:
        return False, f"инструмент {out.get('tool')} вместо {exp['tool']}"
    args = dict(out.get("args") or {})
    used = out.get("tool_result", {}).get("params", {})
    for k, v in exp.get("args", {}).items():
        got = args.get(k, used.get(k))
        if k == "enstru_code":  # код может прийти через product → resolve_enstru
            got = got or used.get("enstru_code")
        if v is None and got not in (None, ""):
            return False, f"лишний параметр {k}={got}"
        if v is not None and str(got) != str(v):
            return False, f"{k}={got} вместо {v}"
    return True, ""


# ---------------- прогон ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="номера вопросов через запятую")
    ap.add_argument("--split", choices=["dev", "holdout"], help="только один набор")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    cases = json.load(open(CASES_FILE, encoding="utf-8"))
    if a.only:
        ids = {int(x) for x in a.only.split(",")}
        cases = [c for c in cases if c["id"] in ids]
    if a.split:
        cases = [c for c in cases if c.get("split", "dev") == a.split]
    d = Data()
    results = []
    for c in cases:
        exp = c["expect"]
        gold_args = {k: v for k, v in {**exp.get("args", {}), **exp.get("gold_args", {})}.items() if v is not None}
        gold_args.pop("product", None)
        try:
            facts = GOLD[exp["tool"]](d, **gold_args)
        except Exception as e:  # noqa: BLE001
            print(f"[{c['id']}] эталон не рассчитан: {e}")
            continue
        t = time.time()
        out = agent.answer(c["question"])
        dt = time.time() - t
        route_ok, route_err = check_route(c, out)
        checks = check_facts(out.get("answer", ""), facts, out.get("lang", "ru"))
        ver = out.get("verification", {})
        r = {"id": c["id"], "lang": c["lang"], "split": c.get("split", "dev"), "class": exp["tool"], "question": c["question"],
             "route_ok": route_ok, "route_err": route_err, "tool": out.get("tool"), "args": out.get("args"),
             "facts": checks, "n_gold": len(checks), "n_found": sum(x["found"] for x in checks),
             "answer_facts": ver.get("n_facts", 0), "answer_unsupported": ver.get("n_unsupported", 0),
             "seconds": round(dt, 2), "answer": out.get("answer", "")}
        results.append(r)
        miss = [x["fact"] for x in checks if not x["found"]]
        print(f"[{c['id']:>2}] {c['lang']} {exp['tool']:<21} route={'OK ' if route_ok else 'BAD'} "
              f"facts {r['n_found']}/{r['n_gold']} unsup {r['answer_unsupported']}/{r['answer_facts']} {dt:4.1f}s"
              + (f"  ! {route_err}" if route_err else "") + (f"  miss: {', '.join(miss)}" if miss else ""))
        if a.verbose:
            for x in checks:
                print("     ", "+" if x["found"] else "-", x["fact"], x["gold"])
    if results:
        report(results)


def summarize(rs):
    n_g, n_f = sum(r["n_gold"] for r in rs), sum(r["n_found"] for r in rs)
    n_a, n_u = sum(r["answer_facts"] for r in rs), sum(r["answer_unsupported"] for r in rs)
    return {"n": len(rs), "route_acc": sum(r["route_ok"] for r in rs) / len(rs),
            "fact_recall": n_f / n_g if n_g else None, "gold_facts": n_g,
            "fact_precision": 1 - n_u / n_a if n_a else None, "answer_facts": n_a,
            "avg_seconds": sum(r["seconds"] for r in rs) / len(rs)}


def report(results):
    total = summarize(results)
    groups = {"набор": defaultdict(list), "класс": defaultdict(list), "язык": defaultdict(list)}
    for r in results:
        groups["набор"][r["split"]].append(r)
        groups["класс"][r["class"]].append(r)
        groups["язык"][r["lang"]].append(r)

    def row(name, s):
        return (f"| {name} | {s['n']} | {s['route_acc']:.0%} | {s['fact_recall']:.1%} ({s['gold_facts']}) | "
                f"{s['fact_precision']:.1%} ({s['answer_facts']}) | {s['avg_seconds']:.1f} |")

    lines = ["# Eval агента", "",
             f"Модель маршрутизации: {agent.LLM_MODEL}; режим вывода: {agent.SUMMARY_MODE}; вопросов: {total['n']}.",
             "", "Цель: точность извлечения фактов (fact_recall) ≥ 85%.", "",
             "| Срез | N | route_acc | fact_recall (эталонных фактов) | fact_precision (фактов в ответах) | с/вопрос |",
             "|---|---|---|---|---|---|", row("**всего**", total)]
    for g, parts in groups.items():
        for k, rs in sorted(parts.items()):
            lines.append(row(f"{g}: {k}", summarize(rs)))
    lines += ["", "## Ошибки", ""]
    for r in results:
        miss = [x for x in r["facts"] if not x["found"]]
        if r["route_ok"] and not miss:
            continue
        lines.append(f"- **[{r['id']}] {r['lang']}** {r['question']}")
        if not r["route_ok"]:
            lines.append(f"  - маршрутизация: {r['route_err']} (args: {r['args']})")
        for x in miss:
            lines.append(f"  - не найден факт «{x['fact']}»: эталон {x['gold']}")
    open("eval_report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"summary": total, "results": results}, open("eval_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    table = lines[lines.index("## Ошибки") - 1::-1][::-1]  # всё до раздела ошибок
    print("\n" + "\n".join(l for l in table if l.startswith("|")))
    print("\nотчёт: eval_report.md, подробности: eval_results.json")


if __name__ == "__main__":
    main()
