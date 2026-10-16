"""
Инструменты аналитики для AI-агента. LLM сама ничего не считает: она вызывает эти функции,
а функции возвращают готовые цифры + всё нужное для объяснимости (выборка, N, метод, ID, ссылки).

Проверка из терминала:
    python tools.py anomalies --enstru 282932.500.000001
    python tools.py lot 80800385-ОИ6
    python tools.py volumes --year 2026
    python tools.py search "бумага офисная"
    python tools.py compare 171012.300.000000
"""
import argparse
import json
import os
import re
from datetime import date, datetime
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
MIN_N = 5                 # минимальный размер выборки для статистики
REGION_SHRINK = 5         # сглаживание регионального коэффициента: w = n_city / (n_city + 5)
YEARS = "2024-2026"
KZ_PLAIN = str.maketrans("әғқңөұүһіё", "агкноуухие")
# латинские буквы, похожие на кириллицу: модели иногда пишут «кaгaз» с латинской a
HOMOGLYPHS = str.maketrans("aceopxyiAKMHTBCEOPXY", "асеорхуіАКМНТВСЕОРХУ")


def _norm_text(text):
    """Нижний регистр, латинские двойники → кириллица (только в кириллических словах)."""
    words = re.findall(r"\w+|\W+", text.lower())
    return "".join(w.translate(HOMOGLYPHS) if re.search(r"[а-яёәғқңөұүһі]", w) else w for w in words)


def _stem(w):
    """Грубый стемминг: до 6 букв отрезаем одну («яйца» → «яйц», «бумаги» → «бумаг»), длиннее — две
    («перчатки» → «перчат»). Короче нельзя: «каг» из «кагаз» находит «силикагель»."""
    n = len(w)
    return w if n <= 3 else w[:n - 1] if n <= 6 else w[:n - 2]


SEARCH_STOP = {"для", "или", "под", "над", "без", "при", "все", "закупки", "закупка", "закупок", "покупка"}


# ---------------- служебное ----------------

def _q(sql, params=None):
    with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(sql, params or {})
        return [_clean(dict(r)) for r in cur.fetchall()]


def _val(v):
    if isinstance(v, Decimal):
        # суммы и цены до копеек; доли и коэффициенты (|v| < 100) до 6 знаков: при 4 знаках
        # 0.378450 -> 0.3785 и в ответе выходит 37,9% вместо 37,8% (двойное округление)
        return round(float(v), 2 if abs(v) >= 100 else 6)
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10]
    if isinstance(v, list):
        return [_val(i) for i in v]
    return v


def _clean(row):
    return {k: _val(v) for k, v in row.items()}


MONTHS_RU = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
             "ноября", "декабря"]
MONTHS_KZ = ["қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым", "шілде", "тамыз", "қыркүйек", "қазан",
             "қараша", "желтоқсан"]


def price_base():
    """К ценам какого месяца приведены цены (price_real): последний месяц ref_cpi. None, если ИПЦ не загружен."""
    r = _q("SELECT max(month) AS m, min(month) AS first FROM ref_cpi")[0]
    if not r["m"]:
        return None
    y, mo = int(r["m"][:4]), int(r["m"][5:7])
    return {"month": r["m"][:7], "first_month": r["first"][:7],
            "ru": f"{MONTHS_RU[mo - 1]} {y}", "kz": f"{y} жылғы {MONTHS_KZ[mo - 1]}"}


def cpi_factor(on_date):
    """Множитель приведения цены месяца on_date к ценам последнего месяца ИПЦ — как price_real в mv_price_base:
    индекс ближайшего месяца не позже даты; если такого нет или ИПЦ не загружен, 1."""
    r = _q("""SELECT (SELECT cpi FROM ref_cpi ORDER BY month DESC LIMIT 1)
                     / (SELECT cpi FROM ref_cpi WHERE month <= date_trunc('month', %(d)s::date) ORDER BY month DESC LIMIT 1)
                     AS f""", {"d": str(on_date)[:10]})[0]["f"]
    return float(r) if r else 1.0


def _inflation_note():
    b = price_base()
    if not b:
        return "Индекс потребительских цен не загружен: цены разных лет сравниваются без поправки на инфляцию."
    return (f"Цены приведены к ценам {b['ru']} по помесячному индексу потребительских цен (Бюро национальной "
            f"статистики РК); цены договоров после {b['month']} не пересчитываются.")


def _confidence(n, n_city=None, n_customers=None):
    # аналоги от одного ведомства — по сути один источник цен, уверенность низкая при любом N
    if n_customers is not None and n_customers < 2:
        return "низкая"
    if n >= 20 and (n_city is None or n_city >= 3) and (n_customers is None or n_customers >= 3):
        return "высокая"
    if n >= MIN_N:
        return "средняя"
    return "низкая"


# ---------------- 1. ценовые аномалии ----------------

def find_price_anomalies(enstru_code=None, customer_bin=None, year=None,
                         threshold=0.30, top_k=10, include_services=False):
    """Позиции договоров, где цена выше средневзвешенной по коду ЕНС ТРУ более чем на threshold.
    По умолчанию только товары: у услуг и работ «цена за единицу» = цена всего договора."""
    where = ["n >= %(min_n)s"]
    if not include_services and not enstru_code:
        where.append("NOT is_service")
    p = {"min_n": MIN_N, "thr": threshold, "k": top_k}
    if enstru_code:
        where.append("enstru_code = %(enstru)s"); p["enstru"] = enstru_code
    if customer_bin:
        where.append("customer_bin = %(bin)s"); p["bin"] = customer_bin
    if year:
        where.append("EXTRACT(YEAR FROM sign_date) = %(year)s"); p["year"] = year
    w = " AND ".join(where)

    summary = _q(f"""
        SELECT count(*) AS n_items_checked,
               count(DISTINCT enstru_code) AS n_enstru_codes,
               count(*) FILTER (WHERE dev_from_wavg > %(thr)s) AS n_anomalies,
               sum(overpay_vs_wavg) FILTER (WHERE dev_from_wavg > %(thr)s) AS total_overpay_kzt,
               avg(CASE WHEN inflation_applied THEN 1 ELSE 0 END) AS share_inflation_adjusted
        FROM v_price_anomalies WHERE {w}""", p)[0]

    rows = _q(f"""
        SELECT contract_id, contract_number, contract_url, lot_id, number_anno, announcement_url,
               customer_bin, customer_name, supplier_bin, supplier_name, sign_date,
               enstru_code, enstru_name, item_name, unit_name, quantity_norm AS quantity,
               unit_price_norm AS unit_price, price_real, wavg, median, n AS sample_n,
               dev_from_wavg, robust_z, above_iqr_fence, overpay_vs_wavg
        FROM v_price_anomalies
        WHERE {w} AND dev_from_wavg > %(thr)s
        ORDER BY overpay_vs_wavg DESC NULLS LAST
        LIMIT %(k)s""", p)

    limitations = [
        f"Статистика считается только по кодам ЕНС ТРУ, где не меньше {MIN_N} позиций договоров.",
        "Средневзвешенная цена взвешена по количеству, крупные закупки влияют на неё сильнее.",
        "Цены без НДС, за нормализованную единицу измерения. Позиции без связи с планом (нет единицы) исключены.",
        "Услуги и работы (единица «Одна услуга», «Работа») по умолчанию исключены: один код ЕНС ТРУ покрывает договоры очень разного объёма, и цена за «единицу» несравнима.",
    ]
    if note := _inflation_note():
        limitations.append(note)
    return {
        "question_type": "anomalies",
        "params": {"period": YEARS, "customer_bin": customer_bin, "enstru_code": enstru_code,
                   "year": year, "threshold": threshold,
                   "scope": "товары и услуги" if (include_services or enstru_code) else "только товары"},
        "method": (f"Для каждой позиции договора цена за единицу сравнивается со средневзвешенной "
                   f"по количеству ценой всех позиций с тем же кодом ЕНС ТРУ и единицей измерения. "
                   f"Аномалия: отклонение > {threshold:.0%}. Дополнительно указаны robust z-score "
                   f"(через MAD) и выход за Q3 + 1.5*IQR."),
        "summary": summary,
        "results": rows,
        "price_base": price_base(),
        "confidence": _confidence(summary["n_items_checked"] or 0),
        "limitations": limitations,
    }


# ---------------- 2. справедливость цены (Fair Price) ----------------

def fair_price(enstru_code, unit_norm, city_code, month_num, exclude_bin=None):
    """
    Fair Price = P_base * K_region * K_season
      P_base   - медиана цен (в ценах последнего месяца ИПЦ) других ведомств по тому же коду ЕНС ТРУ и единице
      K_region - медиана по городу поставки / P_base, сглаженная к 1: K = 1 + w*(raw-1), w = n_city/(n_city+5)
      K_season - сезонный коэффициент раздела КПВЭД для месяца (mv_season), 1 если данных мало
    """
    p = {"e": enstru_code, "u": unit_norm, "c": city_code, "x": exclude_bin or ""}
    base = _q("""
        SELECT count(*) AS n, count(DISTINCT customer_bin) AS n_customers,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY price_real) AS median,
               percentile_cont(0.25) WITHIN GROUP (ORDER BY price_real) AS q1,
               percentile_cont(0.75) WITHIN GROUP (ORDER BY price_real) AS q3
        FROM mv_price_base
        WHERE enstru_code = %(e)s AND unit_norm = %(u)s AND customer_bin <> %(x)s""", p)[0]
    city = _q("""
        SELECT count(*) AS n, percentile_cont(0.5) WITHIN GROUP (ORDER BY price_real) AS median
        FROM mv_price_base
        WHERE enstru_code = %(e)s AND unit_norm = %(u)s AND customer_bin <> %(x)s
          AND city_code = %(c)s""", p)[0]
    season = _q("SELECT coef, n FROM mv_season WHERE category = left(%(e)s, 2) AND month_num = %(m)s",
                {"e": enstru_code, "m": month_num})

    if not base["n"]:
        return {"n": 0}
    raw_k = (city["median"] / base["median"]) if city["n"] and base["median"] else 1.0
    w = city["n"] / (city["n"] + REGION_SHRINK)
    k_region = 1 + w * (raw_k - 1)
    k_season = season[0]["coef"] if season else 1.0
    return {
        "n": base["n"], "n_customers": base["n_customers"],
        "p_base_median": base["median"], "q1": base["q1"], "q3": base["q3"],
        "n_city": city["n"], "city_median": city["median"],
        "k_region": round(k_region, 4), "k_season": round(float(k_season), 4),
        # без округления: по Fair Price дальше считается отклонение, округляет только вывод
        "fair_price": base["median"] * k_region * float(k_season),
    }


def _verdict(dev):
    return ("в пределах рынка" if abs(dev) <= 0.15 else
            "требует внимания" if abs(dev) <= 0.30 else
            "существенно выше рынка" if dev > 0 else "существенно ниже рынка")


MAX_COMPARABLE_ITEMS = 3   # для скольких пунктов лота (с наибольшим отклонением) приводить договоры-аналоги


def assess_lot_price(lot_ref):
    """Оценка адекватности цены лота относительно аналогичных закупок других ведомств.
    Лот может состоять из многих пунктов плана с разными кодами ЕНС ТРУ («Продукты питания»):
    каждый пункт оценивается отдельно, для лота считается отклонение, взвешенное по количеству."""
    lots = _q("""
        SELECT l.id AS lot_id, l.lot_number, l.name_ru, l.description_ru, l.customer_bin,
               s.name_ru AS customer_name, l.count, l.amount, l.status_name,
               a.number_anno, a.url AS announcement_url, a.publish_date
        FROM lots l
        JOIN announcements a ON a.id = l.announcement_id
        LEFT JOIN subjects s ON s.bin = l.customer_bin
        WHERE l.lot_number = %(r)s OR l.id::TEXT = %(r)s OR a.number_anno = %(r)s
        ORDER BY l.lot_number
        LIMIT 300""", {"r": str(lot_ref).strip()})
    if not lots:
        return {"question_type": "fairness", "error": f"Лот {lot_ref} не найден в хранилище "
                f"(период {YEARS}, 27 организаций из перечня)."}

    results = []
    for lot in lots:
        points = _q("""
            SELECT pp.id AS plan_point_id, pp.name_ru AS item_name, pp.enstru_code, e.name_ru AS enstru_name,
                   pp.unit_norm, u.norm_name AS unit_name, pp.count_norm AS quantity, pp.amount,
                   pp.city_code, pp.kato_source, pp.delivery_place
            FROM lot_plan_points lp JOIN plan_points pp ON pp.id = lp.plan_point_id
            LEFT JOIN ref_enstru e ON e.code = pp.enstru_code
            LEFT JOIN ref_units u ON u.code = pp.unit_code
            WHERE lp.lot_id = %(l)s ORDER BY pp.amount DESC""", {"l": lot["lot_id"]})
        # фактические цены позиций договора по лоту, с привязкой к пункту плана
        facts = _q("""
            SELECT b.contract_id, b.contract_number, b.contract_url, b.enstru_code, b.unit_norm,
                   b.price_real, b.quantity_norm, b.month_num, b.sign_date, ci.plan_point_id
            FROM mv_price_base b JOIN contract_items ci ON ci.id = b.item_id
            WHERE b.lot_id = %(l)s ORDER BY b.sign_date DESC""", {"l": lot["lot_id"]})
        items = []
        for pp in points:
            item = dict(pp)
            if not pp["enstru_code"] or not pp["unit_norm"]:
                items.append({**item, "error": "у пункта плана нет кода ЕНС ТРУ или единицы"}); continue
            # позиция договора по этому пункту плана; если пункт в договоре другой редакции, то по коду и единице
            f = (next((x for x in facts if x["plan_point_id"] == pp["plan_point_id"]), None)
                 or next((x for x in facts if x["enstru_code"] == pp["enstru_code"]
                          and x["unit_norm"] == pp["unit_norm"]), None))
            if f:
                price, month = f["price_real"], f["month_num"]
                item.update(price_source="договор", contract_id=f["contract_id"],
                            contract_number=f["contract_number"], contract_url=f["contract_url"])
            elif pp["quantity"] and pp["amount"]:
                # цена плана приводится к ценам последнего месяца ИПЦ, как и цены аналогов (price_real)
                factor = cpi_factor(lot["publish_date"])
                price, month = pp["amount"] / pp["quantity"] * factor, int(lot["publish_date"][5:7])
                item["price_source"] = "план (сумма / количество), договора нет"
                item["inflation_factor"] = round(factor, 6)
            else:
                items.append({**item, "error": "нет цены и количества"}); continue
            fp = fair_price(pp["enstru_code"], pp["unit_norm"], pp["city_code"], month,
                            exclude_bin=lot["customer_bin"])
            if not fp.get("n"):
                items.append({**item, "price_per_unit": round(price, 2), "error": "нет аналогов у других ведомств"})
                continue
            dev = (price - fp["fair_price"]) / fp["fair_price"]
            items.append({**item, "price_per_unit": round(price, 2), "fair": fp, "deviation": round(dev, 6),
                          "verdict": _verdict(dev),
                          "confidence": _confidence(fp["n"], fp["n_city"], fp["n_customers"])})

        ok = [i for i in items if not i.get("error")]
        ok.sort(key=lambda i: -abs(i["deviation"]))
        for i in ok[:MAX_COMPARABLE_ITEMS]:
            # один договор может содержать несколько позиций с тем же кодом: берём ближайшую к Fair Price
            i["comparables"] = _q("""
                SELECT * FROM (
                    SELECT DISTINCT ON (contract_id)
                           contract_id, contract_number, contract_url, customer_name, supplier_name,
                           sign_date, unit_price_norm AS unit_price, price_real, city_code
                    FROM mv_price_base
                    WHERE enstru_code = %(e)s AND unit_norm = %(u)s AND customer_bin <> %(x)s
                    ORDER BY contract_id, abs(price_real - %(fp)s)) t
                ORDER BY (city_code = %(c)s) DESC, abs(price_real - %(fp)s) LIMIT 5""",
                {"e": i["enstru_code"], "u": i["unit_norm"], "x": lot["customer_bin"],
                 "c": i["city_code"], "fp": i["fair"]["fair_price"]})
        summary = {"n_items": len(items), "n_assessed": len(ok)}
        if ok:
            # отклонение лота: фактическая стоимость оценённых пунктов против стоимости по Fair Price
            cost = sum(i["price_per_unit"] * (i["quantity"] or 0) for i in ok)
            fair_cost = sum(i["fair"]["fair_price"] * (i["quantity"] or 0) for i in ok)
            dev = (cost - fair_cost) / fair_cost if fair_cost else None
            # уверенность лота: преобладающая среди пунктов; если оценено меньше половины пунктов, на уровень ниже
            order = ["низкая", "средняя", "высокая"]
            confs = [i["confidence"] for i in ok]
            lvl = order.index(max(set(confs), key=lambda c: (confs.count(c), -order.index(c))))
            if len(ok) < len(items) / 2:
                lvl = max(lvl - 1, 0)
            summary.update(cost_assessed=round(cost, 2), fair_cost=round(fair_cost, 2),
                           deviation=round(dev, 6) if dev is not None else None,
                           verdict=_verdict(dev) if dev is not None else None,
                           overpay_vs_fair=round(cost - fair_cost, 2),
                           by_verdict={v: sum(1 for i in ok if i["verdict"] == v) for v in
                                       ("в пределах рынка", "требует внимания", "существенно выше рынка",
                                        "существенно ниже рынка")},
                           confidence=order[lvl])
        results.append({**lot, "lot_summary": summary, "items": ok + [i for i in items if i.get("error")]})

    limitations = ["Сравнение только с другими ведомствами (заказчик лота исключён из выборки). "
                   "Уверенность высокая при N ≥ 20, n_город ≥ 3 и не меньше 3 ведомствах; если аналоги только "
                   "одного ведомства, уверенность низкая.",
                   "Город поставки определяется по КАТО; если в плане его нет, берётся из лота или адреса заказчика (поле kato_source).",
                   "Лот из нескольких пунктов плана оценивается по каждому пункту отдельно; отклонение лота взвешено "
                   "по количеству и считается только по пунктам, для которых нашлись аналоги."]
    if note := _inflation_note():
        limitations.append(note)
    return {
        "question_type": "fairness",
        "params": {"lot": lot_ref, "period": YEARS},
        "method": ("Fair Price = P_base × K_region × K_season. P_base: медиана цен за единицу других "
                   "ведомств по тому же коду ЕНС ТРУ и единице. K_region: отношение медианы в городе "
                   "поставки к P_base, сглаженное к 1 по числу наблюдений в городе. K_season: сезонный "
                   "коэффициент раздела КПВЭД для месяца закупки. Цена пункта: из позиции договора, иначе "
                   "сумма пункта плана / количество. Пороги: до 15% норма, 15–30% внимание, больше 30% "
                   "существенное отклонение."),
        "results": results,
        "price_base": price_base(),
        "limitations": limitations,
    }


# ---------------- 3. завышение объёмов ----------------

def find_volume_spikes(customer_bin=None, enstru_code=None, year=2026, min_ratio=2.0,
                       min_amount=1_000_000, top_k=10, include_services=False):
    """Пункты плана, где количество в году year выросло в min_ratio+ раз против медианы прошлых лет.
    По умолчанию только товары: у услуг «количество» обычно = число пунктов плана, а не объём."""
    where = ["plan_year = %(y)s", "growth_ratio >= %(r)s", "amount >= %(a)s"]
    if not include_services and not enstru_code:
        where.append("NOT is_service")
    p = {"y": year, "r": min_ratio, "a": min_amount, "k": top_k}
    if customer_bin:
        where.append("customer_bin = %(bin)s"); p["bin"] = customer_bin
    if enstru_code:
        where.append("enstru_code = %(e)s"); p["e"] = enstru_code
    w = " AND ".join(where)
    checked = _q("SELECT count(*) AS n FROM v_volume_growth WHERE plan_year = %(y)s"
                 + ("" if include_services or enstru_code else " AND NOT is_service")
                 + (" AND customer_bin = %(bin)s" if customer_bin else ""), p)[0]["n"]
    rows = _q(f"""
        SELECT customer_bin, customer_name, enstru_code, enstru_name, unit_name, plan_year,
               qty, amount, prev_years, prev_qty, median_prev_qty, growth_ratio,
               qty_increase, extra_amount, plan_point_ids[1:5] AS plan_point_ids
        FROM v_volume_growth WHERE {w}
        ORDER BY extra_amount DESC NULLS LAST LIMIT %(k)s""", p)
    n_flag = _q(f"SELECT count(*) AS n FROM v_volume_growth WHERE {w}", p)[0]["n"]
    # прямые ссылки: последние объявления по пунктам плана (если пункт уже выставлялся на торги)
    for r in rows:
        r["announcements"] = _q("""
            SELECT DISTINCT ON (a.id) a.number_anno, a.url, a.publish_date
            FROM lot_plan_points lp JOIN lots l ON l.id = lp.lot_id
            JOIN announcements a ON a.id = l.announcement_id
            WHERE lp.plan_point_id = ANY(%(ids)s)
            ORDER BY a.id DESC LIMIT 3""", {"ids": r["plan_point_ids"]}) if r["plan_point_ids"] else []
    return {
        "question_type": "volumes",
        "params": {"year": year, "customer_bin": customer_bin, "enstru_code": enstru_code,
                   "min_ratio": min_ratio, "min_amount_kzt": min_amount},
        "method": (f"Для каждой пары (заказчик, код ЕНС ТРУ) плановое количество в {year} году "
                   f"сравнивается с медианой количества за предыдущие годы. Нетипичное завышение: "
                   f"рост в {min_ratio} раза и больше при сумме от {min_amount:,} тг. Сортировка по "
                   f"сумме, приходящейся на прирост: amount × (1 − медиана прошлых лет / текущее количество)."),
        "summary": {"pairs_checked": checked, "n_flagged": n_flag},
        "results": rows,
        "confidence": "средняя" if checked >= 20 else "низкая",
        "limitations": ["История только 2024–2026, то есть максимум 2 предыдущих года для сравнения.",
                        "Используются плановые количества; план мог корректироваться в течение года.",
                        "Рост может быть обоснован (новые подразделения, разовые проекты): нужна ручная проверка.",
                        "Услуги и работы по умолчанию исключены: их количество отражает число пунктов плана, а не объём."],
    }


# ---------------- 4. поиск и сравнение ----------------

def search_purchases(text, customer_bin=None, year=None, top_k=10):
    """Поиск лотов по словам в названии/описании."""
    # каждое слово ищется отдельно по корню: «бумага для офиса» находит «бумага для офисного оборудования»
    words = [w for w in re.findall(r"\w+", _norm_text(text)) if len(w) >= 3 and w not in SEARCH_STOP] \
        or [text.lower().strip()]
    p = {f"t{i}": f"%{_stem(w)}%" for i, w in enumerate(words)}
    p["k"] = top_k
    where = [f"l.name_clean LIKE %(t{i})s" for i in range(len(words))]
    if customer_bin:
        where.append("l.customer_bin = %(bin)s"); p["bin"] = customer_bin
    if year:
        where.append("EXTRACT(YEAR FROM a.publish_date) = %(y)s"); p["y"] = year
    w = " AND ".join(where)
    # один пункт плана может выставляться на торги повторно (закупка не состоялась, смена способа):
    # номер лота = <id пункта плана>-<суффикс>, поэтому группируем по части до дефиса
    # и берём последнюю публикацию
    latest = f"""
        SELECT DISTINCT ON (split_part(l.lot_number, '-', 1), l.customer_bin)
               l.id AS lot_id, l.lot_number, l.name_ru, l.description_ru, l.customer_bin,
               l.count, l.amount, l.status_name, a.number_anno, a.publish_date, a.url,
               count(*) OVER (PARTITION BY split_part(l.lot_number, '-', 1), l.customer_bin) AS n_publications
        FROM lots l JOIN announcements a ON a.id = l.announcement_id
        WHERE {w}
        ORDER BY split_part(l.lot_number, '-', 1), l.customer_bin, a.publish_date DESC"""
    total = _q(f"SELECT count(*) AS n, sum(amount) AS amount FROM ({latest}) t", p)[0]
    rows = _q(f"SELECT * FROM ({latest}) t ORDER BY amount DESC NULLS LAST LIMIT %(k)s", p)
    return {"question_type": "search", "params": {"text": text, "customer_bin": customer_bin,
            "year": year, "period": YEARS}, "method": "Поиск по корням слов запроса (все слова должны встретиться) в очищенном названии и описании лота; повторные публикации одного лота схлопнуты до последней.",
            "summary": {"n_found": total["n"], "total_amount": total["amount"]}, "results": rows,
            "limitations": ["Поиск по тексту, синонимы и казахские названия не сопоставляются."]}


def compare_customers(enstru_code, unit_norm=None):
    """Сравнение цен по одному коду ЕНС ТРУ между заказчиками. Цены сравнимы только в одной единице,
    поэтому без unit_norm берётся самая частая единица кода, остальные перечисляются в ограничениях."""
    units = _q("SELECT s.unit_norm, u.norm_name AS unit_name, s.n FROM mv_enstru_stats s "
               "LEFT JOIN (SELECT DISTINCT norm_code, norm_name FROM ref_units) u ON u.norm_code = s.unit_norm "
               "WHERE s.enstru_code = %(e)s ORDER BY s.n DESC", {"e": enstru_code})
    if unit_norm is None and units:
        unit_norm = units[0]["unit_norm"]
    other_units = [u for u in units if u["unit_norm"] != unit_norm]
    p = {"e": enstru_code, "u": unit_norm}
    rows = _q("""
        SELECT customer_bin, customer_name, unit_name, count(*) AS n_items,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY price_real) AS median_price,
               sum(price_real * quantity_norm) / NULLIF(sum(quantity_norm), 0) AS wavg_price,
               min(price_real) AS min_price, max(price_real) AS max_price,
               sum(quantity_norm) AS total_qty
        FROM mv_price_base
        WHERE enstru_code = %(e)s AND (%(u)s::TEXT IS NULL OR unit_norm = %(u)s)
        GROUP BY 1, 2, 3 ORDER BY median_price DESC""", p)
    overall = _q("SELECT n, median, wavg FROM mv_enstru_stats WHERE enstru_code = %(e)s "
                 "AND (%(u)s::TEXT IS NULL OR unit_norm = %(u)s)", p)
    limitations = [n for n in [_inflation_note()] if n]
    if other_units:
        limitations.append("Сравнение в одной единице измерения; позиции в других единицах не учтены: "
                           + ", ".join(f"{u['unit_name'] or u['unit_norm']} (N = {u['n']})" for u in other_units) + ".")
    return {"question_type": "comparison", "price_base": price_base(),
            "params": {"enstru_code": enstru_code, "unit_norm": unit_norm, "period": YEARS},
            "method": "Медиана и средневзвешенная цена за единицу по каждому заказчику.",
            "summary": overall, "results": rows,
            "confidence": _confidence(sum(r["n_items"] for r in rows)),
            "limitations": limitations}


# ---------------- 5. карточка договора ----------------

SINGLE_SOURCE_METHODS = (6, 23, 131)   # «из одного источника» (по несостоявшимся, прямое заключение, питание)


def contract_lookup(ref, customer_bin=None):
    """Факты по договору: номер, даты, стороны, суммы, способ закупки, статус, позиции, доп. соглашения.
    ref: ID договора, номер договора, номер объявления или номер лота. Номера договоров не уникальны
    («1» встречается сотни раз): без заказчика при нескольких совпадениях возвращается список кандидатов."""
    r = str(ref).strip()
    p = {"r": r, "bin": customer_bin}
    roots = _q("""
        SELECT DISTINCT c.root_id FROM contracts c
        LEFT JOIN contract_items ci ON ci.contract_id = c.id
        LEFT JOIN lots l ON l.id = ci.lot_id
        WHERE (c.id::TEXT = %(r)s OR c.contract_number = %(r)s OR c.number_anno = %(r)s OR l.lot_number = %(r)s)
          AND (%(bin)s::TEXT IS NULL OR c.customer_bin = %(bin)s)""", p)
    base = {"question_type": "contract", "params": {"ref": r, "customer_bin": customer_bin, "period": YEARS}}
    if not roots:
        return {**base, "error": f"Договор {r} не найден в хранилище (период {YEARS}, 27 организаций из перечня)."}
    ids = [x["root_id"] for x in roots]
    head_sql = """
        SELECT c.id AS contract_id, c.root_id, c.contract_number, c.sign_date, c.fin_year,
               c.customer_bin, s.name_ru AS customer_name, c.supplier_bin, c.supplier_name,
               c.contract_sum AS sum_without_vat, c.contract_sum_wnds AS sum_with_vat, c.fakt_sum, c.currency,
               tm.name_ru AS trade_method, st.name_ru AS status, tm.name_kz AS trade_method_kz,
               st.name_kz AS status_kz, c.number_anno,
               'https://goszakup.gov.kz/ru/egzcontract/cpublic/show/' || c.id AS contract_url,
               CASE WHEN c.announcement_id IS NOT NULL
                    THEN 'https://goszakup.gov.kz/ru/announce/index/' || c.announcement_id END AS announcement_url,
               v.n_versions, v.first_sign_date, v.first_sum_with_vat
        FROM contracts c
        LEFT JOIN subjects s ON s.bin = c.customer_bin
        LEFT JOIN ref_trade_methods tm ON tm.id = c.trade_method_id
        LEFT JOIN ref_contract_status st ON st.id = c.status_id
        JOIN LATERAL (
            SELECT count(*) AS n_versions, min(o.sign_date) AS first_sign_date,
                   (array_agg(o.contract_sum_wnds ORDER BY o.sign_date, o.id))[1] AS first_sum_with_vat
            FROM contracts o WHERE o.root_id = c.root_id) v ON TRUE
        WHERE c.is_latest AND c.root_id = ANY(%(ids)s)
        ORDER BY c.sign_date DESC"""
    heads = _q(head_sql, {"ids": ids})
    if len(heads) > 1:
        return {**base, "ambiguous": True, "n_candidates": len(heads), "results": heads[:20],
                "limitations": ["Номер договора не уникален: укажите заказчика (название или БИН) или ID договора."]}
    c = heads[0]
    c["items"] = _q("""
        SELECT ci.id AS item_id, COALESCE(pp.name_ru, e.name_ru) AS item_name, ci.enstru_code,
               u.norm_name AS unit_name, ci.quantity, ci.unit_price AS unit_price_without_vat,
               ci.unit_price_wnds AS unit_price_with_vat, ci.total_sum_wnds AS total_with_vat,
               l.lot_number
        FROM contract_items ci
        LEFT JOIN plan_points pp ON pp.id = ci.plan_point_id
        LEFT JOIN ref_enstru e ON e.code = ci.enstru_code
        LEFT JOIN ref_units u ON u.code = pp.unit_code
        LEFT JOIN lots l ON l.id = ci.lot_id
        WHERE ci.contract_id = %(c)s ORDER BY ci.total_sum_wnds DESC NULLS LAST""", {"c": c["contract_id"]})
    limitations = ["Показана последняя редакция договора (с учётом дополнительных соглашений); "
                   "первоначальная сумма и дата приведены отдельно."]
    if c["currency"] != "KZT":
        limitations.append(f"Договор в валюте {c['currency']}, суммы не пересчитаны в тенге.")
    if any(i["unit_name"] is None for i in c["items"]):
        limitations.append("У части позиций не найден пункт плана, единица измерения неизвестна.")
    return {**base, "method": "Прямое чтение записи договора и его позиций из хранилища, без расчётов.",
            "results": [c], "limitations": limitations}


# ---------------- 6. аналитика закупок ----------------

def purchase_analytics(customer_bin=None, year=None, enstru_code=None, supplier_bin=None, top_k=5):
    """Сводка по договорам: количество, суммы, разбивка по годам и способам закупки, доля закупок
    из одного источника, топ поставщиков, заказчиков и кодов ЕНС ТРУ. Только последние редакции договоров в тенге.
    С кодом ЕНС ТРУ суммы считаются по позициям договоров с этим кодом, иначе по договорам целиком."""
    p = {"bin": customer_bin, "y": year, "e": enstru_code, "s": supplier_bin, "k": top_k,
         "ss": list(SINGLE_SOURCE_METHODS)}
    filt = """c.is_latest AND c.currency = 'KZT'
              AND (%(bin)s::TEXT IS NULL OR c.customer_bin = %(bin)s)
              AND (%(s)s::TEXT IS NULL OR c.supplier_bin = %(s)s)
              AND (%(y)s::INT IS NULL OR EXTRACT(YEAR FROM c.sign_date) = %(y)s)"""
    if enstru_code:
        base = f"""SELECT c.id AS contract_id, c.customer_bin, c.supplier_bin, c.supplier_name, c.trade_method_id,
                          c.sign_date, ci.total_sum_wnds AS amount, ci.enstru_code
                   FROM contracts c JOIN contract_items ci ON ci.contract_id = c.id
                   WHERE {filt} AND ci.enstru_code = %(e)s"""
    else:
        base = f"""SELECT c.id AS contract_id, c.customer_bin, c.supplier_bin, c.supplier_name, c.trade_method_id,
                          c.sign_date, c.contract_sum_wnds AS amount, NULL::TEXT AS enstru_code
                   FROM contracts c WHERE {filt}"""
    w = f"WITH b AS ({base})"
    total = _q(w + """
        SELECT count(DISTINCT contract_id) AS n_contracts, sum(amount) AS total_with_vat,
               count(DISTINCT supplier_bin) AS n_suppliers, count(DISTINCT customer_bin) AS n_customers,
               sum(amount) FILTER (WHERE trade_method_id = ANY(%(ss)s)) AS single_source_with_vat,
               count(DISTINCT contract_id) FILTER (WHERE trade_method_id = ANY(%(ss)s)) AS n_single_source,
               min(sign_date) AS first_date, max(sign_date) AS last_date
        FROM b""", p)[0]
    out = {"question_type": "analytics",
           "params": {"customer_bin": customer_bin, "year": year, "enstru_code": enstru_code,
                      "supplier_bin": supplier_bin, "period": YEARS},
           "summary": total}
    if not total["n_contracts"]:
        return {**out, "results": [], "limitations": ["Договоров по заданным фильтрам нет."]}
    tot = total["total_with_vat"] or 0
    total["single_source_share"] = round((total["single_source_with_vat"] or 0) / tot, 6) if tot else None
    out["by_year"] = _q(w + """SELECT EXTRACT(YEAR FROM sign_date)::INT AS year, count(DISTINCT contract_id) AS n_contracts,
                                      sum(amount) AS total_with_vat FROM b GROUP BY 1 ORDER BY 1""", p)
    out["by_method"] = _q(w + """
        SELECT COALESCE(tm.name_ru, 'id ' || b.trade_method_id) AS trade_method, max(tm.name_kz) AS trade_method_kz,
               count(DISTINCT contract_id) AS n_contracts,
               sum(amount) AS total_with_vat, round(sum(amount) / NULLIF(%(tot)s, 0), 6) AS share
        FROM b LEFT JOIN ref_trade_methods tm ON tm.id = b.trade_method_id
        GROUP BY 1 ORDER BY total_with_vat DESC NULLS LAST""", {**p, "tot": tot})
    out["top_suppliers"] = _q(w + """
        SELECT supplier_bin, max(supplier_name) AS supplier_name, count(DISTINCT contract_id) AS n_contracts,
               sum(amount) AS total_with_vat, round(sum(amount) / NULLIF(%(tot)s, 0), 6) AS share
        FROM b GROUP BY 1 ORDER BY total_with_vat DESC NULLS LAST LIMIT %(k)s""", {**p, "tot": tot})
    if not customer_bin:
        out["top_customers"] = _q(w + """
            SELECT b.customer_bin, max(s.name_ru) AS customer_name, count(DISTINCT contract_id) AS n_contracts,
                   sum(amount) AS total_with_vat, round(sum(amount) / NULLIF(%(tot)s, 0), 6) AS share
            FROM b LEFT JOIN subjects s ON s.bin = b.customer_bin GROUP BY 1 ORDER BY total_with_vat DESC NULLS LAST LIMIT %(k)s""",
            {**p, "tot": tot})
    if not enstru_code:
        out["top_enstru"] = _q(f"""
            SELECT ci.enstru_code, max(e.name_ru) AS enstru_name, count(DISTINCT c.id) AS n_contracts,
                   sum(ci.total_sum_wnds) AS total_with_vat
            FROM contracts c JOIN contract_items ci ON ci.contract_id = c.id
            LEFT JOIN ref_enstru e ON e.code = ci.enstru_code
            WHERE {filt} GROUP BY 1 ORDER BY total_with_vat DESC NULLS LAST LIMIT %(k)s""", p)
    # крупнейшие договоры: прямые ссылки для раздела Top-K
    out["results"] = _q(w + """
        SELECT b.contract_id, c.contract_number, c.sign_date, b.customer_bin, b.supplier_bin, b.supplier_name,
               sum(b.amount) AS amount_with_vat, tm.name_ru AS trade_method, tm.name_kz AS trade_method_kz,
               'https://goszakup.gov.kz/ru/egzcontract/cpublic/show/' || b.contract_id AS contract_url
        FROM b JOIN contracts c ON c.id = b.contract_id LEFT JOIN ref_trade_methods tm ON tm.id = c.trade_method_id
        GROUP BY 1, 2, 3, 4, 5, 6, 8, 9 ORDER BY 7 DESC NULLS LAST LIMIT %(k)s""", p)
    out["method"] = ("Агрегация последних редакций договоров (доп. соглашения учтены), подписанных в " + YEARS +
                     ". Суммы с НДС" + (", по позициям договоров с кодом " + enstru_code if enstru_code else
                     ", по договорам целиком") + ". Из одного источника: способы закупки «Из одного источника "
                     "по несостоявшимся закупкам», «путём прямого заключения договора» (в т.ч. по питанию).")
    lim = ["Учтены только договоры в тенге (1 договор в USD исключён).",
           "Год определяется по дате подписания договора."]
    if year in (None, 2026):
        last = _q("SELECT max(sign_date) AS d FROM contracts")[0]["d"]
        lim[1] += f" 2026 год неполный: договоры в хранилище по {last}."
    if note := _inflation_note():
        lim.append(note)
    out["limitations"] = lim
    out["confidence"] = "высокая"
    return out


# ---------------- справочники для агента ----------------

def resolve_enstru(text, top_k=5):
    """Коды ЕНС ТРУ по словам из названия (RU или KZ), сортировка по числу позиций договоров.
    Нужен, когда в вопросе товар назван словами, а не кодом."""
    words = [w for w in re.findall(r"\w+", _norm_text(text)) if len(w) >= 3 and w not in SEARCH_STOP]
    if not words:
        return []
    # корни слов; казахские буквы сравниваются без диакритики (модель может написать «кагаз» вместо «қағаз»)
    stems = [_stem(w).translate(KZ_PLAIN) for w in words]
    # корень должен начинать слово: «кагаз» не находит «силикагель»
    conds = " AND ".join(f"(lower(r.name_ru) ~ %(w{i})s OR translate(lower(r.name_kz), 'әғқңөұүһіё', "
                         f"'агкноуухие') ~ %(w{i})s)" for i in range(len(stems)))
    p = {f"w{i}": r"\m" + re.escape(s) for i, s in enumerate(stems)}
    p["k"] = top_k
    return _q(f"""
        SELECT r.code AS enstru_code, r.name_ru, r.name_kz, COALESCE(sum(s.n), 0) AS n_items
        FROM ref_enstru r LEFT JOIN mv_enstru_stats s ON s.enstru_code = r.code
        WHERE {conds}
        GROUP BY 1, 2, 3 ORDER BY n_items DESC, r.code LIMIT %(k)s""", p)


def data_freshness():
    """На какой момент данные из API в хранилище: начало последней успешно загруженной выгрузки (etl_runs)."""
    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
            cur.execute("""SELECT data_as_of, mode FROM etl_runs WHERE status = 'ok' AND data_as_of IS NOT NULL
                           ORDER BY data_as_of DESC LIMIT 1""")
            r = cur.fetchone()
    except psycopg2.Error:
        return None
    if not r:
        return None
    # data_as_of записан по местному времени процесса update.py (часы БД в контейнере в UTC)
    return {"as_of": r[0].strftime("%Y-%m-%d %H:%M"), "mode": r[1],
            "hours_ago": round((datetime.now() - r[0]).total_seconds() / 3600, 1)}


def list_subjects():
    return _q("SELECT bin, name_ru FROM subjects ORDER BY bin")


# ---------------- CLI для проверки ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("anomalies"); a.add_argument("--enstru"); a.add_argument("--bin")
    a.add_argument("--year", type=int); a.add_argument("--top", type=int, default=5)
    b = sp.add_parser("lot"); b.add_argument("ref")
    c = sp.add_parser("volumes"); c.add_argument("--bin"); c.add_argument("--year", type=int, default=2026)
    d = sp.add_parser("search"); d.add_argument("text"); d.add_argument("--bin")
    e = sp.add_parser("compare"); e.add_argument("enstru")
    f = sp.add_parser("contract"); f.add_argument("ref"); f.add_argument("--bin")
    g = sp.add_parser("analytics"); g.add_argument("--bin"); g.add_argument("--year", type=int)
    g.add_argument("--enstru"); g.add_argument("--supplier")
    args = ap.parse_args()

    if args.cmd == "anomalies":
        out = find_price_anomalies(args.enstru, args.bin, args.year, top_k=args.top)
    elif args.cmd == "lot":
        out = assess_lot_price(args.ref)
    elif args.cmd == "volumes":
        out = find_volume_spikes(args.bin, year=args.year)
    elif args.cmd == "search":
        out = search_purchases(args.text, args.bin)
    elif args.cmd == "contract":
        out = contract_lookup(args.ref, args.bin)
    elif args.cmd == "analytics":
        out = purchase_analytics(args.bin, args.year, args.enstru, args.supplier)
    else:
        out = compare_customers(args.enstru)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
