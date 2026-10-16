"""
Аналитический агент по госзакупкам: вопрос на RU/KZ -> вызов функции tools.py -> ответ из 6 пунктов.

Конвейер:
  1. route()    LLM (tool calling) выбирает функцию и параметры; регулярки подстраховывают
                извлечение БИН, номеров лотов, кодов ЕНС ТРУ и года; БИН сверяются со справочником.
  2. run_tool() вызывает функцию tools.py, результат JSON.
  3. render()   пункты 2–6 собираются кодом из JSON (цифры не проходят через LLM),
                пункт 1 (краткий вывод) пишет LLM по готовому списку фактов.
  4. verify()   каждое число, дата и БИН в выводе LLM должны быть в JSON инструмента,
                иначе перегенерация (до 2 раз), затем детерминированный шаблон.

LLM: любой OpenAI-совместимый сервер (Ollama, vLLM). Настройки в .env:
  LLM_BASE_URL (по умолчанию http://localhost:11434/v1), LLM_MODEL, LLM_API_KEY.

Запуск:
  python agent.py "Найди закупки с отклонением цены больше 30% от средневзвешенной"
  python agent.py --json "..."              полный ответ с JSON инструмента и проверкой
  uvicorn agent:app --port 8080             HTTP: POST /ask {"question": "..."}
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

import tools

load_dotenv()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
MAX_RETRIES = 2
# template: краткий вывод собирается кодом (по умолчанию, для малых моделей);
# llm: пишет LLM по списку фактов с проверкой чисел и откатом на шаблон
SUMMARY_MODE = os.getenv("SUMMARY_MODE", "template")
DEFAULT_TOP_K = 10
COMPARE_MIN_N = 3        # минимум позиций у заказчика, чтобы считать его самым дорогим/дешёвым
MAX_ITEMS = 10           # пунктов лота в таблице
MAX_LOTS = 10            # лотов объявления в подробном разделе «Сравнение»

# max_retries=1: повтор после тайм-аута на CPU только добавляет нагрузку (брошенные запросы модель ещё считает)
llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT, max_retries=1)
# потолок длины ответа модели: вызов функции занимает десятки токенов, а без потолка модель, не сделавшая вызов,
# может писать текст до конца контекста — на CPU это минуты (16.10.2026 eval в Docker упёрся в тайм-ауты 180 с)
ROUTE_MAX_TOKENS = int(os.getenv("LLM_ROUTE_MAX_TOKENS", "256"))
SUMMARY_MAX_TOKENS = int(os.getenv("LLM_SUMMARY_MAX_TOKENS", "400"))


# ---------------- язык ----------------

KZ_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
KZ_WORDS = {"қандай", "бойынша", "жылы", "сатып", "алу", "баға", "бағасы", "көрсет", "тап", "салыстыр"}


def detect_lang(text):
    if any(c in KZ_LETTERS for c in text):
        return "kz"
    return "kz" if set(re.findall(r"\w+", text.lower())) & KZ_WORDS else "ru"


# ---------------- 1. маршрутизация ----------------

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "find_price_anomalies",
        "description": "Ценовые аномалии: позиции договоров, где цена за единицу выше средневзвешенной "
                       "по коду ЕНС ТРУ больше чем на порог (по умолчанию 30%). Вопросы про завышенные "
                       "цены, переплаты, отклонения цены по многим закупкам.",
        "parameters": {"type": "object", "properties": {
            "customer_bin": {"type": "string", "description": "БИН заказчика (12 цифр), если указана организация"},
            "enstru_code": {"type": "string", "description": "код ЕНС ТРУ (формат 000000.000.000000), только если он написан в вопросе"},
            "product": {"type": "string", "description": "товар словами на русском в именительном падеже, если указан без кода"},
            "year": {"type": "integer", "description": "год подписания договора"},
            "threshold": {"type": "number", "description": "порог отклонения долей: 30% = 0.3"},
            "top_k": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "assess_lot_price",
        "description": "Справедливость (адекватность) цены конкретного лота или объявления относительно "
                       "договоров других ведомств в том же городе поставки (метрика Fair Price).",
        "parameters": {"type": "object", "properties": {
            "lot_ref": {"type": "string", "description": "номер лота (например 83307719-ОК2), ID лота или номер объявления (16831569-1)"}},
            "required": ["lot_ref"]}}},
    {"type": "function", "function": {
        "name": "find_volume_spikes",
        "description": "Нетипичное завышение количества (объёма) закупаемых ТРУ по сравнению с предыдущими годами.",
        "parameters": {"type": "object", "properties": {
            "customer_bin": {"type": "string"},
            "enstru_code": {"type": "string"},
            "product": {"type": "string", "description": "товар словами на русском, если указан без кода"},
            "year": {"type": "integer", "description": "проверяемый год плана, по умолчанию 2026"},
            "min_ratio": {"type": "number", "description": "во сколько раз вырос объём, по умолчанию 2"},
            "top_k": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "search_purchases",
        "description": "Поиск закупок (лотов) по названию товара/услуги: что, кто и на какую сумму закупал.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "ключевые слова на русском, 1–3 слова, корень без окончания"},
            "customer_bin": {"type": "string"},
            "year": {"type": "integer"},
            "top_k": {"type": "integer"}}, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "compare_customers",
        "description": "Сравнение цен на один товар (код ЕНС ТРУ) между заказчиками: медианы и средневзвешенные.",
        "parameters": {"type": "object", "properties": {
            "enstru_code": {"type": "string"},
            "product": {"type": "string", "description": "товар словами на русском, если код не указан"}}}}},
]
TOOL_SPECS += [
    {"type": "function", "function": {
        "name": "contract_lookup",
        "description": "Факты по конкретному договору: дата подписания, заказчик, поставщик, сумма, способ закупки, "
                       "статус, позиции, доп. соглашения. Вопросы «кто поставщик по договору №…», «на какую сумму "
                       "договор …», «когда подписан договор …».",
        "parameters": {"type": "object", "properties": {
            "ref": {"type": "string", "description": "номер или ID договора, номер объявления или лота, как в вопросе"},
            "customer_bin": {"type": "string"}}, "required": ["ref"]}}},
    {"type": "function", "function": {
        "name": "purchase_analytics",
        "description": "Аналитика закупок: сколько договоров и на какую сумму, разбивка по годам и способам закупки, "
                       "доля закупок из одного источника, крупнейшие поставщики, заказчики и товары.",
        "parameters": {"type": "object", "properties": {
            "customer_bin": {"type": "string"},
            "supplier_bin": {"type": "string", "description": "БИН поставщика, только если написан в вопросе"},
            "year": {"type": "integer"},
            "product": {"type": "string", "description": "товар словами на русском, если вопрос про конкретный товар"},
            "top_k": {"type": "integer"}}}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}


@lru_cache(maxsize=1)
def subjects():
    return {r["bin"]: r["name_ru"] for r in tools.list_subjects()}


ABBR = [
    (r"Республиканское государственное предприятие на праве хозяйственного ведения", "РГП на ПХВ"),
    (r"республиканского государственного предприятия на праве хозяйственного ведения", "РГП на ПХВ"),
    (r"Республиканское государственное учреждение", "РГУ"),
    (r"Республиканское государственное предприятие", "РГП"),
    (r"Государственное учреждение", "ГУ"),
    (r"Акционерное общество|Акционерного общества", "АО"),
    (r"Управления [Дд]елами Президента Республики Казахстан", "УДП РК"),
    (r"Медицинского центра", "МЦ"),
    (r"Администрации Президента и Правительства Республики Казахстан", "АП и Правительства РК"),
]


def _short(name):
    s = re.sub(r"\s+", " ", name or "")
    for pat, rep in ABBR:
        s = re.sub(pat, rep, s)
    return s if len(s) <= 120 else s[:117] + "…"


def _router_prompt():
    orgs = "\n".join(f"{b}: {_short(n)}" for b, n in subjects().items())
    return (
        "Ты маршрутизатор аналитического агента по госзакупкам Казахстана (данные 2024–2026, 27 организаций). "
        "Вопрос может быть на русском или казахском. Выбери ровно одну функцию и заполни только те параметры, "
        "которые явно есть в вопросе. Ничего не придумывай. Организацию переводи в БИН только по списку ниже.\n"
        "Правила: «отклонение/завышение цены по многим закупкам» -> find_price_anomalies; "
        "«адекватна ли цена лота №…» -> assess_lot_price; «завышение количества/объёма по сравнению с прошлыми годами» -> "
        "find_volume_spikes; «найди/покажи закупки …» -> search_purchases; «сравни цены на … между заказчиками», "
        "«у кого из заказчиков цена выше/ниже», «кто покупает дороже» -> compare_customers; "
        "«кто поставщик / когда подписан / какая сумма договора №…» -> contract_lookup; «сколько договоров, на какую "
        "сумму, доля из одного источника, крупнейшие поставщики, структура закупок», «шарттар саны», «жалпы сомасы», "
        "«бір көзден үлесі» -> purchase_analytics. search_purchases только если назван конкретный товар.\n"
        "Параметры product и text пиши на русском, даже если вопрос на казахском (жанармай -> топливо, қағаз -> бумага).\n"
        f"Организации (БИН: название):\n{orgs}")


RE_BIN = re.compile(r"(?<!\d)\d{12}(?!\d)")
RE_ENSTRU = re.compile(r"(?<![\d.])\d{6}\.\d{3}\.\d{6}(?![\d.])")
# 83307719-ОК2, 33431522-ОЛ-ЗЦП1
RE_LOT = re.compile(r"(?<![\w-])\d{6,9}(?:-[A-Za-zА-Яа-яЁё]{1,4})+\d{1,3}(?![\w-])")
RE_ANNO = re.compile(r"(?<![\w-])\d{6,9}-\d{1,2}(?![\w-])")
RE_YEAR = re.compile(r"(?<!\d)(202[4-6])(?!\d)")
RE_PCT = re.compile(r"(\d{1,3})\s*%")
RE_TOP = re.compile(r"(?:топ|top)[\s-]*(\d{1,2})", re.I)
RE_CONTRACT = [
    re.compile(r"show/(\d{6,9})"),                                                       # ссылка на портал
    re.compile(r"(?:договор\w*|контракт\w*|шарт\w*)\s*(?:№|N|#|номер)\s*([^\s,;?»\")]+)", re.I),
    re.compile(r"№\s*([^\s,;?»\")]+)\s*(?:договор|контракт|шарт)", re.I),
    re.compile(r"(?:договор\w*|контракт\w*|шарт\w*)\s+(?:с\s+)?(?:ID|id)\s*(\d{6,9})"),
    re.compile(r"(?:ID|id)\s*(\d{6,9})\s+(?:договор|шарт)", re.I),
]
RE_RATIO = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:раз|есе|x|×)", re.I)


ORG_STOP = {"на", "по", "из", "за", "до", "от", "ргп", "пхв", "удп", "рк", "гу", "ргу", "ао", "тоо", "республики", "казахстан", "президента",
            "управления", "делами", "государственное", "государственного", "предприятие", "предприятия",
            "учреждение", "праве", "хозяйственного", "ведения", "республиканское", "филиал", "при"}


KZ_TO_RU = str.maketrans("әғқңөұүһіё", "агкноуухие")


def _org_tokens(text):
    s = text.lower().replace("-", "").translate(KZ_TO_RU)
    toks = re.findall(r"[a-zа-я]+", s)
    return {t[:5] for t in toks if (len(t) >= 2 and t not in ORG_STOP)}


@lru_cache(maxsize=1)
def _org_index():
    return {b: _org_tokens(_short(n)) for b, n in subjects().items()}


def match_org(q):
    """БИН по названию организации в вопросе. Слова взвешены по редкости (1 / число организаций со словом):
    «больница» есть у одной организации и весит 1, «мц» у четырёх — 0,25, предлог «на» почти ничего.
    Принимаем только однозначного лидера."""
    idx = _org_index()
    df = defaultdict(int)
    for toks in idx.values():
        for t in toks:
            df[t] += 1
    qt = _org_tokens(q)
    scores = sorted(((sum(1 / df[t] for t in qt & toks), b) for b, toks in idx.items()), reverse=True)
    best, runner = scores[0][0], scores[1][0] if len(scores) > 1 else 0
    if best >= 0.5 and best - runner >= 0.2:
        return scores[0][1]
    return None


def org_candidates(q):
    """Организации, которые подходят под название в вопросе почти одинаково (для уточняющего вопроса)."""
    idx = _org_index()
    df = defaultdict(int)
    for toks in idx.values():
        for t in toks:
            df[t] += 1
    qt = _org_tokens(q)
    scores = sorted(((sum(1 / df[t] for t in qt & toks), b) for b, toks in idx.items()), reverse=True)
    if not scores or scores[0][0] < 0.5:
        return []
    return [b for s, b in scores if s >= scores[0][0] - 0.2][:6]


def regex_extract(q):
    """Детерминированное извлечение идентификаторов: надёжнее LLM для цифр."""
    out = {}
    bins = [b for b in RE_BIN.findall(q) if b in subjects()]
    if bins:
        out["customer_bin"] = bins[0]
    elif b := match_org(q):
        out["customer_bin"] = b
    if m := RE_ENSTRU.search(q):
        out["enstru_code"] = m.group(0)
    if m := RE_LOT.search(q) or RE_ANNO.search(q):
        out["lot_ref"] = m.group(0)
    elif m := re.search(r"(?:лот\w*|lot)\s*(?:№|#|N)?\s*(\d{6,9})(?!\d)", q, re.I):
        out["lot_ref"] = m.group(1)
    if m := RE_YEAR.search(q):
        out["year"] = int(m.group(1))
    if m := RE_PCT.search(q):
        out["threshold"] = int(m.group(1)) / 100
    if m := RE_TOP.search(q):
        out["top_k"] = min(int(m.group(1)), 50)
    if m := RE_RATIO.search(q):
        out["min_ratio"] = float(m.group(1).replace(",", "."))
    for rx in RE_CONTRACT:
        if m := rx.search(q):
            out["ref"] = m.group(1).rstrip(".")
            break
    others = [b for b in RE_BIN.findall(q) if b not in subjects()]
    if others and re.search(r"поставщик|жеткізуші", q, re.I):
        out["supplier_bin"] = others[0]
    return out


KEYWORDS_ANY = (r"закуп|лот|договор|контракт|цен|тендер|объявлен|план|енс|тру|поставщик|заказчик|"
                r"сатып|баға|шарт|хабарландыру|жоспар|жеткізуші|тапсырыс")


def keyword_route(q):
    """Запасной классификатор, если LLM недоступна или не вызвала функцию."""
    s = q.lower()
    if any(rx.search(q) for rx in RE_CONTRACT):
        return "contract_lookup"
    if re.search(r"сколько договор|сумм[аеуы] договоров|доля|структур|крупнейш\w* поставщик|потратил|"
                 r"шарттар\w* (?:саны|сома)|жалпы сома|қанша шарт|үлес|ірі жеткізуші", s):
        return "purchase_analytics"
    if re.search(r"количеств|объём|объем|көлем|саны", s):
        return "find_volume_spikes"
    if re.search(r"лот|адекват|справедлив|fair|хабарландыру", s) and (RE_LOT.search(q) or RE_ANNO.search(q) or re.search(r"\d{6,9}", q)):
        return "assess_lot_price"
    if re.search(r"сравн|салыстыр", s):
        return "compare_customers"
    if re.search(r"отклон|аномал|завыш|переплат|ауытқ|асыра", s):
        return "find_price_anomalies"
    return "search_purchases"


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


# слова, которые не называют товар: «цена», «баға», «договор», «закупки» и т.п. (иначе «баға» → «Бағдарламалық...»)
GENERIC = ("товар", "услуг", "работ", "тру", "тауар", "қызмет", "жұмыс", "закуп", "позиц", "все", "любые",
           "барлық", "сатып", "продукц", "номенклатур", "предмет", "цен", "стоим", "сумм", "договор", "контракт",
           "шарт", "баға", "бағ", "ауытқ", "отклон", "аномал", "переплат", "лот", "объём", "объем", "көлем",
           "количеств", "сан", "поставщ", "жеткіз", "заказчик", "тапсырыс", "рынок", "рыноч", "нарық")


def _is_specific(product):
    """Товар назван конкретно, а не «товары и услуги» / «тауарлар»."""
    words = [w for w in re.findall(r"\w+", product.lower()) if len(w) >= 3 and w not in ("для", "и", "или")]
    return any(not w.startswith(GENERIC) for w in words)


# служебные слова в названиях кодов ЕНС ТРУ: «Медициналық ... бойынша қызмет көрсетулер» не должно совпадать
# с вопросом только потому, что в нём тоже есть «бойынша»
NAME_STOP = {"бойынша", "арналған", "және", "үшін", "туралы", "мен", "бен", "пен", "жылы", "жылғы", "басқа",
             "қамтамасыз", "ету", "көрсету", "көрсетулер", "для", "при", "под", "над", "без", "или", "кроме"}


def _mentioned(product, question):
    """Товар действительно назван в вопросе: корень значимого слова из product или из названия подходящего
    кода ЕНС ТРУ (вопрос на казахском, product от LLM на русском) начинает одно из слов вопроса."""
    q_words = [w.translate(tools.KZ_PLAIN) for w in re.findall(r"\w+", tools._norm_text(question))]
    names = [product] + [c[f] or "" for c in tools.resolve_enstru(product, 5) for f in ("name_ru", "name_kz")]
    for name in names:
        for w in re.findall(r"\w+", tools._norm_text(name)):
            if len(w) < 3 or w in tools.SEARCH_STOP or w in NAME_STOP or w.startswith(GENERIC):
                continue
            stem = tools._stem(w).translate(tools.KZ_PLAIN)
            if any(qw.startswith(stem) for qw in q_words):
                return True
    return False


def route(question):
    """Шаг 1: тип вопроса и параметры. Возвращает (имя функции, аргументы, журнал)."""
    log = {"router": "llm"}
    name, args = None, {}
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[{"role": "system", "content": _router_prompt()},
                      {"role": "user", "content": question}],
            tools=TOOL_SPECS, tool_choice="required", max_tokens=ROUTE_MAX_TOKENS)
        msg = resp.choices[0].message
        if msg.tool_calls:
            name = msg.tool_calls[0].function.name
            args = _parse_args(msg.tool_calls[0].function.arguments)
        elif msg.content:  # часть моделей пишет вызов текстом
            m = re.search(r"\{.*\}", msg.content, re.S)
            j = _parse_args(m.group(0)) if m else {}
            name, args = j.get("name"), _parse_args(j.get("arguments", {}))
    except Exception as e:  # noqa: BLE001 — недоступность LLM не должна ронять агента
        log["llm_error"] = f"{type(e).__name__}: {e}"[:300]
    if name not in TOOL_NAMES:
        name, args, log["router"] = keyword_route(question), {}, "keywords"
        log["keyword_hit"] = bool(re.search(KEYWORDS_ANY, question.lower()))
    log["llm_args"] = dict(args)
    # поиск без текста запроса — признак неверного выбора: переопределяем по ключевым словам
    if name == "search_purchases" and not (args.get("text") or args.get("product")) \
            and (alt := keyword_route(question)) != "search_purchases":
        log["rerouted"] = {"from": name, "to": alt}
        name = alt

    # идентификаторы из текста вопроса важнее LLM. Малые модели «додумывают» параметры
    # (год, top_k, порог), поэтому числовые параметры берём только из самого вопроса (regex_extract),
    # а БИН от LLM принимаем, только если в вопросе есть слова из названия этой организации.
    args = {k: v for k, v in args.items() if v not in (None, "", [], 0)}
    for k in ("year", "top_k", "threshold"):
        if k in args:
            log.setdefault("dropped", {})[k] = args.pop(k)
    if "min_ratio" in args and not re.search(r"\d+(?:[.,]\d+)?\s*(?:раз|есе|x|×)", question, re.I):
        log.setdefault("dropped", {})["min_ratio"] = args.pop("min_ratio")
    for k in ("product", "text"):
        if args.get(k):
            args[k] = tools._norm_text(str(args[k]))
    if args.get("product") and not (_is_specific(args["product"]) and _mentioned(args["product"], question)):
        log.setdefault("dropped", {})["product"] = args.pop("product")
    if args.get("ref") and str(args["ref"]) not in question:
        log.setdefault("dropped", {})["ref"] = args.pop("ref")
    if args.get("supplier_bin") and str(args["supplier_bin"]) not in question:
        log.setdefault("dropped", {})["supplier_bin"] = args.pop("supplier_bin")
    b = str(args.get("customer_bin", ""))
    if b and (b not in subjects() or not (_org_tokens(question) & _org_index()[b])) and b not in question:
        log.setdefault("dropped", {})["customer_bin"] = args.pop("customer_bin")
    elif b and b not in question and match_org(question) is None and len(org_candidates(question)) > 1:
        # под название подходят несколько организаций («Автохозяйство» — их два): LLM выбирает наугад,
        # поэтому БИН отбрасываем, и агент переспрашивает (см. answer)
        log.setdefault("dropped", {})["customer_bin_ambiguous"] = args.pop("customer_bin")
    if "enstru_code" in args and not RE_ENSTRU.fullmatch(str(args["enstru_code"])):
        args.setdefault("product", args.pop("enstru_code"))
    elif "enstru_code" in args and args["enstru_code"] not in question:
        log.setdefault("dropped", {})["enstru_code"] = args.pop("enstru_code")
    args.update(regex_extract(question))
    if name == "find_volume_spikes" and "threshold" in args:
        args.pop("threshold")
    return name, args, log


# ---------------- 2. вызов инструмента ----------------

def run_tool(name, args, lang="ru"):
    a = dict(args)
    resolved = None
    if a.get("product") and not a.get("enstru_code") and name != "search_purchases":
        cands = tools.resolve_enstru(a["product"])
        if cands:
            a["enstru_code"] = cands[0]["enstru_code"]
            resolved = {"product": a["product"], "chosen": cands[0], "alternatives": cands[1:]}
    top_k = int(a.get("top_k") or DEFAULT_TOP_K)

    if name == "find_price_anomalies":
        res = tools.find_price_anomalies(a.get("enstru_code"), a.get("customer_bin"), a.get("year"),
                                         threshold=float(a.get("threshold") or 0.30), top_k=top_k)
    elif name == "assess_lot_price":
        if not a.get("lot_ref"):
            return {"question_type": "fairness", "error": "В вопросе не найден номер лота или объявления."}
        res = tools.assess_lot_price(a["lot_ref"])
    elif name == "find_volume_spikes":
        res = tools.find_volume_spikes(a.get("customer_bin"), a.get("enstru_code"),
                                       year=int(a.get("year") or 2026),
                                       min_ratio=float(a.get("min_ratio") or 2.0), top_k=top_k)
    elif name == "search_purchases":
        text = a.get("text") or a.get("product") or ""
        if not text:
            return {"question_type": "search", "error": "Не указано, что искать."}
        # лоты называются по-русски: казахский запрос сначала переводим через справочник ЕНС ТРУ
        if lang == "kz" and (cands := tools.resolve_enstru(text, 1)) and cands[0]["name_ru"].lower() != text.lower():
            res = tools.search_purchases(cands[0]["name_ru"], a.get("customer_bin"), a.get("year"), top_k=top_k)
            res["params"]["text_original"] = text
            res["derived"] = derived(res)
            return res
        res = tools.search_purchases(text, a.get("customer_bin"), a.get("year"), top_k=top_k)
        if not res["results"] and (cands := tools.resolve_enstru(text, 1)):
            # название могло прийти на казахском: ищем по русскому названию из справочника ЕНС ТРУ
            res = tools.search_purchases(cands[0]["name_ru"], a.get("customer_bin"), a.get("year"), top_k=top_k)
            res["params"]["text_original"] = text
    elif name == "contract_lookup":
        ref = a.get("ref") or a.get("lot_ref")
        if not ref:
            return {"question_type": "contract", "error": "В вопросе не найден номер или ID договора."}
        res = tools.contract_lookup(ref, a.get("customer_bin"))
    elif name == "purchase_analytics":
        res = tools.purchase_analytics(a.get("customer_bin"), a.get("year"), a.get("enstru_code"),
                                       a.get("supplier_bin"), top_k=min(top_k, 10) if a.get("top_k") else 5)
    else:
        if not a.get("enstru_code"):
            return {"question_type": "comparison", "error": "Не удалось определить код ЕНС ТРУ для сравнения."
                    + (f" Товар «{a['product']}» не найден в справочнике." if a.get("product") else "")}
        res = tools.compare_customers(a["enstru_code"])
    if resolved:
        res["enstru_resolved"] = resolved
    res["derived"] = derived(res)
    res["data_as_of"] = tools.data_freshness()
    return res


def derived(res):
    """Величины, которые агент вычисляет из JSON инструмента (доли, отношения). Кладутся в данные,
    чтобы проверка фактов видела их источник."""
    qt, d = res.get("question_type"), {}
    if qt == "anomalies":
        s = res["summary"]
        d["share_anomalies"] = round((s["n_anomalies"] or 0) / s["n_items_checked"], 4) if s["n_items_checked"] else None
    elif qt == "fairness" and res.get("results"):
        ok = [r for r in res["results"] if r["lot_summary"].get("deviation") is not None]
        d["n_lots"] = len(res["results"])
        d["n_lots_assessed"] = len(ok)
        d["lots_by_verdict"] = {v: sum(1 for r in ok if r["lot_summary"]["verdict"] == v) for v in
                                ("в пределах рынка", "требует внимания", "существенно выше рынка", "существенно ниже рынка")}
        if ok:
            d["max_dev_lot"] = max(ok, key=lambda r: r["lot_summary"]["deviation"])["lot_number"]
    elif qt == "analytics" and res.get("results"):
        s = res["summary"]
        d["avg_contract_with_vat"] = round(s["total_with_vat"] / s["n_contracts"], 2) if s["n_contracts"] else None
    elif qt == "comparison" and res.get("results"):
        # крайние значения только среди заказчиков с N >= COMPARE_MIN_N: одна позиция не характеризует цену
        rel = [r for r in res["results"] if r["n_items"] >= COMPARE_MIN_N] or res["results"]
        hi, lo = rel[0], rel[-1]
        d["min_n_for_extremes"] = COMPARE_MIN_N if len(rel) < len(res["results"]) else 1
        d["n_customers_reliable"] = len(rel)
        d["highest_bin"], d["lowest_bin"] = hi["customer_bin"], lo["customer_bin"]
        d["max_to_min_median"] = round(hi["median_price"] / lo["median_price"], 4) if lo["median_price"] else None
    elif qt == "search" and res.get("results"):
        am = sorted(float(r["amount"]) for r in res["results"] if r["amount"])
        if am:
            k = len(am)
            d["shown_median_amount"] = am[k // 2] if k % 2 else (am[k // 2 - 1] + am[k // 2]) / 2
            d["shown_max_amount"] = am[-1]
    return d


# ---------------- форматирование ----------------

def fnum(x, dec=2):
    """Число с пробелами между разрядами: 1 234 567,89 (целые суммы без копеек)."""
    if x is None:
        return "—"
    x = float(x)
    if abs(x) >= 1000 or x == int(x):
        s = f"{x:,.0f}"
    else:
        s = f"{x:,.{dec}f}"
    return s.replace(",", " ").replace(".", ",")


def fpct(x, signed=True):
    if x is None:
        return "—"
    v = float(x) * 100
    s = f"{v:+,.1f}" if signed else f"{v:,.1f}"
    return s.replace(",", " ").replace(".", ",") + "%"


def plural(n, one, few, many):
    """«1 ведомство», «2 ведомства», «5 ведомств»."""
    n = int(n)
    form = one if n % 10 == 1 and n % 100 != 11 else few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many
    return f"{fnum(n)} {form}"


def link(text, url):
    return f"[{text}]({url})" if url else str(text)


def org(bin_, name=None):
    name = name or subjects().get(bin_, "")
    return f"{_short(name)} (БИН {bin_})" if bin_ else _short(name)


T = {
    "ru": {
        "h": ["Краткий вывод", "Использованные данные", "Сравнение", "Метрика / метод оценки",
              "Ограничения и уверенность", "Top-K закупок"],
        "period": "Период", "orgs": "Организации", "all_orgs": "все 27 организаций из перечня",
        "filters": "Фильтры", "none": "нет", "confidence": "Уверенность", "source": "Источник",
        "store": "хранилище goszakup (OWS v3), таблицы договоров, лотов и планов",
        "enstru_res": "Товар «{p}» сопоставлен с кодом ЕНС ТРУ {c} ({n}, позиций: {k})",
        "enstru_alt": "другие коды с похожим названием: {a}",
        "no_results": "Подходящих записей не найдено.",
    },
    "kz": {
        "h": ["Қысқаша қорытынды", "Пайдаланылған деректер", "Салыстыру", "Метрика / бағалау әдісі",
              "Шектеулер және сенімділік", "Top-K сатып алулар"],
        "period": "Кезең", "orgs": "Ұйымдар", "all_orgs": "тізімдегі барлық 27 ұйым",
        "filters": "Сүзгілер", "none": "жоқ", "confidence": "Сенімділік", "source": "Дереккөз",
        "store": "goszakup қоймасы (OWS v3), шарттар, лоттар және жоспарлар кестелері",
        "enstru_res": "«{p}» тауары ЕНС ТРУ {c} кодымен сәйкестендірілді ({n}, позициялар: {k})",
        "enstru_alt": "ұқсас атауы бар басқа кодтар: {a}",
        "no_results": "Сәйкес жазбалар табылмады.",
    },
}
ERR_KZ = {"нет аналогов у других ведомств": "басқа ведомстволарда аналогтар жоқ",
          "лот не связан с пунктом плана, код ЕНС ТРУ неизвестен": "лот жоспар тармағымен байланыспаған, ЕНС ТРУ коды белгісіз",
          "нет цены и количества": "бағасы мен саны жоқ"}
CONF_KZ = {"высокая": "жоғары", "средняя": "орташа", "низкая": "төмен"}
VERDICT_KZ = {"в пределах рынка": "нарық шегінде", "требует внимания": "назар аударуды қажет етеді",
              "существенно выше рынка": "нарықтан айтарлықтай жоғары",
              "существенно ниже рынка": "нарықтан айтарлықтай төмен"}

# переводы типовых ограничений из tools.py (по началу фразы); неизвестные остаются на русском
LIMITS_KZ = {
    "Показана последняя редакция договора": "Шарттың соңғы редакциясы көрсетілген (қосымша келісімдер ескерілген); бастапқы сомасы мен күні бөлек берілген.",
    "Номер договора не уникален": "Шарт нөмірі бірегей емес: тапсырыс берушіні (атауын немесе БСН) немесе шарт ID-ін көрсетіңіз.",
    "Учтены только договоры в тенге": "Тек теңгедегі шарттар ескерілді (USD-дағы 1 шарт алынып тасталды).",
    "Год определяется по дате подписания": "Жыл шартқа қол қойылған күн бойынша анықталады (2026 жыл толық емес).",
    "У части позиций не найден пункт плана": "Кейбір позициялар үшін жоспар тармағы табылмады, өлшем бірлігі белгісіз.",
    "Статистика считается только по кодам": "Статистика тек кемінде 5 шарт позициясы бар ЕНС ТРУ кодтары бойынша есептеледі.",
    "Средневзвешенная цена взвешена": "Орташа өлшенген баға саны бойынша өлшенеді, ірі сатып алулар оған көбірек әсер етеді.",
    "Цены без НДС": "Бағалар ҚҚС-сыз, нормаланған өлшем бірлігіне. Жоспармен байланысы жоқ позициялар (бірлігі белгісіз) алынып тасталды.",
    "Услуги и работы": "Қызметтер мен жұмыстар әдепкі бойынша алынып тасталды: олардың «бірлік» бағасы бүкіл шарттың бағасына тең және салыстыруға келмейді.",
    "Индекс потребительских цен не загружен": "Тұтыну бағаларының индексі жүктелмеген: әр жылдың бағалары инфляцияға түзетусіз салыстырылады.",
    "Цены приведены к ценам": "Бағалар тұтыну бағаларының ай сайынғы индексі бойынша (ҚР Ұлттық статистика бюросы) соңғы айдың бағаларына келтірілген; индекстің соңғы айынан кейінгі шарт бағалары қайта есептелмейді.",
    "Сравнение только с другими ведомствами": "Тек басқа ведомстволармен салыстырылады (лоттың тапсырыс берушісі іріктемеден алынып тасталды). "
        "Сенімділік N ≥ 20, n_қала ≥ 3 және кемінде 3 ведомство болғанда жоғары; аналогтар бір ведомстводан ғана болса, сенімділік төмен.",
    "Город поставки определяется по КАТО": "Жеткізу қаласы КАТО бойынша анықталады; жоспарда болмаса, лоттан немесе тапсырыс берушінің мекенжайынан алынады (kato_source өрісі).",
    "История только 2024–2026": "Тарих тек 2024–2026 жылдар, яғни салыстыру үшін ең көбі 2 алдыңғы жыл.",
    "Используются плановые количества": "Жоспарлы сандар қолданылады; жоспар жыл ішінде түзетілуі мүмкін.",
    "Рост может быть обоснован": "Өсу негізді болуы мүмкін (жаңа бөлімшелер, бір реттік жобалар): қолмен тексеру қажет.",
    "Поиск по тексту": "Мәтін бойынша іздеу: синонимдер мен қазақша атаулар салыстырылмайды.",
}


def _limits(res, lang):
    items = res.get("limitations", [])
    if lang != "kz":
        return items
    return [next((v for k, v in LIMITS_KZ.items() if s.startswith(k)), s) for s in items]


def _enstru_note(res, lang):
    r = res.get("enstru_resolved")
    if not r:
        return []
    c = r["chosen"]
    lines = [T[lang]["enstru_res"].format(p=r["product"], c=c["enstru_code"],
             n=c["name_kz" if lang == "kz" else "name_ru"], k=fnum(c["n_items"]))]
    if r["alternatives"]:
        lines.append(T[lang]["enstru_alt"].format(a=", ".join(a["enstru_code"] for a in r["alternatives"])))
    return lines


# ---------------- 3. разделы 2–6 по типам вопросов ----------------

def _used_data(res, lang, filters):
    t = T[lang]
    p = res.get("params", {})
    b = p.get("customer_bin")
    lines = [f"- {t['source']}: {t['store']}",
             f"- {t['period']}: {p.get('period', tools.YEARS)}",
             f"- {t['orgs']}: {org(b) if b else t['all_orgs']}",
             f"- {t['filters']}: {'; '.join(filters) if filters else t['none']}"]
    return "\n".join(lines + [f"- {x}" for x in _enstru_note(res, lang)])


def sections_anomalies(res, lang):
    kz = lang == "kz"
    p, s, rows = res["params"], res["summary"], res["results"]
    filters = [f"{'ауытқу шегі' if kz else 'порог отклонения'} > {fpct(p['threshold'], False)}",
               ("тек тауарлар" if kz else "только товары") if p["scope"] == "только товары" else
               ("тауарлар мен қызметтер" if kz else "товары и услуги")]
    if p.get("enstru_code"):
        filters.append(f"ЕНС ТРУ {p['enstru_code']}")
    if p.get("year"):
        filters.append(f"{'шарт жылы' if kz else 'год договора'} {p['year']}")
    filters.append(f"{'код бойынша іріктеме' if kz else 'выборка по коду'} N ≥ {tools.MIN_N}")
    used = _used_data(res, lang, filters)

    checked, n_anom = s["n_items_checked"] or 0, s["n_anomalies"] or 0
    share = res["derived"].get("share_anomalies") or 0
    if kz:
        cmp_ = [f"- Тексерілген шарт позициялары: {fnum(checked)}, ЕНС ТРУ кодтары: {fnum(s['n_enstru_codes'])}",
                f"- Ауытқуы > {fpct(p['threshold'], False)} позициялар: {fnum(n_anom)} ({fpct(share, False)})",
                f"- Орташа өлшенгенге қатысты жалпы артық төлем: {fnum(s['total_overpay_kzt'])} тг"]
    else:
        cmp_ = [f"- Проверено позиций договоров: {fnum(checked)}, кодов ЕНС ТРУ: {fnum(s['n_enstru_codes'])}",
                f"- Позиций с отклонением > {fpct(p['threshold'], False)}: {fnum(n_anom)} ({fpct(share, False)})",
                f"- Суммарная переплата относительно средневзвешенной: {fnum(s['total_overpay_kzt'])} тг"]
    if rows:
        pb = res.get("price_base")
        if pb:
            cmp_.append(f"\n{'Бағалар ' + pb['kz'] + ' бағаларына келтірілген (ТБИ)' if kz else 'Цены приведены к ценам ' + pb['ru'] + ' (ИПЦ)'}; "
                        f"{'шарттағы номиналды баға 6-бөлімде' if kz else 'номинальная цена договора — в разделе 6'}.")
        hdr = ("| № | ЕНС ТРУ | Бірлік | Баға | Орт. өлшенген | Медиана | N | Ауытқу | robust z | IQR-ден жоғары |" if kz else
               "| № | ЕНС ТРУ | Ед. | Цена | Средневзв. | Медиана | N | Отклонение | robust z | Выше Q3+1.5·IQR |")
        cmp_ += ["", hdr, "|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            cmp_.append(f"| {i} | {r['enstru_code']} {r['enstru_name']} | {r['unit_name']} | {fnum(r['price_real'])} | "
                        f"{fnum(r['wavg'])} | {fnum(r['median'])} | {r['sample_n']} | {fpct(r['dev_from_wavg'])} | "
                        f"{fnum(r['robust_z'])} | {('иә' if kz else 'да') if r['above_iqr_fence'] else ('жоқ' if kz else 'нет')} |")
    method = (f"Әр шарт позициясының бірлік бағасы сол ЕНС ТРУ коды мен өлшем бірлігі бар барлық позициялардың "
              f"саны бойынша орташа өлшенген бағасымен салыстырылады. Аномалия: ауытқу > {fpct(p['threshold'], False)}. "
              f"Қосымша robust z-score (MAD арқылы) және Q3 + 1,5·IQR шегінен шығу көрсетілген. "
              f"Артық төлем = (баға − орташа өлшенген) × саны." if kz else
              res["method"] + " Переплата = (цена − средневзвешенная) × количество.")
    top = []
    for i, r in enumerate(rows, 1):
        top.append(f"{i}. {link('Договор ' + str(r['contract_number']) if not kz else 'Шарт ' + str(r['contract_number']), r['contract_url'])}"
                   f" (ID {r['contract_id']}, {r['sign_date']}), "
                   + (f"{link('объявление ' + str(r['number_anno']), r['announcement_url'])}, " if r.get("announcement_url") else "")
                   + f"{org(r['customer_bin'], r['customer_name'])}; {r['item_name'][:120]}; "
                   + (f"саны {fnum(r['quantity'])} {r['unit_name']}, шарттағы баға {fnum(r['unit_price'])} тг, "
                      f"артық төлем {fnum(r['overpay_vs_wavg'])} тг" if kz else
                      f"кол-во {fnum(r['quantity'])} {r['unit_name']}, цена в договоре {fnum(r['unit_price'])} тг, "
                      f"переплата {fnum(r['overpay_vs_wavg'])} тг"))
    return used, "\n".join(cmp_), method, top


def _verdict(v, kz):
    return VERDICT_KZ.get(v, v) if kz else v


def _err(e, kz):
    return ERR_KZ.get(e, e) if kz else e


def _lot_dev(r):
    return r["lot_summary"].get("deviation")


def sections_fairness(res, lang):
    kz = lang == "kz"
    rows = res["results"]
    filters = [f"лот {res['params']['lot']}",
               "басқа ведомстволардың шарттары (тапсырыс беруші алынып тасталды)" if kz else
               "договоры других ведомств (заказчик лота исключён)"]
    used = _used_data({**res, "params": {**res["params"], "customer_bin": rows[0]["customer_bin"] if rows else None}},
                      lang, filters)
    cmp_, top = [], []
    if pb := res.get("price_base"):
        cmp_.append(f"{'Бағалар (шарттағы да, жоспардағы да) ' + pb['kz'] + ' бағаларына келтірілген (ТБИ).' if kz else 'Все цены (из договоров и из плана) приведены к ценам ' + pb['ru'] + ' (ИПЦ).'}")
    # в объявлении может быть много лотов: подробно показываем MAX_LOTS с наибольшим отклонением
    shown = sorted(rows, key=lambda r: (_lot_dev(r) is None, -(_lot_dev(r) or 0)))[:MAX_LOTS]
    if len(rows) > len(shown):
        cmp_.append(f"{'Лоттар' if kz else 'Лотов'}: {len(rows)}; "
                    + (f"төменде ауытқуы ең үлкен {len(shown)} лот" if kz else f"ниже {len(shown)} с наибольшим отклонением"))
    for r in shown:
        s, items = r["lot_summary"], r["items"]
        desc = r.get("description_ru") if r.get("description_ru") not in (None, "", r["name_ru"]) else None
        head = (f"**Лот {r['lot_number']}** ({r['name_ru']}" + (f"; {desc}" if desc else "") + ")"
                + f", {link(('хабарландыру ' if kz else 'объявление ') + str(r['number_anno']), r['announcement_url'])}, "
                + f"{r['publish_date']}, {'мәртебесі' if kz else 'статус'}: {r.get('status_name') or '—'}")
        cmp_.append(("\n" if cmp_ else "") + head)
        ok = [i for i in items if not i.get("error")]
        bad = [i for i in items if i.get("error")]
        if s["n_items"] == 1 and ok:
            i = ok[0]
            f, unit = i["fair"], i.get("unit_name") or ""
            src = ({"договор": "шарт", }.get(i["price_source"], "жоспар (сома / саны), шарт жоқ") if kz else i["price_source"])
            if i.get("contract_url"):
                src += f", {link(('шарт ' if kz else 'договор ') + str(i['contract_number']), i['contract_url'])}"
            if kz:
                cmp_ += [f"- ЕНС ТРУ: {i['enstru_code']} {i.get('enstru_name') or ''}, бірлік: {unit}, жеткізу орны: {i.get('delivery_place') or '—'} "
                         f"(қала коды {i['city_code'] or '—'}, дереккөз {i['kato_source']})",
                         f"- Бірлік бағасы: {fnum(i['price_per_unit'])} тг/{unit} ({src})",
                         f"- P_base (басқа ведомстволардың медианасы): {fnum(f['p_base_median'])} тг, Q1–Q3: {fnum(f['q1'])} – {fnum(f['q3'])}, "
                         f"N = {f['n']}, ведомстволар: {f['n_customers']}",
                         f"- Жеткізу қаласы: N = {f['n_city']}, медиана {fnum(f['city_median'])} тг; K_region = {fnum(f['k_region'], 4)}, K_season = {fnum(f['k_season'], 4)}",
                         f"- Fair Price: {fnum(f['fair_price'])} тг; ауытқу: {fpct(i['deviation'])}; қорытынды: **{_verdict(i['verdict'], kz)}**"]
            else:
                cmp_ += [f"- ЕНС ТРУ: {i['enstru_code']} {i.get('enstru_name') or ''}, единица: {unit}, место поставки: {i.get('delivery_place') or '—'} "
                         f"(код города {i['city_code'] or '—'}, источник {i['kato_source']})",
                         f"- Цена за единицу: {fnum(i['price_per_unit'])} тг/{unit} ({src})",
                         f"- P_base (медиана других ведомств): {fnum(f['p_base_median'])} тг, Q1–Q3: {fnum(f['q1'])} – {fnum(f['q3'])}, "
                         f"N = {f['n']}, {plural(f['n_customers'], 'ведомство', 'ведомства', 'ведомств')}",
                         f"- Город поставки: N = {f['n_city']}, медиана {fnum(f['city_median'])} тг; K_region = {fnum(f['k_region'], 4)}, K_season = {fnum(f['k_season'], 4)}",
                         f"- Fair Price: {fnum(f['fair_price'])} тг; отклонение: {fpct(i['deviation'])}; вывод: **{_verdict(i['verdict'], kz)}**"]
        elif ok:
            bv = s["by_verdict"]
            if kz:
                cmp_.append(f"- Жоспар тармақтары: {s['n_items']}, бағаланғаны: {s['n_assessed']}. Бағаланған тармақтардың құны "
                            f"{fnum(s['cost_assessed'])} тг, Fair Price бойынша {fnum(s['fair_cost'])} тг; лоттың ауытқуы "
                            f"{fpct(s['deviation'])}: **{_verdict(s['verdict'], kz)}**")
                cmp_.append(f"- Тармақтар бойынша: нарықтан айтарлықтай жоғары {bv['существенно выше рынка']}, назар аудару "
                            f"{bv['требует внимания']}, нарық шегінде {bv['в пределах рынка']}, айтарлықтай төмен {bv['существенно ниже рынка']}")
                hdr = "| Тармақ | ЕНС ТРУ | Бірлік | Баға | Fair Price | Ауытқу | N | n_қала | Қорытынды |"
            else:
                cmp_.append(f"- Пунктов плана: {s['n_items']}, оценено: {s['n_assessed']}. Стоимость оценённых пунктов "
                            f"{fnum(s['cost_assessed'])} тг против {fnum(s['fair_cost'])} тг по Fair Price; отклонение лота "
                            f"{fpct(s['deviation'])}: **{_verdict(s['verdict'], kz)}**")
                cmp_.append(f"- По пунктам: существенно выше рынка {bv['существенно выше рынка']}, требует внимания "
                            f"{bv['требует внимания']}, в пределах рынка {bv['в пределах рынка']}, существенно ниже {bv['существенно ниже рынка']}")
                hdr = "| Пункт | ЕНС ТРУ | Ед. | Цена | Fair Price | Откл. | N | n_город | Вывод |"
            cmp_ += ["", hdr, "|---|---|---|---|---|---|---|---|---|"]
            for i in ok[:MAX_ITEMS]:
                cmp_.append(f"| {i['item_name']} | {i['enstru_code']} | {i['unit_name']} | {fnum(i['price_per_unit'])} | "
                            f"{fnum(i['fair']['fair_price'])} | {fpct(i['deviation'])} | {i['fair']['n']} | {i['fair']['n_city']} | "
                            f"{_verdict(i['verdict'], kz)} |")
            if len(ok) > MAX_ITEMS:
                cmp_.append(f"\n{'Тағы' if kz else 'Ещё'} {len(ok) - MAX_ITEMS} " + ("тармақ ауытқуы аз" if kz else "пунктов с меньшим отклонением"))
        if bad:
            reasons = defaultdict(int)
            for i in bad:
                reasons[_err(i["error"], kz)] += 1
            cmp_.append(f"- {'Бағаланбады' if kz else 'Не оценено'}: " + "; ".join(f"{k} — {v}" for k, v in reasons.items()))
        for i in ok[:tools.MAX_COMPARABLE_ITEMS]:
            for c in i.get("comparables", []):
                if len(top) >= 15:
                    break
                top.append(f"{len(top) + 1}. {i['item_name']} ({fpct(i['deviation'])}): "
                           f"{link(('Шарт ' if kz else 'Договор ') + str(c['contract_number']), c['contract_url'])} "
                           f"(ID {c['contract_id']}, {c['sign_date']}), {_short(c['customer_name'])}, "
                           f"{'жеткізуші' if kz else 'поставщик'} {_short(c['supplier_name'] or '—')}: "
                           f"{fnum(c['price_real'])} тг/{i['unit_name']}, {'қала коды' if kz else 'код города'} {c['city_code'] or '—'}")
    method = ("Fair Price = P_base × K_region × K_season. P_base: сол ЕНС ТРУ коды мен бірлік бойынша басқа ведомстволардың "
              "бірлік бағаларының медианасы. K_region: жеткізу қаласындағы медиананың P_base-ке қатынасы, қаладағы бақылаулар "
              "санына қарай 1-ге тегістелген (w = n_қала / (n_қала + 5)). K_season: сатып алу айы үшін КПВЭД бөлімінің маусымдық "
              "коэффициенті. Тармақ бағасы шарт позициясынан, ол болмаса жоспар сомасы / саны. Бірнеше тармақты лот әр тармақ "
              "бойынша бағаланады, лоттың ауытқуы саны бойынша өлшенеді. Шектер: 15%-ға дейін қалыпты, 15–30% назар аудару, "
              "30%-дан жоғары айтарлықтай ауытқу."
              if kz else res["method"] + " Сглаживание: w = n_город / (n_город + 5). Лот из нескольких пунктов: "
              "отклонение лота = (Σ цена × кол-во − Σ Fair Price × кол-во) / Σ Fair Price × кол-во по оценённым пунктам.")
    return used, "\n".join(cmp_), method, top


def sections_volumes(res, lang):
    kz = lang == "kz"
    p, s, rows = res["params"], res["summary"], res["results"]
    filters = [f"{'жоспар жылы' if kz else 'год плана'} {p['year']}",
               f"{'өсу' if kz else 'рост'} ≥ {fnum(p['min_ratio'])}×",
               f"{'сомасы' if kz else 'сумма'} ≥ {fnum(p['min_amount_kzt'])} тг"]
    if p.get("enstru_code"):
        filters.append(f"ЕНС ТРУ {p['enstru_code']}")
    else:
        filters.append("тек тауарлар" if kz else "только товары")
    used = _used_data(res, lang, filters)
    cmp_ = [(f"- Тексерілген жұптар (тапсырыс беруші × ЕНС ТРУ коды): {fnum(s['pairs_checked'])}, белгіленгені: {fnum(s['n_flagged'])}" if kz else
             f"- Проверено пар (заказчик × код ЕНС ТРУ): {fnum(s['pairs_checked'])}, отмечено: {fnum(s['n_flagged'])}")]
    if rows:
        cmp_ += ["", ("| № | Тапсырыс беруші | ЕНС ТРУ | Бірлік | Саны | Алдыңғы жылдар | Медиана | Өсу | Өсімге келетін сома, тг |" if kz else
                      "| № | Заказчик | ЕНС ТРУ | Ед. | Кол-во | Прошлые годы | Медиана | Рост | Сумма на прирост, тг |"),
                 "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            prev = ", ".join(f"{y}: {fnum(q)}" for y, q in zip(r["prev_years"], r["prev_qty"]))
            cmp_.append(f"| {i} | {_short(r['customer_name'])} | {r['enstru_code']} {r['enstru_name']} | {r['unit_name']} | "
                        f"{fnum(r['qty'])} | {prev} | {fnum(r['median_prev_qty'])} | ×{fnum(r['growth_ratio'])} | {fnum(r['extra_amount'])} |")
    method = (f"Әр жұп (тапсырыс беруші, ЕНС ТРУ коды) үшін {p['year']} жылғы жоспарлы сан алдыңғы жылдардағы санның "
              f"медианасымен салыстырылады. Әдеттен тыс өсу: {fnum(p['min_ratio'])} есе және одан көп, сомасы "
              f"{fnum(p['min_amount_kzt'])} тг-ден бастап. Сұрыптау өсімге келетін сома бойынша: "
              f"сома × (1 − алдыңғы жылдар медианасы / ағымдағы сан)." if kz else res["method"])
    top = []
    for i, r in enumerate(rows, 1):
        annos = ", ".join(link(a["number_anno"], a["url"]) for a in r.get("announcements", []))
        top.append(f"{i}. {org(r['customer_bin'], r['customer_name'])}; {r['enstru_code']} {r['enstru_name']}; "
                   + (f"жоспар {r['plan_year']}: {fnum(r['qty'])} {r['unit_name']}, {fnum(r['amount'])} тг; "
                      f"жоспар тармақтарының ID: {', '.join(map(str, r['plan_point_ids']))}" if kz else
                      f"план {r['plan_year']}: {fnum(r['qty'])} {r['unit_name']}, {fnum(r['amount'])} тг; "
                      f"ID пунктов плана: {', '.join(map(str, r['plan_point_ids']))}")
                   + (f"; {'хабарландырулар' if kz else 'объявления'}: {annos}" if annos else ""))
    return used, "\n".join(cmp_), method, top


def sections_search(res, lang):
    kz = lang == "kz"
    p, s, rows = res["params"], res["summary"], res["results"]
    filters = [f"{'мәтін' if kz else 'текст'}: «{p['text']}»"
               + (f" ({'ЕНС ТРУ анықтамалығы бойынша' if kz else 'по справочнику ЕНС ТРУ из'} «{p['text_original']}»)"
                  if p.get("text_original") else "")] + ([f"{'жыл' if kz else 'год'} {p['year']}"] if p.get("year") else [])
    used = _used_data(res, lang, filters)
    cmp_ = [(f"- Табылған лоттар (қайта жарияланымдарсыз): {fnum(s['n_found'])}, жалпы сомасы: {fnum(s['total_amount'])} тг" if kz else
             f"- Найдено лотов (без повторных публикаций): {fnum(s['n_found'])}, общая сумма: {fnum(s['total_amount'])} тг")]
    if rows:
        d = res["derived"]
        if d.get("shown_max_amount"):
            cmp_.append(f"- {'Көрсетілген' if kz else 'Среди показанных'} {len(rows)}: "
                        f"{'медиана сомасы' if kz else 'медиана суммы'} {fnum(d['shown_median_amount'])} тг, "
                        f"{'ең үлкені' if kz else 'максимум'} {fnum(d['shown_max_amount'])} тг")
    method = ("Лоттың тазартылған атауы мен сипаттамасынан ішкі жолды іздеу; бір лоттың қайта жарияланымдары соңғысына дейін біріктірілді. "
              "Сома бойынша сұрыптау." if kz else res["method"] + " Сортировка по сумме лота.")
    top = [f"{i}. {link(('Хабарландыру ' if kz else 'Объявление ') + str(r['number_anno']), r['url'])}, "
           f"{'лот' if kz else 'лот'} {r['lot_number']} (ID {r['lot_id']}), {r['publish_date']}: {r['name_ru']}"
           f"{' — ' + r['description_ru'][:100] if r.get('description_ru') else ''}; {org(r['customer_bin'])}; "
           f"{'саны' if kz else 'кол-во'} {fnum(r['count'])}, {'сомасы' if kz else 'сумма'} {fnum(r['amount'])} тг; {r.get('status_name') or '—'}"
           f"{'; ' + ('жарияланымдар' if kz else 'публикаций') + ': ' + str(r['n_publications']) if r['n_publications'] > 1 else ''}"
           for i, r in enumerate(rows, 1)]
    return used, "\n".join(cmp_), method, top


def sections_comparison(res, lang):
    kz = lang == "kz"
    rows, ov = res["results"], res["summary"]
    unit = res["results"][0]["unit_name"] if res["results"] else res["params"]["unit_norm"]
    used = _used_data(res, lang, [f"ЕНС ТРУ {res['params']['enstru_code']}", f"{'бірлік' if kz else 'единица'}: {unit}"])
    cmp_ = []
    for o in ov:
        cmp_.append(f"- {'Барлық тапсырыс берушілер' if kz else 'Все заказчики'}: N = {o['n']}, "
                    f"{'медиана' if kz else 'медиана'} {fnum(o['median'])} тг, "
                    f"{'орташа өлшенген' if kz else 'средневзвешенная'} {fnum(o['wavg'])} тг")
    if rows:
        cmp_ += ["", ("| Тапсырыс беруші | Бірлік | N | Медиана | Орт. өлшенген | Мин | Макс | Жалпы саны |" if kz else
                      "| Заказчик | Ед. | N | Медиана | Средневзв. | Мин | Макс | Всего кол-во |"),
                 "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            cmp_.append(f"| {org(r['customer_bin'], r['customer_name'])} | {r['unit_name']} | {r['n_items']} | "
                        f"{fnum(r['median_price'])} | {fnum(r['wavg_price'])} | {fnum(r['min_price'])} | "
                        f"{fnum(r['max_price'])} | {fnum(r['total_qty'])} |")
    method = ("Әр тапсырыс беруші бойынша бірлік бағасының медианасы және саны бойынша орташа өлшенген бағасы." if kz
              else res["method"])
    examples = tools._q("""
        SELECT DISTINCT ON (customer_bin) customer_bin, contract_id, contract_number, contract_url, sign_date, price_real
        FROM mv_price_base WHERE enstru_code = %(e)s AND unit_norm = %(u)s ORDER BY customer_bin, sign_date DESC""",
        {"e": res["params"]["enstru_code"], "u": res["params"]["unit_norm"]})
    top = [f"{link(('Шарт ' if kz else 'Договор ') + str(e['contract_number']), e['contract_url'])} (ID {e['contract_id']}, "
           f"{e['sign_date']}), {org(e['customer_bin'])}: {fnum(e['price_real'])} тг" for e in examples]
    return used, "\n".join(cmp_), method, top


def _kz(row, field, kz):
    """Название из справочника API на языке ответа (name_kz есть у способов закупки и статусов)."""
    return (row.get(field + "_kz") or row.get(field)) if kz else row.get(field)


def sections_contract(res, lang):
    kz = lang == "kz"
    p = res["params"]
    filters = [f"{'шарт' if kz else 'договор'}: {p['ref']}"]
    if res.get("ambiguous"):
        used = _used_data(res, lang, filters)
        cmp_ = (f"- «{p['ref']}» нөмірімен {res['n_candidates']} шарт табылды (әртүрлі тапсырыс берушілер)" if kz else
                f"- С номером «{p['ref']}» найдено договоров: {res['n_candidates']} (разные заказчики)")
        method = ("Шарттар нөмірі бірегей емес, сондықтан тапсырыс берушіні нақтылау қажет." if kz else
                  "Номера договоров не уникальны, нужен заказчик или ID договора.")
        top = [f"{i}. {link(('Шарт ' if kz else 'Договор ') + str(c['contract_number']), c['contract_url'])} "
               f"(ID {c['contract_id']}, {c['sign_date']}), {org(c['customer_bin'], c['customer_name'])}, "
               f"{'жеткізуші' if kz else 'поставщик'} {_short(c['supplier_name'] or '—')}, {fnum(c['sum_with_vat'])} тг"
               for i, c in enumerate(res["results"], 1)]
        return used, cmp_, method, top
    c = res["results"][0]
    used = _used_data({**res, "params": {**p, "customer_bin": c["customer_bin"]}}, lang, filters)
    st = _kz(c, "status", kz)
    if kz:
        cmp_ = [f"- Шарт: {link(str(c['contract_number']), c['contract_url'])} (ID {c['contract_id']}), қол қойылған күні {c['sign_date']}, қаржы жылы {c['fin_year']}",
                f"- Тапсырыс беруші: {org(c['customer_bin'], c['customer_name'])}",
                f"- Жеткізуші: {_short(c['supplier_name'] or '—')} (БСН {c['supplier_bin']})",
                f"- Сомасы: ҚҚС-пен {fnum(c['sum_with_vat'])} тг, ҚҚС-сыз {fnum(c['sum_without_vat'])} тг; нақты сома (fakt_sum): {fnum(c['fakt_sum'])} тг",
                f"- Сатып алу тәсілі: {_kz(c, 'trade_method', kz) or '—'}; мәртебесі: {st or '—'}"
                + (f"; хабарландыру {link(c['number_anno'], c['announcement_url'])}" if c.get("number_anno") else ""),
                f"- Редакциялар: {c['n_versions']}; бастапқы қол қойылған күні {c['first_sign_date']}, бастапқы сомасы {fnum(c['first_sum_with_vat'])} тг"]
        hdr = "| Позиция | ЕНС ТРУ | Бірлік | Саны | ҚҚС-сыз баға | ҚҚС-пен сома | Лот |"
    else:
        cmp_ = [f"- Договор: {link(str(c['contract_number']), c['contract_url'])} (ID {c['contract_id']}), подписан {c['sign_date']}, финансовый год {c['fin_year']}",
                f"- Заказчик: {org(c['customer_bin'], c['customer_name'])}",
                f"- Поставщик: {_short(c['supplier_name'] or '—')} (БИН {c['supplier_bin']})",
                f"- Сумма: с НДС {fnum(c['sum_with_vat'])} тг, без НДС {fnum(c['sum_without_vat'])} тг; фактическая сумма (fakt_sum): {fnum(c['fakt_sum'])} тг",
                f"- Способ закупки: {c['trade_method'] or '—'}; статус: {st or '—'}"
                + (f"; объявление {link(c['number_anno'], c['announcement_url'])}" if c.get("number_anno") else ""),
                f"- Редакций: {c['n_versions']}; первоначальная дата {c['first_sign_date']}, первоначальная сумма с НДС {fnum(c['first_sum_with_vat'])} тг"]
        hdr = "| Позиция | ЕНС ТРУ | Ед. | Кол-во | Цена без НДС | Сумма с НДС | Лот |"
    cmp_ += ["", hdr, "|---|---|---|---|---|---|---|"]
    for i in c["items"][:MAX_ITEMS]:
        cmp_.append(f"| {i['item_name'] or '—'} | {i['enstru_code'] or '—'} | {i['unit_name'] or '—'} | {fnum(i['quantity'])} | "
                    f"{fnum(i['unit_price_without_vat'])} | {fnum(i['total_with_vat'])} | {i['lot_number'] or '—'} |")
    if len(c["items"]) > MAX_ITEMS:
        cmp_.append(f"\n{'Тағы' if kz else 'Ещё'} {len(c['items']) - MAX_ITEMS} " + ("позиция" if kz else "позиций"))
    method = ("Шарт жазбасы мен оның позицияларын қоймадан тікелей оқу, есептеусіз. Қосымша келісімдер ескерілген соңғы редакция көрсетілген."
              if kz else res["method"] + " Показана последняя редакция с учётом доп. соглашений.")
    top = [f"1. {link(('Шарт ' if kz else 'Договор ') + str(c['contract_number']), c['contract_url'])} (ID {c['contract_id']})"]
    if c.get("announcement_url"):
        top.append(f"2. {link(('Хабарландыру ' if kz else 'Объявление ') + str(c['number_anno']), c['announcement_url'])}")
    return used, "\n".join(cmp_), method, top


def sections_analytics(res, lang):
    kz = lang == "kz"
    p, s = res["params"], res["summary"]
    filters = []
    if p.get("year"):
        filters.append(f"{'қол қойылған жыл' if kz else 'год подписания'} {p['year']}")
    if p.get("enstru_code"):
        filters.append(f"ЕНС ТРУ {p['enstru_code']}")
    if p.get("supplier_bin"):
        filters.append(f"{'жеткізуші БСН' if kz else 'БИН поставщика'} {p['supplier_bin']}")
    filters.append("шарттардың соңғы редакциялары, теңгемен" if kz else "последние редакции договоров, в тенге")
    used = _used_data(res, lang, filters)
    if not s["n_contracts"]:
        return used, T[lang]["no_results"], res.get("method", ""), []
    d = res["derived"]
    if kz:
        cmp_ = [f"- Шарттар: {fnum(s['n_contracts'])}, ҚҚС-пен сомасы {fnum(s['total_with_vat'])} тг, орташа шарт {fnum(d['avg_contract_with_vat'])} тг",
                f"- Жеткізушілер: {fnum(s['n_suppliers'])}, тапсырыс берушілер: {fnum(s['n_customers'])}; кезең {s['first_date']} – {s['last_date']}",
                f"- Бір көзден: {fnum(s['n_single_source'])} шарт, {fnum(s['single_source_with_vat'])} тг ({fpct(s['single_source_share'], False)} сомадан)"]
    else:
        cmp_ = [f"- Договоров: {fnum(s['n_contracts'])}, сумма с НДС {fnum(s['total_with_vat'])} тг, средний договор {fnum(d['avg_contract_with_vat'])} тг",
                f"- Поставщиков: {fnum(s['n_suppliers'])}, заказчиков: {fnum(s['n_customers'])}; период {s['first_date']} – {s['last_date']}",
                f"- Из одного источника: {plural(s['n_single_source'], 'договор', 'договора', 'договоров')} на {fnum(s['single_source_with_vat'])} тг ({fpct(s['single_source_share'], False)} суммы)"]
    if len(res["by_year"]) > 1:
        cmp_.append(f"- {'Жылдар бойынша' if kz else 'По годам'}: " + "; ".join(
            f"{y['year']}: {fnum(y['n_contracts'])} / {fnum(y['total_with_vat'])} тг" for y in res["by_year"]))

    def table(title, head, rows, fmt):
        if not rows:
            return []
        return ["", f"**{title}**", "", head, "|" + "---|" * (head.count("|") - 1)] + [fmt(r) for r in rows]

    cmp_ += table("Сатып алу тәсілдері" if kz else "Способы закупки",
                  "| Тәсіл | Шарттар | Сома, тг | Үлесі |" if kz else "| Способ | Договоров | Сумма, тг | Доля |",
                  res["by_method"], lambda r: f"| {_kz(r, 'trade_method', kz)} | {fnum(r['n_contracts'])} | {fnum(r['total_with_vat'])} | {fpct(r['share'], False)} |")
    cmp_ += table("Ірі жеткізушілер" if kz else "Крупнейшие поставщики",
                  "| Жеткізуші | БСН | Шарттар | Сома, тг | Үлесі |" if kz else "| Поставщик | БИН | Договоров | Сумма, тг | Доля |",
                  res["top_suppliers"], lambda r: f"| {_short(r['supplier_name'] or '—')} | {r['supplier_bin']} | {fnum(r['n_contracts'])} | {fnum(r['total_with_vat'])} | {fpct(r['share'], False)} |")
    cmp_ += table("Ірі тапсырыс берушілер" if kz else "Крупнейшие заказчики",
                  "| Тапсырыс беруші | Шарттар | Сома, тг | Үлесі |" if kz else "| Заказчик | Договоров | Сумма, тг | Доля |",
                  res.get("top_customers"), lambda r: f"| {org(r['customer_bin'], r['customer_name'])} | {fnum(r['n_contracts'])} | {fnum(r['total_with_vat'])} | {fpct(r['share'], False)} |")
    cmp_ += table("Ірі ЕНС ТРУ кодтары" if kz else "Крупнейшие коды ЕНС ТРУ",
                  "| ЕНС ТРУ | Шарттар | Сома, тг |" if kz else "| ЕНС ТРУ | Договоров | Сумма, тг |",
                  res.get("top_enstru"), lambda r: f"| {r['enstru_code']} {r['enstru_name'] or ''} | {fnum(r['n_contracts'])} | {fnum(r['total_with_vat'])} |")
    method = ("Шарттардың соңғы редакцияларын (қосымша келісімдер ескерілген) агрегаттау, " + tools.YEARS + " жылдары қол қойылған. "
              "Сомалар ҚҚС-пен" + (f", {p['enstru_code']} коды бар шарт позициялары бойынша" if p.get("enstru_code") else ", шарттар бойынша толық")
              + ". Бір көзден: «өткізілмеген сатып алулар бойынша бір көзден», «тікелей шарт жасасу арқылы» тәсілдері." if kz else res["method"])
    top = [f"{i}. {link(('Шарт ' if kz else 'Договор ') + str(r['contract_number']), r['contract_url'])} (ID {r['contract_id']}, "
           f"{r['sign_date']}), {org(r['customer_bin'])}, {'жеткізуші' if kz else 'поставщик'} {_short(r['supplier_name'] or '—')}: "
           f"{fnum(r['amount_with_vat'])} тг, {_kz(r, 'trade_method', kz) or '—'}" for i, r in enumerate(res["results"], 1)]
    return used, "\n".join(cmp_), method, top


SECTIONS = {"contract": sections_contract, "analytics": sections_analytics, "anomalies": sections_anomalies, "fairness": sections_fairness, "volumes": sections_volumes,
            "search": sections_search, "comparison": sections_comparison}


def confidence_of(res):
    if res.get("confidence"):
        return res["confidence"]
    rs = [r["confidence"] for r in res.get("results", []) if isinstance(r, dict) and r.get("confidence")]
    rs += [r["lot_summary"]["confidence"] for r in res.get("results", [])
           if isinstance(r, dict) and r.get("lot_summary", {}).get("confidence")]
    order = ["низкая", "средняя", "высокая"]
    return min(rs, key=order.index) if rs else ("средняя" if res.get("results") else "низкая")


# ---------------- 3. краткий вывод: факты для LLM и запасной шаблон ----------------

def key_facts(res, lang):
    """Короткий список уже отформатированных фактов: LLM копирует их, а не считает."""
    kz, qt, f = lang == "kz", res["question_type"], []
    if qt == "anomalies":
        s, p = res["summary"], res["params"]
        checked = s["n_items_checked"] or 0
        f += [f"проверено позиций договоров: {fnum(checked)}",
              f"порог отклонения: {fpct(p['threshold'], False)}",
              f"позиций с отклонением выше порога: {fnum(s['n_anomalies'])} ({fpct(res['derived'].get('share_anomalies') or 0, False)})",
              f"суммарная переплата относительно средневзвешенной: {fnum(s['total_overpay_kzt'])} тг"]
        if p.get("customer_bin"):
            f.append(f"заказчик: {org(p['customer_bin'])}")
        if res["results"]:
            r = res["results"][0]
            f.append(f"крупнейшая переплата: договор {r['contract_number']}, {_short(r['customer_name'])}, {r['enstru_name']}, "
                     f"цена {fnum(r['price_real'])} тг против средневзвешенной {fnum(r['wavg'])} тг "
                     f"(отклонение {fpct(r['dev_from_wavg'])}), переплата {fnum(r['overpay_vs_wavg'])} тг")
    elif qt == "fairness":
        f.append(template_summary(res, lang))
    elif qt == "volumes":
        s = res["summary"]
        f.append(f"проверено пар заказчик × код: {fnum(s['pairs_checked'])}, отмечено с ростом ≥ "
                 f"{fnum(res['params']['min_ratio'])} раза: {fnum(s['n_flagged'])}, год {res['params']['year']}")
        for r in res["results"][:2]:
            f.append(f"{_short(r['customer_name'])}: {r['enstru_name']} — {fnum(r['qty'])} {r['unit_name']} против медианы "
                     f"прошлых лет {fnum(r['median_prev_qty'])} (рост ×{fnum(r['growth_ratio'])}), сумма на прирост {fnum(r['extra_amount'])} тг")
    elif qt == "search":
        s = res["summary"]
        f.append(f"найдено лотов: {fnum(s['n_found'])} на сумму {fnum(s['total_amount'])} тг по запросу «{res['params']['text']}»")
        if res["results"]:
            r = res["results"][0]
            f.append(f"крупнейший: лот {r['lot_number']} «{r['name_ru']}», {_short(subjects().get(r['customer_bin'], ''))}, "
                     f"{fnum(r['amount'])} тг, {r['publish_date']}")
    elif qt == "comparison":
        rows = res["results"]
        if res["summary"]:
            o = res["summary"][0]
            f.append(f"по коду {res['params']['enstru_code']} всего позиций {o['n']}, медиана {fnum(o['median'])} тг, "
                     f"средневзвешенная {fnum(o['wavg'])} тг")
        if rows:
            by = {r["customer_bin"]: r for r in rows}
            hi, lo = by[res["derived"]["highest_bin"]], by[res["derived"]["lowest_bin"]]
            f.append(f"самая высокая медиана: {_short(hi['customer_name'])} — {fnum(hi['median_price'])} тг (N = {hi['n_items']})")
            f.append(f"самая низкая медиана: {_short(lo['customer_name'])} — {fnum(lo['median_price'])} тг (N = {lo['n_items']})")
            f.append(f"заказчиков: {len(rows)}")
    return f


def template_summary(res, lang):
    """Детерминированный краткий вывод: каждое число подписано тем полем JSON, из которого взято."""
    kz, qt = lang == "kz", res["question_type"]
    rows = res.get("results") or []
    if qt == "anomalies":
        s, p = res["summary"], res["params"]
        checked, n = s["n_items_checked"] or 0, s["n_anomalies"] or 0
        if not checked:
            return ("Таңдалған сүзгілер бойынша кемінде 5 позициядан тұратын іріктемесі бар шарт позициялары жоқ."
                    if kz else "По выбранным фильтрам нет позиций договоров с выборкой по коду не меньше 5.")
        share = fpct(res["derived"]["share_anomalies"], False)
        thr = fpct(p["threshold"], False)
        who = org(p["customer_bin"]) if p.get("customer_bin") else None
        if kz:
            out = (f"{who + ' бойынша ' if who else ''}{fnum(checked)} шарт позициясының {fnum(n)} ({share}) бағасы ЕНС ТРУ "
                   f"бойынша орташа өлшенгеннен {thr}-дан артық жоғары, жалпы артық төлем {fnum(s['total_overpay_kzt'])} тг.")
            if rows:
                r = rows[0]
                out += (f" Ең үлкені: {r['contract_number']} шарты ({r['enstru_name']}), баға {fnum(r['price_real'])} тг, "
                        f"орташа өлшенген {fnum(r['wavg'])} тг ({fpct(r['dev_from_wavg'])}), артық төлем {fnum(r['overpay_vs_wavg'])} тг.")
        else:
            out = (f"{'У заказчика ' + who + ' и' if who else 'И'}з {fnum(checked)} позиций договоров {fnum(n)} ({share}) "
                   f"имеют цену выше средневзвешенной по ЕНС ТРУ более чем на {thr}; суммарная переплата "
                   f"{fnum(s['total_overpay_kzt'])} тг.")
            if rows:
                r = rows[0]
                out += (f" Крупнейшая: договор {r['contract_number']} ({r['enstru_name']}), цена {fnum(r['price_real'])} тг "
                        f"при средневзвешенной {fnum(r['wavg'])} тг ({fpct(r['dev_from_wavg'])}), переплата {fnum(r['overpay_vs_wavg'])} тг.")
        return out
    if qt == "fairness":
        d = res["derived"]

        def item_sentence(lot_number, i):
            f = i["fair"]
            v = _verdict(i["verdict"], kz)
            if kz:
                city = (f"жеткізу қаласында (код {i['city_code'] or '—'}) басқа ведомстволардың шарттары жоқ, сондықтан "
                        f"салыстыру қала коэффициентінсіз барлық іріктеме бойынша жүргізілді"
                        if not f["n_city"] else f"оның ішінде жеткізу қаласында {f['n_city']}")
                return (f"{lot_number} лотының бағасы {fnum(i['price_per_unit'])} тг/{i['unit_name']}, Fair Price "
                        f"{fnum(f['fair_price'])} тг, ауытқу {fpct(i['deviation'])}: {v}. Аналогтар: N = {f['n']}, "
                        f"{f['n_customers']} ведомство; {city}.")
            city = (f"в городе поставки (код {i['city_code'] or '—'}) договоров других ведомств нет, поэтому сравнение "
                    f"выполнено по всей выборке без регионального коэффициента"
                    if not f["n_city"] else f"из них в городе поставки {f['n_city']}")
            return (f"Цена лота {lot_number} {fnum(i['price_per_unit'])} тг/{i['unit_name']} против Fair Price "
                    f"{fnum(f['fair_price'])} тг, отклонение {fpct(i['deviation'])}: {v}. Аналогов N = {f['n']} "
                    f"({plural(f['n_customers'], 'ведомство', 'ведомства', 'ведомств')}); {city}.")

        def lot_sentence(r):
            s = r["lot_summary"]
            ok = [i for i in r["items"] if not i.get("error")]
            if not ok:
                errs = sorted({_err(i["error"], kz) for i in r["items"]}) or ["нет пунктов плана"]
                return f"Лот {r['lot_number']}: " + ("бағалау мүмкін емес: " if kz else "оценка невозможна: ") + "; ".join(errs) + "."
            if s["n_items"] == 1:
                return item_sentence(r["lot_number"], ok[0])
            w = ok[0]  # пункт с наибольшим по модулю отклонением
            hi = s["by_verdict"]["существенно выше рынка"]
            if kz:
                return (f"{r['lot_number']} лоты ({s['n_items']} тармақ, {s['n_assessed']} бағаланды): бағаланған тармақтардың құны "
                        f"{fnum(s['cost_assessed'])} тг, Fair Price бойынша {fnum(s['fair_cost'])} тг, ауытқу {fpct(s['deviation'])}: "
                        f"{_verdict(s['verdict'], kz)}. Нарықтан айтарлықтай жоғары тармақтар: {hi}; ең үлкен ауытқу: "
                        f"{w['item_name']} — {fnum(w['price_per_unit'])} тг, Fair Price {fnum(w['fair']['fair_price'])} тг ({fpct(w['deviation'])}).")
            return (f"Лот {r['lot_number']} ({plural(s['n_items'], 'пункт', 'пункта', 'пунктов')} плана, оценено {s['n_assessed']}): стоимость оценённых пунктов "
                    f"{fnum(s['cost_assessed'])} тг против {fnum(s['fair_cost'])} тг по Fair Price, отклонение {fpct(s['deviation'])}: "
                    f"{_verdict(s['verdict'], kz)}. Существенно выше рынка пунктов: {hi}; наибольшее отклонение: "
                    f"{w['item_name']} — {fnum(w['price_per_unit'])} тг против {fnum(w['fair']['fair_price'])} тг ({fpct(w['deviation'])}).")

        if len(rows) == 1:
            return lot_sentence(rows[0])
        bv = d["lots_by_verdict"]
        if kz:
            out = (f"Сұрау бойынша {d['n_lots']} лот табылды, {d['n_lots_assessed']} бағаланды; оның ішінде нарықтан айтарлықтай "
                   f"жоғары: {bv['существенно выше рынка']}, нарық шегінде: {bv['в пределах рынка']}.")
        else:
            out = (f"По запросу найдено лотов: {d['n_lots']}, оценено: {d['n_lots_assessed']}; из них существенно выше рынка: "
                   f"{bv['существенно выше рынка']}, в пределах рынка: {bv['в пределах рынка']}.")
        if d.get("max_dev_lot"):
            out += " " + lot_sentence(next(r for r in rows if r["lot_number"] == d["max_dev_lot"]))
        return out
    if qt == "volumes":
        s, p = res["summary"], res["params"]
        if kz:
            out = (f"{p['year']} жылы {fnum(s['pairs_checked'])} жұптың (тапсырыс беруші × ЕНС ТРУ) {fnum(s['n_flagged'])}-інде "
                   f"жоспарлы сан алдыңғы жылдар медианасынан {fnum(p['min_ratio'])} және одан көп есе жоғары.")
        else:
            out = (f"В {p['year']} году у {fnum(s['n_flagged'])} из {fnum(s['pairs_checked'])} пар (заказчик × код ЕНС ТРУ) "
                   f"плановое количество в {fnum(p['min_ratio'])}+ раза выше медианы прошлых лет.")
        if rows:
            r = rows[0]
            out += (f" Ең үлкені: {_short(r['customer_name'])}, {r['enstru_name']}: {fnum(r['qty'])} {r['unit_name']}, "
                    f"медиана {fnum(r['median_prev_qty'])} (×{fnum(r['growth_ratio'])}), өсімге {fnum(r['extra_amount'])} тг."
                    if kz else
                    f" Крупнейшее по сумме: {_short(r['customer_name'])}, {r['enstru_name']}: {fnum(r['qty'])} {r['unit_name']} "
                    f"против медианы {fnum(r['median_prev_qty'])} (×{fnum(r['growth_ratio'])}), на прирост приходится {fnum(r['extra_amount'])} тг.")
        return out
    if qt == "search":
        s, p = res["summary"], res["params"]
        out = (f"«{p['text']}» бойынша {fnum(s['n_found'])} лот табылды, жалпы сомасы {fnum(s['total_amount'])} тг." if kz else
               f"По запросу «{p['text']}» найдено лотов: {fnum(s['n_found'])} на сумму {fnum(s['total_amount'])} тг.")
        if rows:
            r = rows[0]
            out += (f" Ең ірісі: {r['lot_number']} лоты «{r['name_ru']}», {fnum(r['amount'])} тг ({r['publish_date']})." if kz else
                    f" Крупнейший: лот {r['lot_number']} «{r['name_ru']}», {fnum(r['amount'])} тг ({r['publish_date']}).")
        return out
    if qt == "contract":
        p = res["params"]
        if res.get("ambiguous"):
            return (f"«{p['ref']}» нөмірімен {res['n_candidates']} шарт табылды; тапсырыс берушіні (атауын немесе БСН) "
                    f"немесе шарт ID-ін нақтылаңыз. Үміткерлер 6-бөлімде." if kz else
                    f"Договоров с номером «{p['ref']}» найдено: {res['n_candidates']}; уточните заказчика (название или БИН) "
                    f"или ID договора. Кандидаты в разделе 6.")
        c = rows[0]
        st = _kz(c, "status", kz)
        if kz:
            out = (f"{c['contract_number']} шарты {c['sign_date']} күні {_short(c['customer_name'])} пен {_short(c['supplier_name'] or '—')} "
                   f"(БСН {c['supplier_bin']}) арасында жасалған, ҚҚС-пен сомасы {fnum(c['sum_with_vat'])} тг, тәсілі: {_kz(c, 'trade_method', kz) or '—'}, мәртебесі: {st or '—'}.")
            if c["n_versions"] > 1:
                out += f" Редакциялар: {c['n_versions']}, бастапқы сомасы {fnum(c['first_sum_with_vat'])} тг."
        else:
            out = (f"Договор {c['contract_number']} подписан {c['sign_date']} между {_short(c['customer_name'])} и "
                   f"{_short(c['supplier_name'] or '—')} (БИН {c['supplier_bin']}) на {fnum(c['sum_with_vat'])} тг с НДС; "
                   f"способ закупки: {c['trade_method'] or '—'}, статус: {st or '—'}.")
            if c["n_versions"] > 1:
                out += f" Редакций: {c['n_versions']}, первоначальная сумма {fnum(c['first_sum_with_vat'])} тг."
        return out
    if qt == "analytics":
        s, p = res["summary"], res["params"]
        if not s["n_contracts"]:
            return T[lang]["no_results"]
        who = org(p["customer_bin"]) if p.get("customer_bin") else None
        when = str(p["year"]) if p.get("year") else tools.YEARS
        sup = res["top_suppliers"][0] if res["top_suppliers"] else None
        if kz:
            out = (f"{(who + ' ') if who else ''}{when} {'жылы' if p.get('year') else 'жылдары'} {fnum(s['n_contracts'])} шарт жасалды, ҚҚС-пен сомасы "
                   f"{fnum(s['total_with_vat'])} тг; бір көзден {fpct(s['single_source_share'], False)} ({fnum(s['single_source_with_vat'])} тг).")
            if sup:
                out += f" Ең ірі жеткізуші: {_short(sup['supplier_name'])} — {fnum(sup['total_with_vat'])} тг ({fpct(sup['share'], False)})."
        else:
            out = (f"{('У ' + who + ' з') if who else 'З'}а {when} подписано договоров: {fnum(s['n_contracts'])} на "
                   f"{fnum(s['total_with_vat'])} тг с НДС; из одного источника {fpct(s['single_source_share'], False)} суммы "
                   f"({fnum(s['single_source_with_vat'])} тг).")
            if sup:
                out += f" Крупнейший поставщик: {_short(sup['supplier_name'])} — {fnum(sup['total_with_vat'])} тг ({fpct(sup['share'], False)})."
        return out
    if qt == "comparison":
        e = res["params"]["enstru_code"]
        if not rows:
            return T[lang]["no_results"]
        d = res["derived"]
        by = {r["customer_bin"]: r for r in rows}
        hi, lo = by[d["highest_bin"]], by[d["lowest_bin"]]
        ratio = d.get("max_to_min_median")
        among = d["min_n_for_extremes"] > 1
        if kz:
            return (f"ЕНС ТРУ {e} бойынша {len(rows)} тапсырыс беруші"
                    + (f" (шеткі мәндер N ≥ {d['min_n_for_extremes']} болатын {d['n_customers_reliable']} арасынан)" if among else "")
                    + f": ең жоғары медиана {_short(hi['customer_name'])} ({fnum(hi['median_price'])} тг, N = {hi['n_items']}), "
                    f"ең төменгі {_short(lo['customer_name'])} ({fnum(lo['median_price'])} тг, N = {lo['n_items']})"
                    + (f", айырма ×{fnum(ratio)}." if ratio else "."))
        return (f"По коду ЕНС ТРУ {e} сравнено заказчиков: {len(rows)}"
                + (f" (крайние значения среди {d['n_customers_reliable']} с N ≥ {d['min_n_for_extremes']})" if among else "")
                + f". Самая высокая медиана у {_short(hi['customer_name'])} ({fnum(hi['median_price'])} тг, N = {hi['n_items']}), "
                f"самая низкая у {_short(lo['customer_name'])} ({fnum(lo['median_price'])} тг, N = {lo['n_items']})"
                + (f", разница ×{fnum(ratio)}." if ratio else "."))
    return T[lang]["no_results"]


SUMMARY_PROMPT = {
    "ru": ("Напиши краткий вывод (1–3 предложения) на русском языке для ответа аналитика по госзакупкам. "
           "Используй ТОЛЬКО факты из списка, числа копируй ровно в том виде, как они записаны (с пробелами между разрядами). "
           "Не вычисляй новые числа, не округляй, не добавляй даты, БИН и проценты, которых нет в списке. "
           "Без общих фраз и рекомендаций, только суть с цифрами. Выведи только текст вывода."),
    "kz": ("Мемлекеттік сатып алулар бойынша талдаушы жауабы үшін қазақ тілінде қысқаша қорытынды жаз (1–3 сөйлем). "
           "ТЕК тізімдегі фактілерді пайдалан, сандарды жазылғандай дәл көшір (разрядтар арасындағы бос орындармен). "
           "Жаңа сандар есептеме, дөңгелектеме, тізімде жоқ күндерді, БСН мен пайыздарды қоспа. "
           "Жалпы сөздер мен ұсыныстарсыз, тек цифрлармен мәні. Тек қорытынды мәтінін шығар."),
}


def llm_summary(question, facts, lang, feedback=None):
    msgs = [{"role": "system", "content": SUMMARY_PROMPT[lang]},
            {"role": "user", "content": f"Вопрос: {question}\nФакты:\n" + "\n".join(f"- {x}" for x in facts)}]
    if feedback:
        msgs.append({"role": "user", "content": feedback})
    resp = llm.chat.completions.create(model=LLM_MODEL, temperature=0, messages=msgs, max_tokens=SUMMARY_MAX_TOKENS)
    text = resp.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ---------------- 4. проверка фактов ----------------

RE_NUM = re.compile(r"(?<![\w.,])[-+−]?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?![\d])|(?<![\w.,])[-+−]?\d+(?:[.,]\d+)?(?![\d])")
RE_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})\b")


def _to_float(s):
    s = s.replace(" ", "").replace(" ", "").replace(" ", "").replace("−", "-").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def allowed_values(*sources):
    """Все числа, даты и идентификаторы из JSON инструмента и списка фактов."""
    nums, strs = set(), set()
    for src in sources:
        for v in _walk(src):
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                nums.add(float(v))
            else:
                s = str(v)
                strs.add(s)
                for m in RE_NUM.finditer(s):
                    if (x := _to_float(m.group(0))) is not None:
                        nums.add(abs(x))
                nums.update(float(d) for d in re.findall(r"\d+", s))
    derived = set()
    for x in nums:
        derived |= {x, abs(x), abs(x) * 100, abs(x) / 1e3, abs(x) / 1e6, abs(x) / 1e9}
    return derived, strs


def _matches(x, allowed):
    for a in allowed:
        if a == x:
            return True
        if a and abs(a - x) / abs(a) <= 0.005:  # округление до 3 значащих цифр
            return True
        # округление до заданного числа знаков: 3.29 -> 3.3, 145186233 -> 145.2 (млн)
        for d in (0, 1, 2):
            if round(a, d) == x:
                return True
    return False


def verify(text, *sources):
    """Каждое число, дата и БИН в тексте должны встречаться в данных. Возвращает (ok, отчёт)."""
    allowed, strs = allowed_values(*sources)
    joined = " ".join(strs)
    facts, bad = [], []
    for m in RE_DATE.finditer(text):
        d = m.group(1)
        iso = d if "-" in d else f"{d[6:]}-{d[3:5]}-{d[:2]}"
        facts.append(d)
        if iso not in joined:
            bad.append(d)
    for b in RE_BIN.findall(text):
        facts.append(b)
        if b not in joined:
            bad.append(b)
    no_dates = RE_DATE.sub(" ", text)
    no_ids = RE_ENSTRU.sub(" ", RE_BIN.sub(" ", no_dates))
    for m in RE_NUM.finditer(no_ids):
        raw = m.group(0).strip()
        x = _to_float(raw)
        if x is None:
            continue
        x = abs(x)
        if x <= 10 and x == int(x):  # мелкие целые: «топ-5», «1–3 предложения»
            continue
        facts.append(raw)
        if not _matches(x, allowed):
            bad.append(raw)
    return not bad, {"n_facts": len(facts), "n_unsupported": len(bad), "unsupported": bad}


# ---------------- сборка ответа ----------------

OUT_OF_SCOPE = {
    "ru": ("Вопрос не относится к данным о госзакупках в хранилище. Агент отвечает на вопросы по закупкам "
           "27 организаций за 2024–2026: ценовые аномалии относительно средневзвешенной по ЕНС ТРУ, справедливость "
           "цены лота (Fair Price), нетипичный рост объёмов, поиск закупок и сравнение цен между заказчиками."),
    "kz": ("Сұрақ қоймадағы мемлекеттік сатып алулар деректеріне қатысты емес. Агент 2024–2026 жылдардағы 27 ұйымның "
           "сатып алулары бойынша жауап береді: ЕНС ТРУ бойынша баға ауытқулары, лот бағасының әділдігі (Fair Price), "
           "көлемнің әдеттен тыс өсуі, сатып алуларды іздеу және тапсырыс берушілер арасындағы бағаны салыстыру."),
}


def log_out_of_scope(route_log, args):
    """LLM не выбрала функцию, ключевые слова не сработали и параметров нет: вопрос не по теме."""
    return route_log.get("router") == "keywords" and not route_log.get("keyword_hit") and not args

def answer(question):
    t0 = time.time()
    lang = detect_lang(question)
    name, args, route_log = route(question)
    t_route = time.time()
    if "customer_bin" not in args and len(cands := org_candidates(question)) > 1:
        lst = "\n".join(f"- {org(b)}" for b in cands)
        msg = (f"Вопрос подходит к нескольким организациям, уточните какую (название или БИН):\n{lst}" if lang == "ru" else
               f"Сұрақ бірнеше ұйымға сәйкес келеді, қайсысы екенін нақтылаңыз (атауы немесе БСН):\n{lst}")
        return {"question": question, "lang": lang, "tool": name, "args": args, "route": route_log,
                "answer": msg, "verification": {"n_facts": 0, "n_unsupported": 0},
                "summary_source": "clarify", "candidates": cands, "timings": {"route": round(t_route - t0, 2)}}
    if log_out_of_scope(route_log, args):
        msg = OUT_OF_SCOPE[lang]
        return {"question": question, "lang": lang, "tool": None, "args": args, "route": route_log,
                "answer": msg, "verification": {"n_facts": 0, "n_unsupported": 0},
                "summary_source": "out_of_scope", "timings": {"route": round(t_route - t0, 2)}}
    res = run_tool(name, args, lang)
    t_tool = time.time()

    if res.get("error"):
        msg = res["error"]
        if lang == "kz":
            msg = "Сұрауды орындау мүмкін болмады: " + msg
        return {"question": question, "lang": lang, "tool": name, "args": args, "route": route_log,
                "answer": msg, "tool_result": res, "verification": {"n_facts": 0, "n_unsupported": 0},
                "summary_source": "error", "timings": {"route": round(t_route - t0, 2)}}

    used, cmp_, method, top = SECTIONS[res["question_type"]](res, lang)
    facts = key_facts(res, lang)

    summary, source, ver, attempts = None, "template", None, 0
    if SUMMARY_MODE == "llm" and res.get("results"):
        feedback = None
        for attempts in range(1, MAX_RETRIES + 2):
            try:
                cand = llm_summary(question, facts, lang, feedback)
            except Exception as e:  # noqa: BLE001
                route_log["summary_error"] = f"{type(e).__name__}: {e}"[:300]
                break
            ok, ver = verify(cand, res, facts)
            if ok and cand:
                summary, source = cand, "llm"
                break
            feedback = ("В тексте есть числа или даты, которых нет в фактах: " + ", ".join(ver["unsupported"])
                        + ". Перепиши, используя только числа из списка фактов в точности.")
    if summary is None:
        summary = template_summary(res, lang)
        if not res.get("results") and res["question_type"] not in ("anomalies",):
            summary = T[lang]["no_results"] + " " + summary
    ver_final = verify(summary, res, facts)[1]
    t_sum = time.time()

    h = T[lang]["h"]
    conf = confidence_of(res)
    limits = _limits(res, lang)
    if fr := res.get("data_as_of"):
        limits.insert(0, (f"Деректер {fr['as_of']} жағдайы бойынша (соңғы сәтті жаңарту)" if lang == "kz" else
                          f"Данные актуальны на {fr['as_of']} (последнее успешное обновление)"))
    conf_line = f"{T[lang]['confidence']}: **{CONF_KZ.get(conf, conf) if lang == 'kz' else conf}**"
    parts = [f"### 1. {h[0]}\n{summary}",
             f"### 2. {h[1]}\n{used}",
             f"### 3. {h[2]}\n{cmp_ or T[lang]['no_results']}",
             f"### 4. {h[3]}\n{method}",
             f"### 5. {h[4]}\n" + "\n".join([f"- {conf_line}"] + [f"- {x}" for x in limits]),
             f"### 6. {h[5]}\n" + ("\n".join(top) if top else T[lang]["no_results"])]
    return {"question": question, "lang": lang, "tool": name, "args": args, "route": route_log,
            "answer": "\n\n".join(parts), "summary_source": source, "summary_attempts": attempts,
            "verification": ver_final, "confidence": conf, "tool_result": res,
            "timings": {"route": round(t_route - t0, 2), "tool": round(t_tool - t_route, 2),
                        "summary": round(t_sum - t_tool, 2), "total": round(time.time() - t0, 2)}}


# ---------------- HTTP ----------------

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Goszakup analytics agent")

    class Ask(BaseModel):
        question: str
        include_tool_result: bool = False

    @app.post("/ask")
    def ask(req: Ask):
        out = answer(req.question)
        if not req.include_tool_result:
            out.pop("tool_result", None)
        return out

    @app.get("/health")
    def health():
        fr = tools.data_freshness()
        return {"db_subjects": len(subjects()), "llm_model": LLM_MODEL, "data_as_of": fr,
                # требование ТЗ: задержка обновления не больше 24 ч
                "data_stale": fr is None or float(fr["hours_ago"]) > 24}
except ImportError:  # FastAPI нужен только для HTTP-режима
    app = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true", help="полный ответ с JSON инструмента")
    a = ap.parse_args()
    out = answer(a.question)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        print(out["answer"])
        print(f"\n[tool={out['tool']} args={out['args']} router={out['route'].get('router')} "
              f"summary={out['summary_source']} facts={out['verification']['n_facts']} "
              f"unsupported={out['verification']['n_unsupported']} t={out['timings']}]")
