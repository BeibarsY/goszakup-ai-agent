"""
Выгрузка сырых данных из OWS v3 (goszakup.gov.kz) по списку БИН за 2024-2026.

Запуск:
    pip install requests python-dotenv
    # .env:      GOSZAKUP_TOKEN=...
    # bins.txt:  по одному БИН на строку
    python fetch_raw.py

Результат: data/raw/<entity>/<bin>.jsonl (одна запись = одна строка JSON)
"""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
URL = "https://ows.goszakup.gov.kz/v3/graphql"
def headers():
    """Токен читается при первом запросе к API, а не при импорте: агенту и загрузке без выгрузки
    (update.py --skip-fetch) он не нужен."""
    token = os.environ.get("GOSZAKUP_TOKEN")
    if not token:
        raise RuntimeError("не задан GOSZAKUP_TOKEN (.env или переменная окружения)")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
YEARS = [2024, 2025, 2026]
DATE_RANGE = ["2024-01-01", "2026-12-31"]
LIMIT = 100  # если API ругается на размер страницы, уменьши до 50
OUT = Path("data/raw")

# ---------- GraphQL-запросы ----------

# Пункты плана: здесь ЕНС ТРУ, единица измерения, цена за единицу, количество, место поставки (КАТО)
PLANS_QUERY = """
query($filter: PlansFiltersInput, $after: Int) {
  Plans(filter: $filter, limit: %d, after: $after) {
    id rootrecordId subjectBiin subjectNameRu plnPointYear
    nameRu nameKz descRu extraDescRu
    refEnstruCode refUnitsCode count price amount
    refMonthsId refTradeMethodsId refPlnPointStatusId refFinsourceId refSubjectTypeId
    dateCreate timestamp isDeleted indexDate
    RefUnits { code nameRu nameKz }
    RefEnstru { code nameRu nameKz }
    PlansKato { refKatoCode fullDeliveryPlaceNameRu count }
    RefSubjectType { id nameRu nameKz }
    RefPlnPointStatus { id nameRu nameKz }
  }
}
""" % LIMIT

# Лоты + краткие данные объявления (TrdBuy) внутри каждого лота
LOTS_QUERY = """
query($filter: LotsFiltersInput, $after: Int) {
  Lots(filter: $filter, limit: %d, after: $after) {
    id lotNumber trdBuyId trdBuyNumberAnno customerBin customerNameRu
    nameRu nameKz descriptionRu count amount
    enstruList pointList plnPointKatoList
    refLotStatusId refTradeMethodsId dumping isDeleted lastUpdateDate indexDate
    RefLotsStatus { nameRu }
    TrdBuy {
      id numberAnno nameRu orgBin customerBin publishDate endDate itogiDatePublic
      totalSum countLots refBuyStatusId refTradeMethodsId finYear kato lastUpdateDate refSubjectTypeId
      RefBuyStatus { id nameRu nameKz }
    }
  }
}
""" % LIMIT

# Договоры + позиции договора (ContractUnits): фактическая цена за единицу и количество
CONTRACT_QUERY = """
query($filter: ContractFiltersInput, $after: Int) {
  Contract(filter: $filter, limit: %d, after: $after) {
    id parentId rootId trdBuyId trdBuyNumberAnno contractNumber contractNumberSys
    customerBin supplierBiin supplierId refContractStatusId refContractTypeId
    faktTradeMethodsId finYear signDate crdate contractSum contractSumWnds
    faktSum refCurrencyCode exchangeRate withNds deleted lastUpdateDate indexDate
    Supplier { bin nameRu }
    ContractUnits {
      id lotId plnPointId itemPrice itemPriceWnds quantity totalSum totalSumWnds
      refEnstruCode deleted
    }
  }
}
""" % LIMIT

SUBJECT_QUERY = """
query($filter: SubjectFiltersInput, $after: Int) {
  Subjects(filter: $filter, limit: %d, after: $after) {
    pid bin nameRu nameKz fullNameRu katoList customer supplier qvazi lastUpdateDate
  }
}
""" % LIMIT


# ---------- общие функции ----------

def gql(query: str, variables: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(URL, json={"query": query, "variables": variables},
                              headers=headers(), timeout=120)
            r.raise_for_status()
            data = r.json()
            if "errors" in data:
                raise RuntimeError(data["errors"])
            return data
        except Exception as e:  # сеть, 5xx, лимиты
            wait = 2 ** attempt
            print(f"  ошибка: {e}; повтор через {wait} c")
            time.sleep(wait)
    raise RuntimeError("API не ответил после нескольких попыток")


def fetch_all(entity: str, query: str, filt: dict):
    """Проходит по всем страницам через after = lastId."""
    after = None
    while True:
        res = gql(query, {"filter": filt, "after": after})
        items = res["data"][entity] or []
        yield from items
        page = res.get("extensions", {}).get("pageInfo", {})
        if not page.get("hasNextPage"):
            break
        after = page["lastId"]
        time.sleep(0.3)  # не долбим API


def save(entity: str, bin_: str, items) -> int:
    path = OUT / entity / f"{bin_}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    bins = [b.strip() for b in Path("bins.txt").read_text().split() if b.strip()]
    for b in bins:
        print(f"БИН {b}")
        jobs = [
            ("Subjects", SUBJECT_QUERY, {"bin": b}),
            ("Plans", PLANS_QUERY, {"subjectBiin": b, "plnPointYear": YEARS}),
            # у лотов нет фильтра по дате: берём все, год отсекаем на этапе ETL по TrdBuy.publishDate
            ("Lots", LOTS_QUERY, {"customerBin": b}),
            ("Contract", CONTRACT_QUERY, {"customerBin": b, "signDate": DATE_RANGE}),
        ]
        for entity, query, filt in jobs:
            n = save(entity, b, fetch_all(entity, query, filt))
            print(f"  {entity}: {n}")


if __name__ == "__main__":
    main()
