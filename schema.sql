-- Схема хранения данных госзакупок (OWS v3) для аналитики и AI-агента
-- Слои: сырые JSONL-файлы (data/raw) -> чистые таблицы (этот файл) -> витрины (следующий шаг)

-- ===================== СПРАВОЧНИКИ =====================

-- Нормализация единиц измерения: код МКЕИ из API -> единая единица + множитель пересчёта
-- Пример: грамм (163) -> кг с множителем 0.001, "Пог.м" (5156) -> метр погонный (018)
CREATE TABLE IF NOT EXISTS ref_units (
    code        TEXT PRIMARY KEY,      -- код из API (refUnitsCode)
    name_ru     TEXT,
    name_kz     TEXT,
    norm_code   TEXT NOT NULL,         -- к какой единице приводим
    norm_name   TEXT NOT NULL,
    factor      NUMERIC NOT NULL DEFAULT 1   -- количество_норм = количество * factor, цена_норм = цена / factor
);

CREATE TABLE IF NOT EXISTS ref_enstru (
    code    TEXT PRIMARY KEY,          -- код ЕНС ТРУ, например 282932.500.000001
    name_ru TEXT,
    name_kz TEXT
);

-- ===================== УЧАСТНИКИ =====================

CREATE TABLE IF NOT EXISTS subjects (
    bin         TEXT PRIMARY KEY,
    pid         BIGINT,
    name_ru     TEXT,
    name_kz     TEXT,
    full_name_ru TEXT,
    kato_list   TEXT[],
    is_customer BOOLEAN,
    is_supplier BOOLEAN,
    is_qvazi    BOOLEAN,
    updated_at  TIMESTAMP
);

-- ===================== ПЛАН =====================

CREATE TABLE IF NOT EXISTS plan_points (
    id              BIGINT PRIMARY KEY,
    rootrecord_id   BIGINT,
    customer_bin    TEXT NOT NULL,
    plan_year       INT NOT NULL,
    name_ru         TEXT,
    name_kz         TEXT,
    desc_ru         TEXT,
    extra_desc_ru   TEXT,
    name_clean      TEXT,              -- очищенное название для поиска
    enstru_code     TEXT REFERENCES ref_enstru(code),
    unit_code       TEXT,              -- как пришло из API
    unit_norm       TEXT,              -- после нормализации
    count_raw       NUMERIC,
    count_norm      NUMERIC,
    price_raw       NUMERIC,           -- плановая цена за единицу
    price_norm      NUMERIC,
    amount          NUMERIC,
    month_id        INT,
    trade_method_id INT,
    status_id       INT,
    kato_code       TEXT,              -- место поставки (первое из PlansKato)
    kato_source     TEXT,              -- plan / lot / customer: откуда взят КАТО
    city_code       TEXT,              -- город поставки, выведенный из КАТО
    delivery_place  TEXT,
    date_create     TIMESTAMP,
    updated_at      TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_plan_enstru ON plan_points(enstru_code, unit_norm, plan_year);
CREATE INDEX IF NOT EXISTS ix_plan_customer ON plan_points(customer_bin, plan_year);

-- ===================== ОБЪЯВЛЕНИЯ И ЛОТЫ =====================

CREATE TABLE IF NOT EXISTS announcements (
    id              BIGINT PRIMARY KEY,
    number_anno     TEXT,
    name_ru         TEXT,
    org_bin         TEXT,              -- организатор (может отличаться от заказчика)
    publish_date    TIMESTAMP,
    end_date        TIMESTAMP,
    itogi_date      TIMESTAMP,
    total_sum       NUMERIC,
    count_lots      INT,
    status_id       INT,
    trade_method_id INT,
    fin_year        INT,
    url             TEXT,
    updated_at      TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lots (
    id              BIGINT PRIMARY KEY,
    announcement_id BIGINT REFERENCES announcements(id),
    lot_number      TEXT,
    customer_bin    TEXT,
    name_ru         TEXT,
    description_ru  TEXT,
    name_clean      TEXT,
    count           NUMERIC,
    amount          NUMERIC,
    status_id       INT,
    status_name     TEXT,
    trade_method_id INT,
    kato_codes      TEXT[],
    dumping         BOOLEAN,
    updated_at      TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_lots_customer ON lots(customer_bin);

-- Лот может объединять несколько пунктов плана (поле pointList)
CREATE TABLE IF NOT EXISTS lot_plan_points (
    lot_id        BIGINT REFERENCES lots(id) ON DELETE CASCADE,
    plan_point_id BIGINT,
    PRIMARY KEY (lot_id, plan_point_id)
);

-- ===================== ДОГОВОРЫ =====================

CREATE TABLE IF NOT EXISTS contracts (
    id              BIGINT PRIMARY KEY,
    root_id         BIGINT,            -- исходный договор; доп.соглашения имеют тот же root_id
    parent_id       BIGINT,
    is_latest       BOOLEAN DEFAULT FALSE,  -- последняя редакция в цепочке root_id
    announcement_id BIGINT,
    contract_number TEXT,
    customer_bin    TEXT,
    supplier_bin    TEXT,
    supplier_name   TEXT,
    status_id       INT,
    type_id         INT,
    trade_method_id INT,
    fin_year        INT,
    sign_date       TIMESTAMP,
    contract_sum    NUMERIC,           -- без НДС
    contract_sum_wnds NUMERIC,         -- с НДС
    fakt_sum        NUMERIC,
    currency        TEXT,
    updated_at      TIMESTAMP
);
-- ALTER только если колонки нет: ADD COLUMN IF NOT EXISTS берёт эксклюзивную блокировку таблицы на всю транзакцию ETL
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'contracts' AND column_name = 'number_anno') THEN
        ALTER TABLE contracts ADD COLUMN number_anno TEXT;
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS ix_contracts_root ON contracts(root_id);
CREATE INDEX IF NOT EXISTS ix_contracts_customer ON contracts(customer_bin, sign_date);

-- Позиции договора: здесь фактическая цена за единицу
CREATE TABLE IF NOT EXISTS contract_items (
    id              BIGINT PRIMARY KEY,
    contract_id     BIGINT REFERENCES contracts(id) ON DELETE CASCADE,
    lot_id          BIGINT,
    plan_point_id   BIGINT,
    enstru_code     TEXT,
    quantity        NUMERIC,
    unit_price      NUMERIC,           -- без НДС
    unit_price_wnds NUMERIC,           -- с НДС
    total_sum       NUMERIC,
    total_sum_wnds  NUMERIC
);
CREATE INDEX IF NOT EXISTS ix_items_enstru ON contract_items(enstru_code);
CREATE INDEX IF NOT EXISTS ix_items_plan ON contract_items(plan_point_id);

-- ===================== СЛУЖЕБНОЕ =====================

-- Журнал проблем качества данных: ничего не удаляем молча
CREATE TABLE IF NOT EXISTS dq_issues (
    entity     TEXT,
    entity_id  BIGINT,
    issue      TEXT,
    detail     TEXT,
    created_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (entity, entity_id, issue)
);

-- Состояние инкрементальной загрузки: с какой даты обновления забирать в следующий раз
CREATE TABLE IF NOT EXISTS etl_state (
    entity      TEXT,
    bin         TEXT,
    last_update TIMESTAMP,
    PRIMARY KEY (entity, bin)
);

-- ===================== ФАКТЫ ЦЕН (основа для аналитики) =====================
-- Одна строка = одна позиция действующей редакции договора
-- с нормализованной ценой, единицей, кодом ЕНС ТРУ и городом поставки.
-- Пункт плана берётся по plan_point_id; если его нет (старая редакция плана),
-- то через лот: пункт плана этого лота с тем же кодом ЕНС ТРУ.
-- removed_at: пункт плана удалён в источнике (флаг isDeleted). Не удаляем строку: на неё могут ссылаться позиции
-- договоров (единица измерения, место поставки). Пропажа из выгрузки API удалением НЕ считается: индекс планов OWS
-- теряет записи без удаления (dq_issues.missing_from_api), см. etl_load.py
-- kato_plan: КАТО из самого плана (сырой), kato_code: итоговый после fallback (лот, адрес заказчика).
-- ALTER только если колонки нет: ADD COLUMN IF NOT EXISTS всё равно берёт эксклюзивную блокировку таблицы
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'plan_points' AND column_name = 'removed_at') THEN
        ALTER TABLE plan_points ADD COLUMN removed_at TIMESTAMP;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'plan_points' AND column_name = 'kato_plan') THEN
        ALTER TABLE plan_points ADD COLUMN kato_plan TEXT;
        UPDATE plan_points SET kato_plan = kato_code WHERE kato_source = 'plan';
    END IF;
END $$;

-- Журнал запусков обновления (update.py): агент показывает дату последнего успешного обновления
CREATE TABLE IF NOT EXISTS etl_runs (
    id          SERIAL PRIMARY KEY,
    started_at  TIMESTAMP NOT NULL DEFAULT now(),
    finished_at TIMESTAMP,
    mode        TEXT,               -- incremental | full
    status      TEXT,               -- running | ok | failed
    stats       JSONB,
    error       TEXT
);

-- data_as_of: на какой момент данные из API (начало выгрузки); по нему агент пишет «Данные актуальны на …»
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'etl_runs' AND column_name = 'data_as_of') THEN
        ALTER TABLE etl_runs ADD COLUMN data_as_of TIMESTAMP;
        UPDATE etl_runs SET data_as_of = started_at WHERE status = 'ok' AND mode <> 'skip-fetch';
    END IF;
END $$;

-- Вид предмета закупки (товар / работа / услуга): из refSubjectTypeId плана и объявления (API OWS v3).
-- Статусы пунктов плана и объявлений: названия из вложенных RefPlnPointStatus и TrdBuy.RefBuyStatus
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'plan_points' AND column_name = 'subject_type_id') THEN
        ALTER TABLE plan_points ADD COLUMN subject_type_id INT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'announcements' AND column_name = 'subject_type_id') THEN
        ALTER TABLE announcements ADD COLUMN subject_type_id INT;
    END IF;
END $$;
CREATE TABLE IF NOT EXISTS ref_subject_type (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);   -- 1 товар, 2 работа, 3 услуга
CREATE TABLE IF NOT EXISTS ref_plan_status (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);
CREATE TABLE IF NOT EXISTS ref_buy_status (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);

-- Справочники способов закупки и статусов договоров (заполняет fetch_refs.py из вложенных полей Contract)
CREATE TABLE IF NOT EXISTS ref_trade_methods (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);
CREATE TABLE IF NOT EXISTS ref_contract_status (id INT PRIMARY KEY, name_ru TEXT, name_kz TEXT);

-- DROP ... CASCADE удаляет и витрины, поэтому после etl_load.py всегда запускается build_marts.py
DROP VIEW IF EXISTS v_price_facts CASCADE;
CREATE VIEW v_price_facts AS
SELECT
    ci.id                         AS item_id,
    c.id                          AS contract_id,
    c.contract_number,
    'https://goszakup.gov.kz/ru/egzcontract/cpublic/show/' || c.id AS contract_url,
    NULLIF(c.announcement_id, 0)  AS announcement_id,
    NULLIF(COALESCE(a.number_anno, c.number_anno), '') AS number_anno,
    CASE WHEN COALESCE(c.announcement_id, 0) > 0
         THEN 'https://goszakup.gov.kz/ru/announce/index/' || c.announcement_id END AS announcement_url,
    ci.lot_id,
    c.customer_bin,
    s.name_ru                     AS customer_name,
    c.supplier_bin,
    c.supplier_name,
    c.sign_date,
    EXTRACT(YEAR FROM c.sign_date)::INT AS sign_year,
    c.trade_method_id,
    ci.enstru_code,
    e.name_ru                     AS enstru_name,
    p.name_clean                  AS item_name,
    p.unit_norm,
    u.norm_name                   AS unit_name,
    p.subject_type_id,
    -- услуга или работа: по виду предмета из API; если его нет (старые и восстановленные пункты плана) — по единице
    COALESCE(p.subject_type_id IN (2, 3), u.norm_name ILIKE '%услуг%' OR u.norm_name ILIKE '%работ%') AS is_service,
    p.plan_source,
    ci.quantity * COALESCE(u.factor, 1)        AS quantity_norm,
    ci.unit_price / NULLIF(COALESCE(u.factor, 1), 0) AS unit_price_norm,
    ci.total_sum,
    p.kato_code,
    p.city_code
FROM contract_items ci
JOIN contracts c         ON c.id = ci.contract_id AND c.is_latest
LEFT JOIN LATERAL (
    SELECT x.* FROM (
        SELECT pp.*, 'direct' AS plan_source, 1 AS prio
          FROM plan_points pp WHERE pp.id = ci.plan_point_id
        UNION ALL
        SELECT pp.*, 'via_lot' AS plan_source, 2 AS prio
          FROM lot_plan_points lp JOIN plan_points pp ON pp.id = lp.plan_point_id
         WHERE lp.lot_id = ci.lot_id AND pp.enstru_code = ci.enstru_code
    ) x ORDER BY x.prio LIMIT 1
) p ON TRUE
LEFT JOIN ref_units u    ON u.code = p.unit_code
LEFT JOIN ref_enstru e   ON e.code = ci.enstru_code
LEFT JOIN announcements a ON a.id = c.announcement_id
LEFT JOIN subjects s     ON s.bin = c.customer_bin
WHERE ci.unit_price > 0 AND ci.quantity > 0;

-- Поставщики: отдельной выгрузки в OWS v3 по поставщикам 27 заказчиков нет, сущность строится из договоров
-- (Contract.supplierBiin, Supplier.nameRu); только последние редакции договоров
CREATE OR REPLACE VIEW v_suppliers AS
SELECT c.supplier_bin,
       (array_agg(c.supplier_name ORDER BY c.sign_date DESC))[1] AS supplier_name,
       count(*)                          AS n_contracts,
       count(DISTINCT c.customer_bin)    AS n_customers,
       sum(c.contract_sum_wnds) FILTER (WHERE c.currency = 'KZT') AS total_with_vat,
       min(c.sign_date)                  AS first_contract,
       max(c.sign_date)                  AS last_contract
FROM contracts c
WHERE c.is_latest AND c.supplier_bin IS NOT NULL
GROUP BY c.supplier_bin;
