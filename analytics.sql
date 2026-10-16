-- Аналитический слой (витрины для AI-агента)
-- Строится поверх v_price_facts и plan_points из schema.sql. Пересборка: python build_marts.py
--
-- Все цены здесь: за нормализованную единицу, без НДС, в тенге.
-- price_real = цена, приведённая к ценам последнего месяца в ref_cpi (учёт инфляции).

-- ---------- индекс потребительских цен ----------
-- Заполняется из cpi.csv (месяц, индекс). База любая, важны отношения между месяцами.
-- Если таблица пустая, инфляция не учитывается (inflation_factor = 1), и агент сообщает об этом.
CREATE TABLE IF NOT EXISTS ref_cpi (
    month DATE PRIMARY KEY,
    cpi   NUMERIC NOT NULL
);

DROP MATERIALIZED VIEW IF EXISTS mv_price_base CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_volumes CASCADE;

-- ---------- 1. база цен ----------
CREATE MATERIALIZED VIEW mv_price_base AS
WITH last_cpi AS (SELECT cpi FROM ref_cpi ORDER BY month DESC LIMIT 1)
SELECT
    f.*,
    date_trunc('month', f.sign_date)::DATE           AS sign_month,
    EXTRACT(MONTH FROM f.sign_date)::INT             AS month_num,
    left(f.enstru_code, 2)                           AS category,      -- раздел КПВЭД
    left(f.kato_code, 2)                             AS region_code,   -- область
    COALESCE((SELECT cpi FROM last_cpi) / c.cpi, 1)  AS inflation_factor,
    f.unit_price_norm * COALESCE((SELECT cpi FROM last_cpi) / c.cpi, 1) AS price_real,
    (c.cpi IS NOT NULL)                              AS inflation_applied
FROM v_price_facts f
-- индекс ближайшего месяца не позже даты подписания (если месяц пропущен в cpi.csv)
LEFT JOIN LATERAL (
    SELECT r.cpi FROM ref_cpi r
    WHERE r.month <= date_trunc('month', f.sign_date)::DATE
    ORDER BY r.month DESC LIMIT 1
) c ON TRUE
WHERE f.unit_norm IS NOT NULL
  AND f.enstru_code IS NOT NULL
  AND f.unit_price_norm > 0;

CREATE INDEX ON mv_price_base (enstru_code, unit_norm);
CREATE INDEX ON mv_price_base (enstru_code, unit_norm, city_code);
CREATE INDEX ON mv_price_base (contract_id);
CREATE INDEX ON mv_price_base (lot_id);

-- ---------- 2. статистика по коду ЕНС ТРУ + единице ----------
CREATE MATERIALIZED VIEW mv_enstru_stats AS
WITH s AS (
    SELECT enstru_code, unit_norm,
           count(*)                                                    AS n,
           count(DISTINCT customer_bin)                                AS n_customers,
           percentile_cont(0.5)  WITHIN GROUP (ORDER BY price_real)    AS median,
           percentile_cont(0.25) WITHIN GROUP (ORDER BY price_real)    AS q1,
           percentile_cont(0.75) WITHIN GROUP (ORDER BY price_real)    AS q3,
           sum(price_real * quantity_norm) / NULLIF(sum(quantity_norm), 0) AS wavg,
           avg(price_real)                                             AS mean
    FROM mv_price_base
    GROUP BY enstru_code, unit_norm
)
SELECT s.*,
       s.q3 - s.q1 AS iqr,
       (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(b.price_real - s.median))
          FROM mv_price_base b
         WHERE b.enstru_code = s.enstru_code AND b.unit_norm = s.unit_norm) AS mad
FROM s;
CREATE UNIQUE INDEX ON mv_enstru_stats (enstru_code, unit_norm);

-- ---------- 3. то же в разрезе города поставки ----------
CREATE MATERIALIZED VIEW mv_enstru_city_stats AS
SELECT enstru_code, unit_norm, city_code,
       count(*)                                                 AS n,
       count(DISTINCT customer_bin)                             AS n_customers,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY price_real)  AS median,
       sum(price_real * quantity_norm) / NULLIF(sum(quantity_norm), 0) AS wavg
FROM mv_price_base
WHERE city_code IS NOT NULL
GROUP BY enstru_code, unit_norm, city_code;
CREATE UNIQUE INDEX ON mv_enstru_city_stats (enstru_code, unit_norm, city_code);

-- ---------- 4. сезонность ----------
-- Для каждого раздела КПВЭД и месяца: медиана отношения цены к медиане своего кода.
-- Коэффициент считается только при n >= 30 и ограничен диапазоном [0.8, 1.25], иначе 1.
CREATE MATERIALIZED VIEW mv_season AS
WITH rel AS (
    SELECT b.category, b.month_num, b.price_real / s.median AS r
    FROM mv_price_base b
    JOIN mv_enstru_stats s USING (enstru_code, unit_norm)
    WHERE s.n >= 5 AND s.median > 0
)
SELECT category, month_num, count(*) AS n,
       CASE WHEN count(*) >= 30
            THEN LEAST(GREATEST(percentile_cont(0.5) WITHIN GROUP (ORDER BY r), 0.8), 1.25)
            ELSE 1 END AS coef
FROM rel
GROUP BY category, month_num;
CREATE UNIQUE INDEX ON mv_season (category, month_num);

-- ---------- 5. ценовые аномалии (позиция договора против своего кода ЕНС ТРУ) ----------
CREATE OR REPLACE VIEW v_price_anomalies AS
SELECT b.item_id, b.contract_id, b.contract_number, b.contract_url, b.lot_id, b.announcement_id,
       b.number_anno, b.announcement_url, b.customer_bin, b.customer_name,
       b.supplier_bin, b.supplier_name, b.sign_date, b.enstru_code, b.enstru_name,
       b.item_name, b.unit_name, b.quantity_norm, b.unit_price_norm, b.price_real,
       b.city_code, b.inflation_applied, b.is_service,
       s.n, s.n_customers, s.median, s.wavg, s.q1, s.q3, s.iqr, s.mad,
       (b.price_real - s.wavg) / NULLIF(s.wavg, 0)             AS dev_from_wavg,
       (b.price_real - s.median) / NULLIF(1.4826 * s.mad, 0)   AS robust_z,
       (b.price_real > s.q3 + 1.5 * s.iqr)                     AS above_iqr_fence,
       (b.price_real - s.wavg) * b.quantity_norm               AS overpay_vs_wavg
FROM mv_price_base b
JOIN mv_enstru_stats s USING (enstru_code, unit_norm);

-- ---------- 6. объёмы закупок по годам (из плана) ----------
CREATE MATERIALIZED VIEW mv_volumes AS
SELECT p.customer_bin, s.name_ru AS customer_name, p.enstru_code, e.name_ru AS enstru_name,
       p.unit_norm, u.norm_name AS unit_name,
       p.plan_year,
       -- услуга или работа: по виду предмета из API, иначе по единице измерения (агрегат: в группе пункты
       -- с видом предмета и без него не должны разбивать пару заказчик × код × год на две строки)
       bool_or(COALESCE(p.subject_type_id IN (2, 3), u.norm_name ILIKE '%услуг%' OR u.norm_name ILIKE '%работ%')) AS is_service,
       sum(p.count_norm)  AS qty,
       sum(p.amount)      AS amount,
       count(*)           AS n_points,
       array_agg(p.id ORDER BY p.amount DESC) AS plan_point_ids
FROM plan_points p
LEFT JOIN subjects s   ON s.bin = p.customer_bin
LEFT JOIN ref_enstru e ON e.code = p.enstru_code
LEFT JOIN ref_units u  ON u.code = p.unit_code
WHERE p.plan_year IN (2024, 2025, 2026) AND p.count_norm > 0
  AND p.removed_at IS NULL   -- пункты, исчезнувшие из источника, в объёмах не учитываются
GROUP BY 1, 2, 3, 4, 5, 6, 7;
CREATE INDEX ON mv_volumes (customer_bin, enstru_code, unit_norm);

-- Рост объёма: год Y против медианы предыдущих лет того же заказчика по тому же коду
CREATE OR REPLACE VIEW v_volume_growth AS
SELECT cur.customer_bin, cur.customer_name, cur.enstru_code, cur.enstru_name,
       cur.unit_name, cur.is_service, cur.plan_year, cur.qty, cur.amount, cur.plan_point_ids,
       prev.n_years AS n_prev_years, prev.prev_years, prev.prev_qty, prev.median_prev_qty,
       cur.qty / NULLIF(prev.median_prev_qty, 0) AS growth_ratio,
       cur.qty - prev.median_prev_qty            AS qty_increase,
       cur.amount * (1 - prev.median_prev_qty / NULLIF(cur.qty, 0)) AS extra_amount
FROM mv_volumes cur
JOIN LATERAL (
    SELECT count(*) AS n_years,
           array_agg(v.plan_year ORDER BY v.plan_year) AS prev_years,
           array_agg(v.qty ORDER BY v.plan_year)       AS prev_qty,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY v.qty) AS median_prev_qty
    FROM mv_volumes v
    WHERE v.customer_bin = cur.customer_bin AND v.enstru_code = cur.enstru_code
      AND v.unit_norm = cur.unit_norm AND v.plan_year < cur.plan_year
) prev ON prev.n_years > 0;
