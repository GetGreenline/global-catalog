WITH weedmaps_source AS (
  SELECT
    weedmaps_product_id::varchar AS external_id,
    product_category,
    updated_at
  FROM archive.arch_us_weedmaps_products
),

weedmaps_uncategorized AS (
  SELECT
    external_id,
    'uncategorized'::varchar AS level_one,
    'uncategorized'::varchar AS level_two,
    'uncategorized'::varchar AS level_three,
    'weedmaps'::varchar       AS source,
    updated_at
  FROM weedmaps_source
  WHERE product_category IS NULL OR product_category = JSON_PARSE('[]')
),

weedmaps_exploded AS (
  SELECT
    external_id,
    updated_at,
    JSON_SERIALIZE(cat_val) AS category_json
  FROM weedmaps_source AS w,
       w.product_category  AS cat_val
),

weedmaps_unnested AS (
  SELECT
    external_id,
    JSON_EXTRACT_PATH_TEXT(category_json, 'l1', true)::varchar AS level_one,
    JSON_EXTRACT_PATH_TEXT(category_json, 'l2', true)::varchar AS level_two,
    JSON_EXTRACT_PATH_TEXT(category_json, 'l3', true)::varchar AS level_three,
    'weedmaps'::varchar AS source,
    updated_at
  FROM weedmaps_exploded
  WHERE JSON_EXTRACT_PATH_TEXT(category_json, 'l1', true) IS NOT NULL
),

weedmaps_union AS (
  SELECT * FROM weedmaps_unnested
  UNION ALL
  SELECT * FROM weedmaps_uncategorized
),

-- Dedupe by PATH ONLY; choose a representative external_id
weedmaps_latest AS (
  SELECT
    external_id,
    level_one,
    level_two,
    level_three,
    source,
    updated_at,
    ROW_NUMBER() OVER (
      PARTITION BY level_one, level_two, level_three, source
      ORDER BY updated_at DESC, external_id ASC
    ) AS rn
  FROM weedmaps_union
)

SELECT
  external_id::varchar,
  level_one::varchar,
  level_two::varchar,
  level_three::varchar,
  source::varchar,
  updated_at::timestamp,
  CURRENT_TIMESTAMP AS load_timestamp
FROM weedmaps_latest
WHERE rn = 1;
