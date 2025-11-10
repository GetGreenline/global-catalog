WITH hoodie_categories AS (
  SELECT
    category_id::varchar,
    level_one::varchar,
    level_two::varchar,
    level_three::varchar,
    source::varchar,
    updated_at::timestamp
  FROM staging.stg_us_categories
  WHERE source = 'hoodie'
)
SELECT * FROM hoodie_categories;
