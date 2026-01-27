select
  m.global_id,
  suc.category_id,
  suc.level_one,
  suc.level_two,
  suc.level_three,
  suc.source,
  suc.updated_at
from staging.stg_us_categories suc
left join staging.stg_us_categories_mapping m
  on m.category_id = suc.category_id;
