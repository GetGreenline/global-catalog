select
  shc.category_id,
  shc.level_one,
  shc.level_two,
  shc.level_three,
  shc.source,
  shc.updated_at
from staging.stg_us_hoodie_categories shc
