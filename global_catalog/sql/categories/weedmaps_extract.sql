select 
    swc.category_id,
    swc.level_one,
    swc.level_two,
    swc.level_three,
    swc.source,
    swc.updated_at
from staging.stg_us_weedmaps_categories swc;

