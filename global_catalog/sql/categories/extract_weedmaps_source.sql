-- this file is used to extract the weedmaps source category data for testing purposes
select 
    swc.category_id,
    swc.level_one,
    swc.level_two,
    swc.level_three,
    'test_source' as source,
    swc.updated_at
from staging.stg_us_categories swc
where swc.source = 'weedmaps';
