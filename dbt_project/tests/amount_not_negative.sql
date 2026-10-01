select txn_id
from {{ ref('stg_transactions') }}
where amount < 0
