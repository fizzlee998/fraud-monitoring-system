select
    cast(txn_id as integer) as txn_id,
    cast(step as integer) as step,
    type as txn_type,
    amount,
    name_orig,
    oldbalance_org,
    newbalance_orig,
    name_dest,
    oldbalance_dest,
    newbalance_dest,
    cast(is_fraud as integer) as is_fraud,
    cast(is_flagged_fraud as integer) as is_flagged_fraud
from {{ source('fraud_raw', 'transactions') }}
