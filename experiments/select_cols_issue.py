#%%
import pandas as pd

import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")

timestamp = pd.to_datetime("2021-07-01 00:00:00", utc=True)

soma_ids = [
    292864,
    291116,
    303149,
    264824,
    292670,
    260541,
    301085,
    294825,
    292649,
    298937,
    262678,
]

old_roots = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": soma_ids},
    timestamp=timestamp,
).set_index("id")["pt_root_id"]
print(old_roots)

old_roots_w_select = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": soma_ids},
    select_columns=["id", "pt_root_id"],
    timestamp=timestamp,
).set_index("id")["pt_root_id"]
print(old_roots_w_select)
