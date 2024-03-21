# %%
import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")

position = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_equal_dict={"id": 730537},
    select_columns=['pt_position'],
    desired_resolution=[8,8,40]
).iloc[0]['pt_position']
position
# %%
position