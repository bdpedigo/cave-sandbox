# %%
from glob import glob

import numpy as np
import pandas as pd

from grotto import GrottoClient

client = GrottoClient("minnie65_phase3_v1")

path = "cave-sandbox/data/processed_local_labels"

dfs = []
for file in glob(f"{path}/*.csv"):
    df = pd.read_csv(file)
    dfs.append(df)

label_df = pd.concat(dfs, ignore_index=True)
label_df["classification"] = label_df["classification"].replace(
    {"myelinated": "thick/myelin", "thick": "thick/myelin"}
)

# %%
label_df["classification"].value_counts()

# %%
root_ids = np.unique(label_df["root_id"])
timestamps = client.get_root_timestamps(root_ids, latest=True)
timestamps = pd.Series(timestamps, name="timestamp", index=root_ids)

# %%
label_df["timestamp"] = label_df["root_id"].map(timestamps)

# # %%
# from tqdm.auto import tqdm

# layer_range = range(3, 6)

# upper_ids_df = pd.DataFrame(
#     index=label_df.index, columns=[f"level{i}_id" for i in layer_range]
# )

# for root_id, sub_table in tqdm(label_df.groupby("root_id")):
#     node_ids = sub_table["level2_id"]
#     for layer in layer_range:
#         upper_ids = client.get_roots(
#             node_ids, timestamp=sub_table["timestamp"].iloc[0], stop_layer=layer
#         )
#         upper_ids_df.loc[sub_table.index, f"level{layer}_id"] = upper_ids

# # %%
# label_df = label_df.join(upper_ids_df)

# %%
label_df.to_csv("cave-sandbox/data/assembled_local_labels.csv", index=False)
