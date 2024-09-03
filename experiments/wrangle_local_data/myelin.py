# %%
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from tqdm.auto import tqdm

from grotto import GrottoClient

client = GrottoClient("minnie65_phase3_v1")

# %%
myelin_table = client.query_table("vortex_manual_myelination_v0")

tables = client.get_tables()
# %%

roots = myelin_table["valid_id"].unique()

# %%


def map_to_closest(source: pd.DataFrame, target: pd.DataFrame):
    target_ilocs = pairwise_distances_argmin(source, target)
    target_index = target.index[target_ilocs]
    source_index = source.index
    return pd.Series(target_index, index=source_index)


tables = []
for root_id, sub_table in tqdm(myelin_table.groupby("valid_id")):
    sub_table = sub_table.query("tag == 't'")
    level2_ids = client.get_leaves(root_id, stop_layer=2)
    level2_data = client.get_l2data(level2_ids)

    myelin_pos = sub_table[["x", "y", "z"]]
    l2_pos = level2_data[["x", "y", "z"]]

    myelin_to_l2_map = map_to_closest(myelin_pos, l2_pos)

    new_table = pd.DataFrame(index=myelin_to_l2_map.values)
    new_table.index.name = "level2_id"
    new_table["root_id"] = root_id
    new_table["classification"] = "myelinated"
    tables.append(new_table)

final_table = pd.concat(tables)


# %%

import pyvista as pv

pv.set_jupyter_backend("client")

plotter = pv.Plotter()

# plotter.add_mesh(mesh_poly)

point_poly = pv.PolyData(l2_pos.values)
plotter.add_mesh(point_poly, point_size=5, color="red")

myelin_poly = pv.PolyData(myelin_pos.values)
plotter.add_mesh(myelin_poly, point_size=5, color="blue")

plotter.show()

# %%
final_table.to_csv("cave-sandbox/data/processed_local_labels/vortex_myelin.csv")

# %%
