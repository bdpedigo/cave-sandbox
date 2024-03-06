# %%
from datetime import datetime

import numpy as np
import pandas as pd

import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
cg = cc.chunkedgraph.ChunkedGraphClient(
    server_address="http://0.0.0.0:5001",
    auth_client=client.auth,
    table_name=client.chunkedgraph._table_name,
    verify=False,
)

# %%

# find the location of an edit for examining
root_id = 864691136812623475
operation_id = 530423
change_log = client.chunkedgraph.get_tabular_change_log([root_id])[root_id].set_index(
    "operation_id"
)
row = change_log.loc[operation_id]
details = cg.get_operation_details([operation_id])[str(operation_id)]
point_in_cg = np.array(details["sink_coords"][0])
timestamp = datetime.utcfromtimestamp(row["timestamp"] / 1000)

seg_res = np.array([8, 8, 40])
point_in_nm = point_in_cg * seg_res

# %%

bbox_halfwidth = 5000


def make_bbox(bbox_halfwidth, point_in_nm):
    x_center, y_center, z_center = point_in_nm

    x_start = x_center - bbox_halfwidth
    x_stop = x_center + bbox_halfwidth
    y_start = y_center - bbox_halfwidth
    y_stop = y_center + bbox_halfwidth
    z_start = z_center - bbox_halfwidth
    z_stop = z_center + bbox_halfwidth

    start_point_cg = np.array([x_start, y_start, z_start]) / seg_res
    stop_point_cg = np.array([x_stop, y_stop, z_stop]) / seg_res

    bbox_cg = np.array([start_point_cg, stop_point_cg], dtype=int).T
    return bbox_cg


bbox_cg = make_bbox(bbox_halfwidth, point_in_nm)
my_edges = cg.level2_chunk_graph(root_id, bounds=bbox_cg)
my_edges = np.array(my_edges, dtype=np.uint64)
my_edges = pd.MultiIndex.from_arrays(np.unique(np.sort(my_edges, axis=1), axis=0).T)

# %%
# get a subgraph of l1 edges
true_l1_edges, affinities, areas = client.chunkedgraph.get_subgraph(root_id, bbox_cg)

# get the unique supervoxel ids mentioned in these edges
unique_supervoxel_ids = np.unique(true_l1_edges)

# get the l2 ids for these supervoxels...
l2_ids = client.chunkedgraph.get_roots(
    unique_supervoxel_ids, stop_layer=2, timestamp=timestamp
)
l1_to_l2_map = dict(zip(unique_supervoxel_ids, l2_ids))

# ...and then map the edges to their l2 counterparts
true_edges = pd.DataFrame(true_l1_edges, columns=["pre", "post"]).map(
    lambda x: l1_to_l2_map[x]
)
# remove edges between l1s in the same l2
true_edges = true_edges.query("pre != post").values

# sort and remove duplicates
true_edges = pd.MultiIndex.from_arrays(np.unique(np.sort(true_edges, axis=1), axis=0).T)

# %%
my_edges.difference(true_edges)
