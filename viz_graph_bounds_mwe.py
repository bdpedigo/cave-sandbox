# %%
from datetime import datetime

import numpy as np
import pandas as pd
from nglui import statebuilder

import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")
# cg = client.chunkedgraph
cg = cc.chunkedgraph.ChunkedGraphClient(
    server_address="http://0.0.0.0:5001",
    auth_client=client.auth,
    table_name=client.chunkedgraph._table_name,
    verify=False,
)
cv = client.info.segmentation_cloudvolume()

seg_resolution = np.array(cv.mip_resolution(0))
viewer_resolution = client.info.viewer_resolution()

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

point_in_nm = point_in_cg * seg_resolution

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

    start_point_cg = np.array([x_start, y_start, z_start]) / seg_resolution
    stop_point_cg = np.array([x_stop, y_stop, z_stop]) / seg_resolution

    bbox_cg = np.array([start_point_cg, stop_point_cg], dtype=int)
    return bbox_cg


bbox_cg = make_bbox(bbox_halfwidth, point_in_nm)

# %%
my_edges = cg.level2_chunk_graph(root_id, bounds=bbox_cg.T)
my_edges = np.array(my_edges, dtype=np.uint64)
my_edges = np.unique(np.sort(my_edges, axis=1), axis=0)
my_edges = pd.DataFrame(my_edges, columns=["source", "target"])

# %%
# get a subgraph of l1 edges
true_l1_edges, affinities, areas = client.chunkedgraph.get_subgraph(root_id, bbox_cg.T)

# get the unique supervoxel ids mentioned in these edges
unique_supervoxel_ids = np.unique(true_l1_edges)

# get the l2 ids for these supervoxels...
l2_ids = client.chunkedgraph.get_roots(
    unique_supervoxel_ids, stop_layer=2, timestamp=timestamp
)
l1_to_l2_map = dict(zip(unique_supervoxel_ids, l2_ids))

# ...and then map the edges to their l2 counterparts
true_edges = (
    pd.DataFrame(true_l1_edges, columns=["pre", "post"])
    .map(lambda x: l1_to_l2_map[x])
    .astype(int)
)

# remove edges between l1s in the same l2
true_edges = true_edges.query("pre != post").values

# sort and remove duplicates
true_edges = pd.DataFrame(
    np.unique(np.sort(true_edges, axis=1), axis=0), columns=["source", "target"]
)


# %%
def spatial_graph_mapper(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    point_columns: str,
    layer_name="",
    resolution=None,
    color="white",
):
    if resolution is None:
        resolution = [1, 1, 1]

    edges_df = edges_df.copy()
    if "pre" in edges_df.columns and "post" in edges_df.columns:
        edges_df = edges_df.rename(columns={"pre": "source", "post": "target"})
    if 0 in edges_df.columns and 1 in edges_df.columns:
        edges_df = edges_df.rename(columns={0: "source", 1: "target"})

    line_rows = []
    for edge in edges_df.iterrows():
        source = edge[1]["source"]
        target = edge[1]["target"]
        source_point = nodes_df.loc[source][point_columns]
        target_point = nodes_df.loc[target][point_columns]
        if isinstance(source_point, pd.Series):
            # preprend "source" to everything index element for this series
            source_point.index = "source_" + source_point.index
            # preprend "target" to everything index element for this series
            target_point.index = "target_" + target_point.index
            source_point = source_point.to_dict()
            target_point = target_point.to_dict()
            line_rows.append(
                {
                    **source_point,
                    **target_point,
                }
            )
        else:
            line_rows.append(
                {
                    "start": source_point,
                    "end": target_point,
                }
            )

    line_df = pd.DataFrame(line_rows)

    line_mapper = statebuilder.LineMapper(
        point_column_a="source",
        point_column_b="target",
        set_position=False,
        split_positions=True,
    )
    line_layer = statebuilder.AnnotationLayerConfig(
        mapping_rules=line_mapper,
        name=layer_name,
        color=color,
    )
    line_sb = statebuilder.StateBuilder(
        [line_layer], client=client, resolution=resolution
    )

    return line_sb, line_df


def chunk_to_nm(xyz_ch, cv):
    """Map a chunk location to Euclidean space

    Parameters
    ----------
    xyz_ch : array-like
        Nx3 array of chunk indices
    cv : cloudvolume.CloudVolume
        CloudVolume object associated with the chunked space
    voxel_resolution : list, optional
        Voxel resolution, by default [4, 4, 40]

    Returns
    -------
    np.array
        Nx3 array of spatial points
    """
    x_vox = np.atleast_2d(xyz_ch) * cv.mesh.meta.meta.graph_chunk_size
    return (x_vox + np.array(cv.mesh.meta.meta.voxel_offset(0))) * cv.mip_resolution(0)


def chunk_dims(cv):
    """Gets the size of a chunk in euclidean space

    Parameters
    ----------
    cv : cloudvolume.CloudVolume
        Chunkedgraph-targeted cloudvolume object

    Returns
    -------
    np.array
        3-element box dimensions of a chunk in nanometers.
    """
    dims = chunk_to_nm([1, 1, 1], cv) - chunk_to_nm([0, 0, 0], cv)
    return np.squeeze(dims)


def get_node_data_for_edges(edges: pd.DataFrame):
    l2_ids = np.unique(edges.values)
    l2_data = pd.DataFrame(client.l2cache.get_l2data(l2_ids)).T
    l2_data.index = l2_data.index.astype(int)
    l2_data["x"] = l2_data["rep_coord_nm"].apply(lambda x: x[0])
    l2_data["y"] = l2_data["rep_coord_nm"].apply(lambda x: x[1])
    l2_data["z"] = l2_data["rep_coord_nm"].apply(lambda x: x[2])
    l2_data[["x", "y", "z"]] = l2_data[["x", "y", "z"]] / viewer_resolution
    chunks = [cv.meta.decode_chunk_position(l2id) for l2id in l2_ids]
    lbs = [chunk_to_nm(ch, cv) / [4, 4, 40] for ch in chunks]
    ubs = [lb + chunk_dims(cv) / [4, 4, 40] for lb in lbs]
    l2_data["lb"] = [lb.squeeze() for lb in lbs]
    l2_data["ub"] = [ub.squeeze() for ub in ubs]
    return l2_data


# %%
l2_node_data = get_node_data_for_edges(my_edges)
l2_line_sb, l2_line_df = spatial_graph_mapper(
    l2_node_data,
    my_edges,
    point_columns=["x", "y", "z"],
    layer_name="cave_l2_graph",
    resolution=viewer_resolution,
    color="red",
)

#%%
sbs = []
dfs = []
img, seg = statebuilder.from_client(client)
seg.add_selection_map(fixed_ids=l2_node_data.index)
seg._view_kws["alpha_3d"] = 0.3
anno = statebuilder.AnnotationLayerConfig(
    "l2chunks",
    mapping_rules=statebuilder.BoundingBoxMapper(
        point_column_a="lb",
        point_column_b="ub",
    ),
    color="white",
)

sb = statebuilder.StateBuilder([img, seg, anno], client=client)
sbs.append(sb)
dfs.append(l2_node_data)

anno = statebuilder.AnnotationLayerConfig(
    "query_bounds",
    mapping_rules=statebuilder.BoundingBoxMapper(
        point_column_a="lb",
        point_column_b="ub",
    ),
    # color='white'
)
anno_sb = statebuilder.StateBuilder([anno], client=client, resolution=viewer_resolution)
query_bbox_df = pd.DataFrame(
    [
        {
            "lb": bbox_cg[0].squeeze() * np.array([2, 2, 1]),
            "ub": bbox_cg[1].squeeze() * np.array([2, 2, 1]),
        }
    ]
)
dfs.append(query_bbox_df)
sbs.append(anno_sb)



dfs.append(l2_line_df)
sbs.append(l2_line_sb)

return_as = 'html'
sb = statebuilder.ChainedStateBuilder(sbs)
statebuilder.helpers.package_state(dfs, sb, client=client, return_as=return_as)

# %%
