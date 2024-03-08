# %%
from datetime import datetime

import numpy as np
import pandas as pd
from nglui import statebuilder

import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")
cg = client.chunkedgraph
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


def get_node_data_for_edges(edges: pd.DataFrame):
    l2_ids = np.unique(edges.values)
    l2_data = pd.DataFrame(client.l2cache.get_l2data(l2_ids)).T
    l2_data.index = l2_data.index.astype(int)
    l2_data["x"] = l2_data["rep_coord_nm"].apply(lambda x: x[0])
    l2_data["y"] = l2_data["rep_coord_nm"].apply(lambda x: x[1])
    l2_data["z"] = l2_data["rep_coord_nm"].apply(lambda x: x[2])
    l2_data[["x", "y", "z"]] = l2_data[["x", "y", "z"]] / viewer_resolution
    return l2_data


def make_bbox_nodes_edges(bbox: np.ndarray):
    # make a dataframe of lines for the edges of the bounding box
    vertices = [
        [bbox[0, 0], bbox[0, 1], bbox[0, 2]],
        [bbox[1, 0], bbox[0, 1], bbox[0, 2]],
        [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
        [bbox[1, 0], bbox[1, 1], bbox[0, 2]],
        [bbox[0, 0], bbox[0, 1], bbox[1, 2]],
        [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
        [bbox[0, 0], bbox[1, 1], bbox[1, 2]],
        [bbox[1, 0], bbox[1, 1], bbox[1, 2]],
    ]

    node_df = pd.DataFrame(
        vertices, columns=["x", "y", "z"], index=range(len(vertices))
    )

    box_edges = [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [4, 5],
        [5, 7],
        [7, 6],
        [6, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    edge_df = pd.DataFrame(box_edges, columns=["source", "target"])
    return node_df, edge_df


sbs = []
dfs = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
)
seg_layer = statebuilder.SegmentationLayerConfig(
    client.info.segmentation_source(), alpha_3d=0.3
)
seg_layer.add_selection_map(selected_ids_column="root_id")

base_sb = statebuilder.StateBuilder(
    [img_layer, seg_layer],
    client=client,
    resolution=viewer_resolution,
    view_kws={"position": point_in_cg * np.array([2, 2, 1])},
)
base_df = pd.DataFrame({"root_id": [root_id]})
sbs.append(base_sb)
dfs.append(base_df)


bbox_ngl = bbox_cg * np.array([2, 2, 1])
node_df, edge_df = make_bbox_nodes_edges(bbox_ngl)

line_sb, lines_df = spatial_graph_mapper(
    node_df,
    edge_df,
    point_columns=["x", "y", "z"],
    resolution=viewer_resolution,
    layer_name="bbox",
)
dfs.append(lines_df)
sbs.append(line_sb)


l2_node_data = get_node_data_for_edges(true_edges)
l2_line_sb, l2_line_df = spatial_graph_mapper(
    l2_node_data,
    true_edges,
    point_columns=["x", "y", "z"],
    layer_name="cave_l2_graph",
    resolution=viewer_resolution,
    color="red",
)
dfs.append(l2_line_df)
sbs.append(l2_line_sb)


l2_id = l2_node_data.index[0]

chunk_loc = cv.meta.decode_chunk_position(l2_id)
offset_vox = np.array(cv.meta.voxel_offset(0))

lb = (
    offset_vox + np.array(np.atleast_2d(chunk_loc) * cv.meta.graph_chunk_size).squeeze()
)
ub = np.array((lb + cv.meta.chunk_size(0)).squeeze())

scaling = np.array(cv.mip_resolution(0) / viewer_resolution)
lb = lb * scaling
ub = ub * scaling

bbox = np.array([lb, ub], dtype=int)

l2_position = l2_node_data.loc[l2_id][["x", "y", "z"]].values

if not (l2_position[0] < bbox[1, 0] and l2_position[0] > bbox[0, 0]):
    print("out of x range")
if not (l2_position[1] < bbox[1, 1] and l2_position[1] > bbox[0, 1]):
    print("out of y range")
if not (l2_position[2] < bbox[1, 2] and l2_position[2] > bbox[0, 2]):
    print("out of z range")


node_df, edge_df = make_bbox_nodes_edges(bbox)

line_sb, lines_df = spatial_graph_mapper(
    node_df,
    edge_df,
    point_columns=["x", "y", "z"],
    resolution=viewer_resolution,
    layer_name="bbox-l2",
)
dfs.append(lines_df)
sbs.append(line_sb)


return_as = "html"
sb = statebuilder.ChainedStateBuilder(sbs)
statebuilder.helpers.package_state(dfs, sb, client=client, return_as=return_as)
