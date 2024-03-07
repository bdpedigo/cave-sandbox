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

# %%
import imageryclient as ic

img_client = ic.ImageryClient(client=client)


width = 400
z_slices = 3
ctr = point_in_cg.copy()
ctr = ctr * np.array([2, 2, 1])
ctr = [169000, 166108, 20759]
bounds_3d = ic.bounds_from_center(ctr, width=width, height=width, depth=z_slices)

image, segs = img_client.image_and_segmentation_cutout(
    bounds_3d, split_segmentations=True, scale_to_bounds=True
)

out = ic.composite_overlay(
    segs,
    imagery=image,
    palette="husl",
    outline=True,
    merge_outline=False,
    alpha=0.8,
    side="in",
    width=2,
)
out[0]

#%%
bounds = bounds_3d
bbox_size=None
bbox = img_client._compute_bounds(bounds, bbox_size)
mip = img_client._base_segmentation_mip
resolution = img_client.resolution
volume_cutout = img_client.segmentation_cv.download(bbox, agglomerate=False, mip=mip,coord_resolution=resolution)
volume_cutout = np.array(np.squeeze(volume_cutout))
volume_cutout

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# make a matplotlib custom colormap which maps each label to a unique color 
# from the above

from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import seaborn as sns

unique_ids = np.unique(volume_cutout)
ids_to_inds = dict(zip(unique_ids, range(len(unique_ids))))

# get the unique colors from seaborn
color_map = sns.color_palette("husl", len(unique_ids))

# create the colormap
cmap = ListedColormap(color_map)

# create the norm
norm = BoundaryNorm(np.arange(-0.5, len(color_map), 1), len(color_map))

# remap the volume cutout to the unique colors
img_slice = np.squeeze(volume_cutout[:, :, 0])
row_inds, col_inds = np.nonzero(img_slice)
vals = volume_cutout[row_inds, col_inds].ravel()
ind_vals = np.array([ids_to_inds[val] for val in vals])


img_slice = np.zeros(img_slice.shape)
img_slice[row_inds, col_inds] = ind_vals

# plot the first slice of the volume
ax.imshow(img_slice, cmap=cmap, norm=norm)



# ax.imshow(volume_cutout[:, :, 0], cmap=color_map)

#%%
mask = volume_cutout == volume_cutout[100,100,0]
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(mask[:, :, 0], cmap="gray")

#%%
image, segs = img_client.image_and_segmentation_cutout(
    ctr,
    split_segmentations=True,
    bbox_size=(1024, 1024),
    scale_to_bounds=True,
    root_ids=[root_id],
)

ic.composite_overlay(
    segs,
    imagery=image,
    palette="husl",
    outline=True,
    merge_outline=False,
    alpha=0.8,
    side="in",
    width=2,
)

# %%
import matplotlib.pyplot as plt

arr_image = image.astype(np.uint8)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(arr_image.T, cmap="grey", vmin=0, vmax=255)


def forward(x):
    return x


def inverse(x):
    return x


secax = ax.secondary_xaxis("top", functions=(forward, inverse))
secax.set_xlabel("angle [rad]")
# secax.scatter(ctr[0], ctr[1], color="red")

# %%
scv = img_client.segmentation_cv
dir(scv)

scv.get_chunk_layer(l2_ids[0])

#%%
out = scv.get_chunk_mappings(l2_ids[0])
svs = out[l2_ids[0]]
len(svs)

#%%
len(client.chunkedgraph.get_children(l2_ids[0]))

#%%
help(scv.get_leaves)

#%%
help(get_chunk)

#%%

scv.download(l2_ids[0])

#%%
scv.