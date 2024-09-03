# %%
import io
import re
from io import BytesIO

import numpy as np
import pandas as pd
from cloudfiles import CloudFiles
from meshparty import meshwork
from tqdm.auto import tqdm


def load_mw(directory, filename):
    # REF: stolen from https://github.com/AllenInstitute/skeleton_plot/blob/main/skeleton_plot/skel_io.py
    # filename = f"{root_id}_{nuc_id}/{root_id}_{nuc_id}.h5"
    '''
    """loads a meshwork file from .h5 into meshparty.meshwork object

    Args:
        directory (str): directory location of meshwork .h5 file. in cloudpath format as seen in https://github.com/seung-lab/cloud-files
        filename (str): full .h5 filename

    Returns:
        meshwork (meshparty.meshwork): meshwork object containing .h5 data
    """'''

    if "://" not in directory:
        directory = "file://" + directory

    cf = CloudFiles(directory)
    binary = cf.get([filename])

    with io.BytesIO(cf.get(binary[0]["path"])) as f:
        f.seek(0)
        mw = meshwork.load_meshwork(f)

    return mw


meshwork_path = "gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons/axon_dendrite_classifier/groundtruth_mws"
meshwork_cf = CloudFiles(meshwork_path)

ground_truth_path = "gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons/axon_dendrite_classifier/groundtruth_feats_and_class"
ground_truth_cf = CloudFiles(ground_truth_path)

# %%


for meshwork_file in meshwork_cf.list():
    if meshwork_file.endswith(".h5"):
        mw = load_mw(meshwork_path, meshwork_file)
        break

# %%

for ground_truth_file in ground_truth_cf.list():
    if ground_truth_file.endswith(".csv"):
        ground_truth = ground_truth_cf.get(ground_truth_file)
        ground_truth_df = pd.read_csv(BytesIO(ground_truth), index_col=0)
        break

# %%


ground_truth_root_ids = [
    int(name.split("_")[0]) for name in ground_truth_cf.list() if name.endswith(".csv")
]
meshwork_root_ids = [
    int(name.split("mesh")[0]) for name in meshwork_cf.list() if name.endswith(".h5")
]
has_ground_truth = np.intersect1d(ground_truth_root_ids, meshwork_root_ids)


def string_to_list(string):
    string = re.sub("\s+", ",", string)
    if string.startswith("[,"):
        string = "[" + string[2:]
    return eval(string)


label_map = {0: "dendrite", 1: "axon"}

all_l2_dfs = []
for root_id in tqdm(has_ground_truth[:]):
    ground_truth_file = f"{root_id}_feats.csv"
    ground_truth_bytes = ground_truth_cf.get(ground_truth_file)
    ground_truth_df = pd.read_csv(BytesIO(ground_truth_bytes), index_col=0)

    meshwork_file = f"{root_id}mesh.h5"
    mw = load_mw(meshwork_path, meshwork_file)

    ground_truth_df["segment"] = ground_truth_df["segment"].apply(string_to_list)

    root_skel_id = list(mw.root_skel)[0]

    ground_truth_df["classification"] = ground_truth_df["classification"].map(label_map)

    has_soma = ground_truth_df["segment"].apply(lambda x: root_skel_id in x)
    ground_truth_df.loc[has_soma, "classification"] = "soma"

    l2_index = pd.Index(mw.anno["lvl2_ids"]["lvl2_id"])
    l2_df = pd.Series(
        data=mw.skeleton.mesh_to_skel_map, index=l2_index, name="skeleton_index"
    ).to_frame()

    skeleton_df = ground_truth_df.explode("segment").set_index("segment")
    skeleton_df.index.name = "skeleton_index"

    l2_df["classification"] = l2_df["skeleton_index"].map(skeleton_df["classification"])
    l2_df["root_id"] = root_id

    all_l2_dfs.append(l2_df)

all_l2_df = pd.concat(all_l2_dfs)


# %%
all_l2_df.classification.value_counts()

# %%

n_samples = 30
level2_ids = all_l2_df.query("classification != 'soma'").sample(n_samples).index

from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")
cv = client.info.segmentation_cloudvolume()
meshes = cv.mesh.get(level2_ids, allow_missing=True)

import pyvista as pv

from neurovista import to_mesh_polydata

pv.set_jupyter_backend("client")


plotter = pv.Plotter(shape=(3, 10))
for i, mesh in enumerate(meshes.values()):
    plotter.subplot(i // 10, i % 10)
    pv_mesh = to_mesh_polydata(mesh.vertices, mesh.faces)
    plotter.add_mesh(pv_mesh)
plotter.show()

# %%
all_l2_df.query("classification != 'soma'", inplace=True)
all_l2_df.index.name = "level2_id"
all_l2_df.drop(columns="skeleton_index")[["root_id", "classification"]].to_csv(
    "cave-sandbox/data/processed_local_labels/ej_skeletons.csv"
)
