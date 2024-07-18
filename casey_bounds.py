# %%

import numpy as np
import pandas as pd
from nglui import statebuilder

import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")
cg = client.chunkedgraph
cv = client.info.segmentation_cloudvolume()

seg_resolution = np.array(cv.mip_resolution(0))
viewer_resolution = client.info.viewer_resolution()


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


l2ids = [
    160032475051983415,
    160032543771460210,
    160032612490936816,
    160032681210413593,
    160032749929890185,
    160032749929890340,
    160032818649367090,
    160102843796161019,
    160102912515637813,
    160102981235115106,
    160103049954591364,
    160103049954591386,
    160103118674068005,
    160103187393544707,
    160173556137722487,
]
chunks = [cv.meta.decode_chunk_position(l2id) for l2id in l2ids]
lbs = [chunk_to_nm(ch, cv) / [4, 4, 40] for ch in chunks]
ubs = [lb + chunk_dims(cv) / [4, 4, 40] for lb in lbs]


img, seg = statebuilder.from_client(client)
seg.add_selection_map(fixed_ids=l2ids)
anno = statebuilder.AnnotationLayerConfig(
    "l2chunks",
    mapping_rules=statebuilder.BoundingBoxMapper(
        point_column_a="lb",
        point_column_b="ub",
    ),
    color='white'
)

sb = statebuilder.StateBuilder([img, seg, anno], client=client)

df = pd.DataFrame(
    {
        "lb": [lb.squeeze() for lb in lbs],
        "ub": [ub.squeeze() for ub in ubs],
    }
)

sb.render_state(df, return_as="html")
