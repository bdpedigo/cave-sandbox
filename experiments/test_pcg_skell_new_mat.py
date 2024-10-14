# %%
import pcg_skel
from pcg_skel import add_volumetric_properties

import caveclient

datastack = "minnie65_public"
client = caveclient.CAVEclient(datastack)
client.materialize.version = 795  # Ensure we will always use this data release

# %%
root_id = 864691135397503777

skel = pcg_skel.pcg_skeleton(root_id=root_id, client=client)

# %%
# Get the location of the soma from nucleus detection:
root_resolution = [
    1,
    1,
    1,
]  # Cold be another resolution as well, but this will mean the location is in nm.
soma_df = client.materialize.views.nucleus_detection_lookup_v1(
    pt_root_id=root_id
).query(desired_resolution=root_resolution)
soma_location = soma_df["pt_position"].values[0]

# Use the above parameters in the skeletonization:

skel = pcg_skel.pcg_skeleton(
    root_id,
    client,
    root_point=soma_location,
    root_point_resolution=root_resolution,
    collapse_soma=True,
    collapse_radius=7500,
)
# %%

nrn = pcg_skel.pcg_meshwork(
    root_id=root_id,
    client=client,
    root_point=soma_location,
    root_point_resolution=root_resolution,
    collapse_soma=True,
    collapse_radius=7500,
    synapses=True,
)

# %%

nrn = pcg_skel.pcg_meshwork(
    root_id=root_id,
    client=client,
    root_point=soma_location,
    root_point_resolution=root_resolution,
    collapse_soma=True,
    collapse_radius=7500,
    synapses=True,
    live_query=True,
)
# %%

add_volumetric_properties(nrn, client)

# %%

client = caveclient.CAVEclient(datastack)

nrn = pcg_skel.pcg_meshwork(
    root_id=root_id,
    client=client,
    root_point=soma_location,
    root_point_resolution=root_resolution,
    collapse_soma=True,
    collapse_radius=7500,
    synapses=True,
    live_query=True,
)
