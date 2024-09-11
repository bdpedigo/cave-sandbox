# %%

import numpy as np

from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")

# %%
# this raises HTTPError: 413 Client Error: Request Entity Too Large
n_nodes = 500_000
state_dict = {}
for i in range(n_nodes):
    state_dict[str(i)] = np.random.uniform()

client.state.upload_property_json(state_dict, max_size=None)
# %%
# this raises HTTPError: 502 Server Error: Bad Gateway for url
n_nodes = 100_000
state_dict = {}
for i in range(n_nodes):
    state_dict[str(i)] = np.random.uniform()

client.state.upload_property_json(state_dict, max_size=None)

# %%
# this works
n_nodes = 90_000
state_dict = {}
for i in range(n_nodes):
    state_dict[str(i)] = np.random.uniform()

client.state.upload_property_json(state_dict, max_size=None)
