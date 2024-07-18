# %%

import numpy as np

import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")
client.chunkedgraph._server_version = "0.1.0"
client.chunkedgraph.level2_chunk_graph(
    2, np.array([[0, 10000], [0, 1000], [0, 1000]])
)

#%%
import inspect
inspect.signature(client.chunkedgraph.level2_chunk_graph)

# client.chunkedgraph.level2_chunk_graph(
#     2, bounds=None

#%%
import inspect 

inspect.getfullargspec(client.chunkedgraph.level2_chunk_graph)

#%%
# )
type(client.chunkedgraph.level2_chunk_graph.__kwdefaults__)

#%%
type(client.chunkedgraph.level2_chunk_graph.__defaults__)