# %%
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")

roots = [
    864691133389721621,
    864691133389721621,
    864691133389721621,
    864691133389721621,
    864691133597902842,
    864691133597902842,
    864691133597902842,
    864691134024938078,
    864691134024938078,
    864691134024938078,
]
client.chunkedgraph.get_leaves_many([roots])