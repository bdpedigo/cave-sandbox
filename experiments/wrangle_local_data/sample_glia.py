# %%
import pandas as pd

from caveclient import CAVEclient

client = CAVEclient("minnie65_public", version=1078)
cell_table = pd.read_csv("cave-sandbox/data/joint_cell_table.csv", low_memory=False)

# %%
from nglui.statebuilder import make_neuron_neuroglancer_link
from IPython.display import display, Markdown
cell_table.query("coarse_type == 'nonneuron'")["cell_type"].unique()

for subtype, subtype_table in cell_table.query("coarse_type == 'nonneuron'").groupby(
    "cell_type"
):
    roots = subtype_table.sample(100)["pt_root_id"]

    display(make_neuron_neuroglancer_link(client, roots, timestamp=client.timestamp, link_text=subtype))
        
    print()

# %%
