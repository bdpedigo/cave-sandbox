# %%
import pandas as pd

pd.read_feather(
    "cave-sandbox/data/segclr_training_data/labeled_cell_m343_df_221011b.feather"
)

# %%
compartment_df = pd.read_feather(
    "cave-sandbox/data/segclr_training_data/microns_compartment_m343.feather"
)
compartment_df["class"].unique()

# %%

pd.read_feather("cave-sandbox/data/segclr_training_data/microns_ct_m343.feather")

# %%
import gcsfs

from connectomics.segclr import reader

PUBLIC_GCSFS = gcsfs.GCSFileSystem(token="anon")
embedding_reader = reader.get_reader("microns_v343", PUBLIC_GCSFS)
test_root = 864691135988665856
embeddings = embedding_reader[test_root]

embeddings = pd.DataFrame(embeddings).T
embeddings

# %%
example_compartment_df = compartment_df.set_index("skeletonid").loc[test_root]

# %%
