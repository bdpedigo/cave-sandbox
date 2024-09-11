# %%
import requests.adapters

from caveclient import CAVEclient, set_session_defaults, get_session_defaults

set_session_defaults()
get_session_defaults()

#%%

pool_maxsize = 21
pool_block = True
max_retries = 5
# SESSION_DEFAULTS["pool_maxsize"] = val
set_session_defaults(
    pool_maxsize=pool_maxsize,
    pool_block=pool_block,
    max_retries=max_retries,
    backoff_factor=0.5,
    backoff_max=240,
    status_forcelist=(502, 503, 504, 505),
)

client = CAVEclient(
    "minnie65_phase3_v1",
    # max_retries=0,
    # pool_maxsize=20,
    # pool_block=True
)

assert client.l2cache.session.adapters["https://"]._pool_maxsize == pool_maxsize
assert client.l2cache.session.adapters["https://"]._pool_block == True
assert client.l2cache.session.adapters["https://"].max_retries.total == max_retries
assert client.l2cache.session.adapters["https://"].max_retries.backoff_factor == 0.5
assert client.l2cache.session.adapters["https://"].max_retries.backoff_max == 240
assert client.l2cache.session.adapters["https://"].max_retries.status_forcelist == (
    502,
    503,
    504,
    505,
)

# assert client.l2cache.session.adapters["https://"]._pool_maxsize == val

client.materialize.get_timestamp(1078)

# %%

dir(client.l2cache.session.adapters["https://"])

# %%
client.l2cache.session.adapters["https://"].max_retries

# %%
import requests

requests.adapters.DEFAULT_POOLBLOCK

# %%
requests.adapters.DEFAULT_POOLSIZE

# %%
client.l2cache.get_l2data([100])

# %%
from io import BytesIO

import pandas as pd
from cloudfiles import CloudFiles

cf = CloudFiles("gs://allen-minnie-phase3/vasculature_feature_pulls/segclr/2024-08-19")


# %%
def load_dataframe(path, **kwargs):
    bytes_out = cf.get(path)
    with BytesIO(bytes_out) as f:
        df = pd.read_csv(f, **kwargs)
    return df


for file_name in cf.list():
    if "_level2_features" in file_name:
        level2_features = load_dataframe(file_name, index_col=[0, 1])
    break

# %%
l2_ids = level2_features.index.get_level_values("level2_id").unique().values

# %%
import time

client = CAVEclient(
    "minnie65_phase3_v1",
    pool_maxsize=12,
    pool_block=True,
)

currtime = time.time()
client.l2cache.get_l2data([level2_features.index.get_level_values("level2_id")])
print(f"{time.time() - currtime:.3f} seconds elapsed with pool_maxsize=12.")

client = CAVEclient(
    "minnie65_phase3_v1",
)

currtime = time.time()
client.l2cache.get_l2data([level2_features.index.get_level_values("level2_id")])
print(f"{time.time() - currtime:.3f} seconds elapsed with no pool_maxsize.")

# %%
import numpy as np
from joblib import Parallel, delayed

rows = []
for i in range(10):
    client = CAVEclient(
        "minnie65_phase3_v1",
    )

    chunk_size = 5_000
    l2_id_chunks = np.array_split(l2_ids, len(l2_ids) // chunk_size)

    currtime = time.time()
    Parallel(n_jobs=-1)(
        delayed(client.l2cache.get_l2data)(l2_id_chunk) for l2_id_chunk in l2_id_chunks
    )
    elapsed = time.time() - currtime
    rows.append({"pool": False, "elapsed": elapsed})

    client = CAVEclient(
        "minnie65_phase3_v1",
        pool_maxsize=20,
        pool_block=True,
    )

    chunk_size = 5_000
    l2_id_chunks = np.array_split(l2_ids, len(l2_ids) // chunk_size)

    currtime = time.time()
    Parallel(n_jobs=-1)(
        delayed(client.l2cache.get_l2data)(l2_id_chunk) for l2_id_chunk in l2_id_chunks
    )
    elapsed = time.time() - currtime
    rows.append({"pool": True, "elapsed": elapsed})

# %%
results = pd.DataFrame(rows)

import seaborn as sns

sns.stripplot(data=results, x="pool", y="elapsed")

# %%
