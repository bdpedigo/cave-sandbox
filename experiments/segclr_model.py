# %%

import gcsfs
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from tqdm_joblib import tqdm_joblib

from connectomics.segclr import reader
from grotto import GrottoClient
from caveclient import CAVEclient

client = GrottoClient("minnie65_phase3_v1")

label_df = pd.read_csv("cave-sandbox/data/assembled_local_labels.csv")
counts = label_df["level2_id"].value_counts()
label_df = label_df[label_df["level2_id"].isin(counts[counts == 1].index)]

assert label_df["level2_id"].is_unique

# %%
timestamp_343 = client.get_timestamp(343)

root_ids = np.unique(label_df["root_id"])


def get_past_id_map_for_chunk(chunk):
    out = client.get_past_ids(chunk, timestamp_past=timestamp_343)
    return out["past_id_map"]


chunk_size = 1000
n_chunks = len(root_ids) // chunk_size + 1
chunks = np.array_split(root_ids, n_chunks)

with tqdm_joblib(desc="Getting past id map", total=n_chunks):
    past_id_maps = Parallel(n_jobs=-1)(
        delayed(get_past_id_map_for_chunk)(chunk) for chunk in chunks
    )

past_id_map = {}
for past_id_map_chunk in past_id_maps:
    past_id_map.update(past_id_map_chunk)


forward_id_map = {}
for current_id, past_ids in past_id_map.items():
    for past_id in past_ids:
        forward_id_map[past_id] = current_id

past_root_ids = np.unique(list(forward_id_map.keys()))

# %%

PUBLIC_GCSFS = gcsfs.GCSFileSystem(token="anon")

embedding_reader = reader.get_reader("microns_v343", PUBLIC_GCSFS)


def get_embeddings_for_past_id(past_id):
    past_id = int(past_id)
    root_id = forward_id_map[past_id]
    try:
        out = embedding_reader[past_id]
        new_out = {}
        for xyz, embedding_vector in out.items():
            new_out[(root_id, past_id, *xyz)] = embedding_vector
    except KeyError:
        new_out = {}
    return new_out


get_embeddings_for_past_id(past_root_ids[0])
# %%

with tqdm_joblib(desc="Getting embeddings", total=len(past_root_ids)):
    embeddings_dicts = Parallel(n_jobs=-1)(
        delayed(get_embeddings_for_past_id)(past_id) for past_id in past_root_ids
    )

embeddings_dict = {}
for d in embeddings_dicts:
    embeddings_dict.update(d)

embedding_df = pd.DataFrame(embeddings_dict).T
embedding_df.index.names = ["root_id", "past_id", "x", "y", "z"]

embedding_df["x_nm"] = embedding_df.index.get_level_values("x") * 32
embedding_df["y_nm"] = embedding_df.index.get_level_values("y") * 32
embedding_df["z_nm"] = embedding_df.index.get_level_values("z") * 40
mystery_offset = np.array([13824, 13824, 14816]) * np.array([8, 8, 40])
embedding_df["x_nm"] += mystery_offset[0]
embedding_df["y_nm"] += mystery_offset[1]
embedding_df["z_nm"] += mystery_offset[2]

# %%


def map_to_level2(root_id, root_data):
    level2_ids = client.get_leaves(root_id, stop_layer=2)
    l2_data = client.get_l2data(level2_ids)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(l2_data[["x", "y", "z"]].values)
    distances, indices = nn.kneighbors(root_data[["x_nm", "y_nm", "z_nm"]].values)
    distances = distances.squeeze()
    indices = indices.squeeze()

    info_df = pd.DataFrame(index=root_data.index)
    info_df["level2_id"] = l2_data.index[indices]
    info_df["distance_to_level2_node"] = distances
    return info_df


# info_dfs = []
# for root_id, root_data in tqdm(
#     embedding_df.groupby("root_id"),
#     total=len(embedding_df.index.get_level_values("root_id").unique()),
# ):
#     info_df = map_to_level2(root_id, root_data)
#     info_dfs.append(info_df)

with tqdm_joblib(
    desc="Mapping to level2",
    total=len(embedding_df.index.get_level_values("root_id").unique()),
):
    info_dfs = Parallel(n_jobs=-1)(
        delayed(map_to_level2)(root_id, root_data)
        for root_id, root_data in embedding_df.groupby("root_id")
    )
info_df = pd.concat(info_dfs)

# %%
info_df["label"] = info_df["level2_id"].map(
    label_df.set_index("level2_id")["classification"]
)

# %%

info_df["distance_to_level2_node"].hist(bins=100)


# %%
info_df = info_df[info_df["label"].notnull()]
info_df = info_df[info_df["distance_to_level2_node"] < 3000]


# %%
if "label" in embedding_df.columns:
    embedding_df = embedding_df.drop(columns="label")
if "level2_id" in embedding_df.columns:
    embedding_df = embedding_df.drop(columns="level2_id")
if "distance_to_level2_node" in embedding_df.columns:
    embedding_df = embedding_df.drop(columns="distance_to_level2_node")
# %%
embedding_df = embedding_df.join(info_df)

# %%
labeled_embedding_df = embedding_df[embedding_df["label"].notnull()]
# labeled_embedding_df = labeled_embedding_df.query("label != 'soma'")
# %%

train_embedding_df = labeled_embedding_df.sample(400_000)
test_embedding_df = labeled_embedding_df.drop(train_embedding_df.index)

# %%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {
    # "quantile_lda": Pipeline(
    #     [
    #         ("scaler", QuantileTransformer(output_distribution="normal")),
    #         ("lda", LinearDiscriminantAnalysis()),
    #     ]
    # ),
    # "std_lda": Pipeline(
    #     [("scaler", StandardScaler()), ("lda", LinearDiscriminantAnalysis())]
    # ),
    # "lda": LinearDiscriminantAnalysis(),
    # "quantile_logistic": Pipeline(
    #     [
    #         ("scaler", QuantileTransformer(output_distribution="normal")),
    #         ("logistic", LogisticRegression(max_iter=500, n_jobs=-1)),
    #     ]
    # ),
    "std_logistic": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=500, n_jobs=-1)),
        ]
    ),
    "std_logistic_balanced": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"),
            ),
        ]
    ),
    # "logistic": LogisticRegression(),
}

embedding_cols = np.arange(64)
for model_name, model in models.items():
    X_train = train_embedding_df[embedding_cols]
    X_test = test_embedding_df[embedding_cols]
    y_train = train_embedding_df["label"]
    y_test = test_embedding_df["label"]

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Model:", model_name)

    print("Train")
    print(classification_report(y_train, y_train_pred))

    print("Test")
    print(classification_report(y_test, y_test_pred))

    print()

import seaborn as sns

if "lda" in models:
    model = models["lda"]
    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)

    transformed_df = test_embedding_df.copy()
    for i, col in enumerate(Z_test.T):
        transformed_df[f"lda{i}"] = col

    pg = sns.PairGrid(
        transformed_df,
        hue="label",
        vars=[f"lda{i}" for i in range(Z_test.shape[1])],
        corner=True,
    )

    pg.map_lower(sns.scatterplot, s=10, linewidth=0, alpha=0.1)

# %%

best_model = models["std_logistic_balanced"]

from sklearn.metrics import confusion_matrix

y_test_pred = best_model.predict(X_test)
conf_mat = confusion_matrix(
    y_test, y_test_pred, labels=best_model.classes_, normalize="true"
)

conf_mat = pd.DataFrame(
    conf_mat, index=best_model.classes_, columns=best_model.classes_
)
conf_mat.index.name = "True"
conf_mat.columns.name = "Predicted"

# %%
sns.heatmap(conf_mat, annot=True, cmap="Reds", fmt=".2f")

# %%
