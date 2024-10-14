# %%

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from caveclient import CAVEclient
from minniemorpho.segclr import SegCLRQuery

# client = GrottoClient("minnie65_phase3_v1")
client = CAVEclient("minnie65_phase3_v1")

label_df = pd.read_csv("cave-sandbox/data/assembled_local_labels.csv")
counts = label_df["level2_id"].value_counts()
label_df = label_df[label_df["level2_id"].isin(counts[counts == 1].index)]

assert label_df["level2_id"].is_unique

# %%

root_ids = np.unique(label_df["root_id"])

query = SegCLRQuery(client, verbose=True, n_jobs=-1, version=943)
query.set_query_ids(root_ids)
query.map_to_version()
query.get_embeddings()
query.map_to_level2()

# %%
feature_cols = np.arange(64)
embedding_df = query.features_[feature_cols]
info_df = query.level2_mapping_

# %%
info_df["label"] = info_df["level2_id"].map(
    label_df.set_index("level2_id")["classification"]
)

# %%

info_df["distance_to_level2_node"].hist(bins=100)


# %%
info_df = info_df[info_df["label"].notnull()]
info_df = info_df[info_df["distance_to_level2_node"] < 2000]


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

all_ids = labeled_embedding_df.index.get_level_values("current_id").unique()

train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

train_embedding_df = labeled_embedding_df.loc[train_ids]
test_embedding_df = labeled_embedding_df.loc[test_ids]

# train_embedding_df = labeled_embedding_df.sample(400_000)
# test_embedding_df = labeled_embedding_df.drop(train_embedding_df.index)

# %%


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
            ("logistic", LogisticRegression(max_iter=2000, n_jobs=-1)),
        ]
    ),
    "std_logistic_balanced": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced"),
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
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
sns.set_context("talk")
sns.heatmap(
    conf_mat,
    annot=True,
    cmap="Reds",
    fmt=".2f",
    ax=ax,
    square=True,
    cbar_kws=dict(shrink=0.7),
)
ax.set_yticklabels(
    [tick.get_text().capitalize() for tick in ax.get_yticklabels()], rotation=0
)
ax.set_xticklabels(
    [tick.get_text().capitalize() for tick in ax.get_xticklabels()], rotation=45
)

plt.savefig('segclr_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('segclr_confusion_matrix.svg', dpi=300, bbox_inches='tight')

# %%

final_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                max_iter=2000, n_jobs=-1, class_weight="balanced", verbose=1
            ),
        ),
    ]
)

final_model.fit(labeled_embedding_df[embedding_cols], labeled_embedding_df["label"])

# %%


# dump(final_model, "cave-sandbox/models/segclr_logreg_bdp.skops")

# %%
