# %%

import datetime

from caveclient import CAVEclient

versioned_client = CAVEclient("minnie65_phase3_v1", version=1078)
print(versioned_client.version)
print(versioned_client.timestamp)

# %%
print(versioned_client.materialize.version)

# %%
print(versioned_client.chunkedgraph.timestamp)

# %%
# should break
try:
    versioned_client.materialize.version = 1000
except Exception as e:
    print(e)

# %%
# should break
try:
    versioned_client.chunkedgraph.timestamp = datetime.datetime.now()
except Exception as e:
    print(e)

# %%

# now actually change the version
versioned_client.version = 661
print(versioned_client.version)
print(versioned_client.timestamp)

# %%
# this is the ts for 661
ts = datetime.datetime(2023, 4, 6, 20, 17, 9, 199182, tzinfo=datetime.timezone.utc)

unversioned_client = CAVEclient("minnie65_phase3_v1")

# this root is valid now, but not at 661
root = 864691135013958390
# %%
# should be valid
print(unversioned_client.chunkedgraph.is_valid_nodes(root))

# should not be valid
print(unversioned_client.chunkedgraph.is_valid_nodes(root, end_timestamp=ts))
print(versioned_client.chunkedgraph.is_valid_nodes(root))

# %%
# should be itself
print(unversioned_client.chunkedgraph.get_latest_roots(root))

# should be other nodes
print(unversioned_client.chunkedgraph.get_latest_roots(root, timestamp=ts))
print(versioned_client.chunkedgraph.get_latest_roots(root))

# %%

# should get a valid nucleus
nuc_df = unversioned_client.materialize.query_table(
    "nucleus_detection_v0", filter_equal_dict={"pt_root_id": root}
)
print(nuc_df["pt_position"].values)

# %%
# should be empty, node didn't exist at that time
nuc_df = unversioned_client.materialize.query_table(
    "nucleus_detection_v0",
    filter_equal_dict={"pt_root_id": root},
    timestamp=ts,
)
print(nuc_df["pt_position"].values)

# %%
# again, should be empty, node didn't exist at that time
nuc_df = versioned_client.materialize.query_table(
    "nucleus_detection_v0", filter_equal_dict={"pt_root_id": root}
)
print(nuc_df["pt_position"].values)

# %%
# this ID is where the nucleus was at 661, should get a response
old_root = 864691135888784265
nuc_df = unversioned_client.materialize.query_table(
    "nucleus_detection_v0",
    filter_equal_dict={"pt_root_id": old_root},
    materialization_version=661,
)
print(nuc_df["pt_position"].values)

# %%
# response should be the same as the above
nuc_df = versioned_client.materialize.query_table(
    "nucleus_detection_v0",
    filter_equal_dict={"pt_root_id": old_root},
)
print(nuc_df["pt_position"].values)

