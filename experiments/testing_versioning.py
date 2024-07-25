# %%

import datetime
import logging
from http.client import HTTPConnection

from caveclient import CAVEclient


def debug_requests_on():
    """Switches on logging of the requests module."""
    HTTPConnection.debuglevel = 1

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

debug_requests_on()

versioned_client = CAVEclient("minnie65_phase3_v1", version=1078)
print(versioned_client.version)
print(versioned_client.timestamp)

# %%
print(versioned_client.materialize.version)

# %%
print(versioned_client.chunkedgraph.timestamp)

versioned_client.chunkedgraph.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")

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


# %%
versioned_client.materialize.get_views()

# %%
print(
    unversioned_client.materialize.query_view(
        "single_neurons", filter_in_dict={"pt_root_id": [root]}
    )["id"].values
)

# should not be there
print(
    unversioned_client.materialize.query_view(
        "single_neurons",
        filter_in_dict={"pt_root_id": [root]},
        materialization_version=661,
    )["id"].values
)

print(
    unversioned_client.materialize.query_view(
        "single_neurons",
        filter_in_dict={"pt_root_id": [old_root]},
        materialization_version=661,
    )["id"].values
)

print(
    versioned_client.materialize.query_view(
        "single_neurons", filter_in_dict={"pt_root_id": [old_root]}
    )["id"].values
)

# %%
now = datetime.datetime(2024, 7, 18, 10, 17, 49, 822856)
unversioned_client.materialize.live_live_query(
    "nucleus_detection_v0",
    timestamp=now,
    filter_equal_dict={"nucleus_detection_v0": {"pt_root_id": root}},
)

# %%
# should be `root`
unversioned_client.chunkedgraph.suggest_latest_roots(old_root)

# %%
# should be `old_root`
versioned_client.chunkedgraph.suggest_latest_roots(old_root)

# %%

# this is right before ID was created
versioned_client.version = 900

print(versioned_client.timestamp)

print(unversioned_client.chunkedgraph.get_root_timestamps(root))

# suggest latest should give something else
# looked at this object in NGL and it looks right
print(versioned_client.chunkedgraph.suggest_latest_roots(root))

# after this time, suggest latest should give root
versioned_client.version = 910
print(versioned_client.chunkedgraph.suggest_latest_roots(root))
