# %%
import datetime

import caveclient

datastack_name = "flywire_fafb_public"
client = caveclient.CAVEclient(datastack_name)
postsyn_df = client.materialize.live_query(
    "synapses_nt_v1",
    filter_in_dict={"post_pt_root_id": [720575940617343316]},
    timestamp=datetime.datetime.now(),
)
