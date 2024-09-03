# %%

import numpy as np
from tqdm.auto import tqdm

from grotto import GrottoClient

client = GrottoClient("minnie65_public", version=1078)

# %%

new_label_map = {
    "perivascular": [
        # "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4890727126401024"
        "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4930623044059136"
    ],
    "soma": [
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6362053351571456",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5716916632027136",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5990023284391936",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4696569841451008",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5111429943263232",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6091125539471360",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4772943218343936",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4534147331653632",
    ],
    "thick": [
        # "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5327511408869376"
        "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4793926851493888",
        "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6497916773466112",
        "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5664197686853632",
    ],
    "glia": [
        "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4520453902172160",
        "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6274720006668288",
    ],
}

labeling_rows = []
for new_label, new_label_links in new_label_map.items():
    # for each link, get all segments for that label
    new_segments = []
    for link in new_label_links:
        state_id = int(link.split("/")[-1])
        state = client.get_state_json(state_id)
        segments = state["layers"][1]["segments"]
        segments = np.array([int(seg) for seg in segments if seg[0] != "!"])
        new_segments.extend(segments)

    new_segments = np.unique(new_segments)

    # for all of those segments, get the leaves and root
    for segment in tqdm(new_segments, desc=new_label):
        leaves = client.get_leaves(segment, stop_layer=2)
        # this root lookup should work since we set the client version to 1078
        root = client.get_root_id(segment)
        for leaf in leaves:
            labeling_rows.append(
                {"level2_id": leaf, "root_id": root, "classification": new_label}
            )

# %%
import pandas as pd

label_df = pd.DataFrame(labeling_rows)

label_df.to_csv("cave-sandbox/data/processed_local_labels/link_labels.csv", index=False)
