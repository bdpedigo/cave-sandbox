# %%
import pandas as pd

import caveclient as cv

print(pd.__version__)
print(cv.__version__)

client = cv.CAVEclient("minnie65_public_v117")
client.materialize.query_table("synapses_pni_2", limit=5)
