#%%
from caveclient import CAVEclient

client = CAVEclient("minnie65_public")

client.annotation.get_table_metadata("allen_v1_column_types")