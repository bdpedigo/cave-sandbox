# %%
import ibis

from caveclient import CAVEclient

ibis.options.interactive = True

client = CAVEclient("minnie65_phase3_v1", version=1078)

# %%

# get cell type, E/I/N, and nucleus info from the cell type table

cell_type_table_name = "aibs_metamodel_celltypes_v661"
cell_type_df = client.materialize.query_table(
    cell_type_table_name, split_positions=True
)

# %%
cell_table = ibis.memtable(cell_type_df, name="cell_type_table")

# %%

# remove duplicate root IDs
# TODO why are there duplicate root IDs here?
cell_table = cell_table.distinct(on="pt_root_id", keep=None)
cell_table

# %%
cell_table = cell_table[
    [
        "target_id",
        "classification_system",
        "cell_type",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
        "pt_supervoxel_id",
        "pt_root_id",
    ]
]

cell_table = cell_table.mutate(cell_type_provenance=ibis.literal(cell_type_table_name))
cell_table

# %%
# add some notation for where the cell type label came from
cell_table.mutate(cell_type_provenance=ibis.literal(cell_type_table_name))

# %%
# merge in the corrections table
cell_type_corrections_table_name = "aibs_metamodel_celltypes_v661_corrections"

cell_type_corrections_df = client.materialize.query_table(
    cell_type_corrections_table_name
)
cell_type_corrections_table = ibis.memtable(
    cell_type_corrections_df, name="cell_type_corrections_table"
)[["target_id", "classification_system", "cell_type"]].mutate(
    cell_type_provenance=ibis.literal(cell_type_corrections_table_name)
)
cell_type_corrections_table

# %%
cell_table = cell_table.join(
    cell_type_corrections_table,
    cell_type_corrections_table.target_id == cell_table.target_id,
    how="left",
    rname="{name}_corrected",
)

cell_table

# %%
# override the classification_system, cell_type, and cell_type_provenance columns
# coalesce witll take the first non-null value, so prefer the corrections table

cell_table = cell_table.mutate(
    classification_system=ibis.coalesce(
        cell_table.classification_system_corrected,
        cell_table.classification_system,
    ),
    cell_type=ibis.coalesce(cell_table.cell_type_corrected, cell_table.cell_type),
    cell_type_provenance=ibis.coalesce(
        cell_table.cell_type_provenance_corrected,
        cell_table.cell_type_provenance,
    ),
).drop(
    [
        "target_id_corrected",
        "classification_system_corrected",
        "cell_type_corrected",
        "cell_type_provenance_corrected",
    ]
)

# %%
cell_table

# %%
cell_table.cell_type_provenance.value_counts()

# %%
assert (
    cell_table["target_id"].value_counts()["target_id_count"].to_pandas().sum()
    == cell_table.count().to_pandas()
)

# %%
cell_table = cell_table.rename(coarse_type="classification_system")

# %%
cell_table = cell_table.rename(
    coarse_type_soma="coarse_type",
    cell_type_soma="cell_type",
    cell_type_soma_source="cell_type_provenance",
).mutate(coarse_type_soma_source="cell_type_soma_source")

cell_table

# %%
# add E/I from Baylor model

ei_table_name = "baylor_log_reg_cell_type_coarse_v1"
ei_df = client.materialize.query_table(ei_table_name)[["target_id", "cell_type"]]

ei_table = ibis.memtable(ei_df, name="ei_table")

ei_table = (
    ei_table.mutate(
        coarse_type_spines=ei_table.cell_type.substitute(
            {"excitatory": "excitatory_neuron", "inhibitory": "inhibitory_neuron"}
        )
    )
    .drop("cell_type")
    .mutate(spines_source=ibis.literal(ei_table_name))
)

ei_table


# %%

cell_table = cell_table.join(
    ei_table,
    cell_table.target_id == ei_table.target_id,
    how="left",
)

# %%
disagreements = cell_table.filter(
    cell_table.coarse_type_soma != cell_table.coarse_type_spines
)

disagreements.group_by(
    [disagreements.coarse_type_soma, disagreements.coarse_type_spines]
).aggregate(n_disagreements=disagreements.count()).pivot_wider(
    names_from="coarse_type_spines", values_from="n_disagreements"
).to_pandas().set_index("coarse_type_soma").rename_axis("baylor", axis=1).rename_axis(
    "metamodel", axis=0
).fillna(0).astype(int)


# %%


# relabel coarse cell type as "unsure" if the models disagree

disagreement_mask = cell_table.coarse_type_soma != cell_table.coarse_type_spines
cell_table = cell_table.mutate(
    coarse_type=disagreement_mask.ifelse("unsure", cell_table.coarse_type_soma)
)

# %%

cell_table.coarse_type.value_counts()

# %%
cell_table = cell_table.drop(["target_id_right"])

cell_table

# %%
cell_table.group_by(["coarse_type", "cell_type_soma"]).aggregate(
    count=cell_table.count()
).to_pandas().set_index(["coarse_type", "cell_type_soma"])


# %%

mtype_table_name = "aibs_metamodel_mtypes_v661_v2"

mtype_df = client.materialize.query_table(mtype_table_name)

mtype_table = ibis.memtable(mtype_df, name="mtype_table")[["target_id", "cell_type"]]

mtype_table = mtype_table.rename(mtype_soma="cell_type")

mtype_table


# %%

cell_table = cell_table.join(
    mtype_table,
    cell_table.target_id == mtype_table.target_id,
    how="left",
).drop("target_id_right")

cell_table

# %%
cell_table = cell_table.mutate(mtype_soma_source=ibis.literal(mtype_table_name))

cell_table

# %%
# TODO add manual labels from the column


# %%
# add visual area information

functional_areas_table_name = "nucleus_functional_area_assignment"

functional_areas_df = client.materialize.query_table(functional_areas_table_name)

functional_areas_table = ibis.memtable(
    functional_areas_df, name="functional_areas_table"
)[["target_id", "tag"]].rename(visual_area="tag")

functional_areas_table

# %%
cell_table = (
    cell_table.join(
        functional_areas_table,
        cell_table.target_id == functional_areas_table.target_id,
        how="left",
    )
    .drop("target_id_right")
    .mutate(visual_area_source=ibis.literal(functional_areas_table_name))
)

cell_table

# %%

proofreading_table_name = "proofreading_status_and_strategy"
proofreading_df = client.materialize.query_table(
    proofreading_table_name, log_warning=False
)

# %%
proofreading_table = ibis.memtable(proofreading_df, name="proofreading_table")
proofreading_table = proofreading_table[
    [
        "pt_root_id",
        "status_dendrite",
        "status_axon",
        "strategy_dendrite",
        "strategy_axon",
    ]
]

# %%
cell_table = cell_table.join(
    proofreading_table, cell_table.pt_root_id == proofreading_table.pt_root_id
)

# %%
cell_table

# %%
ibis.to_sql(cell_table)

# %%
# TODO add coregistration information

# %%
# corrections

column_neuron_corrections_df = client.materialize.query_table(
    "allen_v1_column_types_slanted_ref"
)
column_nonneuron_corrections_df = client.materialize.query_table(
    "aibs_column_nonneuronal_ref"
)

column_nonneuron_corrections_table = ibis.memtable(
    column_nonneuron_corrections_df, name="column_nonneuron_corrections"
)[["target_id", "cell_type", "classification_system"]].rename(
    coarse_type="classification_system"
)
column_nonneuron_corrections_table = column_nonneuron_corrections_table.filter(
    column_nonneuron_corrections_table.cell_type.notin(["unsure", "error"])
)

# %%
column_nonneuron_corrections_df["cell_type"].value_counts()
