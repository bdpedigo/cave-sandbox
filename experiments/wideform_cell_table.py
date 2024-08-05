# %%
import ibis

from caveclient import CAVEclient

ibis.options.interactive = True

client = CAVEclient("minnie65_phase3_v1", version=1078)

# %%
cell_type_table_name = "aibs_metamodel_celltypes_v661"
cell_type_df = client.materialize.query_table(
    cell_type_table_name, split_positions=True
)
# %%
cell_type_table = ibis.memtable(cell_type_df, name="cell_type_table")

# %%
# remove any rows for which there is more than one root id

# cell_type_table = cell_type_table.filter(
#     ~cell_type_table.pt_root_id.isin(
#         cell_type_table.group_by(cell_type_table.pt_root_id)
#         .aggregate(n_roots=cell_type_table.count())
#         .order_by("n_roots")
#         .filter(lambda x: x.n_roots > 1)
#         .pt_root_id
#     )
# )
cell_type_table = cell_type_table.distinct(on="pt_root_id", keep=None)
cell_type_table

# %%
cell_type_table = cell_type_table.drop(
    [
        "created_ref",
        "valid_ref",
        "volume",
        "created",
        "bb_start_position_x",
        "bb_start_position_y",
        "bb_start_position_z",
        "bb_end_position_x",
        "bb_end_position_y",
        "bb_end_position_z",
        "valid",
        "id_ref",
        "id",
    ]
)

cell_type_table = cell_type_table.mutate(
    cell_type_provenance=ibis.literal(cell_type_table_name)
)

# %%
cell_type_table.mutate(cell_type_provenance=ibis.literal(cell_type_table_name))

# %%
cell_type_corrections_table_name = "aibs_metamodel_celltypes_v661_corrections"
cell_type_corrections_df = client.materialize.query_table(
    cell_type_corrections_table_name
)
cell_type_corrections_table = (
    ibis.memtable(cell_type_corrections_df, name="cell_type_corrections_table")
    .drop(
        [
            "id_ref",
            "created_ref",
            "valid_ref",
            "volume",
            "pt_supervoxel_id",
            "id",
            "created",
            "valid",
            "pt_root_id",
            "pt_position",
            "bb_start_position",
            "bb_end_position",
        ]
    )
    .mutate(cell_type_provenance=ibis.literal(cell_type_corrections_table_name))
)
cell_type_corrections_table

cell_type_table = cell_type_table.join(
    cell_type_corrections_table,
    cell_type_corrections_table.target_id == cell_type_table.target_id,
    how="left",
)

cell_type_table

# %%
# override the classification_system, cell_type, and cell_type_provenance columns

cell_type_table = cell_type_table.mutate(
    classification_system=ibis.coalesce(
        cell_type_table.classification_system_right,
        cell_type_table.classification_system,
    ),
    cell_type=ibis.coalesce(cell_type_table.cell_type_right, cell_type_table.cell_type),
    cell_type_provenance=ibis.coalesce(
        cell_type_table.cell_type_provenance_right,
        cell_type_table.cell_type_provenance,
    ),
).drop(
    [
        "target_id_right",
        "classification_system_right",
        "cell_type_right",
        "cell_type_provenance_right",
    ]
)

cell_type_table

# %%

proofreading_table_name = "proofreading_status_and_strategy"
proofreading_df = client.materialize.query_table(proofreading_table_name)

# %%
proofreading_table = ibis.memtable(proofreading_df, name="proofreading_table")
proofreading_table = proofreading_table.drop(
    [
        "id",
        "created",
        "superceded_id",
        "valid",
        "valid_id",
        "pt_supervoxel_id",
        "pt_position",
    ]
)
# %%
joined_table = cell_type_table.join(
    proofreading_table, cell_type_table.pt_root_id == proofreading_table.pt_root_id
)

# %%
ibis.to_sql(joined_table)
