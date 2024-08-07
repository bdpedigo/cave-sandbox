# %%
from typing import Callable, Optional

import ibis

from caveclient import CAVEclient

ibis.options.interactive = True

client = CAVEclient("minnie65_phase3_v1", version=1078)

cell_table_name = "nucleus_detection_v0"

cell_df = client.materialize.query_table(cell_table_name, split_positions=True)

cell_table = ibis.memtable(cell_df, name="cell_table")

cell_table = cell_table[
    [
        "id",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
        "pt_supervoxel_id",
        "pt_root_id",
    ]
]
cell_table = cell_table.rename(target_id="id")
cell_table = cell_table.distinct(on="pt_root_id", keep=None)


# %%


def coagulate(
    table: ibis.Table,
    sources: list,
    join_field: str,
    field_name: str,
    filter_map_logic: Optional[Callable] = None,
    condense_sources: bool = False,
    drop_sources: bool = False,
) -> ibis.Table:
    source_names = [list(source_info.keys())[0] for source_info in sources]
    source_field_names = [f"{field_name}_{source_name}" for source_name in source_names]
    # loop through each source and join on the cell table
    # some custom logic here for mapping field names...
    for source_info in sources[:]:
        source_name, source_field = list(source_info.items())[0]
        df = client.materialize.query_table(source_name)
        this_table = ibis.memtable(df, name=source_name)[join_field, source_field]
        this_field_name = f"{field_name}_{source_name}"
        this_table = this_table.rename({this_field_name: source_field})

        # apply custom logic to map the field names
        if filter_map_logic is not None:
            this_table = filter_map_logic(this_table, this_field_name)

        if condense_sources:
            this_table = this_table.mutate({source_name: ibis.literal(source_name)})

        table = table.join(
            this_table,
            table[join_field] == this_table[join_field],
            how="left",
        ).drop([f"{join_field}_right"])

    # coalesce across these columns in order to make the final call
    table = table.mutate(
        {
            field_name: ibis.coalesce(
                *[table[f"{field_name}_{source_name}"] for source_name in source_names]
            )
        }
    )

    if condense_sources:
        table = table.mutate(
            {
                f"{field_name}_source": ibis.coalesce(
                    *[table[source_name] for source_name in source_names]
                )
            }
        )
        table = table.drop(source_names)

    if drop_sources:
        table = table.drop(source_field_names)

    return table


join_field = "target_id"

field_name = "coarse_type"


def filter_map_logic(table: ibis.Table, field_name: str) -> ibis.Table:
    for type_label in ["excitatory", "inhibitory", "nonneuron"]:
        table = table.mutate(
            {
                field_name: (
                    table[field_name]
                    .contains(type_label)
                    .ifelse(type_label, table[field_name])
                )
            }
        )
    return table


coarse_type_sources = [
    # "bodor_pt_target_proofread",  # uses nucleus_neuron_svm table
    {"aibs_metamodel_celltypes_v661_corrections": "classification_system"},
    {"aibs_column_nonneuronal_ref": "classification_system"},
    {"allen_v1_column_types_slanted_ref": "classification_system"},
    {"baylor_log_reg_cell_type_coarse_v1": "cell_type"},
    {"aibs_metamodel_celltypes_v661": "classification_system"},
]

condense_sources = True
drop_sources = True

cell_table = coagulate(
    cell_table,
    coarse_type_sources,
    join_field,
    field_name,
    filter_map_logic,
    condense_sources=condense_sources,
    drop_sources=drop_sources,
)
cell_table

# %%
cell_table["coarse_type_source"].value_counts()

# %%

cell_type_sources = [
    {"aibs_metamodel_celltypes_v661_corrections": "cell_type"},
    {"aibs_column_nonneuronal_ref": "cell_type"},
    {"allen_v1_column_types_slanted_ref": "cell_type"},
    {"aibs_metamodel_celltypes_v661": "cell_type"},
]
field_name = "cell_type"
join_field = "target_id"


def filter_map_logic(table: ibis.Table, field_name: str) -> ibis.Table:
    table = table.filter(table[field_name].notin(["unsure", "error"]))

    return table


cell_table = coagulate(
    cell_table,
    cell_type_sources,
    join_field,
    field_name,
    filter_map_logic,
    condense_sources=condense_sources,
    drop_sources=drop_sources,
)

cell_table

# %%

mtype_sources = [
    {"allen_column_mtypes_v2": "cell_type"},
    {"aibs_metamodel_mtypes_v661_v2": "cell_type"},
]
field_name = "mtype"
join_field = "target_id"

cell_table = coagulate(
    cell_table,
    mtype_sources,
    join_field,
    field_name,
    filter_map_logic=None,
    condense_sources=condense_sources,
    drop_sources=drop_sources,
)

#%%


# %%

cell_table.drop_null("coarse_type").sample(0.001)


# %%
ibis.to_sql(cell_table)
