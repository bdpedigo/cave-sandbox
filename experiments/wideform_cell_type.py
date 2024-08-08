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
import pandas as pd


def coagulate(
    table: ibis.Table,
    source_info: pd.DataFrame,
    join_field: str,
    field_name: str,
    filter_map_logic: Optional[Callable] = None,
    condense_sources: bool = False,
    drop_sources: bool = False,
) -> ibis.Table:
    source_names = source_info.index
    source_field_names = [f"{field_name}_{source_name}" for source_name in source_names]
    # loop through each source and join on the cell table
    # some custom logic here for mapping field names...
    for source_name, source_info in source_info.iterrows():
        source_field = source_info["field"]
        df = client.materialize.query_table(source_name, log_warning=False)
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


join_field = "pt_root_id"

field_name = "coarse_type"


def filter_map_logic(table: ibis.Table, field_name: str) -> ibis.Table:
    for type_label in ["excitatory", "inhib", "nonneuron"]:
        table = table.mutate(
            {
                field_name: (
                    table[field_name]
                    .contains(type_label)
                    .ifelse(
                        type_label.replace("inhib", "inhibitory"), table[field_name]
                    )
                )
            }
        )
    table = table.mutate(
        {
            field_name: table[field_name]
            .contains("aibs_coarse_unclear")
            .ifelse(ibis.null(str), table[field_name])
        }
    )
    return table


import pandas as pd

coarse_type_source_info = [
    {"aibs_metamodel_celltypes_v661_corrections": "classification_system"},
    {
        "bodor_pt_target_proofread": "classification_system"
    },  # uses nucleus_neuron_svm table so shouldn't use target_id?
    {"aibs_column_nonneuronal_ref": "classification_system"},
    {"allen_v1_column_types_slanted_ref": "classification_system"},
    # the last two are models
    {"baylor_log_reg_cell_type_coarse_v1": "cell_type"},
    {"aibs_metamodel_celltypes_v661": "classification_system"},
]
coarse_type_source_info = {
    "aibs_metamodel_celltypes_v661_corrections": {"field": "classification_system"},
    "bodor_pt_target_proofread": {"field": "classification_system"},
    "aibs_column_nonneuronal_ref": {"field": "classification_system"},
    "allen_v1_column_types_slanted_ref": {"field": "classification_system"},
    "baylor_log_reg_cell_type_coarse_v1": {"field": "cell_type"},
    "aibs_metamodel_celltypes_v661": {"field": "classification_system"},
}
coarse_type_source_info = pd.DataFrame(coarse_type_source_info).T

condense_sources = True
drop_sources = False

cell_table = coagulate(
    cell_table,
    coarse_type_source_info,
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
cell_table["coarse_type"].value_counts()

# %%
cell_table.to_pandas().query("coarse_type_source == 'bodor_pt_target_proofread'")

# %%


cell_type_source_info = {
    "aibs_metamodel_celltypes_v661_corrections": {"field": "cell_type"},
    "aibs_column_nonneuronal_ref": {"field": "cell_type"},
    "allen_v1_column_types_slanted_ref": {"field": "cell_type"},
    "aibs_metamodel_celltypes_v661": {"field": "cell_type"},
}
cell_type_source_info = pd.DataFrame(cell_type_source_info).T


field_name = "cell_type"
join_field = "target_id"


def filter_map_logic(table: ibis.Table, field_name: str) -> ibis.Table:
    table = table.filter(
        table[field_name].notin(["unsure", "error", "Unsure I", "Unsure E", "Unsure"])
    )

    return table


cell_table = coagulate(
    cell_table,
    cell_type_source_info,
    join_field,
    field_name,
    filter_map_logic,
    condense_sources=condense_sources,
    drop_sources=drop_sources,
)

cell_table

# %%
cell_table[field_name].value_counts().to_pandas()

# %%


mtype_source_info = {
    "allen_column_mtypes_v2": {"field": "cell_type"},
    "aibs_metamodel_mtypes_v661_v2": {"field": "cell_type"},
}
mtype_source_info = pd.DataFrame(mtype_source_info).T


field_name = "mtype"
join_field = "target_id"

cell_table = coagulate(
    cell_table,
    mtype_source_info,
    join_field,
    field_name,
    filter_map_logic=None,
    condense_sources=condense_sources,
    drop_sources=drop_sources,
)

# %%
cell_table[field_name].value_counts().to_pandas()

# %%

functional_area_source_info = {"nucleus_functional_area_assignment": {"field": "tag"}}
functional_area_source_info = pd.DataFrame(functional_area_source_info).T

cell_table = coagulate(
    cell_table,
    functional_area_source_info,
    "target_id",
    "visual_area",
    filter_map_logic=None,
    condense_sources=False,
    drop_sources=True,
)

# %%
cell_table["visual_area"].value_counts().to_pandas()

# %%

join_field = "pt_root_id"
proofreading_table_name = "proofreading_status_and_strategy"

proofreading_df = client.materialize.query_table(
    proofreading_table_name, log_warning=False
)

proofreading_table = ibis.memtable(proofreading_df, name=proofreading_table_name)
proofreading_table = proofreading_table[
    [
        "pt_root_id",
        "status_dendrite",
        "status_axon",
        "strategy_dendrite",
        "strategy_axon",
    ]
]
proofreading_table = proofreading_table.mutate(
    {
        "strategy_axon": proofreading_table.strategy_axon.replace("none", "None"),
    }
)

cell_table = cell_table.join(
    proofreading_table,
    cell_table[join_field] == proofreading_table[join_field],
    how="left",
)

cell_table = cell_table.rename(
    {
        "proofreading_status_dendrite": "status_dendrite",
        "proofreading_status_axon": "status_axon",
        "proofreading_strategy_dendrite": "strategy_dendrite",
        "proofreading_strategy_axon": "strategy_axon",
    }
)

# %%

cell_table.drop_null("coarse_type").sample(0.01).drop(
    [
        "pt_root_id",
        "pt_supervoxel_id",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
    ]
).to_pandas().head(10)

# %%
ibis.to_sql(cell_table)

# %%

df = cell_table.to_pandas()

# %%
pd.crosstab(
    df["coarse_type_aibs_metamodel_celltypes_v661"],
    df["coarse_type_baylor_log_reg_cell_type_coarse_v1"],
)

# %%
len(
    df.query(
        "coarse_type_aibs_metamodel_celltypes_v661 != coarse_type_baylor_log_reg_cell_type_coarse_v1"
    )
    .dropna(
        subset=[
            "coarse_type_aibs_metamodel_celltypes_v661",
            "coarse_type_baylor_log_reg_cell_type_coarse_v1",
        ]
    )
    .pt_root_id.to_list()
)

# %%


# n_random = 1
# number_cols = []
# for i in range(n_random):
#     col_name = f"random_{i}"
#     seg_df[col_name] = np.random.permutation(len(seg_df)).astype('int8')
#     number_cols.append(col_name)

number_cols = None

from nglui import statebuilder
from nglui.segmentprops import SegmentProperties

seg_df = cell_table.to_pandas().copy()
seg_df = seg_df.dropna(subset=["coarse_type"])
tag_value_cols = [
    "coarse_type",
    "cell_type",
    "mtype",
    "visual_area",
    "proofreading_strategy_dendrite",
    "proofreading_strategy_axon",
]


def prepend_columns(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].apply(
            lambda x: col + ":" + x if pd.notnull(x) else col + ":None"
        )
    return df


seg_prop = SegmentProperties.from_dataframe(
    prepend_columns(seg_df.reset_index(), tag_value_cols),
    id_col="pt_root_id",
    label_col="target_id",
    tag_value_cols=tag_value_cols,
)


def generate_link_from_segment_properties(seg_prop):
    prop_id = client.state.upload_property_json(seg_prop.to_dict())
    prop_url = client.state.build_neuroglancer_url(
        prop_id, format_properties=True, target_site="mainline"
    )

    img = statebuilder.ImageLayerConfig(
        source=client.info.image_source(),
    )
    seg = statebuilder.SegmentationLayerConfig(
        source=client.info.segmentation_source(),
        segment_properties=prop_url,
        active=True,
        skeleton_source="precomputed://middleauth+https://minnie.microns-daf.com/skeletoncache/api/v1/minnie65_phase3_v1/precomputed/skeleton",
    )

    sb = statebuilder.StateBuilder(
        layers=[img, seg],
        target_site="mainline",
        view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
    )

    return sb.render_state()


print("Overall:")
generate_link_from_segment_properties(seg_prop)

# %%

field = "coarse_type"
tag_value_cols = [f"{field}_{source}" for source in coarse_type_source_info.index]


seg_prop = SegmentProperties.from_dataframe(
    prepend_columns(seg_df.reset_index(), tag_value_cols),
    id_col="pt_root_id",
    label_col="target_id",
    tag_value_cols=tag_value_cols,
)

print("Coarse Type:")
generate_link_from_segment_properties(seg_prop)

# %%
field = "cell_type"
tag_value_cols = [f"{field}_{source}" for source in cell_type_source_info.index]

seg_prop = SegmentProperties.from_dataframe(
    prepend_columns(seg_df.reset_index(), tag_value_cols),
    id_col="pt_root_id",
    label_col="target_id",
    tag_value_cols=tag_value_cols,
)

print("Cell Type:")
generate_link_from_segment_properties(seg_prop)

# %%
field = "mtype"
tag_value_cols = [f"{field}_{source}" for source in mtype_source_info.index]

seg_prop = SegmentProperties.from_dataframe(
    prepend_columns(seg_df.reset_index(), tag_value_cols),
    id_col="pt_root_id",
    label_col="target_id",
    tag_value_cols=tag_value_cols,
)

print("M-Type:")
generate_link_from_segment_properties(seg_prop)
