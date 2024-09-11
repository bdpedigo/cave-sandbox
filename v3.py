logging.warning(
    "Deprecation: this method is to facilitate beta testing of this feature, \
    it will likely get removed in future versions. "
)
timestamp = convert_timestamp(timestamp)
return_df = True
if datastack_name is None:
    datastack_name = self.datastack_name
endpoint_mapping = self.default_url_mapping
endpoint_mapping["datastack_name"] = datastack_name
data = {}
query_args = {}
query_args["return_pyarrow"] = True
query_args["arrow_format"] = True
query_args["merge_reference"] = False
query_args["allow_missing_lookups"] = allow_missing_lookups
query_args["allow_invalid_root_ids"] = allow_invalid_root_ids
if random_sample:
    query_args["random_sample"] = random_sample
data["table"] = table
data["timestamp"] = timestamp
url = self._endpoints["live_live_query"].format_map(endpoint_mapping)
if joins is not None:
    data["join_tables"] = joins
if filter_in_dict is not None:
    data["filter_in_dict"] = filter_in_dict
if filter_out_dict is not None:
    data["filter_notin_dict"] = filter_out_dict
if filter_equal_dict is not None:
    data["filter_equal_dict"] = filter_equal_dict
if filter_spatial_dict is not None:
    data["filter_spatial_dict"] = filter_spatial_dict
if filter_regex_dict is not None:
    data["filter_regex_dict"] = filter_regex_dict
if select_columns is not None:
    data["select_columns"] = select_columns
if offset is not None:
    data["offset"] = offset
if limit is not None:
    assert limit > 0
    data["limit"] = limit
if suffixes is not None:
    data["suffixes"] = suffixes
if desired_resolution is None:
    desired_resolution = self.desired_resolution
if desired_resolution is not None:
    data["desired_resolution"] = desired_resolution
encoding = DEFAULT_COMPRESSION
response = self.session.post(
    url,
    data=json.dumps(data, cls=BaseEncoder),
    headers={
        "Content-Type": "application/json",
        "Accept-Encoding": encoding,
    },
    params=query_args,
    stream=~return_df,
    verify=self.verify,
)
self.raise_for_status(response, log_warning=log_warning)
with MyTimeIt("deserialize"):
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        df = deserialize_query_response(response)
        if desired_resolution is not None:
            if not response.headers.get("dataframe_resolution", None):
                if len(desired_resolution) != 3:
                    raise ValueError(
                        "desired resolution needs to be of length 3, for xyz"
                    )
                vox_res = self.get_table_metadata(
                    table,
                    datastack_name,
                    log_warning=False,
                )["voxel_resolution"]
                df = convert_position_columns(df, vox_res, desired_resolution)

    if not split_positions:
        concatenate_position_columns(df, inplace=True)
if metadata:
    try:
        attrs = self._assemble_attributes(
            table,
            join_query=False,
            filters={
                "inclusive": filter_in_dict,
                "exclusive": filter_out_dict,
                "equal": filter_equal_dict,
                "spatial": filter_spatial_dict,
                "regex": filter_regex_dict,
            },
            select_columns=select_columns,
            offset=offset,
            limit=limit,
            live_query=timestamp is not None,
            timestamp=string_format_timestamp(timestamp),
            materialization_version=None,
            desired_resolution=response.headers.get(
                "dataframe_resolution", desired_resolution
            ),
        )
        df.attrs.update(attrs)
    except HTTPError as e:
        raise Exception(
            e.message
            + " Metadata could not be loaded, try with metadata=False if not needed"
        )
