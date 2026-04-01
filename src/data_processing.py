# all data processing related functions

import os, time, sys
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import norm, gamma
from scipy.interpolate import interp1d
########################################################################################################################
# data transformation

# Skipped


# def data_transformation(
#     data, method, settings, mode="transform", times=None, cdfs=None
# ):
#     if method == "boxcox":
#         if mode == "transform":
#             data = boxcox_transform(data, settings["exponent"])
#         elif mode == "back_transform":
#             data = boxcox_back_transform(data, settings["exponent"])
#         else:
#             print("Unknown transformation mode: entry=", mode)
#             sys.exit()
#     elif method == "ecdf":
#         if mode == "transform":
#             data = normal_quantile_transform(data, times, cdfs, settings)
#             data = data.T
#         elif mode == "back_transform":
#             data = inverse_normal_quantile_transform(data, times, cdfs, settings)
#             if data.ndim == 2:
#                 data = data.T
#         else:
#             print("Unknown transformation mode: entry=", mode)
#             sys.exit()
#     else:
#         print("Unknown transformation method: entry=", method)
#         sys.exit()
#     return data


########################################################################################################################
# input station data processing


def merge_stndata_into_single_file(config):
    """
    This is the main function of this module.

    GMET v2.0 assumes that each station has an independent file. GPEP will merge the station data into one file,
    which can speed up i/o in subsequent runs of GPEP using the same dataset.

    """

    t1 = time.time()

    ########################################################################################################################
    # parse and change configurations
    outpath_parent = config["outpath_parent"]
    path_stn_info = f"{outpath_parent}/stn_info"
    file_allstn = f"{path_stn_info}/all_station_data.nc"  # set default name for a merged station data file
    os.makedirs(path_stn_info, exist_ok=True)

    config["path_stn_info"] = path_stn_info
    config["file_allstn"] = (
        file_allstn  # store default name for a merged station data file in config
    )

    # in/out information to this function
    if "input_stn_list" in config:
        input_stn_list = config["input_stn_list"]
    else:
        input_stn_list = ""
    if "input_stn_path" in config:
        input_stn_path = config["input_stn_path"]
    else:
        input_stn_path = ""
    if "input_stn_all" in config:
        input_stn_all = config["input_stn_all"]
    else:
        input_stn_all = ""

    file_allstn = config["file_allstn"]
    sensor_depth_cm = config["sensor_depth_cm"]
    input_vars = config["input_vars"]
    target_vars = config["target_vars"]
    predictor_name_static_stn = config["predictor_name_static_stn"]
    static_stn_vars = [var for var in predictor_name_static_stn]

    if "minRange_vars" in config:
        minRange_vars = config["minRange_vars"]
        if isinstance(minRange_vars, (int, float)):
            minRange_vars = [minRange_vars] * len(target_vars)
    else:
        minRange_vars = [-np.inf] * len(target_vars)

    if "maxRange_vars" in config:
        maxRange_vars = config["maxRange_vars"]
        if isinstance(maxRange_vars, (int, float)):
            maxRange_vars = [maxRange_vars] * len(target_vars)
    else:
        maxRange_vars = [np.inf] * len(target_vars)

    if "onlytrans_ens" in config:
        onlytrans_ens = config["onlytrans_ens"]
    else:
        onlytrans_ens = False

    if onlytrans_ens == True:
        transform_vars = [""] * len(target_vars)
    elif "transform_vars" in config:
        transform_vars = config["transform_vars"]
        if not isinstance(transform_vars, list):
            transform_vars = [transform_vars] * len(target_vars)
    else:
        transform_vars = [""] * len(target_vars)

    if "transform" in config:
        transform_settings = config["transform"]
    else:
        transform_settings = {}

    if "mapping_InOut_var" in config:
        mapping_InOut_var = config["mapping_InOut_var"]
    else:
        mapping_InOut_var = []

    if "overwrite_merged_stnfile" in config:
        overwrite_merged_stnfile = config["overwrite_merged_stnfile"]
    else:
        overwrite_merged_stnfile = True

    ########################################################################################################################
    # settings and prints
    print("#" * 50)
    print("Merging individual station files to one single file")
    print("#" * 50)
    print("Input station list:     ", input_stn_list)
    print("Input station folder:   ", input_stn_path)
    print("Output station file:    ", file_allstn)
    print("Target variables:       ", input_vars)
    print("Static station variables: ", predictor_name_static_stn)

    if os.path.isfile(file_allstn):
        print("NOTE: Merged station file exists")
        if overwrite_merged_stnfile:
            print("overwrite_merged_stnfile is True. Continue.")
        else:
            print("overwrite_merged_stnfile is False. Skip station merging.\n")
            return config

    # load station data ########################################################################################################################
    if "input_stn_all" in config and os.path.isfile(input_stn_all):
        print("input_stn_all exists:    ", input_stn_all)
        print("reading station info from", input_stn_all, "instead of individual files")
        # print('reading station information from', file_allstn, 'instead of individual files')  # this looks incorrect

        ds_stn = xr.load_dataset(input_stn_all)

        if os.path.isfile(input_stn_list):
            df_stn = pd.read_csv(input_stn_list)
            for col in df_stn.columns:
                ds_stn[col] = xr.DataArray(df_stn[col].values, dims=("stn"))

    else:
        print(
            "A merged station data file does not exist: reading station information from individual files"
        )

        ########################################################################################################################
        # Read station information from the list file
        df_stn = pd.read_csv(input_stn_list)
        df_stn["mask"] = 1.0

        ########################################################################################################################
        # Read all station data files into a list of dataframes
        all_dfs = []
        for i, stnid in enumerate(df_stn.stnid):
            infilei = f"{input_stn_path}/sm_{stnid}_depth{sensor_depth_cm}.csv"
            if not os.path.isfile(infilei):
                print(f"{infilei} does not exist. Skip {stnid}.")
                continue

            df_datai = pd.read_csv(infilei, index_col="time", parse_dates=True)

            # drop if there is duplicate time (time is now the index, not a column)
            df_datai = df_datai[~df_datai.index.duplicated(keep="first")]

            all_dfs.append(
                df_datai[input_vars]
            )  # TODO: this only works for 1 variables

        _all_dfs_concat = pd.concat(all_dfs, axis=1)

        ########################################################################################################################
        # Resample the data to the desired frequency
        # TODO: make the frequency flexible
        all_dfs_concat = _all_dfs_concat.resample("D").mean()

        # create empty xarray dataset with coordinates
        coor_stn_vars = ["lat", "lon", "mask"] + static_stn_vars
        coords_stn = {var: df_stn[var].values for var in coor_stn_vars}
        coords_stn["time"] = all_dfs_concat.index
        coords_stn["stn"] = df_stn["stnid"].values
        ds_stn = xr.Dataset(coords=coords_stn)

        # add input variables
        for var in input_vars:
            ds_stn[var] = xr.DataArray(all_dfs_concat.values, dims=("time", "stn"))

    ########################################################################################################################
    # constrain variables
    for i in range(len(target_vars)):
        vari = target_vars[i]
        if vari in ds_stn.data_vars:
            v = ds_stn[vari].values
            if np.any(v < minRange_vars[i]):
                print(
                    f"{vari} station data has values < {minRange_vars[i]}. Adjust those to {minRange_vars[i]}."
                )
                v[v < minRange_vars[i]] = minRange_vars[i]
            if np.any(v > maxRange_vars[i]):
                print(
                    f"{vari} station data has values > {maxRange_vars[i]}. Adjust those to {maxRange_vars[i]}."
                )
                v[v > maxRange_vars[i]] = maxRange_vars[i]
            ds_stn[vari].values = v

    ########################################################################################################################
    # Check the combination of available station data per each timestep for each target variable
    for vari in target_vars:
        # Check station data availability per each timestep
        available_array = ~ds_stn[vari].isnull()
        ds_stn[vari + "_avail"] = available_array

        # Get the list of available station data per each timestep
        available_stn_list = np.empty(available_array.shape[0], dtype=object)
        for t in range(available_array.shape[0]):
            available_stn = ds_stn.stn.values[available_array[t]]
            available_stn_list[t] = available_stn

        # Get the unique combinations of available station data
        ds_stn[vari + "_avail_stn"] = xr.DataArray(available_stn_list, dims=("time",))

        # Convert to a list of tuples for comparison
        available_stn_list_tuples = [
            tuple(arr) for arr in ds_stn[vari + "_avail_stn"].values
        ]
        available_stn_unique = list(set(available_stn_list_tuples))
        available_stn_unique_dict = {
            i: combo for i, combo in enumerate(available_stn_unique)
        }

        print(
            "Unique combinations of available station data found for",
            vari,
            ":",
            len(available_stn_unique_dict),
        )

        # Assgin the unique combinations an index, which will be used for the weight calculation
        ds_stn[vari + "_avail_stn_idx"] = xr.DataArray(
            np.arange(len(available_stn_unique)), dims=("avail_stn_idx",)
        )

        # Assign the index to the available stations for each timestep
        avail_stn_index_values = np.empty(available_array.shape[0], dtype=int)
        for t in range(available_array.shape[0]):
            available_stn = tuple(ds_stn.stn.values[available_array[t]])

            # Find the index by searching through the dictionary values
            found = False
            for idx, combo in available_stn_unique_dict.items():
                if combo == available_stn:
                    avail_stn_index_values[t] = idx
                    found = True
                    break
            if not found:
                print(
                    f"Warning: No index found for available stations {available_stn} at time {t}"
                )
                avail_stn_index_values[t] = -1

        ds_stn[vari + "_avail_stn_idx_values"] = xr.DataArray(
            avail_stn_index_values, dims=("time",)
        )

        # Save the unique combinations to a csv file
        available_stn_unique_df = pd.DataFrame(
            available_stn_unique_dict.items(), columns=["avail_stn_index", "avail_stn"]
        )
        available_stn_unique_df.to_csv(
            f"{outpath_parent}/stn_info/available_stn_unique_{vari}.csv", index=False
        )

    # Remove a few variables that are not needed
    for var in target_vars:
        ds_stn = ds_stn.drop_vars([var + "_avail_stn"])

    # # transform variables
    # print("Transform variables if relevant settings are provided")
    # for i in range(len(transform_vars)):
    #     if len(transform_vars[i]) > 0:
    #         tvar = target_vars[i] + "_" + transform_vars[i]
    #         print(
    #             f"Perform {transform_vars[i]} transformation for {target_vars[i]}. Add a new variable {tvar} to output station file."
    #         )
    #         if tvar in ds_stn:
    #             print(f"{tvar} exists in ds_stn. no need to perform transformation")
    #             continue
    #         ds_stn[tvar] = ds_stn[vari].copy()
    #         if transform_vars[i] == "ecdf":
    #             cdfs = calculate_monthly_cdfs(
    #                 ds_stn, target_vars[i], transform_settings[transform_vars[i]]
    #             )
    #             ds_stn[tvar].values = data_transformation(
    #                 ds_stn[target_vars[i]].values,
    #                 transform_vars[i],
    #                 transform_settings[transform_vars[i]],
    #                 "transform",
    #                 times=ds_stn["time"].values,
    #                 cdfs=cdfs,
    #             )
    #         else:
    #             ds_stn[tvar].values = data_transformation(
    #                 ds_stn[target_vars[i]].values,
    #                 transform_vars[i],
    #                 transform_settings[transform_vars[i]],
    #                 "transform",
    #             )
    #     else:
    #         print(f"Do not perform transformation for {target_vars[i]}")

    ########################################################################################################################
    # Save to output files
    encoding = {}
    for var in ds_stn.data_vars:
        encoding[var] = {"zlib": True, "complevel": 4}

    ds_stn.to_netcdf(file_allstn, encoding=encoding)

    t2 = time.time()
    print("Time cost (s) for merging station data file:", t2 - t1, "\n")

    return config
