# all data processing related functions

import os, time, sys
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import norm, gamma
from scipy.interpolate import interp1d
########################################################################################################################
# data transformation


def boxcox_transform(data, texp=4):
    # transform prcp to approximate normal distribution
    # mode: box-cox; power-law
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data = data.copy()
    data[data < 0] = 0
    datat = (data ** (1 / texp) - 1) / (1 / texp)
    return datat


def boxcox_back_transform(data, texp=4):
    # transform prcp to approximate normal distribution
    # mode: box-cox; power-law
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data = data.copy()
    data[data < -texp] = -texp
    datat = (data / texp + 1) ** texp
    return datat


def boxcox_back_transform_biasadjustment(data, sigma_square, texp=4):
    # Box-Cox back transformation can lead to bias
    # This function is put here for future use
    # mode: box-cox; power-law
    # Reference: https://otexts.com/fpp2/transformations.html
    data = data.copy()
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data[data < -texp] = -texp
    datat = (data / texp + 1) ** texp * (
        1 + sigma_square * (1 - 1 / texp) / (2 * (data / texp + 1) ** 2)
    )

    datat[data == -texp] = 0

    return datat


def create_cdf_df(data):
    """Build empirical CDF columns (Value, CDF) from positive finite samples."""
    valid_data = np.asarray(data).ravel()
    valid_data = valid_data[np.isfinite(valid_data) & (valid_data > 0)]
    if valid_data.size == 0:
        return pd.DataFrame(
            {"Value": pd.Series(dtype=float), "CDF": pd.Series(dtype=float)}
        )
    sorted_data = np.sort(valid_data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return pd.DataFrame({"Value": sorted_data, "CDF": cdf_values})


def calculate_monthly_cdfs(ds, var_name, settings):
    """Create empirical CDFs stratified by calendar month (1–12)."""

    pooled = settings.get("pooled", True)

    df = pd.DataFrame(data=ds[var_name].values, index=pd.to_datetime(ds["time"]))

    if pooled:
        cdfs = {
            month: create_cdf_df(df[df.index.month == month].values.flatten())
            for month in range(1, 13)
        }
    else:
        cdfs = {
            station: {
                month: create_cdf_df(df[station][df.index.month == month].dropna())
                for month in range(1, 13)
            }
            for station in df.columns
        }
    return cdfs


def calculate_global_cdfs(ds, var_name, settings):
    """Create one empirical CDF over all times (pooled across stations, or per station)."""

    pooled = settings.get("pooled", True)

    df = pd.DataFrame(data=ds[var_name].values, index=pd.to_datetime(ds["time"]))

    if pooled:
        return create_cdf_df(df.values.flatten())
    return {
        station: create_cdf_df(df[station].dropna().values) for station in df.columns
    }


def calculate_ecdf_cdfs(ds, var_name, settings):
    """Build CDF tables for ECDF transform; monthly vs global is controlled by settings['monthly']."""

    if settings.get("monthly", True):
        return calculate_monthly_cdfs(ds, var_name, settings)
    return calculate_global_cdfs(ds, var_name, settings)


def normal_quantile_transform_monthly(data, times, monthly_cdfs, settings):
    """
    Normal quantile transform using calendar-month empirical CDFs (pooled or per station).
    """

    # Read settings and if not available, assign default value
    pooled = settings.get("pooled", True)
    min_z_value = settings.get("min_z_value", -4)

    df = pd.DataFrame(data=data, index=times)
    transformed_data = pd.DataFrame(
        index=df.index, columns=(df.columns if pooled else None)
    )

    # Read all stations that only contain nan values
    nan_columns = [col for col in df.columns if df[col].isna().all()]

    for month in range(1, 13):
        for station in df.columns if not pooled else [None]:
            month_data = (
                df[station][df.index.month == month]
                if not pooled
                else df[df.index.month == month]
            )
            month_data = month_data[month_data > 0]

            empirical_cdf = (
                monthly_cdfs.get(month) if pooled else monthly_cdfs[station][month]
            )

            if empirical_cdf is not None and not empirical_cdf.empty:
                cdf_interp = interp1d(
                    empirical_cdf["Value"], empirical_cdf["CDF"], bounds_error=True
                )
                cum_probs = np.clip(cdf_interp(month_data), 0, 0.9999)
                z_scores = norm.ppf(cum_probs)

                if pooled:
                    transformed_data.loc[month_data.index, :] = z_scores
                else:
                    transformed_data.loc[month_data.index, station] = z_scores
            else:
                if pooled:
                    transformed_data.loc[month_data.index, :] = np.nan
                else:
                    transformed_data.loc[month_data.index, station] = np.nan

        transformed_data_array = transformed_data.astype(float).to_numpy()

        # Assign min value to all nan
        transformed_data_filled = np.nan_to_num(transformed_data_array, nan=min_z_value)

        # Remove stations that had nan from start
        for col_index, col_name in enumerate(df.columns):
            if col_name in nan_columns:
                transformed_data_filled[:, col_index] = np.nan

    return transformed_data_filled


def normal_quantile_transform_global(data, times, cdfs, settings):
    """
    Normal quantile transform using a single empirical CDF over all time (pooled or per station).
    """

    pooled = settings.get("pooled", True)
    min_z_value = settings.get("min_z_value", -4)

    df = pd.DataFrame(data=data, index=pd.to_datetime(times))
    nan_columns = [col for col in df.columns if df[col].isna().all()]

    transformed_data = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    if pooled:
        empirical_cdf = cdfs
        if empirical_cdf is not None and not empirical_cdf.empty:
            cdf_interp = interp1d(
                empirical_cdf["Value"], empirical_cdf["CDF"], bounds_error=True
            )
            for col in df.columns:
                col_data = df[col]
                pos = col_data[col_data > 0]
                if len(pos) > 0:
                    cum_probs = np.clip(cdf_interp(pos.values), 0, 0.9999)
                    transformed_data.loc[pos.index, col] = norm.ppf(cum_probs)
    else:
        for station in df.columns:
            empirical_cdf = cdfs[station]
            col_data = df[station]
            pos = col_data[col_data > 0]
            if empirical_cdf is not None and not empirical_cdf.empty and len(pos) > 0:
                cdf_interp = interp1d(
                    empirical_cdf["Value"], empirical_cdf["CDF"], bounds_error=True
                )
                cum_probs = np.clip(cdf_interp(pos.values), 0, 0.9999)
                transformed_data.loc[pos.index, station] = norm.ppf(cum_probs)

    transformed_data_array = transformed_data.astype(float).to_numpy()
    transformed_data_filled = np.nan_to_num(transformed_data_array, nan=min_z_value)
    for col_index, col_name in enumerate(df.columns):
        if col_name in nan_columns:
            transformed_data_filled[:, col_index] = np.nan

    # min_z_value to nan
    transformed_data_filled[transformed_data_filled <= min_z_value] = np.nan

    return transformed_data_filled


def inverse_normal_quantile_transform_monthly(data, time, monthly_cdfs, settings):
    """
    Inverse of normal_quantile_transform_monthly.
    """

    # Read settings value, and if not assign default value
    pooled = settings.get("pooled", True)
    interp_method = settings.get("interp_method", "interp1d")
    min_est_value = settings.get("min_est_value", 0.01)

    if data.ndim == 3:  # Grid regression
        flattened_data = data.reshape(
            -1,
            len(
                data[
                    0,
                    0,
                ]
            ),
        )
        transformed_data = pd.DataFrame(flattened_data.T, index=time)
        back_transformed_data = pd.DataFrame(
            index=transformed_data.index, columns=transformed_data.columns
        )
    elif data.ndim == 2:  # Station regression
        transformed_data = pd.DataFrame(data.T, index=time)
        back_transformed_data = pd.DataFrame(
            index=transformed_data.index, columns=transformed_data.columns
        )

    for month in range(1, 13):
        for station in transformed_data.columns if not pooled else [None]:
            # Get z scores for each month, and for unpooled approach for each station

            z_scores = (
                transformed_data[station][transformed_data.index.month == month]
                if not pooled
                else transformed_data[transformed_data.index.month == month]
            )

            # Get precomputed ecdf values
            if pooled:
                empirical_cdf = monthly_cdfs.get(month)
            else:
                empirical_cdf = monthly_cdfs[station][month]

            if empirical_cdf is not None and not empirical_cdf.empty:
                if interp_method == "interp1d":
                    # Use linear interpolation for the inverse transformation
                    value_interp = interp1d(
                        empirical_cdf["CDF"],
                        empirical_cdf["Value"],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    # Calculate cumulative probabilities from z scores
                    z_score_float = z_scores.values.astype(float)
                    cum_probs = norm.cdf(z_score_float)
                    # Use linear interpolation to sample from probabilities back to values
                    original_values = value_interp(cum_probs)

                elif interp_method == "gamma":
                    # Fit a gamma distribution to the empirical CDF
                    a, loc, scale = gamma.fit(empirical_cdf["Value"])
                    # Define the inverse CDF (percent point function) of the fitted gamma distribution
                    gamma_ppf = lambda cum_probs: gamma.ppf(cum_probs, a, loc, scale)
                    # Calculate cumulative probabilities from z scores
                    z_score_float = z_scores.values.astype(float)
                    cum_probs = norm.cdf(z_score_float)
                    # Use gamma distribution to sample from probabilities back to values
                    original_values = gamma_ppf(cum_probs)

                # Filter small values created from filling of nan during first transform
                original_values[original_values < min_est_value] = np.nan
                # Convert all nan to zero
                original_values = np.nan_to_num(original_values)

                if pooled:
                    back_transformed_data.loc[z_scores.index, :] = original_values
                else:
                    back_transformed_data.loc[z_scores.index, station] = original_values
            else:
                if pooled:
                    back_transformed_data.loc[z_scores.index, :] = np.zeros(
                        len(z_scores)
                    )
                else:
                    back_transformed_data.loc[z_scores.index, station] = np.zeros(
                        len(z_scores)
                    )

    # DataFrame is (ntime, nspatial); transpose to (nspatial, ntime) to match
    # forward flatten: data.reshape(-1, ntime) with columns = time.
    _back_transform_array = back_transformed_data.astype(float).to_numpy()
    back_transform_array = _back_transform_array.T
    if data.ndim == 3:
        back_transform_array = back_transform_array.reshape(np.shape(data))

    return back_transform_array


def inverse_normal_quantile_transform_global(data, time, cdfs, settings):
    """
    Inverse of normal_quantile_transform_global.
    """

    pooled = settings.get("pooled", True)
    interp_method = settings.get("interp_method", "interp1d")
    min_est_value = settings.get("min_est_value", 0.01)

    if data.ndim == 3:
        flattened_data = data.reshape(
            -1,
            len(
                data[
                    0,
                    0,
                ]
            ),
        )
        transformed_data = pd.DataFrame(flattened_data.T, index=pd.to_datetime(time))
        back_transformed_data = pd.DataFrame(
            index=transformed_data.index,
            columns=transformed_data.columns,
            dtype=float,
        )
    elif data.ndim == 2:
        transformed_data = pd.DataFrame(data, index=pd.to_datetime(time))
        back_transformed_data = pd.DataFrame(
            index=transformed_data.index,
            columns=transformed_data.columns,
            dtype=float,
        )
    else:
        print("inverse_normal_quantile_transform_global: unsupported ndim=", data.ndim)
        sys.exit()

    def _inverse_from_cdf(z_vals, empirical_cdf):
        if empirical_cdf is None or empirical_cdf.empty:
            return np.zeros(len(z_vals))
        z_score_float = z_vals.values.astype(float)
        cum_probs = norm.cdf(z_score_float)
        if interp_method == "interp1d":
            value_interp = interp1d(
                empirical_cdf["CDF"],
                empirical_cdf["Value"],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            original_values = value_interp(cum_probs)
        elif interp_method == "gamma":
            a, loc, scale = gamma.fit(empirical_cdf["Value"])
            original_values = gamma.ppf(cum_probs, a, loc, scale)
        else:
            print("Unknown interp_method for ecdf: ", interp_method)
            sys.exit()
        original_values = np.asarray(original_values)
        original_values[original_values < min_est_value] = np.nan
        return np.nan_to_num(original_values)

    if pooled:
        empirical_cdf = cdfs
        for col in transformed_data.columns:
            z_scores = transformed_data[col]
            original_values = _inverse_from_cdf(z_scores, empirical_cdf)
            back_transformed_data.loc[z_scores.index, col] = original_values
    else:
        for station in transformed_data.columns:
            z_scores = transformed_data[station]
            try:
                original_values = _inverse_from_cdf(z_scores, cdfs[station])
                back_transformed_data.loc[z_scores.index, station] = original_values
            except:
                print(
                    "Error in inverse_normal_quantile_transform_global for station:",
                    station,
                )

    back_transform_array = back_transformed_data.astype(float).to_numpy()

    if data.ndim == 3:
        # The back_transformed_data DataFrame has shape (ntime, nstations), and data has shape (nrow, ncol, ntime)
        # Flatten the 3D data to (nrow*ncol, ntime), process, then reshape back to (nrow, ncol, ntime)
        nrow, ncol, ntime = data.shape
        # back_transform_array at this point is (ntime, nrow*ncol) due to .T in DataFrame creation
        # Transpose back to (nrow*ncol, ntime), then reshape to (nrow, ncol, ntime)
        back_transform_array = back_transform_array.T.reshape(nrow, ncol, ntime)

    return back_transform_array


def data_transformation(
    data, method, settings, mode="transform", times=None, cdfs=None
):
    # if method == "boxcox":
    #     if mode == "transform":
    #         data = boxcox_transform(data, settings["exponent"])
    #     elif mode == "back_transform":
    #         data = boxcox_back_transform(data, settings["exponent"])
    #     else:
    #         print("Unknown transformation mode: entry=", mode)
    #         sys.exit()
    if method == "ecdf":
        monthly = settings.get("monthly", True)
        if mode == "transform":
            if monthly:
                data = normal_quantile_transform_monthly(data, times, cdfs, settings)
            else:
                data = normal_quantile_transform_global(data, times, cdfs, settings)

        elif mode == "back_transform":
            if monthly:
                data = inverse_normal_quantile_transform_monthly(
                    data, times, cdfs, settings
                )
            else:
                data = inverse_normal_quantile_transform_global(
                    data, times, cdfs, settings
                )
            if data.ndim == 2:
                None
                # data = data.T
        else:
            print("Unknown transformation mode: entry=", mode)
            sys.exit()
    else:
        print("Unknown transformation method: entry=", method)
        sys.exit()
    return data


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
    if target_vars[0] == "q_mean":
        include_large_q = bool(config["include_large_q"])

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
            if input_vars[0] == "sm":
                infilei = f"{input_stn_path}/sm_{stnid}_depth{sensor_depth_cm}.csv"
            elif input_vars[0] == "q_mean":
                if include_large_q:
                    infilei = (
                        f"{input_stn_path}/qmean_{sensor_depth_cm}_{stnid}_seasonal.csv"
                    )
                else:
                    infilei = f"{input_stn_path}/qmean_{sensor_depth_cm}_{stnid}_seasonal_no_large_q.csv"
            else:
                print(f"Unknown input variable: {input_vars[0]}")
                sys.exit()

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
        if input_vars[0] == "sm":
            all_dfs_concat = _all_dfs_concat.resample("D").mean()
        elif input_vars[0] == "q_mean":
            all_dfs_concat = _all_dfs_concat

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

    # transform variables
    print("Transform variables if relevant settings are provided")
    for i in range(len(transform_vars)):
        if len(transform_vars[i]) > 0:
            tvar = target_vars[i] + "_" + transform_vars[i]
            print(
                f"Perform {transform_vars[i]} transformation for {target_vars[i]}. Add a new variable {tvar} to output station file."
            )
            if tvar in ds_stn:
                print(f"{tvar} exists in ds_stn. no need to perform transformation")
                continue
            ds_stn[tvar] = ds_stn[target_vars[i]].copy()
            if transform_vars[i] == "ecdf":
                cdfs = calculate_ecdf_cdfs(
                    ds_stn, target_vars[i], transform_settings[transform_vars[i]]
                )
                ds_stn[tvar].values = data_transformation(
                    ds_stn[target_vars[i]].values,
                    transform_vars[i],
                    transform_settings[transform_vars[i]],
                    "transform",
                    times=ds_stn["time"].values,
                    cdfs=cdfs,
                )
            else:
                ds_stn[tvar].values = data_transformation(
                    ds_stn[target_vars[i]].values,
                    transform_vars[i],
                    transform_settings[transform_vars[i]],
                    "transform",
                )
        else:
            print(f"Do not perform transformation for {target_vars[i]}")

    ########################################################################################################################
    # Save to output files
    encoding = {}
    for var in ds_stn.data_vars:
        encoding[var] = {"zlib": True, "complevel": 4}

    ds_stn.to_netcdf(file_allstn, encoding=encoding)

    t2 = time.time()
    print("Time cost (s) for merging station data file:", t2 - t1, "\n")

    return config
