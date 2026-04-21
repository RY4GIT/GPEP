import os, time
import xarray as xr
import numpy as np
from tqdm import tqdm


def distanceweight(dist, maxdist=100, exp=3):
    weight = (1 - (dist / maxdist) ** exp) ** exp
    weight[weight < 0] = 0
    return weight


def distanceweight_userdefined(dist, maxdist, weight_formula):
    weight = eval(weight_formula)
    weight[weight < 0] = 0
    return weight


def calculate_weights_from_distance(
    nearDistance, weight_max_distance=100, initial_distance=100, exp=3, formula=""
):
    # calculate weights

    # Handle 1D input (single location)
    if nearDistance.ndim == 1:
        # Prepare
        nearWeight = np.full_like(nearDistance, np.nan, dtype=np.float32)
        valid = np.isfinite(nearDistance) & (nearDistance >= 0)
        if not np.any(valid):
            return nearWeight
        nearDistance_valid = nearDistance[valid]
        max_dist = np.max([weight_max_distance, np.max(nearDistance_valid) + 1])

        # Calculate weights
        if len(formula) == 0:
            nearWeight[valid] = distanceweight(nearDistance_valid, max_dist, exp)
        else:
            nearWeight[valid] = distanceweight_userdefined(
                nearDistance_valid, max_dist, formula
            )
        return nearWeight

    else:
        print("Error: nearDistance must be 1D or 2D")
        return None


def calculate_weight_using_nearstn_info(config):
    """
    This is the main function of this module.

    This module is used to calculate weights based on near station info.

    """

    t1 = time.time()

    # parse and change configurations
    path_stn_info = config["path_stn_info"]
    file_stn_weight = f"{path_stn_info}/all_stn_weight.nc"
    config["file_stn_weight"] = file_stn_weight

    # in/out information to this function
    file_stn_nearinfo = config["file_stn_nearinfo"]
    file_stn_weight = config["file_stn_weight"]
    initial_distance = config["initial_distance"]
    weight_max_distance = config["weight_max_distance"]

    if "weight_formula" in config:
        weight_formula = config["weight_formula"]
    else:
        weight_formula = ""

    if "overwrite_weight" in config:
        overwrite_weight = config["overwrite_weight"]
    else:
        overwrite_weight = False

    # default settings
    keyword = "nearDistance"  # defined in near station search
    keywords_drop = [
        "nearDistance",
        "nearIndex",
    ]  # TODO: why do I wet drop this variable? For later?
    truncation_dist = (
        np.inf
    )  # stations beyond this distance have zero weights. this is not activated for now

    print("#" * 50)
    print("Calculate weights for near stations")
    print("#" * 50)
    print("input file_stn_nearinfo:", file_stn_nearinfo)
    print("output file_stn_weight: ", file_stn_weight)

    if os.path.isfile(file_stn_weight):
        print("Note! Weight file exists")
        if overwrite_weight == True:
            print("overwrite_weight is True. Continue.")
        else:
            print("overwrite_weight is False. Skip weight calculation.\n")
            return config

    ########################################################################################################################
    # calculate weights
    ds_inout = xr.load_dataset(file_stn_nearinfo)

    for var in ds_inout.data_vars:
        if keyword in var:
            print("Processing:", var)
            nearDistance = ds_inout[var].values
            dims = ds_inout[var].dims

            # Determine the structure and calculate weights accordingly
            if "Grid" in var:
                # Grid variables: (stn_combo, y, x, near)
                print(f"  Dimensions: {dims} - Processing as Grid data")
                n_combo, nrow, ncol, n_near = (
                    nearDistance.shape
                )  # ('stn_combo_sm', 'y', 'x', 'near')
                nearWeight = np.nan * np.ones(
                    [n_combo, nrow, ncol, n_near], dtype=np.float32
                )

                for combo in tqdm(range(n_combo), desc="Processing combos"):
                    for i in range(nrow):
                        for j in range(ncol):
                            nearWeight[combo, i, j, :] = (
                                calculate_weights_from_distance(
                                    nearDistance[combo, i, j, :],
                                    weight_max_distance,
                                    initial_distance,
                                    3,
                                    weight_formula,
                                )
                            )

            elif "InStn" in var:
                # Station variables: (stn_combo, stn, near)
                print(f"  Dimensions: {dims} - Processing as Station data")
                n_combo, nstn, n_near = nearDistance.shape
                nearWeight = np.nan * np.ones([n_combo, nstn, n_near], dtype=np.float32)

                for combo in tqdm(range(n_combo), desc="Processing combos"):
                    for s in range(nstn):
                        nearWeight[combo, s, :] = calculate_weights_from_distance(
                            nearDistance[combo, s, :],
                            weight_max_distance,
                            initial_distance,
                            3,
                            weight_formula,
                        )

            else:
                print(f"  WARNING: Unknown variable type for {var}, skipping")
                continue

            # Apply truncation distance
            nearWeight[nearDistance > truncation_dist] = 0

            # Add to dataset
            ds_inout[var.replace(keyword, "nearWeight")] = xr.DataArray(
                nearWeight, dims=dims
            )

    # drop some vars
    for var in ds_inout.data_vars:
        if np.any([k in var for k in keywords_drop]):
            ds_inout = ds_inout.drop_vars([var])

    # save to output files
    encoding = {}
    for var in ds_inout.data_vars:
        encoding[var] = {"zlib": True, "complevel": 4}
    ds_inout.to_netcdf(file_stn_weight, encoding=encoding)

    t2 = time.time()
    print("Time cost (s):", t2 - t1)
    print("Weight calculation completed successfully!\n\n")

    return config
