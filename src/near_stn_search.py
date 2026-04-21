import os, sys, time
import xarray as xr
import numpy as np
import multiprocessing


def distance(lat1, lon1, lat2, lon2):
    # distance from lat/lon to km
    lat1r, lon1r, lat2r, lon2r = (
        np.radians(lat1),
        np.radians(lon1),
        np.radians(lat2),
        np.radians(lon2),
    )
    d = ((180 * 60) / np.pi) * (
        2
        * np.arcsin(
            np.sqrt(
                (np.sin((lat1r - lat2r) / 2)) ** 2
                + np.cos(lat1r) * np.cos(lat2r) * (np.sin((lon1r - lon2r) / 2)) ** 2
            )
        )
    )
    d = d * 1.852  # nautical mile to km
    return d


def find_nearstn_for_one_target(
    lat_tar,
    lon_tar,
    lat_stn,
    lon_stn,
    try_radius,
    initial_radius,
    nearstn_min,
    nearstn_max,
):
    # lat_tar/lon_tar: one value
    # lat_stn/lon_stn: vector lat/lon of stations
    # try_radius: a large radius depending on the station density. this helps to reduce computation time
    # initial_radius: initial radius to find stations
    # near station number will >= nearstn_min and <= nearstn_max

    # criteria: (1) find all stations within initial_radius, (2) if the number < nearstn_min, find nearstn_min nearby
    # stations without considering initial_radius

    near_index = -99999 * np.ones(nearstn_max, dtype=int)
    near_dist = np.nan * np.ones(nearstn_max, dtype=np.float32)
    stnID = np.arange(len(lat_stn))

    # basic control to reduce the number of input stations
    # Want to keep try_radius large if including all stations
    try_index = (np.abs(lat_stn - lat_tar) < try_radius) & (
        np.abs(lon_stn - lon_tar) < try_radius
    )
    lat_stn_try = lat_stn[try_index]
    lon_stn_try = lon_stn[try_index]
    stnID_try = stnID[try_index]

    # calculate distance (km)
    dist_try = distance(lat_tar, lon_tar, lat_stn_try, lon_stn_try)

    # Keep initial_radius large if including all stations
    index_use = dist_try <= initial_radius

    nstn = np.sum(index_use)

    # If there is more than nearstn_max stations, use the closest nearstn_max stations
    # Keep nearstn_max large if including all stations
    if nstn >= nearstn_max:  # strategy-1
        dist_try = dist_try[index_use]  # delete redundant stations
        stnID_try = stnID_try[index_use]
        index_final = np.argsort(dist_try)[:nearstn_max]
        near_index[0:nearstn_max] = stnID_try[index_final]
        near_dist[0:nearstn_max] = dist_try[index_final]
    # If there is less than nearstn_min stations, use the closest nearstn_min stations
    # Keep nearstn_min small if including all stations
    else:  # strategy-2
        dist = distance(lat_tar, lon_tar, lat_stn, lon_stn)
        index_use = dist <= initial_radius

        # If there is more than nearstn_min stations, use all of the stations within radius
        if np.sum(index_use) >= nearstn_min:
            stnID = stnID[try_index]
            dist = dist[try_index]
            nearstn_use = min(len(stnID), nearstn_max)

        else:
            nearstn_use = nearstn_min

        index_final = np.argsort(dist)[:nearstn_use]
        near_index[0:nearstn_use] = stnID[index_final]
        near_dist[0:nearstn_use] = dist[index_final]

    return near_index, near_dist


# parallel version
def find_nearstn_for_Grids(
    lat_stn,
    lon_stn,
    lat_grid,
    lon_grid,
    mask_grid,
    try_radius,
    nearstn_min,
    nearstn_max,
    initial_distance,
    num_processes=4,
):
    if lat_grid.ndim != 2:
        sys.exit("Error! Wrong dim of lat_grid!")

    # lon_stn/lat_stn can contain nan

    # simple distance threshold
    try_radius = (
        try_radius / 100
    )  # try within this degree (assume 1 degree ~= 100 km). if failed, expanding to all stations.

    # initialization
    nrows, ncols = np.shape(lat_grid)
    nearIndex = -99999 * np.ones([nrows, ncols, nearstn_max], dtype=int)
    nearDistance = np.nan * np.ones([nrows, ncols, nearstn_max], dtype=np.float32)

    # divide the grid into chunks for parallel processing
    chunk_size = nrows // num_processes
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    chunks[-1] = (
        chunks[-1][0],
        nrows,
    )  # last chunk may be larger if nrows is not a multiple of num_processes

    # process each chunk in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for chunk in chunks:
            result = pool.apply_async(
                process_chunk,
                (
                    chunk,
                    lat_stn,
                    lon_stn,
                    lat_grid,
                    lon_grid,
                    mask_grid,
                    try_radius,
                    nearstn_min,
                    nearstn_max,
                    initial_distance,
                ),
            )
            results.append(result)

        for result, chunk in zip(results, chunks):
            chunk_ni, chunk_nd = result.get()
            nearIndex[chunk[0] : chunk[1], :, :] = chunk_ni
            nearDistance[chunk[0] : chunk[1], :, :] = chunk_nd

    return nearIndex, nearDistance


# serial version
def find_nearstn_for_Grids_serial(
    lat_stn,
    lon_stn,
    lat_grid,
    lon_grid,
    mask_grid,
    try_radius,
    nearstn_min,
    nearstn_max,
    initial_distance,
):
    if lat_grid.ndim != 2:
        sys.exit("Error! Wrong dim of lat_grid!")

    # lon_stn/lat_stn can contain nan

    # simple distance threshold
    try_radius = (
        try_radius / 100
    )  # try within this degree (assume 1 degree ~= 100 km). if failed, expanding to all stations.

    # initialization
    nrows, ncols = np.shape(lat_grid)
    nearIndex = -99999 * np.ones([nrows, ncols, nearstn_max], dtype=int)
    nearDistance = np.nan * np.ones([nrows, ncols, nearstn_max], dtype=np.float32)

    # process sequentially without multiprocessing
    for rr in range(nrows):
        for cc in range(ncols):
            if mask_grid[rr, cc] == 1:
                ni, nd = find_nearstn_for_one_target(
                    lat_grid[rr, cc],
                    lon_grid[rr, cc],
                    lat_stn,
                    lon_stn,
                    try_radius,
                    initial_distance,
                    nearstn_min,
                    nearstn_max,
                )

                nearIndex[rr, cc, :], nearDistance[rr, cc, :] = ni, nd

    return nearIndex, nearDistance


def process_chunk(
    chunk,
    lat_stn,
    lon_stn,
    lat_grid,
    lon_grid,
    mask_grid,
    try_radius,
    nearstn_min,
    nearstn_max,
    initial_distance,
):
    nearIndex = -99999 * np.ones(
        [chunk[1] - chunk[0], lat_grid.shape[1], nearstn_max], dtype=int
    )
    nearDistance = np.nan * np.ones(
        [chunk[1] - chunk[0], lat_grid.shape[1], nearstn_max], dtype=np.float32
    )

    for rr in range(chunk[0], chunk[1]):
        for cc in range(lat_grid.shape[1]):
            if mask_grid[rr, cc] == 1:
                ni, nd = find_nearstn_for_one_target(
                    lat_grid[rr, cc],
                    lon_grid[rr, cc],
                    lat_stn,
                    lon_stn,
                    try_radius,
                    initial_distance,
                    nearstn_min,
                    nearstn_max,
                )
                nearIndex[rr - chunk[0], cc, :], nearDistance[rr - chunk[0], cc, :] = (
                    ni,
                    nd,
                )

    return nearIndex, nearDistance


def find_nearstn_for_InStn(
    lat_stn, lon_stn, try_radius, nearstn_min, nearstn_max, initial_distance
):
    # InStn: input stations themselves
    # lon_stn/lat_stn can contain nan
    # t1 = time.time()
    # print(f'Find near station for input stations')

    # simple distance threshold
    try_radius = (
        try_radius / 100
    )  # try within this degree (assume 1 degree ~= 100 km). if failed, expanding to all stations.

    # initialization
    nstn = len(lon_stn)
    nearIndex = -99999 * np.ones([nstn, nearstn_max], dtype=int)
    nearDistance = np.nan * np.ones([nstn, nearstn_max], dtype=np.float32)

    for i in range(nstn):
        lat_stni = lat_stn.copy()
        lon_stni = lon_stn.copy()
        lat_stni[i] = np.nan
        lon_stni[i] = np.nan
        if ~np.isnan(lat_stn[i]):
            nearIndex[i, :], nearDistance[i, :] = find_nearstn_for_one_target(
                lat_stn[i],
                lon_stn[i],
                lat_stni,
                lon_stni,
                try_radius,
                initial_distance,
                nearstn_min,
                nearstn_max,
            )

    return nearIndex, nearDistance


def get_near_station_info(config):
    """
    This is the main function of this module.

    This module is used to get the near station information for each station/grid.

    """

    t1 = time.time()

    # parse and change configurations
    path_stn_info = config["path_stn_info"]
    file_stn_nearinfo = f"{path_stn_info}/all_stn_nearinfo.nc"
    config["file_stn_nearinfo"] = file_stn_nearinfo

    # in/out information to this function
    file_allstn = config["file_allstn"]
    infile_grid_domain = config["infile_grid_domain"]
    file_stn_nearinfo = config["file_stn_nearinfo"]
    try_radius = config["try_radius"]
    initial_distance = config["initial_distance"]
    nearstn_min = config["nearstn_min"]
    nearstn_max = config["nearstn_max"]
    # target_vars = ['prcp', 'tmean', 'trange']
    target_vars = config["target_vars"]
    num_processes = config["num_processes"]
    stn_lat_name = config["stn_lat_name"]
    stn_lon_name = config["stn_lon_name"]
    grid_lat_name = config["grid_lat_name"]
    grid_lon_name = config["grid_lon_name"]
    grid_mask_name = config["grid_mask_name"]

    if "overwrite_stninfo" in config:
        overwrite_stninfo = config["overwrite_stninfo"]
    else:
        overwrite_stninfo = False

    print("#" * 50)
    print("Get near station information")
    print("#" * 50)
    print("input file_allstn:       ", file_allstn)
    print("input infile_grid_domain:", infile_grid_domain)
    print("output file_stn_nearinfo:", file_stn_nearinfo)
    print("nearstn_min:             ", nearstn_min)
    print("nearstn_max:             ", nearstn_max)
    print("try_radius:              ", try_radius)
    print("initial_distance:        ", initial_distance)
    print("Number of processes:     ", num_processes)

    if os.path.isfile(file_stn_nearinfo):
        print("NOTE: Nearest neighbor station info file exists")
        if overwrite_stninfo == True:
            print("overwrite_stninfo is True. Continue.")
        else:
            print(
                "overwrite_stninfo is False. Skip finding nearest neighbor station information.\n"
            )
            return config

    ########################################################################################################################
    # read station information
    ds_stn = xr.load_dataset(file_allstn)
    lat_stn_raw = ds_stn[stn_lat_name].values
    lon_stn_raw = ds_stn[stn_lon_name].values

    ########################################################################################################################
    # read domain information
    ds_domain = xr.load_dataset(infile_grid_domain)
    # ds_domain = ds_domain.rename({'x':'lon', 'y':'lat'})
    lat_grid = ds_domain[grid_lat_name].values
    lon_grid = ds_domain[grid_lon_name].values
    mask_grid = ds_domain[grid_mask_name].values
    ds_domain.coords["y"] = np.arange(lat_grid.shape[0])
    ds_domain.coords["x"] = np.arange(lat_grid.shape[1])

    # initialize output
    ds_nearinfo = ds_domain.copy()
    ds_nearinfo.coords["near"] = np.arange(nearstn_max)  # Max number of nearby stations
    ds_nearinfo.coords["stn"] = ds_stn.stn.values
    for v in ds_stn.data_vars:
        if not "time" in ds_stn[v].dims:
            ds_nearinfo["stn_" + v] = ds_stn[v]

    ########################################################################################################################
    # generate near info
    for i, vari in enumerate(target_vars):
        print(f"Processing variable: {vari}")

        # Get unique station combinations for this variable
        unique_stn_list_idx = np.unique(ds_stn[vari + "_avail_stn_idx_values"].values)
        n_unique_combos = len(unique_stn_list_idx)

        # Initialize arrays indexed by station combination for this variable
        nrows, ncols = lat_grid.shape
        nstn = len(ds_stn.stn)

        nearIndex_Grid_all = -99999 * np.ones(
            [n_unique_combos, nrows, ncols, nearstn_max], dtype=int
        )
        nearDistance_Grid_all = np.nan * np.ones(
            [n_unique_combos, nrows, ncols, nearstn_max], dtype=np.float32
        )
        nearIndex_InStn_all = -99999 * np.ones(
            [n_unique_combos, nstn, nearstn_max], dtype=int
        )
        nearDistance_InStn_all = np.nan * np.ones(
            [n_unique_combos, nstn, nearstn_max], dtype=np.float32
        )

        ########################################################################################################################
        # find nearby stations for each unique station combination
        for combo_idx, stn_list_idx in enumerate(unique_stn_list_idx):
            print(
                f"  Processing station combination {combo_idx + 1}/{n_unique_combos} (stn_list_idx={stn_list_idx})"
            )

            # Get timesteps with this station combination
            time_mask = ds_stn[vari + "_avail_stn_idx_values"].values == stn_list_idx

            # Get the mean value to identify available stations
            vm = ds_stn[vari].isel(time=time_mask).copy().values.mean(axis=0)

            # If the data is not available, skip this station combination
            if np.isnan(vm).all():
                print(
                    f"Data is not available for station combination {stn_list_idx}. Skip."
                )
                continue

            # Get the lat and lon of available stations
            lat_stn = lat_stn_raw.copy()
            lon_stn = lon_stn_raw.copy()
            lat_stn[np.isnan(vm)] = np.nan
            lon_stn[np.isnan(vm)] = np.nan

            # Find nearby stations for grids
            nearIndex_Grid, nearDistance_Grid = find_nearstn_for_Grids_serial(
                lat_stn,
                lon_stn,
                lat_grid,
                lon_grid,
                mask_grid,
                try_radius,
                nearstn_min,
                nearstn_max,
                initial_distance,
            )

            # nearIndex_Grid, nearDistance_Grid = find_nearstn_for_Grids(lat_stn, lon_stn, lat_grid, lon_grid, mask_grid, try_radius, nearstn_min,
            #                                                            nearstn_max, initial_distance, num_processes)

            # Find nearby stations for input stations
            nearIndex_InStn, nearDistance_InStn = find_nearstn_for_InStn(
                lat_stn, lon_stn, try_radius, nearstn_min, nearstn_max, initial_distance
            )

            # Store results indexed by station combination
            nearIndex_Grid_all[combo_idx, :, :, :] = nearIndex_Grid
            nearDistance_Grid_all[combo_idx, :, :, :] = nearDistance_Grid
            nearIndex_InStn_all[combo_idx, :, :] = nearIndex_InStn
            nearDistance_InStn_all[combo_idx, :, :] = nearDistance_InStn

        ########################################################################################################################
        # add near info to output file indexed by station combination
        # Create a coordinate for the station combination index
        ds_nearinfo.coords["stn_combo_" + vari] = unique_stn_list_idx

        ds_nearinfo["nearIndex_Grid_" + vari] = xr.DataArray(
            nearIndex_Grid_all, dims=("stn_combo_" + vari, "y", "x", "near")
        )
        ds_nearinfo["nearDistance_Grid_" + vari] = xr.DataArray(
            nearDistance_Grid_all, dims=("stn_combo_" + vari, "y", "x", "near")
        )
        ds_nearinfo["nearIndex_InStn_" + vari] = xr.DataArray(
            nearIndex_InStn_all, dims=("stn_combo_" + vari, "stn", "near")
        )
        ds_nearinfo["nearDistance_InStn_" + vari] = xr.DataArray(
            nearDistance_InStn_all, dims=("stn_combo_" + vari, "stn", "near")
        )

    # save to output files
    # Performance bottleneck: Writing large netCDF files with compression is slow
    # Potential optimizations:
    # 1. Reduce complevel (e.g., from 4 to 1-2) for faster writing with slightly larger files
    # 2. Use chunking to optimize I/O: encoding[var] = {"zlib": True, "complevel": 4, "chunksizes": (1, y_size, x_size, near_size)}
    # 3. Consider using netCDF4 engine explicitly: ds_nearinfo.to_netcdf(..., engine='netcdf4')
    # 4. For very large files, consider writing without compression first, then compress separately
    # encoding = {}
    # for var in ds_nearinfo.data_vars:
    #     encoding[var] = {"zlib": True, "complevel": 2}
    print(f"Saving {file_stn_nearinfo}")
    ds_nearinfo.to_netcdf(file_stn_nearinfo)  # , encoding=encoding)

    t2 = time.time()
    print("Time cost (s):", t2 - t1)
    print("Completed search of station nearest neighbors!\n")

    return config
