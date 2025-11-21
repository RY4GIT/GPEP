import sys, os, pathlib, time
import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool
from data_processing import data_transformation, calculate_monthly_cdfs
from sklearn import *
import statsmodels.api as sm

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_allpoint
from tqdm import tqdm
# from scipy.interpolate import griddata, RegularGridInterpolator
# from sklearn.model_selection import KFold

# from functions import (
#     ludcmp,
#     lubksb,
#     linearsolver,
#     logistic_regression,
#     least_squares_numpy,
#     least_squares_ludcmp,
# )
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.neural_network import MLPRegressor, MLPClassifier


def weight_linear_regression(y, X, weights=None, x_test=None):
    """https://www.statsmodels.org/stable/examples/notebooks/generated/wls.html"""
    model = sm.WLS(y, X, weights=weights, missing="drop").fit(disp=0)
    y_pred = model.predict(x_test)
    return y_pred


########################################################################################################################
# dynmaic predictor-related functions


def check_dynamic_filepath(dynamic_predictor_filelist):
    # check whether files in dynamic_predictor_filelist use relative or absolute path
    # if the relative path is used, change it to the absolute path for simplicity
    with open(dynamic_predictor_filelist, "r") as f:
        file0 = f.readlines()

    file1 = []
    path0 = str(pathlib.Path(os.path.abspath(dynamic_predictor_filelist)).parent)
    flag = False
    for f in file0:
        f = f.strip()
        if not os.path.isabs(f):
            file1.append(f"{path0}/{f}")
            flag = True
        else:
            file1.append(f)

    if flag == True:
        print(
            f"Changing the relative path in {dynamic_predictor_filelist} to the absolute path"
        )
        with open(dynamic_predictor_filelist, "w") as f:
            for fi in file1:
                _ = f.write(f"{fi}\n")


def initial_check_dynamic_predictor(
    dynamic_predictor_name, dynamic_predictor_filelist, target_vars
):
    if not isinstance(dynamic_predictor_name, list):
        sys.exit(f"Error! dynamic_predictor_name must be a list.")

    if (len(target_vars) != len(dynamic_predictor_name)) and (
        len(dynamic_predictor_name) > 0
    ):
        sys.exit(
            f"Error! len(dynamic_predictor_name)>0 but is different from len(target_vars)!"
        )

    flag = False
    if not os.path.isfile(dynamic_predictor_filelist):
        print("Do not find dynamic_predictor_filelist:", dynamic_predictor_filelist)
    elif len(dynamic_predictor_name) == 0:
        print("dynamic_predictor_name length is 0")
    else:
        # change relative path to absolute path if needed
        check_dynamic_filepath(dynamic_predictor_filelist)

        with open(dynamic_predictor_filelist, "r") as f:
            file0 = f.readline().strip()
        if not os.path.isfile(file0):
            print(
                f"Do not find the first file: {file0} in dynamic_predictor_filelist: {dynamic_predictor_filelist}"
            )
        else:
            # with nc.Dataset(file0) as ds:
            with xr.open_dataset(file0) as ds:
                for i in range(len(dynamic_predictor_name)):
                    tmp = []
                    for v in dynamic_predictor_name[i]:
                        # if v in ds.variables.keys():
                        if v in ds.data_vars:
                            print(
                                f"Include {v} as a dynamic predictor for {target_vars[i]}"
                            )
                            tmp.append(v)
                        else:
                            print(
                                f"Cannot find {v} in {file0}. Do not include it as a dynamic predictor for {target_vars[i]}"
                            )
                    if len(tmp) > 0:
                        flag = True
                    dynamic_predictor_name[i] = tmp

    if flag == False:
        print("Dynamic predictors are not activated")
    else:
        print(
            f"Dynamic predictors are activated. Dynamic predictors are {dynamic_predictor_name}"
        )
    return flag


def map_filelist_timestep(dynamic_predictor_filelist, timeseries):
    # for every time step, find the corresponding input file

    # get file list
    filelist = []
    with open(dynamic_predictor_filelist, "r") as f:
        for line in f:
            filelist.append(line.strip())

    # make a dateframe containing all time steps
    df = pd.DataFrame()
    for i in range(len(filelist)):
        with xr.open_dataset(filelist[i]) as ds:
            timei = ds.time.values
            files = [filelist[i]] * len(timei)
            dfi = pd.DataFrame(
                {"intime": timei, "file": files, "ind": np.arange(len(timei))}
            )
            df = pd.concat((df, dfi))

    # for each time step, find the closest
    df_mapping = pd.DataFrame()
    for t in timeseries:
        if (
            t >= df["intime"].iloc[0] and t <= df["intime"].iloc[-1]
        ):  # don't extrapolate
            indi = np.argmin(np.abs(df["intime"] - t))  # nearest neighbor
            df_mapping = pd.concat((df_mapping, df.iloc[[indi]]))
        else:
            df_mapping = pd.concat(
                (
                    df_mapping,
                    pd.DataFrame({"intime": [np.nan], "file": [""], "ind": [np.nan]}),
                )
            )

    df_mapping["tartime"] = timeseries
    df_mapping.index = np.arange(len(df_mapping))

    return df_mapping


def read_period_input_data(df_mapping, varnames):
    # read and
    # select period
    # select variables

    # tarlon/tarlat: target lat/lon vector
    # default: dynmaic input files must have lat/lon dimensions

    files = np.unique(df_mapping["file"].values)
    files = [f for f in files if len(f) > 0]

    intime = df_mapping["intime"].values
    intime = intime[~np.isnan(intime)]

    tartime = df_mapping["tartime"].values

    if len(files) == 0:
        print("Warning! Cannot find any valid dynamic input files")
        ds = xr.Dataset()
    else:
        ds = xr.open_mfdataset(files)
        # select ...
        ds = ds[varnames]
        ds = ds.sel(time=slice(tartime[0], tartime[-1]))
        ds = ds.load()
        ds = ds.interp(time=tartime, method="linear")
    return ds


def regrid_xarray(ds, tarlon=None, tarlat=None, target=None, interp_like=None):
    # if target='1D', tarlon and tarlat are vector of station points
    # if target='2D', tarlon and tarlat are vector defining grids

    if target == "1D":
        if (len(tarlat) < len(ds.lat)) and (len(tarlon) < len(ds.lon)):
            # If the target grid is coarser than the source grid, use nearest neighbor interpolation
            method = "nearest"
        else:
            # If the target grid is finer than the source grid, use linear interpolation
            method = "linear"

        ds_out = ds.interp(
            lat=tarlat,
            lon=tarlon,
            method=method,
        )

    elif target == "2D":
        if (len(interp_like.x) < len(ds.lat)) and (len(interp_like.y) < len(ds.lon)):
            method = "nearest"
        else:
            method = "linear"

        ds_out = ds.interp_like(interp_like, method=method)

    return ds_out


def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


# ########################################################################################################################
# # machine learning regression


def train_and_return_test(
    Xtrain, ytrain, Xtest, method, probflag, settings={}, weight=[]
):
    """Unit train and return for ML method (currently only Random Forest)"""
    indexvalid = ~np.isnan(np.sum(Xtrain, axis=1) + ytrain)

    if Xtest.ndim == 1:
        Xtest = Xtest[np.newaxis, :]

    indexnan_test = np.isnan(np.sum(Xtest, axis=1))
    Xtest = Xtest.copy()
    Xtest[indexnan_test, :] = 0

    Xtrain = Xtrain[indexvalid, :]
    ytrain = ytrain[indexvalid]

    ldict = {"settings": settings, "method": method}
    exec(f"model = sklearn.{method}(**settings)", globals(), ldict)
    # exec(f"model = {method}(**settings)", globals(), ldict)
    model = ldict["model"]

    if len(weight) == 0:
        model.fit(Xtrain, ytrain)
    else:
        model.fit(Xtrain, ytrain, weight)

    if probflag == False:
        ytest = model.predict(Xtest)
    else:
        try:
            ytest = model.predict_proba(Xtest)[:, 1]
        except:
            ytest = model.predict(Xtest)

    ytest[indexnan_test] = np.nan
    return ytest


########################################################################################################################
# WEIGHTED REGRESSION
# parallel version of loop regression: independent processes and large memory use if there are many cpus


def init_worker(
    stn_data,
    stn_predictor,
    tar_nearIndex,
    tar_nearWeight,
    tar_predictor,
    method,
    probflag,
    settings,
    dynamic_predictors,
    importmodules,
):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    global mppool_ini_dict
    mppool_ini_dict = {}
    mppool_ini_dict["stn_data"] = stn_data
    mppool_ini_dict["stn_predictor"] = stn_predictor
    mppool_ini_dict["tar_nearIndex"] = tar_nearIndex
    mppool_ini_dict["tar_nearWeight"] = tar_nearWeight
    mppool_ini_dict["tar_predictor"] = tar_predictor
    mppool_ini_dict["method"] = method
    mppool_ini_dict["probflag"] = probflag
    mppool_ini_dict["settings"] = settings
    mppool_ini_dict["dynamic_predictors"] = dynamic_predictors

    for im in importmodules:
        if "." in im:
            im2 = im.split(".")[-1]
            im1 = im.replace("." + im2, "")
            exec(f"from {im1} import {im2}", globals())
        else:
            exec(f"import {im}", globals())

    # global Ridge
    # from sklearn.linear_model import Ridge


def regression_for_a_chunk(r1, r2, c1, c2, data=None):
    """
    Train regression model based on station data (weighted based on nearest stations)
    and use it to predict the values of the target grid cells at each time step.
    """
    if data is None:
        # Parallel regression (using global variables)
        stn_data = mppool_ini_dict["stn_data"]
        stn_predictor = mppool_ini_dict["stn_predictor"]
        tar_nearIndex = mppool_ini_dict["tar_nearIndex"]
        tar_nearWeight = mppool_ini_dict["tar_nearWeight"]
        tar_predictor = mppool_ini_dict["tar_predictor"]
        method = mppool_ini_dict["method"]
        probflag = mppool_ini_dict["probflag"]
        settings = mppool_ini_dict["settings"]
        dynamic_predictors = mppool_ini_dict["dynamic_predictors"]
    else:
        # Serial
        stn_data = data["stn_data"]
        stn_predictor = data["stn_predictor"]
        tar_nearIndex = data["tar_nearIndex"]
        tar_nearWeight = data["tar_nearWeight"]
        tar_predictor = data["tar_predictor"]
        method = data["method"]
        probflag = data["probflag"]
        settings = data["settings"]
        dynamic_predictors = data["dynamic_predictors"]

    (ntime, nstn) = np.shape(stn_data)
    ydata_tar = np.nan * np.zeros([r2 - r1, c2 - c1, ntime])

    # Loop through each grid cell in the chunk
    for r in range(r1, r2):
        for c in range(c1, c2):
            # prepare xdata and sample weight for training the regression model
            # tar_nearIndex (nrow, ncol, stn)
            sample_nearIndex = tar_nearIndex[r, c, :]
            index_valid = sample_nearIndex >= 0

            if np.sum(index_valid) > 0:
                # tar_nearWeight (nrow, ncol, stn)
                # stn_predictor (stn, predictor)
                # tar_predictor(nrow, ncol, predictor)
                # xdata_near0 (stn, predictor)
                # xdata_g0 (predictor)

                sample_nearIndex = sample_nearIndex[index_valid]
                sample_weight = tar_nearWeight[r, c, :][index_valid]
                xdata_near0 = stn_predictor[sample_nearIndex, :]
                xdata_g0 = tar_predictor[r, c, :]

                # interpolation for every time step
                for t in range(ntime):
                    ydata_near = np.squeeze(stn_data[t, sample_nearIndex])
                    if len(np.unique(ydata_near)) == 1:  # e.g., for prcp, all zero
                        ydata_tar[r - r1, c - c1, t] = ydata_near[0]
                    else:
                        # add dynamic predictors if flag is true and predictors are good
                        xdata_near = xdata_near0
                        xdata_g = xdata_g0

                        if dynamic_predictors["flag"] == True:
                            xdata_near_add = dynamic_predictors[
                                "stn_predictor_dynamic"
                            ][
                                :, t, sample_nearIndex, sample_nearIndex
                            ].T  # (1, time, lat_stn, lon_stn)
                            xdata_g_add = dynamic_predictors["tar_predictor_dynamic"][
                                :, t, r, c
                            ]  # (1, time, lat_grid, lon_grid)
                            if np.all(~np.isnan(xdata_near_add)) and np.all(
                                ~np.isnan(xdata_g_add)
                            ):
                                # whether use a dynamic predictor
                                diff = np.max(xdata_near_add, axis=0) - np.min(
                                    xdata_near_add, axis=0
                                )
                                tolerance = 1e-10

                                xdata_near_add = xdata_near_add[:, diff > tolerance]
                                xdata_g_add = xdata_g_add[diff > tolerance]

                                if xdata_near_add.size > 0:
                                    xdata_near_try = np.hstack(
                                        (xdata_near, xdata_near_add)
                                    )
                                    xdata_g_try = np.hstack((xdata_g, xdata_g_add))

                                    # The below codes using check_predictor_matrix_behavior are not necessary
                                    xdata_near = xdata_near_try
                                    xdata_g = xdata_g_try

                        # Main regression part
                        if method == "Linear":
                            ydata_tar[r - r1, c - c1, t] = weight_linear_regression(
                                y=ydata_near,
                                X=xdata_near,
                                weights=sample_weight,
                                x_test=xdata_g,
                            )
                        # elif method == "Logistic":
                        #     ydata_tar[r - r1, c - c1, t] = weight_logistic_regression(
                        #         xdata_near, sample_weight, ydata_near, xdata_g
                        #     )
                        # else:
                        # Machine learing methods
                        #     ydata_tar[r - r1, c - c1, t] = train_and_return_test(
                        #         xdata_near,
                        #         ydata_near,
                        #         xdata_g,
                        #         method,
                        #         probflag,
                        #         settings,
                        #         sample_weight,
                        #     )
                        else:
                            sys.exit(f"Unknonwn regression method: {method}")
                        # End of main regression part

                        # apply max limit (not implemented yet)

    return ydata_tar


def loop_regression(
    stn_data,
    stn_predictor,
    tar_nearIndex,
    tar_nearWeight,
    tar_predictor,
    method,
    probflag,
    settings,
    dynamic_predictors={},
    num_processes=4,
    importmodules=[],
    maxlimit={},
):
    t1 = time.time()

    if len(dynamic_predictors) == 0:
        dynamic_predictors["flag"] = False

    (ntime, nstn) = np.shape(stn_data)
    nrow, ncol, nearmax = np.shape(tar_nearIndex)

    # Create a list of chunks to process
    if nrow > ncol:
        chunks = [(r, r + 1, 0, ncol) for r in range(nrow)]
    else:
        chunks = [(0, nrow, c, c + 1) for c in range(ncol)]

    # # Serial regression
    # result = #fix dimension
    # for item in tqdm(items):
    #     data = {}
    #     data["stn_data"] = stn_data
    #     data["stn_predictor"] = stn_predictor
    #     data["tar_nearIndex"] = tar_nearIndex
    #     data["tar_nearWeight"] = tar_nearWeight
    #     data["tar_predictor"] = tar_predictor
    #     data["method"] = method
    #     data["probflag"] = probflag
    #     data["settings"] = settings
    #     data["dynamic_predictors"] = dynamic_predictors
    #     result[item] = regression_for_a_chunk(
    #         item[0], item[1], item[2], item[3], data=data
    #     )

    # Parallel regression
    with Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(
            stn_data,
            stn_predictor,
            tar_nearIndex,
            tar_nearWeight,
            tar_predictor,
            method,
            probflag,
            settings,
            dynamic_predictors,
            importmodules,
        ),
    ) as pool:
        result = pool.starmap(regression_for_a_chunk, chunks)

    # fill estimates to matrix
    estimates = np.nan * np.zeros([nrow, ncol, ntime], dtype=np.float32)
    for i, chunk in enumerate(chunks):
        estimates[chunk[0] : chunk[1], chunk[2] : chunk[3], :] = result[i]

    t2 = time.time()
    print("Regression time cost (sec):", t2 - t1)
    return np.squeeze(estimates)


########################################################################################################################


def main_regression(config, target):
    """Main function for regression"""

    ########################################################################################################################
    # PREPARATION
    ########################################################################################################################

    # target: loo (leave one out station) or grid
    t1 = time.time()

    # parse and change configurations
    case_name = config["case_name"]

    outpath_parent = config["outpath_parent"]
    path_regression = f"{outpath_parent}/regression/"
    os.makedirs(path_regression, exist_ok=True)

    datestamp = (
        f"{config['date_start'].replace('-', '')}-{config['date_end'].replace('-', '')}"
    )

    if "append_date_to_output_filename" in config:
        append_date_to_output_filename = config["append_date_to_output_filename"]
    else:
        append_date_to_output_filename = False

    if target == "grid":
        if append_date_to_output_filename == True:
            outfile = f"{path_regression}/{case_name}_grid_regression_{datestamp}.nc"  # regression without cross-validation
        else:
            outfile = f"{path_regression}/{case_name}_grid_regression.nc"
        config["file_grid_reg"] = outfile
        if "overwrite_grid_reg" in config:
            overwrite_flag = config["overwrite_grid_reg"]
        else:
            overwrite_flag = False
        predictor_name_static_target = config["predictor_name_static_grid"]

    elif target == "cval":
        if append_date_to_output_filename == True:
            outfile = f"{path_regression}/{case_name}_stn_CV_regression_{datestamp}.nc"  # leave one out regression
        else:
            outfile = f"{path_regression}/{case_name}_stn_CV_regression.nc"
        config["file_cval_reg"] = outfile

        if "overwrite_stn_cv_reg" in config:
            overwrite_flag = config["overwrite_stn_cv_reg"]
        else:
            overwrite_flag = False
        predictor_name_static_target = config["predictor_name_static_stn"]
    else:
        sys.exit("Unknown target!")

    # in/out information to this function

    file_allstn = config["file_allstn"]
    file_stn_nearinfo = config["file_stn_nearinfo"]
    file_stn_weight = config["file_stn_weight"]
    file_infile_grid_domain = config["infile_grid_domain"]
    outfile = outfile  # just to make sure all in/out settings are in this section

    target_vars = config["target_vars"]

    if "target_vars_WithProbability" in config:
        target_vars_WithProbability = config["target_vars_WithProbability"]
    else:
        target_vars_WithProbability = []

    if "probability_thresholds" in config:
        probability_thresholds = config["probability_thresholds"]
    else:
        probability_thresholds = [0] * len(target_vars_WithProbability)

    date_start = config["date_start"]
    date_end = config["date_end"]

    predictor_name_static_stn = config["predictor_name_static_stn"]
    predictor_name_static_target = predictor_name_static_target

    if "minRange_vars" in config:
        minRange_vars = config["minRange_vars"]
        if not isinstance(minRange_vars, list):
            minRange_vars = [minRange_vars] * len(target_vars)
        minRange_vars = minRange_vars.copy()
    else:
        minRange_vars = [-np.inf] * len(target_vars)

    if "maxRange_vars" in config:
        maxRange_vars = config["maxRange_vars"]
        if not isinstance(maxRange_vars, list):
            maxRange_vars = [maxRange_vars] * len(target_vars)
        maxRange_vars = maxRange_vars.copy()
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

    # this has not been really implemented yet
    if "target_vars_max_constrain" in config:
        target_vars_max_constrain = config["target_vars_max_constrain"]
    else:
        target_vars_max_constrain = []

    if "dynamic_predictor_filelist" in config:
        dynamic_predictor_filelist = config["dynamic_predictor_filelist"]
        dynamic_predictor_name = config["dynamic_predictor_name"]
        dynamic_predictor_operation = config["dynamic_predictor_operation"]
    else:
        dynamic_predictor_filelist = ""
        dynamic_predictor_name = []
        dynamic_predictor_operation = []

    num_processes = config["num_processes"]
    if "master_seed" in config:
        master_seed = config["master_seed"]
    else:
        master_seed = -1
    overwrite_flag = overwrite_flag

    gridcore_classification = config["gridcore_classification"]
    gridcore_continuous = config["gridcore_continuous"]

    n_splits = config["n_splits"]

    ensemble_flag = config["ensemble_flag"]
    backtransform = (
        not ensemble_flag
    )  # for example, if ensemble_flag=False, no need to create ensemble outputs,
    #   the regression outputs should be backtransformed in this step

    if "sklearn" in config:
        sklearn_config = config["sklearn"]
    else:
        sklearn_config = {}

    if target == "cval":
        # keyword for near information (default setting in this script)
        near_keyword = "InStn"  # input stations
    else:
        near_keyword = "Grid"

    stn_lat_name = config["stn_lat_name"]
    stn_lon_name = config["stn_lon_name"]

    grid_lat_name = config["grid_lat_name"]
    grid_lon_name = config["grid_lon_name"]

    dynamic_grid_lat_name = config["dynamic_grid_lat_name"]
    dynamic_grid_lon_name = config["dynamic_grid_lon_name"]

    print("#" * 50)
    if target == "cval":
        print(
            "step 1:  station point cross-validated regression to estimate predictive uncertainty"
        )
    elif target == "grid":
        print("step 2:  regression for all grid points")
    else:
        print("regression target not recognized:", target)
    print("#" * 50)
    print("Input file_allstn:      ", file_allstn)
    print("Input file_stn_nearinfo:", file_stn_nearinfo)
    print("Input file_stn_weight:  ", file_stn_weight)
    print("Output regression file: ", outfile)
    print("Output target:          ", target)
    print("Target variables:       ", target_vars)
    print("Number of processes:    ", num_processes)

    if os.path.isfile(outfile):
        print("Note! Output regression file exists")
        if overwrite_flag == True:
            print("overwrite_flag is True. Continue.")
        else:
            print("overwrite_flag is False. Skip regression.\n")
            return config

    dynamic_flag = initial_check_dynamic_predictor(
        dynamic_predictor_name, dynamic_predictor_filelist, target_vars
    )

    if master_seed < 0:
        master_seed = np.random.randint(1e9)
    np.random.seed(master_seed)

    ########################################################################################################################
    # load sklearn methods and settings
    gnew = []
    importmodules = []
    for g in [gridcore_classification, gridcore_continuous]:
        if g.startswith("LWR:"):
            g = g.replace("LWR:", "")

        # load sklearn modules
        if "." in g:
            m1 = g.split(".")[0]
            m2 = g.split(".")[1]
            # exec(f'global {m2}')
            # exec(f"from sklearn.{m1} import {m2}")
            g = m2
            importmodules.append(f"sklearn.{m1}.{m2}")

        gnew.append(g)

        # sklearn settings
        if not g in sklearn_config:
            sklearn_config[g] = {}

    gridcore_classification_short, gridcore_continuous_short = gnew

    ########################################################################################################################
    # initialize outputs
    with xr.open_dataset(file_allstn) as ds_stn:
        ds_stn = ds_stn.sel(time=slice(date_start, date_end))
        timeaxis = ds_stn.time.values

    ds_out = xr.Dataset()
    ds_out.coords["time"] = timeaxis
    if target == "grid":
        with xr.open_dataset(file_stn_nearinfo) as ds_nearinfo:
            xaxis = ds_nearinfo[grid_lon_name].isel(y=0).values
            yaxis = ds_nearinfo[grid_lat_name].isel(x=0).values
            ds_out.coords["x"] = ds_nearinfo.coords["x"]
            ds_out.coords["y"] = ds_nearinfo.coords["y"]
            ds_out[grid_lat_name] = ds_nearinfo[grid_lat_name]
            ds_out[grid_lon_name] = ds_nearinfo[grid_lon_name]

    elif target == "cval":
        ds_out.coords["stn"] = ds_stn.stn.values

    else:
        sys.exit(f"Unknown target: {target}")

    ########################################################################################################################
    # load dynamic factors if dynamic_flag == True
    # this costs more memory than reading for each time step, but reduces interpolation time. this could be a challenge for large domain

    if dynamic_flag:
        allvars = flatten_list(dynamic_predictor_name)

        df_mapping = map_filelist_timestep(dynamic_predictor_filelist, timeaxis)
        ds_dynamic = read_period_input_data(df_mapping, allvars)
        ds_dynamic = ds_dynamic.rename(
            {dynamic_grid_lat_name: "lat", dynamic_grid_lon_name: "lon"}
        )

        # transformation dynamic variables if necessary
        dyn_operation_trans = {}
        dyn_operation_interp = {}
        for op in dynamic_predictor_operation:
            info = op.split(":")
            for i in range(1, len(info)):
                if info[i].split("=")[0] == "transform":
                    dyn_operation_trans[info[0]] = info[i].split("=")[1]
                elif info[i].split("=")[0] == "interp":
                    dyn_operation_interp[info[0]] = info[i].split("=")[1]

        for v in ds_dynamic.data_vars:
            if v in dyn_operation_trans:
                print("Transform dynamic predictor:", v)
                ds_dynamic[v].values = data_transformation(
                    ds_dynamic[v].values,
                    dyn_operation_trans[v],
                    transform_settings[dyn_operation_trans[v]],
                    "transform",
                )

        # Interpolate dynamic predictors to station points
        ds_dynamic_stn = regrid_xarray(
            ds_dynamic,
            tarlon=ds_stn[stn_lon_name].values,
            tarlat=ds_stn[stn_lat_name].values,
            target="1D",
        )

        # Dynamic predictors for grids
        if target == "grid":
            # Get static predictor domain
            ds_domain = xr.open_dataset(file_infile_grid_domain)

            # Interpolate dynamic predictors to grid points
            ds_dynamic_tar = regrid_xarray(
                ds_dynamic,
                target="2D",
                interp_like=ds_domain,
            )

        # Dynamic predictors for station points
        elif target == "cval":
            ds_dynamic_tar = ds_dynamic_stn.copy()

    ########################################################################################################################
    # MAIN REGRESSION
    ########################################################################################################################

    ########################################################################################################################
    # loop variables
    for vn in range(len(target_vars)):
        var_name = target_vars[vn]
        print("Regression for:", var_name)

        # transformed or not
        if len(transform_vars[vn]) > 0:
            var_name_trans = var_name + "_" + transform_vars[vn]
            print(f"Variable {var_name} is transformed using {transform_vars[vn]}")
            print(
                f"{var_name_trans} instead of {var_name} will be loaded from the station data file {file_allstn}."
            )

            # adjust max/min limits
            if minRange_vars[vn] != -np.inf:
                minRange_vars[vn] = data_transformation(
                    minRange_vars[vn],
                    transform_vars[vn],
                    transform_settings[transform_vars[vn]],
                    "transform",
                )
            if maxRange_vars[vn] != np.inf:
                maxRange_vars[vn] = data_transformation(
                    maxRange_vars[vn],
                    transform_vars[vn],
                    transform_settings[transform_vars[vn]],
                    "transform",
                )

        else:
            var_name_trans = ""

        ########################################################################################################################
        # load data for regression

        # station data
        with xr.open_dataset(file_allstn) as ds_stn:
            ds_stn = ds_stn.sel(time=slice(date_start, date_end))

            if len(var_name_trans) > 0:
                stn_value = ds_stn[var_name_trans].values
            else:
                stn_value = ds_stn[var_name].values

            nstn = len(ds_stn.stn)
            predictor_static_stn = np.ones(
                [nstn, len(predictor_name_static_stn) + 1]
            )  # first column used for regression
            for i in range(len(predictor_name_static_stn)):
                predictor_static_stn[:, i + 1] = ds_stn[
                    predictor_name_static_stn[i]
                ].values

        ### near information ###
        with xr.open_dataset(file_stn_nearinfo) as ds_nearinfo:
            # nearDistance = ds_nearinfo['nearDistance_InStn_' + var_name].values
            vtmp = f"nearIndex_{near_keyword}_{var_name}"
            if vtmp in ds_nearinfo.data_vars:
                if target == "grid":
                    nearIndex = ds_nearinfo[vtmp].values.copy()
                elif target == "cval":
                    # nearIndex = ds_nearinfo[vtmp][np.newaxis, :, :].copy()
                    nearIndex = np.expand_dims(ds_nearinfo[vtmp].values, axis=0)
            else:
                sys.exit(
                    f"Cannot find nearIndex_{near_keyword}_{var_name} in {file_stn_nearinfo}"
                )

        ### predictor information ###
        if target == "grid":
            with xr.open_dataset(file_stn_nearinfo) as ds_nearinfo:
                nrow, ncol, nearmax = np.shape(nearIndex)
                predictor_static_target = np.ones(
                    [nrow, ncol, len(predictor_name_static_stn) + 1]
                )  # first column used for regression
                for i in range(len(predictor_name_static_target)):
                    prei = ds_nearinfo[predictor_name_static_target[i]].values
                    predictor_static_target[:, :, i + 1] = prei

        elif target == "cval":
            predictor_static_target = predictor_static_stn[np.newaxis, :, :].copy()

        ### weights ###
        with xr.open_dataset(file_stn_weight) as ds_weight:
            vtmp = f"nearWeight_{near_keyword}_{var_name}"
            if vtmp in ds_weight.data_vars:
                if target == "grid":
                    nearWeight = ds_weight[vtmp].values.copy()
                elif target == "cval":
                    nearWeight = np.expand_dims(
                        ds_weight[vtmp].values, axis=0
                    )  # ds_weight[vtmp][np.newaxis, :, :].copy()
            else:
                sys.exit(
                    f"Cannot find nearWeight_{near_keyword}_{var_name} in {file_stn_weight}"
                )

        ########################################################################################################################
        # produce predictor matrix for regression
        # other static or dynamic predictors can be added in the future

        stn_predictor = predictor_static_stn
        tar_predictor = predictor_static_target
        del predictor_static_stn, predictor_static_target

        # dynmaic predictors
        predictor_dynamic = {}
        predictor_dynamic["flag"] = dynamic_flag

        if dynamic_flag == True:
            # stn_predictor_dynamic dim: [n_feature, n_time, n_station]
            # tar_predictor_dynamic dim: [n_feature, n_time, n_station] or [n_feature, n_time, n_row, n_col]
            predictor_dynamic["stn_predictor_dynamic"] = np.stack(
                [ds_dynamic_stn[v].values for v in dynamic_predictor_name[vn]], axis=0
            )
            predictor_dynamic["tar_predictor_dynamic"] = np.stack(
                [ds_dynamic_tar[v].values for v in dynamic_predictor_name[vn]], axis=0
            )

            if target == "cval":
                # change raw dim: [n_feature, n_time, n_station] to [n_feature, n_time, 1, n_station]
                predictor_dynamic["tar_predictor_dynamic"] = predictor_dynamic[
                    "tar_predictor_dynamic"
                ][:, :, np.newaxis, :]

        ########################################################################################################################
        # get estimates at station points
        probflag = False  # for continuous variables

        if gridcore_continuous.startswith("LWR:"):
            # Transform max limit if necessary
            if var_name in target_vars_max_constrain:
                print("Perform max constraint for ", var_name)

                if len(transform_vars[vn]) > 0:
                    maxlimit = {
                        "flag": True,
                        "method": transform_vars[vn],
                        "setting": transform_settings[transform_vars[vn]],
                    }
                else:
                    maxlimit = {"flag": True}
            else:
                maxlimit = {"flag": False}

            print("Perform regression for ", var_name)
            # Get estimates via regression
            estimates = loop_regression(
                stn_value,
                stn_predictor,
                nearIndex,
                nearWeight,
                tar_predictor,
                gridcore_continuous[4:],
                probflag,
                sklearn_config[gridcore_continuous_short],
                predictor_dynamic,
                num_processes,
                maxlimit=maxlimit,
            )

        estimates = np.squeeze(estimates)

        ############################################################
        # constrain variables
        min_val = minRange_vars[vn]
        max_val = maxRange_vars[vn]

        if np.any(estimates < min_val):
            print(
                f"{var_name} estimates have values < {min_val}. Adjust those to {min_val}."
            )

        if np.any(estimates > max_val):
            print(
                f"{var_name} estimates have values > {max_val}. Adjust those to {max_val}."
            )

        estimates = np.clip(estimates, min_val, max_val)

        ########################################################################################################################
        # add to output ds
        if len(var_name_trans) > 0:
            if backtransform == False:
                var_name_save = var_name_trans
            else:
                var_name_save = var_name
                if "ecdf" in var_name_trans:
                    cdfs = calculate_monthly_cdfs(
                        xr.open_dataset(file_allstn),
                        var_name,
                        transform_settings[transform_vars[vn]],
                    )
                    estimates = data_transformation(
                        estimates,
                        transform_vars[vn],
                        transform_settings[transform_vars[vn]],
                        "back_transform",
                        times=ds_out["time"].values,
                        cdfs=cdfs,
                    )
                else:
                    estimates = data_transformation(
                        estimates,
                        transform_vars[vn],
                        transform_settings[transform_vars[vn]],
                        "back_transform",
                    )
        else:
            var_name_save = var_name

        if estimates.ndim == 3:
            ds_out[var_name_save] = xr.DataArray(estimates, dims=("y", "x", "time"))
        elif estimates.ndim == 2:
            ds_out[var_name_save] = xr.DataArray(estimates, dims=("stn", "time"))

            # evaluation
            dtmp1 = ds_stn[var_name].values
            if (len(var_name_trans) > 0) and (backtransform == False):
                if "ecdf" in var_name_trans:
                    cdfs = calculate_monthly_cdfs(
                        xr.open_dataset(file_allstn),
                        var_name,
                        transform_settings[transform_vars[vn]],
                    )
                    dtmp2 = data_transformation(
                        estimates,
                        transform_vars[vn],
                        transform_settings[transform_vars[vn]],
                        "back_transform",
                        times=ds_out["time"].values,
                        cdfs=cdfs,
                    )
                else:
                    dtmp2 = data_transformation(
                        estimates,
                        transform_vars[vn],
                        transform_settings[transform_vars[vn]],
                        "back_transform",
                    )
            else:
                dtmp2 = estimates.T

            metvalue, metname = evaluate_allpoint(dtmp1, dtmp2, np.nan)
            ds_out.coords["met"] = metname
            ds_out[var_name_save + "_metric"] = xr.DataArray(
                metvalue, dims=("stn", "met")
            )
            del dtmp1, dtmp2

    # reduce coordinate dims if needed
    if "x" in ds_out.dims and "y" in ds_out.dims:
        grid_lat_diff = np.abs(
            ds_out[grid_lat_name].isel(x=0).values
            - ds_out[grid_lat_name].isel(x=-1).values
        )
        grid_lon_diff = np.abs(
            ds_out[grid_lon_name].isel(y=0).values
            - ds_out[grid_lon_name].isel(y=-1).values
        )
        if (np.nanmax(grid_lat_diff) < 1e-10) and (np.nanmax(grid_lon_diff) < 1e-10):
            ds_out.coords["x"] = ds_out[grid_lon_name].isel(y=0).values
            ds_out.coords["y"] = ds_out[grid_lat_name].isel(x=0).values
            ds_out = ds_out.drop_vars([grid_lat_name, grid_lon_name])
            ds_out = ds_out.rename({"y": "lat", "x": "lon"})

    # save output file
    encoding = {}
    for var in ds_out.data_vars:
        encoding[var] = {"zlib": True, "complevel": 4}

    ds_out.to_netcdf(outfile, encoding=encoding)

    t2 = time.time()
    print("Time cost (s):", t2 - t1)
    print("Regression step completed successfully!\n")

    return config
