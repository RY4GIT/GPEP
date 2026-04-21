import sys, os, pathlib, time
import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool

from data_processing import data_transformation, calculate_ecdf_cdfs
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


def weight_linear_regression(y, x, weights=None, x_test=None):
    """
    https://www.statsmodels.org/stable/examples/notebooks/generated/wls.html
    https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.WLS.html
    """
    # X = sm.add_constant(x) # Looks like already has constant? From where?
    # X_test = sm.add_constant(x_test)
    X = x
    X_test = x_test

    model = sm.WLS(y, X, weights=weights, missing="drop").fit(disp=0)
    y_pred = model.predict(X_test)
    return y_pred, model


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
        """Sample dynamic predictor at station points (interp)"""
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
        """Regrid dynamic predictor to target grids (interp_like)"""
        if (len(interp_like.x) < len(ds.lat)) and (len(interp_like.y) < len(ds.lon)):
            method = "nearest"
        else:
            method = "linear"

        interp_like_latlon = interp_like.rename({"x": "lon", "y": "lat"})
        ds_out = ds.interp_like(interp_like_latlon, method=method)

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
    npredictor = (
        np.shape(stn_predictor)[1]
        + np.shape(dynamic_predictors["stn_predictor_dynamic"])[0]
    )

    ydata_tar = np.nan * np.zeros([r2 - r1, c2 - c1, ntime])
    model_stats = {
        "r_squared": np.nan * np.zeros([r2 - r1, c2 - c1, ntime]),
        "p_value": np.nan * np.zeros([r2 - r1, c2 - c1, ntime, npredictor]),
        "std_error": np.nan * np.zeros([r2 - r1, c2 - c1, ntime, npredictor]),
        "t_value": np.nan * np.zeros([r2 - r1, c2 - c1, ntime, npredictor]),
        "coefficients": np.nan * np.zeros([r2 - r1, c2 - c1, ntime, npredictor]),
    }

    # Loop through each grid cell in the chunk
    for r in range(r1, r2):
        for c in range(c1, c2):
            # interpolation for every time step
            for t in range(ntime):
                # prepare xdata and sample weight for training the regression model
                _sample_nearIndex = tar_nearIndex[t, r, c, :]
                _sample_weight = tar_nearWeight[t, r, c, :]
                index_valid = _sample_weight >= 0
                n_valid = np.sum(index_valid)

                if n_valid <= npredictor:
                    # If not enough valid predictors (Number of data point is less than the number of predictors),
                    # set the output to NaN
                    # print(
                    #     f"Warning: Not enough valid predictors at r={r}, c={c}, t={t}"
                    # )
                    ydata_tar[r - r1, c - c1, t] = np.nan
                    continue
                else:
                    # tar_nearIndex (ntime, nrow, ncol, stn)
                    # tar_nearWeight (ntime, nrow, ncol, stn)
                    # stn_predictor (stn, predictor)
                    # tar_predictor(nrow, ncol, predictor)
                    # xdata_near0 (stn, predictor)
                    # xdata_g0 (predictor)

                    sample_nearIndex = _sample_nearIndex[index_valid]
                    sample_weight = _sample_weight[index_valid]
                    xdata_near0 = stn_predictor[sample_nearIndex, :]
                    xdata_g0 = tar_predictor[r, c, :]

                    #########################################################
                    # PREPARE PREDICTOR MATRIX
                    #########################################################

                    # Get station data
                    ydata_near = np.squeeze(stn_data[t, sample_nearIndex])

                    # Get static predictors
                    xdata_near = xdata_near0
                    xdata_g = xdata_g0

                    # Get dynamic predictors
                    if dynamic_predictors["flag"]:
                        # Station-based dynamic predictors
                        xdata_near_add = dynamic_predictors["stn_predictor_dynamic"][
                            :, t, sample_nearIndex, sample_nearIndex
                        ].T  # (1, time, lat_stn, lon_stn)

                        # Grid-based dynamic predictors
                        xdata_g_add = dynamic_predictors["tar_predictor_dynamic"][
                            :, t, r, c
                        ]  # (nvars, time, lat_grid, lon_grid)

                        # Check if the dynamic predictors are valid
                        if np.all(~np.isnan(xdata_near_add)) and np.all(
                            ~np.isnan(xdata_g_add)
                        ):
                            # # Whether use a dynamic predictor (if it is static, diff will be close to 0, and we don't need to use it): Dimension gets messed up; deactivate it for now
                            # diff = np.max(xdata_near_add, axis=0) - np.min(
                            #     xdata_near_add, axis=0
                            # )
                            # tolerance = 1e-10

                            # # Remove stationary dynamic predictors
                            # xdata_near_add = xdata_near_add[:, diff > tolerance]
                            # xdata_g_add = xdata_g_add[diff > tolerance]

                            # Add dynamic predictors
                            if xdata_near_add.size > 0:
                                # If the dynamic predictors are valid, add them to the predictor matrix
                                xdata_near = np.hstack((xdata_near, xdata_near_add))
                                xdata_g = np.hstack((xdata_g, xdata_g_add))
                            else:
                                # The dynamic predictiors are not available, so not get added to the predictor matrix
                                print(
                                    f"No valid dynamic predictors at r={r}, c={c}, t={t}"
                                )

                    #########################################################
                    # MAIN REGRESSION PART
                    #########################################################

                    # Implement regression
                    if method == "Linear":
                        ydata_tar[r - r1, c - c1, t], model = weight_linear_regression(
                            y=ydata_near,
                            x=xdata_near,
                            weights=sample_weight,
                            x_test=xdata_g,
                        )

                        model_stats["r_squared"][r - r1, c - c1, t] = model.rsquared
                        model_stats["p_value"][r - r1, c - c1, t, :] = model.pvalues
                        model_stats["std_error"][r - r1, c - c1, t, :] = (
                            model.bse
                        )  # standard error of the beta coefficients
                        model_stats["t_value"][r - r1, c - c1, t, :] = model.tvalues
                        model_stats["coefficients"][r - r1, c - c1, t, :] = model.params

                    else:
                        sys.exit(f"Unknonwn regression method: {method}")

    return ydata_tar, model_stats


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

    # Initialize dynamic predictor flags
    if len(dynamic_predictors) == 0:
        dynamic_predictors["flag"] = False

    # Get the dimensions
    (ntime, nstn) = np.shape(stn_data)
    _, nrow, ncol, _ = np.shape(
        tar_nearIndex
    )  # ntime, 1, nstn, nstn or ntime, nrow, ncol, nstn

    # Create a list of chunks to process
    if nrow > ncol:
        chunks = [(r, r + 1, 0, ncol) for r in range(nrow)]
    else:
        chunks = [(0, nrow, c, c + 1) for c in range(ncol)]

    # Serial regression
    y_estimates = []
    model_stats = []
    for chunk in tqdm(chunks, leave=False):
        data = {}
        data["stn_data"] = stn_data
        data["stn_predictor"] = stn_predictor
        data["tar_nearIndex"] = tar_nearIndex
        data["tar_nearWeight"] = tar_nearWeight
        data["tar_predictor"] = tar_predictor
        # data["stn_combo_idx"] = stn_combo_idx
        data["method"] = method
        data["probflag"] = probflag
        data["settings"] = settings
        data["dynamic_predictors"] = dynamic_predictors

        y_estimate, model_stats_chunk = regression_for_a_chunk(
            chunk[0], chunk[1], chunk[2], chunk[3], data=data
        )
        y_estimates.append(y_estimate)
        model_stats.append(model_stats_chunk)

    # # Parallel regression
    # with Pool(
    #     processes=num_processes,
    #     initializer=init_worker,
    #     initargs=(
    #         stn_data,
    #         stn_predictor,
    #         tar_nearIndex,
    #         tar_nearWeight,
    #         tar_predictor,
    #         method,
    #         probflag,
    #         settings,
    #         dynamic_predictors,
    #         importmodules,
    #     ),
    # ) as pool:
    #     y_estimates, model_stats = pool.starmap(regression_for_a_chunk, chunks)

    # fill estimates to matrix
    estimates = np.nan * np.zeros([nrow, ncol, ntime], dtype=np.float32)
    npredictor = (
        np.shape(stn_predictor)[1]
        + np.shape(dynamic_predictors["stn_predictor_dynamic"])[0]
    )
    stats = dict(
        r_squared=np.nan * np.zeros([nrow, ncol, ntime], dtype=np.float32),
        p_value=np.nan * np.zeros([nrow, ncol, ntime, npredictor], dtype=np.float32),
        std_error=np.nan * np.zeros([nrow, ncol, ntime, npredictor], dtype=np.float32),
        t_value=np.nan * np.zeros([nrow, ncol, ntime, npredictor], dtype=np.float32),
        coefficients=np.nan
        * np.zeros([nrow, ncol, ntime, npredictor], dtype=np.float32),
    )
    for i, chunk in enumerate(chunks):
        estimates[chunk[0] : chunk[1], chunk[2] : chunk[3], :] = y_estimates[i]
        stats["r_squared"][chunk[0] : chunk[1], chunk[2] : chunk[3], :] = model_stats[
            i
        ]["r_squared"]
        stats["p_value"][chunk[0] : chunk[1], chunk[2] : chunk[3], :, :] = model_stats[
            i
        ]["p_value"]
        stats["std_error"][chunk[0] : chunk[1], chunk[2] : chunk[3], :, :] = (
            model_stats[i]["std_error"]
        )
        stats["t_value"][chunk[0] : chunk[1], chunk[2] : chunk[3], :, :] = model_stats[
            i
        ]["t_value"]
        stats["coefficients"][chunk[0] : chunk[1], chunk[2] : chunk[3], :, :] = (
            model_stats[i]["coefficients"]
        )

    t2 = time.time()
    print("Regression time cost (sec):", t2 - t1)
    return np.squeeze(estimates), stats


########################################################################################################################


class Conf:
    def __init__(self, d):
        self.__dict__ = d


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
    infile_grid_domain = config["infile_grid_domain"]
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
    ds_out["reg_predictor"] = xr.DataArray(
        ["constant"]
        + predictor_name_static_target
        + dynamic_predictor_name[0],  # only the first dynamic predictor is used for now
        dims=("reg_predictor"),
    )
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
            ds_domain = xr.open_dataset(infile_grid_domain)

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
        stn_combo_dim = f"stn_combo_{var_name}"

        # transformed or not
        if len(transform_vars[vn]) > 0:
            var_name_trans = var_name + "_" + transform_vars[vn]
            print("Regression for transformed variable:", var_name_trans)
            print(f"Variable {var_name} is transformed using {transform_vars[vn]}")
            print(
                f"{var_name_trans} instead of {var_name} will be loaded from the station data file {file_allstn}."
            )

            # # adjust max/min limits
            # if minRange_vars[vn] != -np.inf:
            #     minRange_vars[vn] = data_transformation(
            #         minRange_vars[vn],
            #         transform_vars[vn],
            #         transform_settings[transform_vars[vn]],
            #         "transform",
            #     )
            # if maxRange_vars[vn] != np.inf:
            #     maxRange_vars[vn] = data_transformation(
            #         maxRange_vars[vn],
            #         transform_vars[vn],
            #         transform_settings[transform_vars[vn]],
            #         "transform",
            #     )

        else:
            print("Regression for original variable:", var_name)
            var_name_trans = ""

        ########################################################################################################################
        # load data for regression

        ################################
        # station data
        ################################
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

            # Get combo index values per timestep for the station data
            ds_stn_combo_idx = ds_stn[var_name + "_avail_stn_idx_values"].values

        ################################
        # near information ###
        ################################
        with xr.open_dataset(file_stn_nearinfo) as ds_nearinfo:
            # Get the near index data with station combination dimension
            nearIndex_combo = ds_nearinfo[f"nearIndex_{near_keyword}_{var_name}"]
            if f"stn_combo_{var_name}" in nearIndex_combo.dims:
                ntime = len(ds_stn_combo_idx)

                # Initialize near index array
                if target == "grid":
                    # nearIndex_combo dims: (stn_combo, y, x, near)
                    _, ny, nx, n_near = nearIndex_combo.shape
                    nearIndex = np.zeros(
                        [ntime, ny, nx, n_near], dtype=nearIndex_combo.dtype
                    )

                elif target == "cval":
                    # nearIndex_combo dims: (stn_combo, stn, near)
                    _, nstn, n_near = nearIndex_combo.shape
                    # Expand dimension: (time, 1, stn, near) to match tar_predictor structure
                    nearIndex = np.zeros(
                        [ntime, 1, nstn, n_near], dtype=nearIndex_combo.dtype
                    )

                # Map each timestep to its corresponding combo
                for t in range(ntime):
                    nearIndex[t, :, :, :] = nearIndex_combo.sel(
                        {stn_combo_dim: ds_stn_combo_idx[t]}
                    ).values

            else:
                sys.exit(
                    f"Cannot find nearIndex_{near_keyword}_{var_name} in {file_stn_nearinfo}"
                )
        print(nearIndex.shape)
        ### predictor information ###
        if target == "grid":
            with xr.open_dataset(file_stn_nearinfo) as ds_nearinfo:
                # ds_nearinfo = ds_nearinfo.sel(time=slice(date_start, date_end))
                _, nrow, ncol, _ = np.shape(nearIndex)
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
                # Get the nearWeight data with station combination dimension
                nearWeight_combo = ds_weight[
                    vtmp
                ]  # dims: (stn_combo_X, stn/y, near/x, near)

                if f"stn_combo_{var_name}" in nearWeight_combo.dims:
                    ntime = len(ds_stn_combo_idx)

                    if target == "grid":
                        # nearWeight_combo dims: (stn_combo, y, x, near)
                        _, ny, nx, n_near = nearWeight_combo.shape
                        nearWeight = np.zeros(
                            [ntime, ny, nx, n_near], dtype=nearWeight_combo.dtype
                        )

                    elif target == "cval":
                        # nearWeight_combo dims: (stn_combo, stn, near)
                        _, nstn, n_near = nearWeight_combo.shape
                        # Expand dimension: (time, 1, stn, near) to match tar_predictor structure
                        nearWeight = np.zeros(
                            [ntime, 1, nstn, n_near], dtype=nearWeight_combo.dtype
                        )

                    # Map each timestep to its corresponding combo
                    if stn_combo_dim not in nearWeight_combo.dims:
                        sys.exit(
                            f"Unknown stn_combo dimension for variable: {var_name}"
                        )

                    for t in range(ntime):
                        nearWeight[t, :, :, :] = nearWeight_combo.sel(
                            {stn_combo_dim: ds_stn_combo_idx[t]}
                        ).values

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

        if dynamic_flag:
            # stn_predictor_dynamic dim: [n_feature, n_time, n_station]
            # tar_predictor_dynamic dim: [n_feature, n_time, n_station] or [n_feature, n_time, n_row, n_col]
            predictor_dynamic["stn_predictor_dynamic"] = np.stack(
                [ds_dynamic_stn[v].values for v in dynamic_predictor_name[vn]], axis=0
            )
            predictor_dynamic["tar_predictor_dynamic"] = np.stack(
                [ds_dynamic_tar[v].values for v in dynamic_predictor_name[vn]], axis=0
            )

            # if target == "cval":
            # If there is only one dynamic predictor, change raw dim: [n_feature, n_time, n_station] to [n_feature, n_time, 1, n_station]
            if len(dynamic_predictor_name[vn]) == 1:
                predictor_dynamic["tar_predictor_dynamic"] = predictor_dynamic[
                    "tar_predictor_dynamic"
                ][:, :, np.newaxis, :]

        ########################################################################################################################
        # get estimates at station points
        probflag = False  # for continuous variables

        if gridcore_continuous.startswith("LWR:"):
            # Transform max limit if necessary
            if var_name in target_vars_max_constrain:
                if len(transform_vars[vn]) > 0:
                    print(
                        "Perform max constraint for transformed variable:",
                        var_name_trans,
                    )
                    maxlimit = {
                        "flag": True,
                        "method": transform_vars[vn],
                        "setting": transform_settings[transform_vars[vn]],
                    }
                else:
                    maxlimit = {"flag": True}
            else:
                print("Perform max constraint for ", var_name, ": ", target)
                maxlimit = {"flag": False}

            if len(var_name_trans) > 0:
                print("Perform regression for transformed variable:", var_name_trans)
            else:
                print("Perform regression for original variable:", var_name)

            # Get estimates via regression
            estimates, stats = loop_regression(
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
                    cdfs = calculate_ecdf_cdfs(
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

        # Grid regression
        if estimates.ndim == 3:
            ds_out[var_name_save] = xr.DataArray(estimates, dims=("y", "x", "time"))
            for stat_name in stats:
                if stats[stat_name].ndim == 4:
                    ds_out["reg_" + stat_name] = xr.DataArray(
                        stats[stat_name].squeeze(),
                        dims=("y", "x", "time", "reg_predictor"),
                    )
                elif stats[stat_name].ndim == 3:
                    ds_out["reg_" + stat_name] = xr.DataArray(
                        stats[stat_name].squeeze(), dims=("y", "x", "time")
                    )

        # Station regression
        elif estimates.ndim == 2:
            ds_out[var_name_save] = xr.DataArray(estimates, dims=("stn", "time"))
            for stat_name in stats:
                if stats[stat_name].ndim == 4:
                    ds_out["reg_" + stat_name] = xr.DataArray(
                        stats[stat_name].squeeze(),
                        dims=("stn", "time", "reg_predictor"),
                    )
                elif stats[stat_name].ndim == 3:
                    ds_out["reg_" + stat_name] = xr.DataArray(
                        stats[stat_name].squeeze(), dims=("stn", "time")
                    )

            # evaluation
            dtmp1 = ds_stn[var_name].values
            if (len(var_name_trans) > 0) and (backtransform == False):
                # Ensemble is done on transformed variable
                if "ecdf" in var_name_trans:
                    # Evaluate using original target variable values even if the regression is done on transformed variable
                    cdfs = calculate_ecdf_cdfs(
                        xr.open_dataset(file_allstn),
                        var_name,
                        transform_settings[transform_vars[vn]],
                    )
                    _dtmp2 = data_transformation(
                        estimates,
                        transform_vars[vn],
                        transform_settings[transform_vars[vn]],
                        "back_transform",
                        times=ds_out["time"].values,
                        cdfs=cdfs,
                    )
                    dtmp2 = _dtmp2.T
                else:
                    _dtmp2 = data_transformation(
                        estimates,
                        transform_vars[vn],
                        transform_settings[transform_vars[vn]],
                        "back_transform",
                    )
                    dtmp2 = _dtmp2.T
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
