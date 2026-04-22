import read_config
import data_processing
import near_stn_search
import weight_calculation
import regression
import probabilistic_auxiliary
import data_correlation
import probabilistic_estimation

import sys, time
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    t1 = time.time()

    # config_file = sys.argv[1]
    config_file = r"C:\Users\flipl\dev\GPEP\config\config.GALR.dynamic.q.toml"

    ########################################################################################################################
    # load configuration file

    config = read_config.read_config(config_file)

    ########################################################################################################################
    # assemble individual stations and station attributes (e.g., lat, lon) to one netcdf file
    # See "data_processing.py" for more details
    config = data_processing.merge_stndata_into_single_file(config)

    ########################################################################################################################
    # get near station info for each station/grid
    # See "near_stn_search.py" for more details
    config = near_stn_search.get_near_station_info(config)

    ########################################################################################################################
    # calculate weights based on near station info. this step is independent to enable flexible weight test if needed.
    # See "weight_calculation.py" for more details
    config = weight_calculation.calculate_weight_using_nearstn_info(config)

    ########################################################################################################################
    # perform regression
    # (1) estimate predictive uncertainty using cross-validated (i.e., leave one out, LOO) regression at station points
    # See "regression.py" for more details
    config = regression.main_regression(config, "cval")

    # (2) estimate regression coefficients at all grid points
    # See "regression.py" for more details
    config = regression.main_regression(config, "grid")

    ########################################################################################################################
    # probabilistic / ensemble estimation
    if config["ensemble_flag"] == False:
        print(
            "ensemble_flag is false in the configuration file -- ensemble generation will be skipped."
        )

    else:
        ########################################################################################################################
        # estimate gridcell uncertainty based on interpolating estimation error from station locations (the LOO regression)
        # See "probabilistic_auxiliary.py" for more details
        config = probabilistic_auxiliary.extrapolate_auxiliary_info((config))

        ########################################################################################################################
        # probabilistic estimation (ensemble generation)

        # 1. get space and time correlations
        # See "data_correlation.py" for more details
        config = data_correlation.station_space_time_correlation(config)

        # 2. probabilistic estimation
        # See "probabilistic_estimation.py" for more details
        config = probabilistic_estimation.generate_prob_estimates(config)

    t2 = time.time()
    print("Total time cost (s):", t2 - t1)
    print("Successfully finished GPEP run!")
