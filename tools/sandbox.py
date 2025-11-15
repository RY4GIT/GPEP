# %%
import xarray as xr
import numpy as np
import os

# %%
file_path = r"G:\Shared drives\Ryoko and Hilary\q-field\out\gpep\AZWG\regression\AZWGstatic_grid_regression_20180101-20180210.nc"

ds = xr.load_dataset(file_path)
ds
# %%
file_path = r"G:\Shared drives\Ryoko and Hilary\q-field\out\gpep\AZWG\regression\AZWGstatic_stn_CV_regression_20180101-20180210.nc"

ds = xr.load_dataset(file_path)
ds
# %%
file_path = r"G:\Shared drives\Ryoko and Hilary\q-field\out\gpep\AZWG\regression\AZWGstatic_auxiliary_20180101-20180210.nc"
ds = xr.load_dataset(file_path)
ds


# %%
