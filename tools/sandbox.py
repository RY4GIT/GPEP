# %%
import xarray as xr
import numpy as np
import os
import geopandas as gpd

# %%
file_path = r"G:\Shared drives\Ryoko and Hilary\q-field\out\gpep\AZWG\ensembles\AZWGstatic_ensMember_20180101-20180102_001.nc"
ds = xr.load_dataset(file_path)

shp_filepath = r"G:\Shared drives\Ryoko and Hilary\q-field\data\USDA_ARS\AZWG\preprocessed\site_metadata\AZWG_boundary.shp"
shp = gpd.read_file(shp_filepath)
ds.sel(time="2018-01-01 00:00:00").sm.plot()
# %%
