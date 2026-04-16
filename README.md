# GPEP-*θ*

This code is modified from the original GPEP code: https://github.com/NCAR/GPEP

[![DOI](https://zenodo.org/badge/674783032.svg)](https://zenodo.org/badge/latestdoi/674783032)

Desgined to run GPEP analysis for: 
- In-situ soil moisture data (*θ*), daily
- Soil moisture drydown parameter (*q*), seasonally

## Changes made
- `data_processing.py` is adjusted to my specific input file type
- `data_processing.py`, `near_stn_search.py`, `regression.py`, `weight_calculation.py`, and `probabilistic_auxiliary.py` are modified to allow time-varying data availability (nearDistance and nearWeight changes according to station data availability). 
- Functions not used in this experiment is removed for readability (e.g., probablistic variables treatment, unused regression functions)

### Notes on `model.settings.toml`

**Need to run again?** A checkmark means rerun that step (`overwrite=true`); leave earlier steps as-is when only the regression setup changes.

| Step | Variable name | Change data | Change regression design |
| --- | :---: | :---: | :---: |
| Merged station file |`overwrite_merged_stnfile`| ✓ | |
| Nearinfo |`overwrite_stninfo`| ✓ | |
| Weight|`overwrite_weight`| ✓ | |
| Spatial correlation (spcorr) |`overwrite_spcorr`| ✓ | |
| Regression (CV and Grid) |`overwrite_stn_cv_reg` `overwrite_grid_reg`| ✓ | ✓ |
| Spatio-temporal correlation of errors |`overwrite_station_cc`| ✓ | ✓ |


## Related References
### GPEP reference
- Tang, G, AW Wood, AJ Newman, MP Clark, and SM Papalexiou, 2023. GPEP v1.0: a Geospatial Probabilistic Estimation Package to support Earth Science applications. Geosci. Mod. Dev.  https://doi.org/10.5194/gmd-2023-172, 2024. 

### GMET (FORTRAN-based) datasets and methods
- __GMET v2.0__:  Bunn, PTW, AW Wood, AJ Newman, H Chang, CL Castro, MP Clark and JR Arnold, 2022, Improving station-based ensemble surface meteorological analyses using numerical weather prediction:  A case study of the Oroville Dam crisis precipitation event. J. Hydromet. 23(7), 1155-1169. https://doi.org/10.1175/JHM-D-21-0193.1  
- Liu, Hongli, AW Wood, AJ Newman and MP Clark, 2021, Ensemble dressing of meteorological fields: using spatial regression to estimate uncertainty in deterministic gridded meteorological datasets, AMS J. Hydromet., https://doi.org/10.1175/JHM-D-21-0176.1   
- Newman, A. J. et al. (2020) Probabilistic Spatial Meteorological Estimates for Alaska and the Yukon, Journal of Geophysical Research: Atmospheres, 125(22), pp. 1–21. https://doi.org/10.1029/2020JD032696  
- Newman, A. J. et al. (2019) Use of daily station observations to produce high-resolution gridded probabilistic precipitation and temperature time series for the Hawaiian Islands, Journal of Hydrometeorology, 20(3), pp. 509–529. https://doi.org/10.1175/JHM-D-18-0113.1  
- Mendoza, PA, AW Wood, EA Clark, E Rothwell, MP Clark, B Nijssen, LD Brekke, and JR Arnold, 2017, An intercomparison of approaches for improving predictability in operational seasonal streamflow forecasting, Hydrol. Earth Syst. Sci., 21, 3915–3935, 2017 (used a real-time implementation of GMET)
- Newman, AJ, MP Clark, J Craig, B Nijssen, AW Wood, E Gutmann, N Mizukami, L Brekke, and JR Arnold, 2015, Gridded Ensemble Precipitation and Temperature Estimates for the Contiguous United States, J. Hydromet., doi: http://dx.doi.org/10.1175/JHM-D-15-0026.1   
- Clark, M. P. and Slater, A. G. (2006) Probabilistic Quantitative Precipitation Estimation in Complex Terrain, Hydrometeorology, Journal O F, (2000), pp. 3–22. https://doi.org/10.1175/JHM474.1  

### Other datasets created using python regression scripts and GMET ensemble generation
- Tang, G., Clark, M. P., & Papalexiou, S. M. (2022). EM-Earth: The Ensemble Meteorological Dataset for Planet Earth. Bulletin of the American Meteorological Society, 103(4), E996–E1018. https://doi.org/10.1175/BAMS-D-21-0106.1  
- Tang, G., Clark, M. P., Papalexiou, S. M., Newman, A. J., Wood, A. W., Brunet, D., & Whitfield, P. H. (2021). EMDNA: an Ensemble Meteorological Dataset for North America. Earth System Science Data, 13(7), 3337–3362. https://doi.org/10.5194/essd-13-3337-2021

## Contacts
- Ryoko Araki, adapted this version for soil moisture analysis
- Guoqiang Tang (guoqiang@ucar.edu), GPEP primary developer
- Andy Wood (andywood@ucar.edu), GPEP project(s) lead





