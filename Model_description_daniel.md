# Models

## General Information

- [x] AROME
- [x] UKMO
- [x] ICON
- [x] ICON-2TE_BLM-GUF
- [x] WRF_ACINN


## AROME
-> rewrite read 3D function with fm_dataset!

problem: i don't have proper lat/lon coordinates...
nun transformierte Daten von Hannes: läuft
Cosma hatte die falschen Daten verwendet, bzw hat sie die Transformation nicht verwendet!

- Resolution: 1 kilometer
- Time (renamed: time): half an hour steps from 2017.10.15 12:00:00 to 2017.10.16 12:00:00 
- Longitude extend from: 7.405 to 15.395 degrees (0.1 degrees) in total 800 steps
- Latitude extends from 45.1 to 49.49 degrees by (0.1 degrees) in total 440 steps
- nz (renamed: height) extends from 1.0 to 90.0 (90 is at the ground)

2D AROME (`AROME_TEAMx_CAP_2D_fields`):

- are tuned, in the night the temperature is not taken from 2m but from 0.5 m (lower temperature)
- hfs = Sensible heat flux at the surface (Wm-2)
- hgt = Surface geopotential height (m)
- lfs = Latent heat flux at the surface (Wm-2)
- lwd = Longwave incoming radiation at the surface (Wm-2)
- lwnet = Longwave net radiation at the surface (Wm-2)
- lwu = Longwave outgoing radiation at the surface (Wm-2) derived lwnet - lwd
- pre = surface precipitation (kgm-2) same as mm
- ps = Surface pressure (Pa)
- swd = Shortwave incoming radiation at the surface (Wm-2)
- swnet = Shortwave net radiation at the surface (Wm-2)
- swu = Shortwave reflected radiation at the surface (Wm-2) derived swnet - swd
- tsk = Surface temperature (K) (Oberflächentemperatur auf 0m)

3D AROME (`AROME_TEAMx_CAP_3D_fields`):

- ciwc = Specific cloud ice water content (kg/kg)
- clwc = Specific cloud liquid water content (kg/kg)
- p = pressure (pa)
- q = Specific humidity (kg/kg)
- th = Potential temperature (K)
- tke = Turbulent kinetic energy (m2s-2)
- u = Zonal wind component (ms-1)
- v = Meridional wind component (ms-1)
- w = vertical wind velocity (ms-1)
- z = geopotential height (m)

Missing Vars:

- lcv (Land cover)
- zs (Depth of center of soil layers)
- ds (Thickness of soil layers)
- ste (Soil temperature)
- smo (Soil moisture)

Timeseries (`AROME_TEAMx_CAP_timeseries`):

- are interpolated to station location
- `ts_t` = AIR temperature (unit: K)
- `ts_q` = 2m specific humidity (unit: kg kg^-1)
- `ts_u` = 10m zonal wind component (unit: m s^-1)
- `ts_v` = 10m meridional wind component (unit: m s^-1)
- `ts_swd` = Shortwave incoming radiation at the surface (unit: W m^-2)
- `ts_swnet` = Shortwave net radiation at the surface (unit: W m^-2)
- `ts_lwd` = Longwave incoming radiation at the surface (unit: W m^-2)
- `ts_lwnet` = Longwave net radiation at the surface (unit: W m^-2)
- `ts_hfs` = Sensible heat flux at the surface (unit: W m^-2)
- `ts_lfs` = Latent heat flux at the surface (unit: W m^-2)
- `ts_swu` = Shortwave reflected radiation at the surface (unit: W m^-2)
- `ts_lwu` = Longwave outgoing radiation at the surface (unit: W m^-2)


## ICON
-> metpy calculations now much faster
maybe add subset for variables (would need a lot of if's as in wrf-fct if not all var's are read)
-> added read multiple hours fct => now easy 

untersch gitter je nach variable -> google
wie mit clat/clon umgehen? -> google
evtl use package: https://psyplot.github.io/psy-view/index.html
-> plot temp w z_ifc not possible 

Basically, the ICON grid is a triangular unstructured grid, so ncells is the total number of cells you may find in the
domain. As it is an unstructured grid, you need information on the neighbors of every cell to "re-construct" the grid
when plotting. If what you would like to do is a vertical profile, z_ifc is the variable you need. The model uses and
Arakawa-c grid, so you have mass points in the center of the cell, and complete level points. Some variables are
calculated in the cell's center (qv), and others at the full level (v). Technically, you should interpolate the
variables at the full model level to the mass points, but due to the resolution we used in the boundary layer and the
preliminary study you are doing, I do not think you need to do it. I would use for all variables z_ifc. Regarding
height, I think that for the time being you can ignore the first
point `take [1:90] of z_ifc height to have same coordinates` in the array (the highest level), so you have a 90 level
array (I forgot to storage the z_mc, which is the height for the full level).

The meteogram takes the values in the grid point nearest to the station's coordinate.

model storage data every 8 seconds.

`import xarray as xr`
`ds_met = xr.open_dataset("the_file_loc/the_file_name.nc")`

`station_name = np.char.decode(ds_met.station_name.values)
variables = np.char.decode(ds_met.var_name.values)
date = np.char.decode(ds_met.date)`

Meteogram (nvars=0 temperature (K), nvars=1 u_wind, nvars=2 v_wind, nvars=3 w_wind, nvars=4 tke, nvars=5
water_vapor_mixing_ratio):

- `time` - Timestamp of the data
- `t2m_C` - Temperature at 2 meters (unit: °C)
- `u` - Zonal wind (unit: m s^-1)
- `v` - Meridional wind (unit: m s^-1)
- `w` - Vertical velocity (unit: m s^-1)
- `tke` - Turbulent kinetic energy (unit: J kg^-1)
- `sh` - Specific humidity (unit: g kg^-1)

3D Variables:

- `clon_bnds` - No long name available (Units not specified)
- `clat_bnds` - No long name available (Units not specified)
- `elon_bnds` - No long name available (Units not specified)
- `elat_bnds` - No long name available (Units not specified)
- `height_bnds` - No long name available (Units not specified)
- `height_3_bnds` - No long name available (Units not specified)
- `depth_2_bnds` - No long name available (Units not specified)
- `u` - Zonal wind (Units: m s^-1)
- `v` - Meridional wind (Units: m s^-1)
- `w` - Vertical velocity (Units: m s^-1)
- `temp` - Temperature (Units: K)
- `pres` - Pressure (Units: Pa)
- `qv` - Specific humidity (Units: kg kg^-1)
- `qc` - Specific cloud water content (Units: kg kg^-1)
- `qi` - Specific cloud ice content (Units: kg kg^-1)
- `qr` - Specific rain content (Units: kg kg^-1)
- `qs` - Specific snow content (Units: kg kg^-1)
- `qg` - Specific graupel content (Units: kg kg^-1)
- `clc` - Cloud cover (Units: %)
- `tke` - Turbulent kinetic energy (Units: m^2 s^-2)
- `slope_angle` - Slope angle (Units: rad)
- `z_ifc` - Geometric height at half level center (Units: m)
- `pres_msl` - Mean sea level pressure (Units: Pa)
- `pres_sfc` - Surface pressure (Units: Pa)
- `rh_2m` - Relative humidity at 2m (Units: %)
- `qv_s` - Specific humidity at the surface (Units: kg kg^-1)
- `t_2m` - Temperature at 2m (Units: K)
- `tmin_2m` - Minimum temperature at 2m (Units: K)
- `td_2m` - Dew-point temperature at 2m (Units: K)
- `t_g` - Weighted surface temperature (Units: K)
- `u_10m` - Zonal wind at 10m (Units: m s^-1)
- `v_10m` - Meridional wind at 10m (Units: m s^-1)
- `w_i` - Weighted water content of interception water (Units: kg m^-2)
- `t_so` - Weighted soil temperature (main level) (Units: K)
- `w_so` - Total water content (ice + liquid water) in soil (Units: kg m^-2)
- `w_so_ice` - Ice content in soil (Units: kg m^-2)
- `smi` - Soil moisture index (Units not specified)
- `vn` - Velocity normal to edge (Units: m s^-1)
- `rho` - Density (Units: kg m^-3)
- `theta_v` - Virtual potential temperature (Units: K)
- `fr_land` - Fraction of land (Units not specified)
- `gz0` - Roughness length (Units: m)
- `t_ice` - Sea/lake-ice temperature (Units: K)
- `h_ice` - Sea/lake-ice depth (Units: m)
- `alb_si` - Sea ice albedo (diffuse) (Units: %)
- `t_mnw_lk` - Mean temperature of the water column (Units: K)
- `t_wml_lk` - Mixed-layer temperature (Units: K)
- `h_ml_lk` - Mixed-layer thickness (Units: m)
- `t_bot_lk` - Temperature at the water-bottom sediment interface (Units: K)
- `c_t_lk` - Shape factor (temperature profile in lake thermocline) (Units not specified)
- `fr_seaice` - Fraction of sea ice (Units: 1)
- `t_sk` - Skin temperature (Units: K)
- `t_seasfc` - Sea surface temperature (Units: K)
- `hsnow_max` - Maximum snow depth (Units: m)
- `snow_age` - Duration of snow cover (Units: d)
- `plantevap` - Time-integrated plant evaporation (Units: kg m^-2)
- `t_snow` - Weighted temperature of the snow surface (Units: K)
- `w_snow` - Water equivalent of snow (unit: kg m^-2)
- `rho_snow` - Snow density (unit: kg m^-3)
- `h_snow` - Snow depth (unit: m)
- `freshsnow` - Age of snow in top of snow layer (unit: 1)
- `snowfrac_lc` - Snow-cover fraction (unit: %)

## UKMO
orig als pd df eingelesen!
why does hannes interpolate pressure at wind levels if the same vertical coordinate is used (model_level_number or height for both?!)
maybe I could also use the "IRIS"-package for reading this data?
https://scitools-iris.readthedocs.io/en/stable/userguide/index.html

-> transformieren von ges. daten in lat, lon möglich? -> dzt nur für 1 pkt?
-> für 2d plots mehrere gitterpunkte einlesen: zB mit fixed point fkt, dann halt jeden gitterpkt einzeln
-> would need fixed_point_all_levels-fkt!
-> METPY-einbindung ist Pfusch: fehlende coordinaten! => ändere evtl auf metpy w. xarray?
-> bei wrf funktioniert ausrechnen ohne coordinaten zu verlieren!
original dimensions: "Time"; "bottom_top", changed to "time", "height"
-> is rh: relative humidity working correct (%?)

Available variables:

- `u` - transformed x wind (unit: m s^-1)
- `v` - transformed y wind (unit: m s^-1)
- `w` - upward air velocity (unit: m s^-1)
- `z` - geopotential height (unit: m^2 s^-2)
- `th` - air potential temperature (unit: K)
- `q` - specific humidity (unit: kg kg^-1)
- `p` - air pressure (unit: Pa)

## WRF_ACINN (ETH not used cause start of simulation is midnight!)

read_wrf_fixed_time: possible to read in only box of lat, lon -> program this also for other models?
-> fkt eig recht gut. Aber warum wieder erst dataframe erstellen und dann erst dataset draus machen?!
- definition of function "generate_datasets" inside function -> maybe rewrite this?!  
-> rewrite with read fm_dataset!
- metpy calcs could be re-written to be a bit faster...

Meteogram:

- `ts_time` - Time (Units: s)
- `ts_t` - Temperature (Units: K)
- `ts_q_mixingratio` - Mixing ratio of specific humidity (Units: kg kg^-1)
- `ts_u` - Zonal wind component (Units: m s^-1)
- `ts_v` - Meridional wind component (Units: m s^-1)
- `ts_sw_net` - Net shortwave radiation at the surface (Units: W m^-2)
- `ts_lwd` - Longwave downward radiation at the surface (Units: W m^-2)
- `ts_tsk` - Surface skin temperature (Units: K)
- `ts_hfs` - Sensible heat flux at the surface (Units: W m^-2)
- `ts_lfs` - Latent heat flux at the surface (Units: W m^-2)

3D Variables:

- `alb` - Albedo (Units not specified)
- `cwc` - Cloud water mixing ratio (Units: kg kg^-1)
- `ds` - Thicknesses of soil layers (Units: m)
- `emiss` - Surface emissivity (Units not specified)
- `hfs` - Upward heat flux at the surface (Units: W m^-2)
- `hgt` - Terrain height (Units: m)
- `ice` - Ice mixing ratio (Units: kg kg^-1)
- `lat` - Latitude, south is negative (Units: degree_north)
- `lcv` - Land use category (Units not specified)
- `lfs` - Latent heat flux at the surface (Units: W m^-2)
- `lon` - Longitude, west is negative (Units: degree_east)
- `lwd` - Downward longwave flux at ground surface (Units: W m^-2)
- `lwu` - Upward longwave flux at ground surface (Units: W m^-2)
- `p` - Air pressure (Units: Pa)
- `pre` - Precipitation rate (Units: kg m^-2)
- `ps` - Surface pressure (Units: Pa)
- `q_mixingratio` - Water vapor mixing ratio (Units: kg kg^-1)
- `smo_kgkg` - Soil moisture (Units: m^3 m^-3)
- `ste` - Soil temperature (Units: K)
- `swd` - Downward shortwave flux at ground surface (Units: W m^-2)
- `swu` - Upward shortwave flux at ground surface (Units: W m^-2)
- `th` - Perturbation potential temperature (theta-t0) (Units: K)
- `time` - Time since reference point (Units not specified)
- `tke` - Turbulent kinetic energy from PBL (Units: m^2 s^-2)
- `tsk` - Surface skin temperature (Units: K)
- `u` - X-wind component (Units: m s^-1)
- `v` - Y-wind component (Units: m s^-1)
- `w` - Z-wind component (Units: m s^-1)
- `z` - Geometric height (Units: m)
- `zs` - Depths of centers of soil layers (Units: m)