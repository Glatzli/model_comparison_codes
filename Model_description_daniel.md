# Models

## model overview

- AROME
- ICON
- ICON-2TE_BLM-GUF (same var's & setup as ICON)
- UKMO
- WRF_ACINN


## AROME
already improved:
	+ read_in_arome_fixed_point: with open_mfdataset insanely fast
	+ conver_calc_variables: had 0 calcs of temperature or rh before...
	+ read_in_arome_fixed_time: reads full domain at 1 timestamp (~2GB)

topography plot is lowest level of geopotential height

ToDo:
- maybe add functionality to fixed_time function to read only part of domain?
	

(vertikale coordinate in dataset ist ursprünglich nz, hannes nutzt für comparison_temp_icon_wrf_arome-plot einfach geopotentielle Höhe und nicht die echte Höhe?!? -> geopot. height ändert sich mit höhe!
er definiert für diesen plot read-in-arome nochmal neu & berechnet zeugs mit metpy => schiach!
dann erst erstellt er dataset mit den berechneten variablen, vorher ist es dataframe...)

- Resolution: 1 kilometer
- Time (renamed: time): half an hour steps from 2017.10.15 12:00:00 to 2017.10.16 12:00:00 
- Longitude extend from: 7.405 to 15.395 degrees (0.1 degrees) in total 800 steps
- Latitude extends from 45.1 to 49.49 degrees by (0.1 degrees) in total 440 steps
- nz extends from 1.0 to 90.0 (orig: 90 is at the ground) -> now inversed (1 is at the ground)

for plotting new created datasets:
- height (geopot. height [m] at timestep 20)
with vars:
- th = pot temp [K]
- p = pressure [hPa]

for PCGP created .tif files as ICON_geometric_height_3dlowest_level_w_crs.tif:
- x, y instead of lat, lon
- band_data instead of z
- added crs info (WGS84)

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
already improved:
	+ metpy calculations now much faster
	+ added read multiple hours fct => now easy
	+ add step to cut out only one point! (?)
 	+ regridded, cut to smaller extent
  	+ read ICON full extent for 1 timestep

ToDo:
- plot 2d
- Is staggering changed in regridding?

staggered/unstaggered grid: google!
evtl use package: https://psyplot.github.io/psy-view/index.html -> only plot of one distinct level possible!
-> plot temp w z_ifc not possible 

DOMAIN subset with CDO sellonlatbox to:
lat: 46.5 - 48.2° N
lon: 9.2° - 13° E


Basically, the ICON grid is a triangular unstructured grid, so ncells is the total number of cells you may find in the
domain. As it is an unstructured grid, you need information on the neighbors of every cell to "re-construct" the grid
when plotting. If what you would like to do is a vertical profile, z_ifc is the variable you need. The model uses and
Arakawa-c grid, so you have mass points in the center of the cell, and complete level points. Some variables are
calculated in the cell's center (qv), and others at the full level (v). Technically, you should interpolate the
variables at the full model level to the mass points, but due to the resolution we used in the boundary layer and the
preliminary study you are doing, I do not think you need to do it. I would use for all variables z_ifc. Regarding
height, I think that for the time being you can ignore the first point `take [1:90] of z_ifc height to have same coordinates` in the array (the highest level), so you have a 90 level
array (I forgot to storage the z_mc, which is the height for the full level).

The meteogram takes the values in the grid point nearest to the station's coordinate.
model storage data every 8 seconds.

`station_name = np.char.decode(ds_met.station_name.values)
variables = np.char.decode(ds_met.var_name.values)
date = np.char.decode(ds_met.date)`

Meteogram (nvars=0 temperature (K), nvars=1 u_wind, nvars=2 v_wind, nvars=3 w_wind, nvars=4 tke, nvars=5
water_vapor_mixing_ratio):

coordinates:
- time: half an hour steps from 2017.10.15 12:00:00 to 2017.10.16 12:00:00, read in only 1h steps!
- height: 1.0 ... 90.0 (90 is at the ground)
- height_2: 1.0 ... 91.0 (91 at ground)
- height_3: 1.0 ... 91.0

calculated vars:
- p = pressure [hPa]
- th = pot temp [K]
- temp = temperature [°C]

for plotting new created datasets:
- height (geometric height [m] at timestep 20)
with METPY calc vars:
- th = pot temp [K]
- temp = temperature [°C]

regridded to latlon variables, longname, coordinates:
- height_bnds, Kein long_name vorhanden, None, ['height']
- height_3_bnds, Kein long_name vorhanden, None, ['height_3']
- u, Zonal wind, m s-1, ['time', 'lon', 'lat', 'height']
- v, Meridional wind, m s-1, ['time', 'lon', 'lat', 'height']
- w, Vertical velocity, m s-1, ['time', 'lon', 'lat', 'height_2']
- temp, Temperature, K, ['time', 'lon', 'lat', 'height']
- pres, Pressure, Pa, ['time', 'lon', 'lat', 'height']
- qv, Specific humidity, kg kg-1, ['time', 'lon', 'lat', 'height']
- clc, cloud cover, %, ['time', 'lon', 'lat', 'height']
- tke, turbulent kinetic energy, m2 s-2, ['time', 'lon', 'lat', 'height_2']
- slope_angle, Slpe angle, rad, ['lon', 'lat']
- z_ifc, geometric height at half level center, m, ['lon', 'lat', 'height_3']
- rho, density, kg m-3, ['time', 'lon', 'lat', 'height']
- theta_v, virtual potential temperature, K, ['time', 'lon', 'lat', 'height']

chunk chosen with "auto":
    height_bnds    (height, bnds) float64 1kB dask.array<chunksize=(90, 2), meta=np.ndarray>
    height_3_bnds  (height_3, bnds) float64 1kB dask.array<chunksize=(91, 2), meta=np.ndarray>
    u              (time, height, lat, lon) float32 14GB dask.array<chunksize=(15, 28, 226, 350), meta=np.ndarray>
    v              (time, height, lat, lon) float32 14GB dask.array<chunksize=(15, 28, 226, 350), meta=np.ndarray>
    w              (time, height_2, lat, lon) float32 14GB dask.array<chunksize=(15, 28, 226, 350), meta=np.ndarray>
    temp           (time, height, lat, lon) float32 14GB dask.array<chunksize=(15, 28, 226, 350), meta=np.ndarray>
    ...             ...
    clc            (time, height, lat, lon) float32 14GB dask.array<chunksize=(15, 28, 226, 350), meta=np.ndarray>
    tke            (time, height_2, lat, lon) float32 14GB dask.array<chunksize=(15, 28, 226, 350), meta=np.ndarray>
    slope_angle    (lat, lon) float32 3MB dask.array<chunksize=(712, 1097), meta=np.ndarray>
    z_ifc          (height_3, lat, lon) float32 284MB dask.array<chunksize=(70, 557, 858), meta=np.ndarray>
    rho            (time, height, lat, lon) float32 14GB dask.array<chunksize=(15, 28, 226, 350), meta=np.ndarray>
    theta_v        (time, height, lat, lon) float32 14GB dask.array<chunksize=(15, 28, 226, 350), meta=np.ndarray>

 REGRIDDING:
- deleted original ICON files from D: to have space (had errors...) (if drive isn't working from WSL: sudo mount -t drvfs H: /mnt/h)
select variables and create new .nc file: CDO merges timestamps together. => only 1 gigantic ICON file w all timesteps!

cdo select,name=u,v,w,temp,pres,qv,clc,tke,slope_angle,z_ifc,rho,theta_v ICON_BLM-GUF_*.nc ICON_20171015Tall_selvars.nc
cdo -remap,latlon_grid_1km.txt,Wfile_TEAMX.nc ICON_20171015Tall_selvars.nc latlon_TEAMX.nc

DOMAIN extent (from clat & clon bnds to degree):
lat: 42.67218 - 49.728592
lon: 0.9697978 - 16.333878

## UKMO
already improved:
	+ uniformly xarray, no pandas!
	+ read_ukmo_fixed_point with open_mfdataset now insanely fast!

questions:
- what is the bnds-variable?
- why does hannes interpolate pressure at wind levels if the same vertical coordinate is used (model_level_number or height for both?!)
maybe I could also use the "IRIS"-package for reading this data?
https://scitools-iris.readthedocs.io/en/stable/userguide/index.html
- rh metpy calc: working correct (%?) -> looks o.k. from plotting

ToDo:
- no read_fixed_time: is loosing lat/lon info => transform them w. fct get_rotated_index_of_lat_lon (works currently only for 1 point)... 
evtl with fixed point fkt, dann halt jeden gitterpkt einzeln

for plotting new created datasets:
- height (geopot. height [m] at timestep 20)
with vars:
- th = pot temp [K]
- p = pressure [hPa]

coordinates:
- time: half an hour steps from 2017.10.15 12:00:00 to 2017.10.16 12:00:00 (drop first 2h)
- model_level_number (renamed height): 1 ... 70 (0 is at the ground)
- grid_latitude ?
- grid_longitude ?
- bnds ?

Available variables:

- `u` - transformed x wind (unit: m s^-1)
- `v` - transformed y wind (unit: m s^-1)
- `w` - upward air velocity (unit: m s^-1)
- `z` - geopotential height (unit: m^2 s^-2)
- `th` - air potential temperature (unit: K)
- `q` - specific humidity (unit: kg kg^-1)
- `p` - air pressure (unit: Pa)
When read with open_mfdataset vars get long names, renamed them!

calculated variables with MetPy:
- pressure [hPa]
- temperature [°C]

## WRF_ACINN (WRF_ETH not used cause start of simulation is midnight!)
already improved: 
	+ work uniformly with xarrray datasets, no pandas! 
	+ read_wrf_fixed_time much faster (w. read_mfdataset)
	+ metpy calculations: only necessary & faster
	+ time/height coords: uniform w other models
	+ generate_datasets function properly defined

topography plot still doesn't work, I don't know why...

ToDo:
- I don't have geopotential height! Only geometric height or (best) terrain height... but miss this for other vars!
read_wrf_fixed_time: possible to read in only box of lat, lon -> dimensions are south_north & west_east 
-> impossible to find lat/lon! -> change projection with pyproj?
	I have to define goal projection => subset it first with CDO?
  
Solution:
- xWRF? https://xarray.dev/blog/introducing-xwrf

Projection Info:
from loading wrf-files directily w XARRAY: Map Proj char: Lambert Conformal
OUTPUT FROM WRF V4.4.1 model

coordinates:
- Time (renamed time): half an hour steps from 2017.10.15 12:00:00 to 2017.10.16 12:00:00
- bottom_top (renamed height): 0 ... 69 (0 is at the ground)
- south_north: ?
- west_east: ?

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
- `time` - Time since reference point (Units not specified) -> deleted
- `tke` - Turbulent kinetic energy from PBL (Units: m^2 s^-2)
- `tsk` - Surface skin temperature (Units: K)
- `u` - X-wind component (Units: m s^-1)
- `v` - Y-wind component (Units: m s^-1)
- `w` - Z-wind component (Units: m s^-1)
- `z` - Geometric height (Units: m)
- `zs` - Depths of centers of soil layers (Units: m)


--------------------------------------------------------------------------------------------
OLD stuff:
ICON:
? 
- `time` - Timestamp of the data
- `t2m_C` - Temperature at 2 meters (unit: °C)
- `u` - Zonal wind (unit: m s^-1)
- `v` - Meridional wind (unit: m s^-1)
- `w` - Vertical velocity (unit: m s^-1)
- `tke` - Turbulent kinetic energy (unit: J kg^-1)
- `sh` - Specific humidity (unit: g kg^-1)

original HEXA 3D Variables:
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

attrs from orig netcdf file:
{'CDI': 'Climate Data Interface version 1.8.3rc (http://mpimet.mpg.de/cdi)',
 'Conventions': 'CF-1.6',
 'number_of_grid_used': 1,
 'uuidOfHGrid': '655488b8-6e60-ac09-a653-9b1ce37a2b20',
 'uuidOfVGrid': '5210aca5-6684-c009-3731-0a31182a3180',
 'institution': 'Max Planck Institute for Meteorology/Deutscher Wetterdienst',
 'title': 'ICON simulation',
 'source': 'git@gitlab.dkrz.de:icon/icon-nwp.git@1638fcbef3269d733d8bc637d523f31663fb60c3',
 'history': '/work/bb1096/b380910/models/icon/icon-nwp_2TE//bin/icon at 20230331 144353',
 'references': 'see MPIM/DWD publications',
 'comment': 'Julian Quimbayo-Duarte (b380910) on l30537 (Linux 4.18.0-348.el8.x86_64 x86_64)'}
