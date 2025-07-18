Modeling a cold air pool in the Inn Valley: A
model intercomparison study

ToDo:
- Do I have to interpolate WRF-model in attrs of netCDF file: lambertian conformal? UM (which projection)?
   Is ICON data now on a regular lat/lon grid or in WGS84?
- complete PCGP calc for Arome & ICON: works now, need to implement calculation of AD_gamma (calc aspect ratio)


Questions for next meeting: 
- for PCGP LU dataset I would need:
    Slope angle (β): The steepness of the terrain.
    Slope aspect (γ): The orientation of the slope.
    Roughness length (z₀): A measure of surface roughness affecting wind and turbulence.
    Albedo (α): The reflectivity of the surface.
  I don't have albedo measured, and LU spec is secondary.
  => Use only slope angle & aspect ratio:
  -> calc slope angle with numpy and aspect with xDEM due to strange errors (ValueError: Surface fit and
  rugosity require the same X and Y resolution ((0.013980000000000005, 0.009879999999999977) was given).
  This was required by: ['slope'].)
  works for AROME & ICON data

- maybe subset all models to have smaller datafiles? would save time each read in! Easy w. CDO
   -> would need to rewrite read in routines...
   UM & WRF is anyway not working for extent with lat/lon...

ToDo till next meeting:

- calc VHD


Input from presentation/last meetings:
- temp timeseries 2d plot: uncertainty from standard atm. is equal to interp. of AROME (manuela)
	=> maybe compute comparison...
	not that important for 2-d plot like that, more important for calc of advection f.e.: real values!
	created new dataset with geopot heigt as variable in z, take 20th timestep (16.10. at 06:30 UTC) for geopot. height
- humidity:
  	- rather use specific humidity for comparing between models cause rel. humidity is largely temp dependent! 
- (rotach: compare radiation to know what's causing temp difference (not in model vars available!), compute from pot temp: makes no sense, zirkelschluss...)


To Do:
- create one file "general_calculations"? for slope angels and future stuff?
- PCGP evaluation:
	would need:
	Slope angle (β): The steepness of the terrain.
	Slope aspect (γ): The orientation of the slope.
	(Roughness length (z₀): A measure of surface roughness affecting wind and turbulence.
	Albedo (α): The reflectivity of the surface.)

	+ LU dataset? -> not that significant! Topo params more important

	only from geopot height (+ICON slope angle) possible?
  	-> just compare slope angle & slope aspect from 8 model grid points with real, DEM one
  	calculate angle & aspect options:
  		- numpy: only slope angle: easy!
  		- xDEM: false Layout of weather models, not equal distance in lat-lon in DEM:
  		ValueError: Surface fit and rugosity require the same X and Y resolution ((0.004166666666666668, 0.004166666666666669) was given). This was required by: ['slope']
  		(saving geopot height of each model as .tiff) https://richdem.readthedocs.io/en/latest/python_api.html#richdem.TerrainAttribute
			=> calculate slope with numpy and aspect with xDEM? would work...
  			compute everything in "calculate_slope"-function. Problems with different file/data layout... (before nc file, now tif file for aspect w xDEM...)
  			
    		- RichDEM: failed installation, probably same as xDEM
  		- Google earth engine: complicated ...  https://developers.google.com/earth-engine/apidocs/ee-terrain-aspect#colab-python

  	Numpy:
	works! calculate PCGP now!
	tried to save .tif DEM-file with lat&lon instead x&y but didn't work...

    
- compute VHD: for spatial extent
  
- find extent of CAP: look at 2m temperature => get overview of valley in models
  
- heat budget calc:
	- find extent: look at 2m temperature => get overview of valley res.
- model topogrpahy: files: AROME_geopot_height_3dlowest_level.nc saved with "lat" & "lon"
- Preliminary work:
  	- Hannes only looked at SEB in AROME and WRF right? for UM no output -> manuela will ask peter for missing data
  	- 2nd research question: in Rauchoecker et al 2023 they looked at budget only for WRF model with this -> i don't have all these vars!
 
Feedback presentation:
- introduce first models, then show results and not else!
- Understand all variables written in a figure (f.e. Hz = vertical comp of heat flux...)

general notes:
- first plot: temp timeseries 2d: (differences are calculated from 0.5 hourly timesteps and *2 to get K/hr)
HATPRO: interpolated HATPRO data to AROME levels & used AROME pressure to calculate HATPRO pot temp
now pot temp timeseries for ibk for all models incl HATPRO
- regridded ICON equal to normal grid (pot temp plot ibk): looks good!



general notes:
- netcdf file chunking: how to find saved chunks? used chunks="auto", works o.k.
- model topography: probably in 2D variables (?) hannes used geopotential height! how best? (model topo or real DEM?): Rachoecker hat auch schiachen overview-plot: mache Ausschnitt v AROME topo etw größer, hau Stadtnamen rein u slope profiles? Inn-Valley Beschriftung & passt scho?
search grid cells & plot temp along the cells to get along-valley cross section -> Hannes made topo plot from geopot height, hgt variable is only tuned to 2m!
HOBO dataset: https://zenodo.org/records/4672313

ICON: tried psy-view, but it's only possible to visualize one single level!
tried plotting with cartopy but somehow dimensions doesn't really fit... is this "grid-file" missing?
"All approaches have in common that the NetCDF
grid file must be read in together with the data file." (Icon tutorial 2023)
look at 2D files (WRF), land cover (forest, etc)
AROME: hgt (2D), ICON: z_ifc, UM: hgt, WRF: hgt
-> make own overview plot till concept presentation?


- rather look at 3D data, not extrapolation to surface! all models extrapolate differently... probably extrapolate by myself to have it consistent -> later
- manuela' variable guidelines, search them to find topo variable? -> not really helpful!

- WSL workspace folder:
\\wsl.localhost\Ubuntu-24.04\mnt\wslg\distro\home\daniel\workspace\regrid_icon



Verbesserungs ToDo's für die ich mir keine Zeit nehmen will:
- maybe overthink read in function setup again (to have enough functions with 1 task...)
- Tests! zmd Tests für die Einleseroutinen (maybe use chatgpt if read in routines are nearly finished?)...

erledigt:
- Einleseroutinen viel um/neugeschrieben
- hannes' code funktioniert nun auch bei mir
- cosma haben eig daten gefehlt, bzw hat sie AROME-Einleseroutine von hannes gar nicht verwendet!
- all metpy calculations now much faster
- create uniform time & height coordinates! (rename them) -> done
- ...

erledigt:
- Einleseroutinen viel um/neugeschrieben
- hannes' code funktioniert nun auch bei mir
- cosma haben eig daten gefehlt, bzw hat sie AROME-Einleseroutine von hannes gar nicht verwendet!
- all metpy calculations now much faster
- create uniform time & height coordinates! (rename them) -> done

code optimization: 
- merge datasets every iteration through the variables?
- create a list, put datasets into it, then merge it with xr.merge => muuuuuch faster!!! 
quantify makes code pretty slow, use less often?!
still not sure how metpy calcs can be done fast with variables:
- rather define new variable with units
- or assign units to dataset and then dequantify again? -> small effect!)

cosma just adapted, often hard coded codes from Hannes. "zeitreihen"-notebooks meist sinnlos. Hannes hat 0 tests geschrieben...

heights of cosma: 
87: (icon, lowest lvl where?)
80: (icon2te)
56: hafelekar


old stuff:

Codes follow those provided by Hannes Wieser, but using Jupyter Notebooks (.ipynb).
All read-in's, calculations and plots are provided for the cities Innsbruck, Kufstein and Imst.

## Data

* Radiosounding
* HOBOs
* HATPRO
* AROME
* WRF (also called WRF_acinn or WRF_helen)
* UM (also called UKMO)
* ICON and ICON2TE


## Calculations and plots

Contain calculations of stability parameters, CAP depth and CAP characteristics such as T, P.

##### Stability parameters

* absolute T differences
* th gradients
* Brunt-Väisälä frequency
* Non-dimensional Valley depth
* Valley Heat deficit


## Additional Calculations

* time series (Zeitreihen)
* principal component analysis (PCA)
* Model domains and locations of Innsbruck, Kufstein, Imst


## Requirements

* confg.py
* adjust directory of all files which are read on top of each .ipynb.

##### Package requirements

* cartopy 0.22.0
* matplotlib 3.5.2
* metpy 1.4.1
* netcdf4 1.6.2
* numpy 1.26.1
* pandas 2.2.3
* rasterio 1.4.1
* salem 0.3.11
* skipy 1.13.1
* wrf_python 1.3.4.1
* xarray 2024.7.0



