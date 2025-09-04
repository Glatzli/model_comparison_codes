Modeling a cold air pool in the Inn Valley: A
model intercomparison study

ToDo: 
- revise ICON read in code (variables & calculation of missing vars)

Points: 
- How did Manuela interpolate the UM data? Manuela already compared qualitatively, I can also look at the code...
- writing: Methodology immediately, important to not forget how I've done what!
	create file with plots and notes! Note what I did and already found!

- PCGP calc:
	-> calc slope angle with numpy and aspect with xDEM due to strange errors (ValueError: Surface fit and
  	rugosity require the same X and Y resolution ((0.013980000000000005, 0.009879999999999977) was given).
 	This was required by: ['slope'].)
	created .tif file from topography data with rioxarray, added WGS84 projection attribute. wrong!
		PROBLEM: would need vCRS: vertical coordinate reference system.

  	richDEM: looks finally good & consistent for DEM & models!
	
  	- WRF: took geometric height as topo (multiple levels at one one point, terrain height is only 2D) ~ 20m difference for ibk gridpoint or woergl gridpoint... 

- calc VHD:
  - plot HATPRO data only for ibk point?(timeseries plot)
  - add radiosonde (point plot)!
  
  can it be correct with only 0.3 MJ/m^2 (in model!) -> calc for radiosounding & HATPRO (cosma already did that a bit, but I have
  different calc with reference at Hafelekar and not at max temp.)
  
  general notes for calc:
	- take care that all needed vars are read if some are calculated! No checks included due to time-reasons...
	- are units for VHD calc right? I have 3*10^6 values, cosma 1.1*10^6 -> YES, everything ok! VHD larger as for cosma
 - plot VHD as hourly small multiples :)
 - result of full domain calc & then selecting PCGP is different from point calc?
   		-> different levels of hafelekar-level!   
   	check for ibk villa point (index HAF height there):
   	AROME: height = 37 at Hafelekar 
   	ICON: height = 32 at Hafelekar 
    UM: height = 22 at Hafelekar 
   	WRF: height = 31
   	HATPRO: height = 43
   		=> take the geopot height of IBK gridpoint! Finally the same!

 - zmax: how to find stable layers/inversion: plot pot temp profiles for mult stations, maybe take same threshold as
   lukas umek in his paper? in vert. profiles it looked pretty shallow. Man braucht irgendein Maß für stärke und dicke des
   CAPs. zB Inversionshöhe & VHD. Maybe search for profiles at my plots? Warum zB ist mein VHD größer am Taleingang-> profil!



general notes/inputs:
- heat budget calc:
	- find extent: look at 2m temperature => get overview of valley res.
- model topogrpahy: files: AROME_geopot_height_3dlowest_level.nc saved with "lat" & "lon"
- Preliminary work:
  	- Hannes only looked at SEB in AROME and WRF right? for UM no output -> manuela will ask peter for missing data
  	- 2nd research question: in Rauchoecker et al 2023 they looked at budget only for WRF model with this -> i don't have all these vars!
 
- introduce first models, then show results and not else!
- Understand all variables written in a figure (f.e. Hz = vertical comp of heat flux...)
- temp timeseries 2d plot: uncertainty from standard atm. is equal to interp. of AROME (manuela)
	=> maybe compute comparison...
	not that important for 2-d plot like that, more important for calc of advection f.e.: real values!
	created new dataset with geopot heigt as variable in z, take 20th timestep (16.10. at 06:30 UTC) for geopot. height
- humidity:
  	- rather use specific humidity for comparing between models cause rel. humidity is largely temp dependent! 
- (rotach: compare radiation to know what's causing temp difference (not in model vars available!), compute from pot temp: makes no sense, zirkelschluss...)
- find extent of CAP: look at 2m temperature => get overview of valley in models

- first plot: temp timeseries 2d: (differences are calculated from 0.5 hourly timesteps and *2 to get K/hr)
HATPRO: interpolated HATPRO data to AROME levels & used AROME pressure to calculate HATPRO pot temp
now pot temp timeseries for ibk for all models incl HATPRO
- regridded ICON equal to normal grid (pot temp plot ibk): looks good!



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

- Projections of Models:
	- AROME: 
  	- ICON is in regular lat/lon grid, which doesn't have to be WGS84 conformal. 
	- UM: rotated pole projection was used (to have least error near Ibk)
       		grid_mapping_name:            rotated_latitude_longitude
     		longitude_of_prime_meridian:  0.0
    		earth_radius:                 6371229.0
     		grid_north_pole_latitude:     42.70000076293945
     		grid_north_pole_longitude:    191.39999389648438
     		north_pole_grid_longitude:    0.0
  	- WRF: Do I have to interpolate WRF-model? from attrs of netCDF file: lambertian conformal projection
  	  '+proj=lcc +lat_0=47.3000068664551 +lon_0=11.3999996185303 +lat_1=44 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs'
  		Manuela's input: For projecting the data into a new coordinate system (e.g., WGS84), you can use cartopy.crs (https://scitools.org.uk/cartopy/docs/latest/reference/crs.html). But even in WGS84, you will
		not have a regular grid. For a regular grid, you will need to interpolate the data. Interpolation always means some sort of loss, but on the other hand having all models on the same grid has the advantage
		that you could also calculate differences between the simulations.

  	I would like to have UM & WRF data also in a regular lat/lon grid. How can I reproject/transform that?
  	-> plotting with defining a projection in cartopy is easy (for UM!), for WRF i get: ValueError: operands could not be broadcast together with shapes (1,189,289) (3,3) ...
  	-> how interpolating/transforming?
  		- Pyproj: wrong! documentation states that Area of use is important. Is not defined.
  		Maybe I have to set Area of use somehow to get transformation with pyproj working? Only for coordinate transformation into new projection?
  		- xESMF: not for Win, setup WSL env, and calc but get still error: Missing cf conventions: dataset is not cf-compatible...
  		- stay with scipy? -> probably have to use loop for different levels, times etc?! -> probably slow and a bit complicated!   	
   	- UM: subtract rotated_latitude_longitude from coords to have regular grid?
   	  tried with pyproj but isn't correct, check calcs again!
  	- WRF: scipy 2d interpolation? or rather pyproj with smth like:
  		pyproj.Proj(proj="lcc",
                           lat_1 = um.attrs["TRUELAT1"], lat_2 = um.attrs["TRUELAT2"],
                           lat_0 = um.attrs["MOAD_CEN_LAT"], lon_0 = um.attrs["STAND_LON"],
                           a=6370000, b=6370000)
    		-> for now: continue with AROME & ICON, maybe include others later 
  	  	having completely same grid is most important for cross sections: that one can compare completely the same points.
  		can be calculated (interpolated) also later for 1 var if cross sect wanted.

		=> Manuela regridded WRF & UM, no worries anymore!



Verbesserungs ToDo's für die ich mir keine Zeit nehmen will:
- maybe us pickle for make read in faster? binary data formats. But it works for now.
- create one file "general_calculations"? for slope angles and future stuff?
- maybe overthink read in function setup again (to have enough functions with 1 task...)
- Tests! zmd Tests für die Einleseroutinen (maybe use chatgpt if read in routines are nearly finished?)...

erledigt:
- Einleseroutinen viel um/neugeschrieben
- hannes' code funktioniert nun auch bei mir
- cosma haben eig daten gefehlt, bzw hat sie AROME-Einleseroutine von hannes gar nicht verwendet!
- all metpy calculations now much faster
- create uniform time & height coordinates! (rename them) -> works now perfectly! also for regridded data...
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











































