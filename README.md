Modeling a cold air pool in the Inn Valley: A
model intercomparison study

What i would probably need for my analysis/plots:
2d slices: 
- time-height/pressure visualization at certain point -> calculate valley-heat deficit -> proper read in for all models!
- longitude (x) & height-slices (get overview of full valley) or lat (y) & height-slices (comparison with SPs)
=> works only for AROME-model w lat/lons!

erledigt:
- Einleseroutinen viel um/neugeschrieben
- hannes' code funktioniert nun auch bei mir
- cosma haben eig daten gefehlt, bzw hat sie AROME-Einleseroutine von hannes gar nicht verwendet!
- all metpy calculations now much faster
- create uniform time & height coordinates! (rename them) -> done

ToDo till next meeting:
- focus on presentation & concept: make first draft of presentation slides for concept pres.
- 1 plot to motivate problem from hannes f.e.
- imagine presenting to present to colleaugue or physics student

general notes:
- first plot: temp timeseries 2d:
1. plot hatpro data: only normal temp [K] available, no pot temp! => need pressure for calculation!
	make plot w temp [°C]: Problems with calcs & dask read ins... AND you don't see anything really! -> take pressure values from any (AROME) model and calculate pot temp for HATPRO data to plot it?
=> now interpolated HATPRO data (linearly?) onto AROME levels and used AROME pressure levels to calc pot temp

	=> read only th & p for temp-plot?
2. make vertical coord uniform w geopot height to show at meeting: -> create new dataset with new vertical coordinate in read in routines for all models! geopot height is not changing a lot (just use timestep in middle of period) and plot temp etc along geopot height
(- vertical coordinates: now use standard-coordinate 0 ... 90 approx. could also use pressure (from first timestep?-> changes w time! Problem: it's not a coordinate for any model!)


- model topography: probably in 2D variables (?) hannes used geopotential height! how best? (model topo or real DEM?): Rachoecker hat auch schiachen overview-plot: mache Ausschnitt v AROME topo etw größer, hau Stadtnamen rein u slope profiles? Inn-Valley Beschriftung & passt scho?
search grid cells & plot temp along the cells to get along-valley cross section: 
ICON: tried psy-view, but it's only possible to visualize one single level!
tried plotting with cartopy but somehow dimensions doesn't really fit... is this "grid-file" missing?
"All approaches have in common that the NetCDF
grid file must be read in together with the data file." (Icon tutorial 2023)
look at 2D files (WRF), land cover (forest, etc)
AROME: hgt (2D), ICON: z_ifc, UM: hgt, WRF: hgt
-> make own overview plot till concept presentation?


- concept work 2nd research question, time onset of cap -> write concept!
- horizontal plots: need to solve problem with lat/lons to know which gridpoint is chosen in the end 
look at hannes' 2d temp plots for lat/lon things...
- for presentation look at radiosonde, hatpro data (hannes plots), maybe include in pres.

- rather look at 3D data, not extrapolation to surface! all models extrapolate differently... probably extrapolate by myself to have it consistent -> later
- manuela' variable guidelines, search them to find topo variable? -> not really helpful!
- heat budget: look at which variables are in model ouput, which can i calculate for the budget eq? advection for all
- rather use specific humidity for comparing between models cause rel. humidity is largely temp dependent! ()


Verbesserungs ToDo's für die ich mir keine Zeit nehmen will:
- maybe overthink read in function setup again (to have enough functions with 1 task...)
- Tests! zmd Tests für die Einleseroutinen (maybe use chatgpt if read in routines are nearly finished?)...

created additionalcalc/timeseries notebook for first draft of timeseries & 2d plots

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



