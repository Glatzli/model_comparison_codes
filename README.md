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

ToDo presentation:
- state of research Rauchoecker
- topo overview AROME or WRF, PIANO
- preliminary work (Fabian Schöni, cosma)

Questions for meeting: 
- research goals: find extent: look at 2m temperature => get overview of valley resolution (i.e. where valley is)

- interpolation: matters more for quantitative analysis, for timerseries in vertical plots not that necessary (maybe use hydrostatic approach for HATPRO pot temp calculation (pressure))

"missing" data:
- Hannes only looked at SEB in AROME and WRF right? for UM no output -> manuela will ask peter for missing data
- make table for each model for surface & above surface: what can i calc where: schöni's thesis helps probably smth
- Why are models too warm research goal: look at temp equation and probably only radiation/advection which dominates? other terms parametrised probably & all different?...

in Rauchoecker et al 2023 they looked at budget only for WRF model with this -> i don't have all these vars!


general notes:
- first plot: temp timeseries 2d: (differences are calculated from 0.5 hourly timesteps and *2 to get K/hr)
HATPRO: interpolated HATPRO data to AROME levels & used AROME pressure to calculate HATPRO pot temp
now pot temp timeseries for ibk for all models incl HATPRO

models: create new dataset with geopot heigt as variable in z, take 20th timestep (16.10. at 06:30 UTC) for geopot. height
	


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



