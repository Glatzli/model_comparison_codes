# Why are NWMs too warm in a cold air pool? A
model intercomparison study

cosma just adapted, often hard coded codes from Hannes. "zeitreihen"-notebooks meist sinnlos. Hannes hat auch 0.0 tests geschrieben...
vertical coordinates: 
- cosma used in plotting routine again metpy to calculate pressure & pot temp => bullshit
- hannes sometimes used geopotential height for plotting model topography

What i would probably need for my analysis/plots:
2d slices: 
- time-height/pressure visualization at certain point -> calculate valley-heat deficit -> proper read in for all models!
- longitude (x) & height-slices (get overview of full valley) or lat (y) & height-slices (as in Schöni's thesis)
-> proper read-in only for AROME-model

(Hannes codes laufen jetzt auch alle bei mir! -> kann einiges recyceln
code optimization: 
- merge datasets every iteration through the variables?
- create a list, put datasets into it, then merge it with xr.merge => muuuuuch faster!!! 
quantify makes code pretty slow, use less often?!
still not sure how metpy calcs can be done fast with variables:
- rather define new variable with units
- or assign units to dataset and then dequantify again? -> small effect!)

erledigt:
- Einleseroutinen viel um/neugeschrieben
- hannes' code funktioniert nun auch bei mir
- cosma haben eig daten gefehlt, bzw hat sie AROME-Einleseroutine von hannes gar nicht verwendet?!
- all metpy calculations now much faster
- create uniform time & height coordinates! (rename them) -> done

ToDo till next meeting:
- make overview topo plot of domain (ICON) => domain of model, probably in 2D variables
- temp timeseries 2d: which vertical coordinate is where?
- rather look at 3D data, not extrapolation to surface! all models extrapolate differently... probably extrapolate by myself to have it consistent -> later
- manuela' variable guidelines, search them to find topo variable? -> not really helpful!
- concept work 2nd research question, time onset of cap -> write concept!
- make first plots: make timeseries of surroundings of ibk? first 2d, timeseries plot? -> for ibk done, not for surroundings
- need to solve problem with lat/lons to know which gridpoint is chosen in the end 
- for presentation look at radiosonde, hatpro data (hannes plots), maybe include in pres.

FIRST:
- what are vertical levels? make plot uniform to compare them -> maybe use pressure? Problem: it's not a coordinate for any model. 
- helen: plot topography of each model -> where is Inn-valley? search grid cells & plot temp along the cells to get along-valley cross section: 
ICON: tried psy-view, but it's only possible to visualize one single level!
tried plotting with cartopy but somehow dimensions doesn't really fit... is this "grid-file" missing?
"All approaches have in common that the NetCDF
grid file must be read in together with the data file." (Icon tutorial 2023)

Verbesserungs ToDo's für die ich mir keine Zeit nehmen will:
- Tests! zmd Tests für die Einleseroutinen...

created additionalcalc/timeseries notebook for first draft of timeseries & 2d plots


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



