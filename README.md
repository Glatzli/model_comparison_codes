# Why are NWMs too warm in a cold air pool? A
model intercomparison study
=> note changes in this file!

cosma just adapted, often hard coded codes from Hannes. "zeitreihen"-notebooks von cosma, alle sinnlos.

ToDo till next meeting:
- make overview topo plot of domain (ICON) => domain of model
- f.e. also temp timeseries 2d 
- revise other code: Hannes Einleseroutinen vereinheitlichen... für UM pandas, meist xarray => vereinheitlichen tlw wird nur einzelner timestamp, oder einzelner gitterpkt eingelesen...
- codes in normale .py fktionen abändern? ICON "erledigt" => nicht alle Daten (variablen) einlesen?
- concept work 2nd research question, time onset of cap
- make first plots: make timeseries of surroundings of ibk? first 2d, timeseries plot?

created additionalcalc/timeseries notebook for first draft of timeseries & 2d plots
changed read in fct of icon to make call of multiple hours easier 


heights of cosma: 
87 (icon, lowest lvl where?)
80 (icon2te)
56: hafelekar


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



