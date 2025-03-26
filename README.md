# Why are NWMs too warm in a cold air pool? A
model intercomparison study
=> note changes in this file!

cosma just adapted, often hard coded codes from Hannes. "zeitreihen"-notebooks von cosma meist sinnlos.

Meeting m Hannes:
1. stimmt es, dass nicht alle Daten auf dem Gitlab-repo raufgeladen 
wurden? Also die Modelldaten sowieso nicht, aber die exportierte 
ZAMG-Stations-Datei finde ich zB nicht. (Die hat Cosma iwie auch nicht 
benutzt, sondern nur die HOBOs soweit ich das bis jetzt gesehen hab.)

2. check ich das einlesen des AROME-Modells noch nicht ganz. Das 
Skript: "change_coords_of_arome_nc.py" beinhaltet ja Funktionen, die 
nur 1x verwendet werden um die Dateigröße zu verringern und die 
Dimensionen des Datasets zu ändern:
"""This script was used to convert the `variables` lat lon time (nz if 
3D) to `coordinates`
Initially the coords were the indexes record, X,Y (level) first, so we 
reduced the filesize by a lot"""
Aber damit hab ich ein paar Probleme weil ich 1. die veränderten Daten 
nicht habe? Also die Funktion "read_timeSeries_AROME(location)" kann 
ich nicht ausführen, weil mir die Daten dazu fehlen.
Dazu habe ich die 3 .tar.gz - komprimierte Dateien, eine davon mit 
über 20GB?

ICON fkt eig, Idee wie es schneller wäre? 

3. UM: Why pandas df?!? -> fixed point fkt jetzt als dataset
-> einlesen von ganzen Daten mit dask auf einmal möglich (~40GB) aber berechnen von lat, lon & temp in °C schwierig!
-> transformieren von ges. daten in lat, lon möglich? -> dzt nur für 1 pkt
-> für 2d plots mehrere gitterpunkte einlesen: zB mit fixed point fkt, dann halt jeden gitterpkt einzeln



ToDo till next meeting:
- make overview topo plot of domain (ICON) => domain of model
- f.e. also temp timeseries 2d 
- revise other code: Hannes Einleseroutinen vereinheitlichen... für UM pandas, meist xarray => vereinheitlichen tlw wird nur einzelner timestamp, oder einzelner gitterpkt eingelesen... -> mit hannes absprechen!
- hannes codes durchgeschaut: fkt nicht wirklich, iwie Problem mit Arome-Dimensionsänderung?
- codes in normale .py fktionen abändern? ICON "erledigt" => nicht alle Daten (variablen) einlesen?
- concept work 2nd research question, time onset of cap
- make first plots: make timeseries of surroundings of ibk? first 2d, timeseries plot?

created additionalcalc/timeseries notebook for first draft of timeseries & 2d plots
changed read in fct of icon to make call of multiple hours easier 


heights of cosma: 
87: (icon, lowest lvl where?)
80: (icon2te)
56: hafelekar

What i did:
vertical plot of icon with time, probably wrong vert. coord (rather use z_ifc = geopot. height) (timeseries notebook)



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



