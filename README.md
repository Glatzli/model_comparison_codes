Code for MSc Thesis:
# Modeling a cold air pool in the Inn Valley: A model intercomparison study

## Program structure 
Each model & observation has a seperate file for reading in the data, cause every raw file looks a bit different. For the model data there are always 2 ways: 
1. either read a timeseries at a specific coordinate (first the data is extracted, manipulated and then saved as a .nc file for later use): used for timeseries plotting, vertical profiles etc
2. or read the full domain at a specific timestamp (much bigger data due to a lot of height levels): used for spatial plots of full domain as temp, VHD (valley heat deficit) etc


VHD calculation: (If the timeseries isn't already calculated for that point)
The models are read per default at the PCGP (physically consistent grid point) around that GP and the timeseries of the model is saved as .nc file. Then
the timeseries at that PCGP is opened, the VHD is calculated, and then saved. The functions for that are defined in
calc_vhd.py and are only called from plot_vhd.py
