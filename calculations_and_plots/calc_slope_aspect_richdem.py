"""
NOT USED!

written by Manuela Lehner for calculating slope and aspect for COSMO data, changed to work for my data
Problems... what's zscale?!
"""

import sys
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import richdem as rd

import confg
import read_in_arome
from plot_topography import calculate_km_for_lon_extent
mpl.use('Qt5Agg')

# read data
"""
original from manuela
infile = "/home/c707201/teaching/Student_supervising/BS_Students/Brueckner_BS/cosmo1_surfaceheight.nc"
ncds = nc.Dataset(infile)
hgt = ncds['h'][:].data.squeeze()
lonr = ncds['rlon'][:].data
latr = ncds['rlat'][:].data"""

ncds = read_in_arome.read_in_arome_fixed_time(16, 12, 0, variables=["z"])

ds = ncds.sel(height=1)
hgt = ds['z'].values.squeeze()
lonr = ds['lon'].values
latr = ds['lat'].values

proj_ll = ccrs.PlateCarree()
"""
# convert to WGS84
lon0 = -170
lat0 = 43
proj_rot = ccrs.RotatedPole(pole_longitude=lon0, pole_latitude=lat0)

lonr2d, latr2d = np.meshgrid(lonr, latr)
lonlat = proj_ll.transform_points(proj_rot, lonr2d, latr2d)
lon2d = lonlat[:, :, 0]
lat2d = lonlat[:, :, 1]
"""
# calculate slope angle
hgtrd = rd.rdarray(hgt, no_data=-9999)
# zscale (float) – How much to scale the z-axis by prior to calculation
# i.e. how much smaller is the distance in z between the levels than in x or y?
# original by Manuela: 1/(how much km is 1° * dist_x in deg)
# => I calculate it with searching the distance between points in lon [deg] and converting those to [m] with the function
# "calculate_km_for_lon_extent", divide that through 1 cause Manuela did that also -> hopefully right?
dist_x = calculate_km_for_lon_extent(confg.ibk_uni["lat"], (ncds.isel(lat=1, lon=2).lon - ncds.isel(lat=1, lon=1).lon)) * 1000
z_scale = 1 / dist_x

slope = rd.TerrainAttribute(hgtrd, attrib='slope_degrees', zscale=z_scale)
aspect = rd.TerrainAttribute(hgtrd, attrib='aspect', zscale=z_scale)
aspect = np.mod(360 - aspect + 180, 360)

# plot terrain height, slope, and aspect
# plt.ion()
plt.figure()
ax = plt.axes(projection=proj_ll)  # orig proj_rot
im = ax.contourf(lonr, latr, hgt, label="height")
ax.gridlines(draw_labels=True)
# ax.set_xlim([0.5, 1.5])
# ax.set_ylim([-0.2, 0.5])
plt.colorbar(im)
plt.show()

plt.figure()
ax = plt.axes(projection=proj_ll)
im = ax.contourf(lonr, latr, slope, label="slope")
ax.gridlines(draw_labels=True)
# ax.set_xlim([0.5, 1.5])
# ax.set_ylim([-0.2, 0.5])
plt.colorbar(im)
plt.show()

plt.figure()
ax = plt.axes(projection=proj_ll)
im = ax.contourf(lonr, latr, aspect, label="aspect")
ax.gridlines(draw_labels=True)
# ax.set_xlim([0.5, 1.5])
# ax.set_ylim([-0.2, 0.5])
plt.colorbar(im)
plt.show()

"""
# write netcdf file
outfile = infile[:-3] + 'rotated_slope_aspect.nc'
ncout = nc.Dataset(outfile, mode='w', format='NETCDF4')

ncout.desription = 'derived COSMO1 terrain data in subdomain'
ncout.author = 'Manuela Lehner (University of Innsbruck)'
ncout.date = '22 March 2023'

ncout.createDimension('lon', ncds.dimensions['rlon'].size)
ncout.createDimension('lat', ncds.dimensions['rlat'].size)

nclon = ncout.createVariable('lon', np.float64, ('lat','lon',)) 
nclon.long_name='longitude in unrotated grid'
nclon.units='degrees east'
nclon[:,:] = lon2d

nclat = ncout.createVariable('lat', np.float64, ('lat','lon')) 
nclat.long_name='latitude in unrotated grid'
nclat.units='degrees north'
nclat[:,:] = lat2d

nchgt = ncout.createVariable('hgt', np.float64, ('lat','lon')) 
nchgt.long_name='Terrein height'
nchgt.units='m MSL'
nchgt[:,:] = hgt

ncslp = ncout.createVariable('slope', np.float64, ('lat','lon')) 
ncslp.long_name='Terrain slope angle'
ncslp.units='deg'
ncslp[:,:] = np.array(slope)

ncasp = ncout.createVariable('aspect', np.float64, ('lat','lon')) 
ncasp.long_name='Terrain slope aspect'
ncasp.units='deg'
ncasp[:,:] = np.array(aspect)

ncout.close()
"""