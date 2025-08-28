import sys
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import richdem as rd

# read data
infile = "D:\MSc_Arbeit\AROME\AROME_TEAMx_CAP_3D_fields\AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_p.nc"
ncds = nc.Dataset(infile)
hgt = ncds['h'][:].data.squeeze()
lonr = ncds['rlon'][:].data
latr = ncds['rlat'][:].data

# convert to WGS84
lon0 = -170
lat0 = 43
proj_rot = ccrs.RotatedPole(pole_longitude=lon0, pole_latitude=lat0)
proj_ll = ccrs.PlateCarree()

lonr2d, latr2d = np.meshgrid(lonr, latr)
lonlat = proj_ll.transform_points(proj_rot, lonr2d, latr2d)
lon2d = lonlat[:, :, 0]
lat2d = lonlat[:, :, 1]

# calculate slope angle
hgtrd = rd.rdarray(hgt, no_data=-9999)
slope = rd.TerrainAttribute(hgtrd, attrib='slope_degrees', zscale=1/(111120*0.01))
aspect = rd.TerrainAttribute(hgtrd, attrib='aspect', zscale=1/(111120*0.01))
aspect = np.mod(360 - aspect + 180, 360)

# plot terrain height, slope, and aspect
plt.ion()
plt.figure()
ax = plt.axes(projection=proj_rot)
im = ax.contourf(lonr, latr, hgt)
ax.gridlines(draw_labels=True)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([-0.2, 0.5])
plt.colorbar(im)
plt.show()

plt.figure()
ax = plt.axes(projection=proj_rot)
im = ax.contourf(lonr, latr, slope)
ax.gridlines(draw_labels=True)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([-0.2, 0.5])
plt.colorbar(im)
plt.show()

plt.figure()
ax = plt.axes(projection=proj_rot)
im = ax.contourf(lonr, latr, aspect)
ax.gridlines(draw_labels=True)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([-0.2, 0.5])
plt.colorbar(im)
plt.show()

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
