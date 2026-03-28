""" compare regridded dataset to original UM output

Manuela Lehner
August 2025
"""

import os

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from regrid_um import get_lonlat
from ipdb import set_trace as bp


# original and regridded files
ncfile0 = '/Users/manu/SeaDrive/My Libraries/My Library/div/gratzl_regrid/um_regrid/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_v.nc'
ncfile1 = '/Users/manu/SeaDrive/My Libraries/My Library/div/gratzl_regrid/um_regrid/regridded/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_v_regrid.nc'

# plots to check output
plotvars = ['transformed_y_wind']
# surface_altitude, (land_binary_mask), air_pressure, specific_humidity, air_potential_temperature
# geopotential_height, upward_air_velocity, transformed_x_wind, transformed_y_wind
plotdir = '/Users/manu/SeaDrive/My Libraries/My Library/div/gratzl_regrid/um_regrid/test_plots'


# ----- TEST FUNCTIONS ------
def test_attrs(nc0, nc1):
    ''' check if attributes are identical for both datasets '''

    attrs0 = nc0.ncattrs()
    attrs1 = nc1.ncattrs()
    assert len(attrs0) == len(attrs1)

    for attr in attrs0:
        assert getattr(nc0, attr) == getattr(nc1, attr)


def test_dims_glob(nc0, nc1):
    ''' check if dataset dimensions are set correctly '''

    dims0 = nc0.dimensions
    dims1 = nc1.dimensions

    # except for grid_latitude and grid_longitude dimensions must be identical for both datasets
    for dim in dims1:
        if dim not in ['grid_latitude', 'grid_longitude']:
            np.testing.assert_array_equal(dims1[dim].size, dims0[dim].size)


def test_dims_var(nc0, nc1):
    ''' check if variable dimensions are set correctly '''

    dims0 = nc0.dimensions
    dims1 = nc1.dimensions

    for dim in dims0:
        assert dim in dims1


def test_vars(nc0, nc1):
    ''' check if all variables are in the new dataset '''

    vars0 = nc0.variables
    vars1 = nc1.variables
    assert len(vars0) == len(vars1)

    for var in vars0:
        assert var in vars1


def test_shape(nc0, nc1):
    ''' check if the shape of the variables is consistent '''

    dims0 = nc0.shape
    dims1 = nc1.shape
    if ('grid_latitude' in nc1.dimensions) or ('grid_longitude' in nc1.dimensions):
        np.testing.assert_array_equal(dims0[:-2], dims1[:-2])
    else:
        np.testing.assert_array_equal(dims0, dims1)



# ----- PLOT FUNCTIONS -----
def test_plot(nc0, nc1, var):
    ''' plot horizontal cross section '''

    # open figure and axes
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.0, 4.0))
    axs[0].set_position([0.07, 0.12, 0.38, 0.8])
    axs[1].set_position([0.51, 0.12, 0.38, 0.8])
    axs[0].set_title(f'{var}: orig')
    axs[1].set_title(f'{var}: regridded')
    axs[0].set_xlabel('lon')
    axs[1].set_xlabel('lon')
    axs[0].set_ylabel('lat')

    # lon/lat limits
    lonmin = np.min(nc1['grid_longitude'])
    lonmax = np.max(nc1['grid_longitude'])
    latmin = np.min(nc1['grid_latitude'])
    latmax = np.max(nc1['grid_latitude'])
    axs[0].set_xlim([lonmin, lonmax])
    axs[0].set_ylim([latmin, latmax])
    axs[1].set_xlim([lonmin, lonmax])
    axs[1].set_ylim([latmin, latmax])

    # rotate orig lon/lat values
    lon0, lat0 = get_lonlat(nc0)

    # 2D fields
    if len(nc0[var].shape) == 2:
        vmin = np.min(nc0[var])
        vmax = np.max(nc0[var])
        co = axs[0].contourf(lon0, lat0, nc0[var], vmin=vmin, vmax=vmax)
        axs[1].contourf(nc1['grid_longitude'][:], nc1['grid_latitude'][:], 
                        nc1[var], vmin=vmin, vmax=vmax, levels=co.levels)
    # 3D fields: plot lowest model level
    else:
        vmin = np.min(nc0[var][0,0,...])
        vmax = np.max(nc0[var][0,0,...])
        co = axs[0].contourf(lon0, lat0, np.squeeze(nc0[var][0,0,...]), vmin=vmin, vmax=vmax)
        axs[1].contourf(nc1['grid_longitude'][:], nc1['grid_latitude'][:],
                        np.squeeze(nc1[var][0,0,...]), vmin=vmin, vmax=vmax, levels=co.levels)

    # colorbar
    cbar = fig.colorbar(co)
    cbar.ax.set_ylabel(nc0[var].units)
    cbar.ax.set_position([0.91, 0.12, 0.01, 0.8])
    axs[0].set_position([0.07, 0.12, 0.38, 0.8])

    # save figure
    figpath = os.path.join(plotdir, f'test_{var}.png')
    fig.savefig(figpath)



# ----- MAIN FUNCTION ------
def test_um_regrid():
    ''' main function to call individual tests '''

    # open both datasets
    nc0 = nc.Dataset(ncfile0, 'r')
    nc1 = nc.Dataset(ncfile1, 'r')

    # check if global attributes are set correctly
    test_attrs(nc0, nc1)

    # check if dimensions are set correctly
    test_dims_glob(nc0, nc1)

    # check if all variables are in the dataset
    test_vars(nc0, nc1)

    # loop over all variables and check if attributes, dims, and shape are correct
    vars1 = nc1.variables
    for var in vars1:
        test_attrs(nc0[var], nc1[var])
        test_dims_var(nc0[var], nc1[var])
        test_shape(nc0[var], nc1[var])
    
    # plot 
    for var in plotvars:
        test_plot(nc0, nc1, var)

    # close both datasets
    nc0.close()
    nc1.close()

if __name__ == '__main__':
    test_um_regrid()
