""" regrid WRF data to a regular lat/lon grid

Manuela Lehner
August 2025
"""

import os
import glob
import argparse
import netCDF4 as nc
import numpy as np
from scipy.interpolate import LinearNDInterpolator
# from ipdb import set_trace as bp

# lat/lon of the new regular grid
lat = np.array([46.501804, 46.511684, 46.521564, 46.531444, 46.541324, 46.551204,
                46.561084, 46.570964, 46.580844, 46.590724, 46.600604, 46.610484,
                46.620364, 46.630244, 46.640124, 46.650004, 46.659884, 46.669764,
                46.679644, 46.689524, 46.699404, 46.709284, 46.719164, 46.729044,
                46.738924, 46.748804, 46.758684, 46.768564, 46.778444, 46.788324,
                46.798204, 46.808084, 46.817964, 46.827844, 46.837724, 46.847604,
                46.857484, 46.867364, 46.877244, 46.887124, 46.897004, 46.906884,
                46.916764, 46.926644, 46.936524, 46.946404, 46.956284, 46.966164,
                46.976044, 46.985924, 46.995804, 47.005684, 47.015564, 47.025444,
                47.035324, 47.045204, 47.055084, 47.064964, 47.074844, 47.084724,
                47.094604, 47.104484, 47.114364, 47.124244, 47.134124, 47.144004,
                47.153884, 47.163764, 47.173644, 47.183524, 47.193404, 47.203284,
                47.213164, 47.223044, 47.232924, 47.242804, 47.252684, 47.262564,
                47.272444, 47.282324, 47.292204, 47.302084, 47.311964, 47.321844,
                47.331724, 47.341604, 47.351484, 47.361364, 47.371244, 47.381124,
                47.391004, 47.400884, 47.410764, 47.420644, 47.430524, 47.440404,
                47.450284, 47.460164, 47.470044, 47.479924, 47.489804, 47.499684,
                47.509564, 47.519444, 47.529324, 47.539204, 47.549084, 47.558964,
                47.568844, 47.578724, 47.588604, 47.598484, 47.608364, 47.618244,
                47.628124, 47.638004, 47.647884, 47.657764, 47.667644, 47.677524,
                47.687404, 47.697284, 47.707164, 47.717044, 47.726924, 47.736804,
                47.746684, 47.756564, 47.766444, 47.776324, 47.786204, 47.796084,
                47.805964, 47.815844, 47.825724, 47.835604, 47.845484, 47.855364,
                47.865244, 47.875124, 47.885004, 47.894884, 47.904764, 47.914644,
                47.924524, 47.934404, 47.944284, 47.954164, 47.964044, 47.973924,
                47.983804, 47.993684, 48.003564, 48.013444, 48.023324, 48.033204,
                48.043084, 48.052964, 48.062844, 48.072724, 48.082604, 48.092484,
                48.102364, 48.112244, 48.122124, 48.132004, 48.141884, 48.151764,
                48.161644, 48.171524, 48.181404, 48.191284])
lon = np.array([ 9.2112581,  9.2252381,  9.2392181,  9.2531981,  9.2671781,
                 9.2811581,  9.2951381,  9.3091181,  9.3230981,  9.3370781,
                 9.3510581,  9.3650381,  9.3790181,  9.3929981,  9.4069781,
                 9.4209581,  9.4349381,  9.4489181,  9.4628981,  9.4768781,
                 9.4908581,  9.5048381,  9.5188181,  9.5327981,  9.5467781,
                 9.5607581,  9.5747381,  9.5887181,  9.6026981,  9.6166781,
                 9.6306581,  9.6446381,  9.6586181,  9.6725981,  9.6865781,
                 9.7005581,  9.7145381,  9.7285181,  9.7424981,  9.7564781,
                 9.7704581,  9.7844381,  9.7984181,  9.8123981,  9.8263781,
                 9.8403581,  9.8543381,  9.8683181,  9.8822981,  9.8962781,
                 9.9102581,  9.9242381,  9.9382181,  9.9521981,  9.9661781,
                 9.9801581,  9.9941381, 10.0081181, 10.0220981, 10.0360781,
                10.0500581, 10.0640381, 10.0780181, 10.0919981, 10.1059781,
                10.1199581, 10.1339381, 10.1479181, 10.1618981, 10.1758781,
                10.1898581, 10.2038381, 10.2178181, 10.2317981, 10.2457781,
                10.2597581, 10.2737381, 10.2877181, 10.3016981, 10.3156781,
                10.3296581, 10.3436381, 10.3576181, 10.3715981, 10.3855781,
                10.3995581, 10.4135381, 10.4275181, 10.4414981, 10.4554781,
                10.4694581, 10.4834381, 10.4974181, 10.5113981, 10.5253781,
                10.5393581, 10.5533381, 10.5673181, 10.5812981, 10.5952781,
                10.6092581, 10.6232381, 10.6372181, 10.6511981, 10.6651781,
                10.6791581, 10.6931381, 10.7071181, 10.7210981, 10.7350781,
                10.7490581, 10.7630381, 10.7770181, 10.7909981, 10.8049781,
                10.8189581, 10.8329381, 10.8469181, 10.8608981, 10.8748781,
                10.8888581, 10.9028381, 10.9168181, 10.9307981, 10.9447781,
                10.9587581, 10.9727381, 10.9867181, 11.0006981, 11.0146781,
                11.0286581, 11.0426381, 11.0566181, 11.0705981, 11.0845781,
                11.0985581, 11.1125381, 11.1265181, 11.1404981, 11.1544781,
                11.1684581, 11.1824381, 11.1964181, 11.2103981, 11.2243781,
                11.2383581, 11.2523381, 11.2663181, 11.2802981, 11.2942781,
                11.3082581, 11.3222381, 11.3362181, 11.3501981, 11.3641781,
                11.3781581, 11.3921381, 11.4061181, 11.4200981, 11.4340781,
                11.4480581, 11.4620381, 11.4760181, 11.4899981, 11.5039781,
                11.5179581, 11.5319381, 11.5459181, 11.5598981, 11.5738781,
                11.5878581, 11.6018381, 11.6158181, 11.6297981, 11.6437781,
                11.6577581, 11.6717381, 11.6857181, 11.6996981, 11.7136781,
                11.7276581, 11.7416381, 11.7556181, 11.7695981, 11.7835781,
                11.7975581, 11.8115381, 11.8255181, 11.8394981, 11.8534781,
                11.8674581, 11.8814381, 11.8954181, 11.9093981, 11.9233781,
                11.9373581, 11.9513381, 11.9653181, 11.9792981, 11.9932781,
                12.0072581, 12.0212381, 12.0352181, 12.0491981, 12.0631781,
                12.0771581, 12.0911381, 12.1051181, 12.1190981, 12.1330781,
                12.1470581, 12.1610381, 12.1750181, 12.1889981, 12.2029781,
                12.2169581, 12.2309381, 12.2449181, 12.2588981, 12.2728781,
                12.2868581, 12.3008381, 12.3148181, 12.3287981, 12.3427781,
                12.3567581, 12.3707381, 12.3847181, 12.3986981, 12.4126781,
                12.4266581, 12.4406381, 12.4546181, 12.4685981, 12.4825781,
                12.4965581, 12.5105381, 12.5245181, 12.5384981, 12.5524781,
                12.5664581, 12.5804381, 12.5944181, 12.6083981, 12.6223781,
                12.6363581, 12.6503381, 12.6643181, 12.6782981, 12.6922781,
                12.7062581, 12.7202381, 12.7342181, 12.7481981, 12.7621781,
                12.7761581, 12.7901381, 12.8041181, 12.8180981, 12.8320781,
                12.8460581, 12.8600381, 12.8740181, 12.8879981, 12.9019781,
                12.9159581, 12.9299381, 12.9439181, 12.9578981, 12.9718781,
                12.9858581, 12.9998381])


def add_attrs(nc1, nc0, ncvar=None):
    ''' copy dimensions from the old file
    Parameters:
    -----------
    nc1: nc dataset
        dataset to which dimensions are copied
    nc0: nc dataset
        dataset from which dimensions are copied
    ncvar: str or None
        if present, copy variable attributes
        if None, copy global attributes
    '''

    if ncvar:
        for ncatt in nc0[ncvar].ncattrs():
            # regridded dataset is not staggered in the horizontal anymore
            if ncatt == 'stagger':
                if getattr(nc0[ncvar], 'stagger') in ['X', 'Y']:
                    setattr(nc1[ncvar], ncatt, '')
                else:
                    setattr(nc1[ncvar], ncatt, getattr(nc0[ncvar], ncatt))
            elif ncatt == 'coordinates':
                attr = getattr(nc0[ncvar], 'coordinates')
                attr.replace('XLONG_U', 'XLONG')
                attr.replace('XLONG_V', 'XLONG')
                attr.replace('XLAT_U', 'XLAT')
                attr.replace('XLAT_V', 'XLAT')
                setattr(nc1[ncvar], ncatt, attr)
            # fill value already set when creating the variable
            elif ncatt == '_FillValue':
                continue
            else:
                setattr(nc1[ncvar], ncatt, getattr(nc0[ncvar], ncatt))
    else:
        ncatts = nc0.ncattrs()
        for ncatt in ncatts:
            setattr(nc1, ncatt, getattr(nc0, ncatt))

    return


def add_dims(nc1, nc0, nlon1, nlat1):
    ''' copy time and vertical dimensions from the old file
        and add new horizontal dimensions 

    Parameters:
    -----------
    nc1: nc dataset
        dataset to which dimensions are copied
    nc0: nc dataset
        dataset from which dimensions are copied
    nlon1: integer
        number of longitude points in the new grid
    nlat1: integer
        number of latitude points in the new grid
    '''

    ncdims = nc0.dimensions
    for ncdim in ncdims:
        # skip horizontal dimensions
        if ncdim.startswith('south_north') or ncdim.startswith('west_east'):
            continue

        # keep information for time and vertical dimensions (including 
        # information on unlimited dimensions
        if ncdims[ncdim].isunlimited():
            nc1.createDimension(ncdim, None)
        else:
            nc1.createDimension(ncdim, ncdims[ncdim].size)

    # add new horizontal dimensions
    nc1.createDimension('south_north', nlat1)
    nc1.createDimension('west_east', nlon1)

    return


def add_data(nc1, nc0, lon1, lat1):
    ''' regrid data and add regridded data with variable attributes to dataset

    Parameters:
    -----------
    nc1: nc dataset
        new dataset with regridded data
    nc0: nc dataset
        original dataset
    lon1: 1D numpy array
        longitude values for the new regular grid
    lat1: 1D numpy array
        latitude values for the new regular grid
    '''

    # get lon/lat values of dataset
    lon, lat, lonu, latu, lonv, latv = get_lonlat(nc0)

    # loop over all variables in nc0
    for ncvar in nc0.variables:

        # get variable dimensions and replace horiz staggered dim names
        lon0, lat0 = lon, lat
        dims = list(nc0[ncvar].dimensions)
        if 'south_north_stag' in dims:
            dind = dims.index('south_north_stag')
            dims[dind] = 'south_north'
            lon0, lat0 = lonv, latv
        if 'west_east_stag' in dims:
            dind = dims.index('west_east_stag')
            dims[dind] = 'west_east'
            lon0, lat0 = lonu, latu

        # create variable and add attributes
        if '_FillValue' in nc0[ncvar].ncattrs():
            temp = nc1.createVariable(ncvar, nc0[ncvar].dtype, dims, 
                                      fill_value=getattr(nc0[ncvar], '_FillValue'))
        else:
            temp = nc1.createVariable(ncvar, nc0[ncvar].dtype, dims)
        add_attrs(nc1, nc0, ncvar=ncvar)

        # get data, regrid if necessary, and add to new dataset
        data = nc0[ncvar][:]
        if 'west_east' in dims:
            temp[:] = regrid(data, lon0, lat0, lon1, lat1)
        else:
            temp[:] = data

    return



def regrid(data0, lon0, lat0, lon1, lat1):
    ''' interpolate WRF data to new regular dataset

    Parameters:
    -----------
    data0: numpy array
         variable to be regridded
    lon0: 2D numpy array
         WRF longitude
    lat0: 2D numpy array
         WRF latitude
    lon1: 1D numpy array
         longitude for the new regular grid
    lat1: 1D numpy array
         latitude for the new regular grid

    Returns:
    --------
    data1: numpy array
         regridded variable
    '''

    # LinearNDInterpolator works with 1D array of points
    temp = np.reshape(data0, list(data0.shape[:-2]) + [-1])
    temp = np.transpose(temp, [-1]+list(range(0,temp.ndim-1)))
    interp = LinearNDInterpolator((lon0.flatten(), lat0.flatten()), temp)

    # interpolated data array needs be rearranged to the WRF order of dims
    data1 = interp(np.meshgrid(lon1, lat1))
    data1 = np.transpose(data1, list(range(2,data1.ndim))+[0,1])

    return data1


def get_lonlat(nc0):
    ''' get staggered lon/lat values from unstaggered coordinates

    Parameters:
    -----------
    nc0: nc dataset
        original WRF dataset with only unstaggered lon/lat

    Returns:
    --------
    lon: 2D numpy array
        longitude of unstaggered grid
    lat: 2D numpy array
        latitude of unstaggered grid
    lonu: 2D numpy array
        longitude of staggered (u points) grid
    latu: 2D numpy array
        latitude of staggered (u points) grid
    lonv: 2D numpy array
        longitude of staggered (v points) grid
    latv: 2D numpy array
        latitude of staggered (v points) grid
    '''

    # unstaggered grid
    lon = nc0['lon'][:].squeeze()
    lat = nc0['lat'][:].squeeze()

    # x staggered
    lonu = 0.5 * (lon[:,:-1] + lon[:,1:])
    lonuw = lonu[:,0] - (lon[:,1] - lon[:,0])
    lonue = lonu[:,-1] + (lon[:,-1] - lon[:,-2])
    lonu = np.hstack([lonuw[:,np.newaxis], lonu, lonue[:,np.newaxis]])

    latu = 0.5 * (lat[:,:-1] + lat[:,1:])
    latuw = latu[:,0] - (lat[:,1] - lat[:,0])
    latue = latu[:,-1] + (lat[:,-1] - lat[:,-2])
    latu = np.hstack([latuw[:,np.newaxis], latu, latue[:,np.newaxis]])

    # y staggered
    lonv = 0.5 * (lon[:-1,:] + lon[1:,:])
    lonvs = lonv[0,:] - (lon[1,:] - lon[0,:])
    lonvn = lonv[-1,:] + (lon[-1,:] - lon[-2,:])
    lonv = np.vstack([lonvn[np.newaxis,:], lonv, lonvs[np.newaxis,:]])

    latv = 0.5 * (lat[:-1,:] + lat[1:,:])
    latvs = latv[0,:] - (lat[1,:] - lat[0,:])
    latvn = latv[-1,:] + (lat[-1,:] - lat[-2,:])
    latv = np.vstack([latvn[np.newaxis,:], latv, latvs[np.newaxis,:]])

    return lon, lat, lonu, latu, lonv, latv


def create_wrf_regrid(nc0, lon1, lat1, wrffile):
    ''' create regridded dataset

    Parameters:
    -----------
    nc0: nc dataset
        original WRF dataset
    lon1: 1D numpy array
        longitude for new regular grid
    lat1: 1D numpy array
        latitude for new regular grid
    wrffile: string
        filename of original WRF dataset
    '''

    # create new nc file
    ncfile = f"""{os.path.join(os.path.dirname(wrffile), "regridded", 
                os.path.splitext(os.path.basename(wrffile))[0])}_regrid.nc"""
    nc1 = nc.Dataset(ncfile, 'w', format='NETCDF4')

    # copy global attributes from the original file
    add_attrs(nc1, nc0)

    # add dimensions for new grid
    add_dims(nc1, nc0, len(lon1), len(lat1))

    # add (regridded) data
    add_data(nc1, nc0, lon1, lat1)

    # close new nc file
    nc1.close()

    return


# ----- MAIN FUNCTION -----
def regrid_wrf(wrfdir):
    ''' loop through all nc files, regrid, and create new data file

    Parameters:
    -----------
    wrfdir: string
        directory containing WRF files to be regridded
    '''

    # get lon/lat for new regular grid
    lon1 = lon
    lat1 = lat

    # check if directory exists
    if not os.path.isdir(wrfdir):
        raise ValueError(f'{wrfdir} does not exist')
    
    # list of WRF nc files
    ncfiles = glob.glob(os.path.join(wrfdir, '*.nc'))

    # loop through all nc files
    for ncfile in ncfiles:

        # open original nc file for reading
        nc0 = nc.Dataset(ncfile, 'r')

        # create regridded dataset
        nc1 = create_wrf_regrid(nc0, lon1, lat1, ncfile)

        # close original nc file
        nc0.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
             'Program to regrid WRF data to a regular lat/lon grid')
    parser.add_argument('wrfdir', type=str, help='WRF directiory')
    args = parser.parse_args()
    regrid_wrf(args.wrfdir)




    """
def my_old_pfusch:
    
    # shouldn't it normally work like that? according to https://pyproj4.github.io/pyproj/stable/examples.html ...
    # but maybe some info is missing: Description: Inverse of unknown + unknown??
    crs_orig = pyproj.CRS.from_proj4(wrf.attrs["pyproj_srs"])  # get CRS info from original WRF dataset
    crs_proj = pyproj.CRS.from_epsg(4326)
    proj = pyproj.Transformer.from_crs(crs_orig, crs_orig)

    # try setting projection directly with cartopy?
    wrf_proj_cartopy = ccrs.LambertConformal(central_longitude=wrf.attrs["STAND_LON"],
                                             central_latitude=wrf.attrs["MOAD_CEN_LAT"],
                                             false_easting=0, false_northing=0,
                                             standard_parallels=(wrf.attrs["TRUELAT1"], wrf.attrs["TRUELAT2"]),
                                             cutoff=0)

    # try to reproject wrf to WGS84, or regular lat lon grid (to have uniform projection for all models)
    # after https://fabienmaussion.info/2018/01/06/wrf-projection/ but some things are deprecated...
    wrf_proj = pyproj.Proj(proj="lcc",
                           lat_1=wrf.attrs["TRUELAT1"], lat_2=wrf.attrs["TRUELAT2"],
                           lat_0=wrf.attrs["MOAD_CEN_LAT"], lon_0=wrf.attrs["STAND_LON"],
                           a=6370000, b=6370000)
    wgs_proj = pyproj.Proj(proj="latlong", datum="WGS84")
    pyproj.Transformer.from_pipeline(wrf.attrs["pyproj_srs"])  # use projection string from wrf dataset for transforming
    # the projection...
    trans = pyproj.Transformer.from_crs("epsg:4326", wrf.attrs["pyproj_srs"])  # or does it work like that?

    # but it somehow still doesn't work - maybe there is some information missing for the transformation?


    transformer = Transformer.from_proj(wgs_proj, wrf_proj, always_xy=True)
    x, y = transformer.transform(wrf.CEN_LON, wrf.CEN_LAT)

    # Easting and Northings of the domains center point
    wgs_proj = Proj(proj='latlong', datum='WGS84')
    e, n = pyproj.transform(wgs_proj, wrf_proj, wrf.CEN_LON, wrf.CEN_LAT)
    # Grid parameters
    dx, dy = wrf.DX, wrf.DY
    nx, ny = wrf.dims['west_east'], wrf.dims['south_north']
    # Down left corner of the domain
    x0 = -(nx - 1) / 2. * dx + e
    y0 = -(ny - 1) / 2. * dy + n
    # 2d grid
    xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)

    our_lons, our_lats = pyproj.transform(wrf_proj, wgs_proj, xx, yy)
    wrf['DIFF'] = np.sqrt((our_lons - wrf.XLONG_M) ** 2 + (our_lats - wrf.XLAT_M) ** 2)
    wrf.salem.quick_map('DIFF', cmap='Reds');

    wrf.z.isel(height=0).salem.quick_map(cmap='topo')"""