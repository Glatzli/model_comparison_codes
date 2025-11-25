"""In this "confg-script" are all the data filepaths listed
You have to change it!
original by hannes, daniel adapted it a bit and added/edited some coords and definitions that are multiple times used
to avoid double definitions & confusions...

"""

import os

# -------------------------------------------To change --------------------------------------------------------
# Folder where the model output is saved:
model_folder = os.path.normpath("D:/MSc_Arbeit")
# Folder where the data is saved:
data_folder = os.path.join(model_folder, "data")
# Plot directory (where to save the plots)
dir_PLOTS = os.path.join(model_folder, "plots")
dir_topo_plots = os.path.join(dir_PLOTS, "topography_comparison")

# -------------------------------------------constants needed for calculations---------------------------------
hafelekar_height = 2279  # m, highest HOBO from https://zenodo.org/records/4672313 hobo dataset, used for VHD calc
c_p = 1005  # J/(kg*K), specific heat capacity of air at constant pressure, for VHD calc

# All point locations defined below (created for Daniel's thesis):
ALL_POINTS = {"ibk_villa": {"name": "ibk villa", "lat": 47.25971, "lon": 11.38420, "height": 579},
              # same lat & lon of Ibk cosma already used: which
              # is for Ibk_Villa (2m temp recording); changed point to coords of https://acinn-data.uibk.ac.at/pages/meteodat.html
              # now 4 m higher as Cosma's point (she had 575m ...)

              "ibk_uni": {"name": "ibk uni", "lat": 47.264, "lon": 11.385, "height": 612},
              # hatpro, uni coords rounded to 3 digits after comma
              "ibk_airport": {"name": "ibk airport", "lat": 47.26, "lon": 11.34, "height": 577},
              "hafelekar": {"name": "hafelekar", "lat": 47.312, "lon": 11.383, "height": hafelekar_height},
              # 2279m, with 3 digits
              "slope_north_patscherkofel": {"name": "slope north patscherkofel", "lat": 47.23, "lon": 11.5,
                                            "height": 1750},
              "woergl": {"name": "woergl", "lat": 47.494, "lon": 12.059, "height": 504},
              # coords for wörgl (504m), lower Inn valley
              "kiefersfelden": {"name": "kiefersfelden", "lat": 47.62, "lon": 12.2, "height": 480},
              # coords for kiefersfelden (480m), Germany, entrance Inn valley
              "telfs": {"name": "telfs", "lat": 47.3, "lon": 11.1, "height": 622},  # 622m
              # valley points in wipp & ziller valley for stability plots (where valleys are narrow):
              "wipp_valley": {"name": "wipp valley", "lat": 47.13, "lon": 11.45, "height": 1044},
              # between Schönberg & Matrei
              "ziller_valley": {"name": "ziller valley", "lat": 47.25, "lon": 11.9, "height": 565},
              # between Zell am Ziller & Zillertal
              "ziller_ried": {"name": "ziller ried", "lat": 47.3, "lon": 11.87, "height": 572}  # Zillertal, Kaltenbach
              }
POINT_NAMES = list(ALL_POINTS.keys())  # list w. all point names
# coordinates of points used for Daniels' Analysis; all points that should include HATPRO or Radiosonde data in
# the point plots need "ibk" in the beginning of point definition
# heights in m from https://www.freemaptools.com/elevation-finder.htm

# List of point keys for easy iteration - coordinates of points used for Daniels' Analysis
# Points that should include HATPRO plot (VHD) need "ibk" in name
# Heights in m from https://www.freemaptools.com/elevation-finder.htm
POINT_NAMES = list(ALL_POINTS.keys())

# Define point categories for easy filtering; hardcoded list to distinguish valley and mountain points
VALLEY_POINTS = ["ibk_villa", "ibk_uni", "ibk_airport", "woergl", "kiefersfelden", "telfs", "wipp_valley", "ziller_valley", "ziller_ried"]
MOUNTAIN_SLOPE_POINTS = ["hafelkar", "slope_north_patscherkofel"]

def get_valley_points_only():
    """Get only valley points (excludes mountains and slopes)"""
    return {key: value for key, value in ALL_POINTS.items() if key in VALLEY_POINTS}

def get_points_excluding_mountains():
    """Get all points except mountain/slope points"""
    return {key: value for key, value in ALL_POINTS.items() if key not in MOUNTAIN_SLOPE_POINTS}


lat_hf_min, lat_hf_max = 47, 47.6
lon_hf_min, lon_hf_max = 11.1, 12.1

lat_min_vhd, lat_max_vhd = 47, 47.7  # orig: 47, 47.7    # lat & lon values for vhd domain plotting
lon_min_vhd, lon_max_vhd = 10.8, 12  # 10.8, 12

lat_min_cap_height, lat_max_cap_height = 47, 48.2
lon_min_cap_height, lon_max_cap_height = 10.6, 13

lat_min, lat_max = 46.5, 48.2
lon_min, lon_max = 9.2, 13

# -------------------------------------------------------------------------------------------------------------
radiosonde_folder = os.path.join(data_folder, "Observations", "Radiosonde")
radiosonde_csv = os.path.join(radiosonde_folder, "2017101603_bufr309052.csv")  # radiosonden aufstieg at innsbruck airport
# deprecated: radiosonde_edited = os.path.join(radiosonde_folder, "radiosonde_ibk_2017101603.csv")  # calculated pot. temp & rho
# from other vars
radiosonde_dataset = os.path.join(radiosonde_folder, "radiosonde_ibk_2017101603.nc")  # for same handling for plots & calcs  save
# Radiosonde as dataset
# radiosonde_dataset_height_as_z = os.path.join(radiosonde_folder, "radiosonde_ibk_2017101603_height_as_z.nc")  # deprecated,
# use fct in réad_in_hatpro_radiosonde.py
# only geopot. height instead of "height values"
# deprecated? radiosonde_smoothed = os.path.join(radiosonde_folder, "radiosonde_ibk_smoothed.nc")

all_model_topographies = os.path.join(model_folder, "AROME", "all_model_topographies.nc")  # all topography-values extracted (lowest
# lvl) of geopot. height + "hgt" - vars for AROME & WRF and put into one file

JSON_TIROL = os.path.join(data_folder, "Height", "gadm41_AUT_1.json")  # tirol json file
DEMFILE_CLIP = os.path.join(data_folder, "Height", "dem_clipped.tif")  # dem file (höhe)
TIROL_DEMFILE = os.path.join(data_folder, "Height", "dem.tif")  # changed dem file: indexed and renamed coords
dem_smoothed = os.path.join(data_folder, "Height", "dem_smoothed.tif")
filepath_arome_height = os.path.join(model_folder, "AROME", "AROME_TEAMx_CAP_3D_fields",
                                     "AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_z.nc")
dem_file_hobos_extent = os.path.join(data_folder, "Height", "dem_cut_hobos.tif")  # created DEM (in model_topography) to see real
# heights with HOBOS

# ZAMG Datahub files
kufstein_zamg = os.path.join(data_folder, "Observations", "ZAMG_Tawes", "data_station9016-Kufstein_20171012_20171018.csv")
innsbruck_uni_zamg = os.path.join(data_folder, "Observations", "ZAMG_Tawes",
                                  "data_station11803-InnsbruckUniversity_20171012_20171018.csv")
innsbruck_airport_zamg = os.path.join(data_folder, "Observations", "ZAMG_Tawes",
                                      "data_station11804-InnsbruckAirport_20171012_20171018.csv")
jenbach_zamg = os.path.join(data_folder, "Observations", "ZAMG_Tawes", "data_station11901-Jenbach_20171012_20171018.csv")
rinn_zamg = os.path.join(data_folder, "Observations", "ZAMG_Tawes", "data_station11123-Rinn_20171015T1200_20171016T1210.csv")
munchen_zamg = os.path.join(data_folder, "Observations", "ZAMG_Tawes", "data_munich_T2m.csv")

# mobile stations, cut to our period
momma_our_period_file = os.path.join(data_folder, "Observations", "MOMMA", "MOMMA_our_period.nc")

# ----------------------------------Models-----------------------------------------------------

# absolute paths AROME
dir_AROME = os.path.join(model_folder, "AROME")
dir_2D_AROME = os.path.join(dir_AROME, "AROME_TEAMx_CAP_2D_fields")
dir_3D_AROME = os.path.join(dir_AROME, "AROME_TEAMx_CAP_3D_fields")
dir_timeseries_AROME = os.path.join(dir_AROME, "AROME_TEAMx_CAP_timeseries")

# absolute paths WRF
wrf_folder = os.path.join(model_folder, "WRF_ACINN")  # before: wrf_ACINN

# absolute paths ICON
icon_folder_3D = os.path.join(model_folder, "ICON")
icon_folder_meteogram = os.path.join(model_folder, "icon", "ICON_Meteogram")  # ?

# absolute paths ICON2TE
icon2TE_folder_3D = os.path.join(model_folder, "ICON2TE")

# absolute Path UKMO
ukmo_folder = os.path.join(model_folder, "ukmo")

# ------------------------------colormaps for plotting ---------------------------------------------
from colorspace import qualitative_hcl

# --- Color scheme for models (consistent with plot_cap_height) ---
qualitative_colors_temp = qualitative_hcl(palette="Dark 3").colors()
qualitative_colors_wind = qualitative_colors_temp
qualitative_colors_humidity = qualitative_hcl(palette="Dark 3").colors()

# Model color mapping for temperature - ICON and ICON2TE share the same color; take same color for LIDAR88 and HATPRO
model_colors_temp_wind = {"AROME": qualitative_colors_temp[0], "ICON": qualitative_colors_temp[2],
                          "ICON2TE": qualitative_colors_temp[2], "UM": qualitative_colors_temp[4],
                          "WRF": qualitative_colors_temp[6], "Radiosonde": qualitative_colors_temp[10],
                          "HATPRO": qualitative_colors_temp[8], "LIDAR88": qualitative_colors_temp[8],
                          "LIDAR142": qualitative_colors_temp[9]}

# Model color mapping for humidity
model_colors_humidity = {"AROME": qualitative_colors_humidity[0], "ICON": qualitative_colors_humidity[2],
                         "ICON2TE": qualitative_colors_humidity[2], "UM": qualitative_colors_humidity[4],
                         "WRF": qualitative_colors_humidity[6], "Radiosonde": qualitative_colors_humidity[10],
                         "HATPRO": qualitative_colors_humidity[8]}

# define linestyle for ICON2TE
icon_2te_hatpro_linestyle = "dot"

# -------------------------------Data and Plot paths -----------------------------------------------

# EC stations
dir_EC_stations = f"{data_folder}/Observations/EC_4_stations"
EC_30min_final = f"{dir_EC_stations}/EC_30min_file.nc"
EC_1min_final = f"{dir_EC_stations}/EC_1min_file.nc"

# Ibox dir
ibox_folder = f"{data_folder}/Observations/Ibox"

# HOBOS station
hobos_file = f"{data_folder}/Observations/HOBOS/hobos_final.nc"  # Observations/HOBOS/

# Lidar obs
lidar_obs_folder = os.path.join(data_folder, "Observations", "Lidar_obs")
lidar_sl88 = os.path.join(lidar_obs_folder, "SL88_vad_l2")  # add trailing separator for consistency
lidar_slxr142 = os.path.join(lidar_obs_folder, "SLXR142_vad_l2")
# merged files, subsetted to period and 30 min intervals are saved in:
lidar_sl88_merged_path = os.path.join(lidar_sl88, 'sl88_merged.nc')
lidar_slxr142_merged_path = os.path.join(lidar_slxr142, 'slxr142_merged.nc')

# HATPRO obs
hatpro_folder = os.path.join(data_folder, "Observations", "HATPRO_obs")  # nicht vorhanden?
hatpro_merged = os.path.join(hatpro_folder, "hatpro_merged.nc")  #  + "hatpro_merged.nc"
hatpro_smoothed = os.path.join(hatpro_folder, "hatpro_smoothed.nc")
hatpro_calced_vars = os.path.join(hatpro_folder, "hatpro_calced_vars_from_arome_p_height_as_z.nc")
hatpro_with_cap_height = os.path.join(hatpro_folder, "hatpro_interpolated_arome_height_as_z_with_cap_height.nc")

# Radiosonde CAP height is 1537 m (searched by hand) - ibk_airport["height"] = 960 m
radiosonde_cap_height = 1537 - ALL_POINTS["ibk_airport"]["height"]
# deprecated ones?
# hatpro_interp_arome = hatpro_folder + "hatpro_interpolated_arome.nc"
# hatpro_interp_arome_height_as_z = hatpro_folder + "hatpro_interpolated_arome_height_as_z.nc"


# Define colors for the models to use the same in each plot:
colordict = {"HOBOS": "purple", "ICON": "orange", "RADIOSONDE": "black", "AROME": "red", "HATPRO": "gray",
             "UKMO": "green", "WRF_ACINN": "blue"}

# Create a dictionary with information of the TAWES stations
station_files_zamg = {
    "IAO": {"filepath": innsbruck_uni_zamg, "name": "Innsbruck Uni", 'lon': 11.384167, 'lat': 47.259998,
            'hoehe': 578, },
    "JEN": {"filepath": jenbach_zamg, "name": "Jenbach", 'lat': 47.388889, 'lon': 11.758056, 'hoehe': 530, },
    "KUF": {"filepath": kufstein_zamg, "name": "Kufstein", 'lon': 12.162778, 'lat': 47.575279, 'hoehe': 490, },
    "LOWI": {"filepath": innsbruck_airport_zamg, "name": "Innsbruck Airport", 'lat': 47.2598, 'lon': 11.3553,
             'hoehe': 578, }, "IMST": {  # "filepath": imst_zamg,
        "name": "Imst", 'lat': 47.2419, 'lon': 10.7218, 'hoehe': 828, }}

# create a dict with info about the IBOX stations
stations_ibox = {
    "VF0": {"filepath": f"{ibox_folder}/vf0.csv", "name": "Kolsass", "latitude": 47.305, "longitude": 11.622,
            "height": 545},
    "SF8": {"filepath": f"{ibox_folder}/sf8.csv", "name": "Terfens", "latitude": 47.326, "longitude": 11.652,
            "height": 575},
    "SF1": {"filepath": f"{ibox_folder}/sf1.csv", "name": "Eggen", "latitude": 47.317, "longitude": 11.616,
            "height": 829},
    "NF10": {"filepath": f"{ibox_folder}/nf10.csv", "name": "Weerberg", "latitude": 47.300, "longitude": 11.673,
             "height": 930},
    "NF27": {"filepath": f"{ibox_folder}/nf27.csv", "name": "Hochhaeuser", "latitude": 47.288, "longitude": 11.631,
             "height": 1009}}

# dict with infos about EC stations
ec_station_names = {1: {"name": "Patsch_EC_South", "lat": 47.209068, "lon": 11.411932},
                    0: {"name": "Innsbruck_Airport_EC_West", "lat": 47.255375, "lon": 11.342832},
                    2: {"name": "Thaur_EC_East", "lat": 47.281335, "lon": 11.474532},
                    3: {"name": "IAO_Centre_Innsbruck_EC_Center", "lat": 47.264035, "lon": 11.385707}}

# variables, units 2D AROME
variables_units_2D_AROME = {'hfs': 'W/m²',  # Sensible heat flux at the surface
                            'hgt': 'm',  # Surface geopotential height
                            'lfs': 'W/m²',  # Latent heat flux at the surface
                            'lwd': 'W/m²',  # Longwave incoming radiation at the surface
                            'lwnet': 'W/m²',  # Longwave net radiation at the surface
                            'lwu': 'W/m²',  # Longwave outgoing radiation at the surface (derived: lwnet - lwd)
                            'pre': 'kg/m²',  # Surface precipitation (same as mm)
                            'ps': 'Pa',  # Surface pressure
                            'swd': 'W/m²',  # Shortwave incoming radiation at the surface
                            'swnet': 'W/m²',  # Shortwave net radiation at the surface
                            'swu': 'W/m²',  # Shortwave reflected radiation at the surface (derived: swnet - swd)
                            'tsk': 'K'  # Surface temperature (Oberflächentemperatur)
                            }

# variables, units 3D AROME
variables_units_3D_AROME = {'ciwc': 'kg/kg',  # Specific cloud ice water content
                            'clwc': 'kg/kg',  # Specific cloud liquid water content
                            'p': 'Pa',  # Pressure
                            'q': 'kg/kg',  # Specific humidity
                            'th': 'K',  # Potential temperature
                            'tke': 'm²/s²',  # Turbulent kinetic energy
                            'u': 'm/s',  # Zonal wind component
                            'v': 'm/s',  # Meridional wind component
                            'w': 'm/s',  # Vertical wind velocity
                            'z': 'm',  # Geopotential height
                            }

# Define colors for the cities, used e.g. in temperature timeseries
cities = {
    'Innsbruck Uni': {'lon': 11.384167, 'lat': 47.259998, 'csv': innsbruck_uni_zamg, 'color': "red", 'hoehe': 578, },
    'Kufstein': {'lon': 12.162778, 'lat': 47.575279, 'csv': kufstein_zamg, 'color': "blue", 'hoehe': 490, },
    'Innsbruck Airport': {'lat': 47.2598, 'lon': 11.3553, 'csv': innsbruck_airport_zamg, 'color': "green",
                          'hoehe': 578, },
    'Jenbach': {'lat': 47.388889, 'lon': 11.758056, 'csv': jenbach_zamg, 'color': "gray", 'hoehe': 530, },
    'Rinn': {'lat': 47.249168, 'lon': 11.503889, 'csv': rinn_zamg, 'color': "purple", 'hoehe': 924},
    'Muenchen': {'lat': 48.149723, 'lon': 11.540523, 'color': "orange", 'hoehe': 521, 'csv': munchen_zamg
                 # Replace 'munchen_zamg' with the actual path to your CSV file if you have data for München
                 }}

# information about MOMMA stations
MOMMA_stations = {"0": "Völs", "1": "Innsbruck_Bergisel", "2": "Patsch_Pfaffenbichl", "3": "Innsbruck_Ölberg",
                  "4": "Innsbruck_Hotel Hilton", "5": "Innsbruck_Saggen_Kettenbrücke", "6": "Volders",
                  "7": "Unterperfuss", "8": "Inzing_Zirl_Modellflugplatz"}

MOMMA_stations_PM = {
    "PM02": {"name": "Völs", "latitude": 47.2614791608, "longitude": 11.3117537274, "height": 583, "key": 0},
    "PM03": {"name": "Innsbruck_Bergisel", "latitude": 47.2472604421, "longitude": 11.3986000093, "height": 726,
             "key": 1},
    "PM04": {"name": "Patsch_Pfaffenbichl", "latitude": 47.21030188, "longitude": 11.4105114057, "height": 983,
             "key": 2},
    "PM05": {"name": "Innsbruck_Ölberg", "latitude": 47.2784241867, "longitude": 11.3902967638, "height": 722,
             "key": 3},
    "PM06": {"name": "Innsbruck_Hotel Hilton", "latitude": 47.2620425014, "longitude": 11.3959606669, "height": 629,
             "key": 4},
    "PM07": {"name": "Innsbruck_Saggen_Kettenbrücke", "latitude": 47.2787431973, "longitude": 11.4123320657,
             "height": 569, "key": 5},
    "PM08": {"name": "Volders", "latitude": 47.2930516284, "longitude": 11.5697988436, "height": 552, "key": 6},
    "PM09": {"name": "Unterperfuss", "latitude": 47.2615210341, "longitude": 11.2607050096, "height": 594, "key": 7},
    "PM10": {"name": "Inzing_Zirl_Modellflugplatz", "latitude": 47.2744017492, "longitude": 11.2143291427,
             "height": 597, "key": 8}}

# units of lidar observations
vars_lidar = {'u': 'm/s', 'v': 'm/s', 'w': 'm/s', 'ff': 'm/s', 'dd': 'degree'}

# hatpro height information
hatpro_vertical_levels = {
    "height_name": ["V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10", "V11", "V12", "V13", "V14",
                    "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
                    "V29", "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38", "V39"],
    "height": ["0", "10", "30", "50", "75", "100", "125", "150", "200", "250", "325", "400", "475", "550", "625", "700",
               "800", "900", "1000", "1150", "1300", "1450", "1600", "1800", "2000", "2200", "2500", "2800", "3100",
               "3500", "3900", "4400", "5000", "5600", "6200", "7000", "8000", "9000", "10000"]}

