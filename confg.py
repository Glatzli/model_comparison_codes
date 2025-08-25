"""In this "confg-script" are all the data filepaths listed
You have to change it!
original by hannes, daniel adapted it a bit and added/edited some coords etc

"""

# -------------------------------------------To change --------------------------------------------------------
# Folder where the data is saved:
data_folder = "D:/MSc_Arbeit/data/"
# Folder where the model output is saved:
model_folder = "D:/MSc_Arbeit/"
# Plot directory (where to save the plots)
dir_PLOTS = "D:/MSc_Arbeit/plots/"

# -------------------------------------------constants needed for calculations---------------------------------
hafelekar_height = 2279  # m, highest HOBO from https://zenodo.org/records/4672313 hobo dataset, used for VHD calc
c_p = 1005  # J/(kg*K), specific heat capacity of air at constant pressure, for VHD calc

# coordinates of points used for Daniels' Analysis
ibk_villa = {"name": "ibk_villa", "lat": 47.259998, "lon": 11.384167}  # same lat & lon of Ibk cosma already used: which
# is for Ibk_Villa (2m temp recording)
ibk_uni = {"name": "ibk_uni", "lat": 47.264, "lon": 11.385}  # hatpro, uni coords rounded to 3 digits after comma
woergl = {"name": "woergl", "lat": 47.494, "lon": 12.059}  # coords for wörgl, lower Inn valley
kiefersfelden = {"name": "kiefersfelden", "lat": 47.62, "lon": 12.2}  # coords for kiefersfelden, Germany, entrance Inn valley


lat_min, lat_max = 47, 47.7  # orig: 47, 47.7    # lat & lon values for vhd domain plotting
lon_min, lon_max = 10.8, 12  # 10.8, 12
# -------------------------------------------------------------------------------------------------------------

radiosonde_csv = f"{data_folder}/2017101603_bufr309052.csv"  # radiosonden aufstieg at innsbruck airport
JSON_TIROL = f"{data_folder}/Height/gadm41_AUT_1.json"  # tirol json file
DEMFILE_CLIP = f"{data_folder}/Height/dem_clipped.tif"  # dem file (höhe)
TIROL_DEMFILE = f"{data_folder}/Height/dem.tif"  # changed dem file: indexed and renamed coords
dem_smoothed = f"{data_folder}/Height/dem_smoothed.tif"
filepath_arome_height = f"{model_folder}/AROME/AROME_TEAMx_CAP_3D_fields/AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_z.nc"
dem_file_hobos_extent = f"{data_folder}/Height/dem_cut_hobos.tif"  # created DEM (in model_topography) to see real heights with HOBOS

# ZAMG Datahub files
kufstein_zamg = f"{data_folder}/Observations/ZAMG_Tawes/data_station9016-Kufstein_20171012_20171018.csv"
innsbruck_uni_zamg = f"{data_folder}/Observations/ZAMG_Tawes/data_station11803-InnsbruckUniversity_20171012_20171018.csv"
innsbruck_airport_zamg = f"{data_folder}/Observations/ZAMG_Tawes/data_station11804-InnsbruckAirport_20171012_20171018.csv"
jenbach_zamg = f"{data_folder}/Observations/ZAMG_Tawes/data_station11901-Jenbach_20171012_20171018.csv"
rinn_zamg = f"{data_folder}/Observations/ZAMG_Tawes/data_station11123-Rinn_20171015T1200_20171016T1210.csv"
munchen_zamg = f"{data_folder}/Observations/ZAMG_Tawes/data_munich_T2m.csv"

# mobile stations, cut to our period
momma_our_period_file = f"{data_folder}/Observations/MOMMA/MOMMA_our_period.nc"

# ----------------------------------Models-----------------------------------------------------

# absolute paths AROME
dir_AROME = model_folder + "/AROME/"
dir_2D_AROME = f"{dir_AROME}AROME_TEAMx_CAP_2D_fields"
dir_3D_AROME = f"{dir_AROME}AROME_TEAMx_CAP_3D_fields"
dir_timeseries_AROME = f"{dir_AROME}AROME_TEAMx_CAP_timeseries/"

# absolute paths WRF
wrf_folder = f"{model_folder}WRF_ACINN"  # before: wrf_ACINN

# absolute paths ICON
icon_folder_3D = f"{model_folder}ICON"
icon_folder_meteogram = f"{model_folder}icon/ICON_Meteogram"  # ?

# absolute paths ICON
icon2TE_folder_3D = f"{model_folder}ICON2TE"

# absolute Path UKMO
ukmo_folder = f"{model_folder}ukmo/"

# -------------------------------Data and Plot paths -----------------------------------------------

# EC stations
dir_EC_stations = f"{data_folder}/Observations/EC_4_stations"
EC_30min_final = f"{dir_EC_stations}/EC_30min_file.nc"
EC_1min_final = f"{dir_EC_stations}/EC_1min_file.nc"

# Ibox dir
ibox_folder = f"{data_folder}/Observations/Ibox"

# HOBOS station
hobos_file = f"{data_folder}/Observations/hobos_final.nc"  # Observations/HOBOS/

# LIDAR obs
lidar_obs_folder = f"{data_folder}/Observations/LIDAR"

# HATPRO obs
hatpro_folder = f"{data_folder}/Observations/HATPRO_obs"  # nicht vorhanden?

# Define colors for the models to use the same in each plot:
colordict = {"HOBOS": "purple",
             "ICON": "orange",
             "RADIOSONDE": "black",
             "AROME": "red",
             "HATPRO": "gray",
             "UKMO": "green",
             "WRF_ACINN": "blue"}

# Create a dictionary with information of the TAWES stations
station_files_zamg = {
    "IAO": {
        "filepath": innsbruck_uni_zamg,
        "name": "Innsbruck Uni",
        'lon': 11.384167,
        'lat': 47.259998,
        'hoehe': 578,
    },
    "JEN": {
        "filepath": jenbach_zamg,
        "name": "Jenbach",
        'lat': 47.388889,
        'lon': 11.758056,
        'hoehe': 530,
    },
    "KUF": {
        "filepath": kufstein_zamg,
        "name": "Kufstein",
        'lon': 12.162778,
        'lat': 47.575279,
        'hoehe': 490,
    },
    "LOWI": {
        "filepath": innsbruck_airport_zamg,
        "name": "Innsbruck Airport",
        'lat': 47.2598,
        'lon': 11.3553,
        'hoehe': 578,
    },
    "IMST": {
        #"filepath": imst_zamg,
        "name": "Imst",
        'lat': 47.2419,
        'lon': 10.7218,
        'hoehe': 828,
    }
}

# create a dict with info about the IBOX stations
stations_ibox = {
    "VF0": {
        "filepath": f"{ibox_folder}/vf0.csv",
        "name": "Kolsass",
        "latitude": 47.305,
        "longitude": 11.622,
        "height": 545
    },
    "SF8": {
        "filepath": f"{ibox_folder}/sf8.csv",
        "name": "Terfens",
        "latitude": 47.326,
        "longitude": 11.652,
        "height": 575
    },
    "SF1": {
        "filepath": f"{ibox_folder}/sf1.csv",
        "name": "Eggen",
        "latitude": 47.317,
        "longitude": 11.616,
        "height": 829
    },
    "NF10": {
        "filepath": f"{ibox_folder}/nf10.csv",
        "name": "Weerberg",
        "latitude": 47.300,
        "longitude": 11.673,
        "height": 930
    },
    "NF27": {
        "filepath": f"{ibox_folder}/nf27.csv",
        "name": "Hochhaeuser",
        "latitude": 47.288,
        "longitude": 11.631,
        "height": 1009
    }
}

# dict with infos about EC stations
ec_station_names = {
    1: {"name": "Patsch_EC_South", "lat": 47.209068, "lon": 11.411932},
    0: {"name": "Innsbruck_Airport_EC_West", "lat": 47.255375, "lon": 11.342832},
    2: {"name": "Thaur_EC_East", "lat": 47.281335, "lon": 11.474532},
    3: {"name": "IAO_Centre_Innsbruck_EC_Center", "lat": 47.264035, "lon": 11.385707}
}

# variables, units 2D AROME
variables_units_2D_AROME = {
    'hfs': 'W/m²',  # Sensible heat flux at the surface
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
variables_units_3D_AROME = {
    'ciwc': 'kg/kg',  # Specific cloud ice water content
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
    'Innsbruck Uni': {
        'lon': 11.384167,
        'lat': 47.259998,
        'csv': innsbruck_uni_zamg,
        'color': "red",
        'hoehe': 578,
    },
    'Kufstein': {
        'lon': 12.162778,
        'lat': 47.575279,
        'csv': kufstein_zamg,
        'color': "blue",
        'hoehe': 490,
    },
    'Innsbruck Airport': {
        'lat': 47.2598,
        'lon': 11.3553,
        'csv': innsbruck_airport_zamg,
        'color': "green",
        'hoehe': 578,
    },
    'Jenbach': {
        'lat': 47.388889,
        'lon': 11.758056,
        'csv': jenbach_zamg,
        'color': "gray",
        'hoehe': 530,
    }, 'Rinn': {
        'lat': 47.249168,
        'lon': 11.503889,
        'csv': rinn_zamg,
        'color': "purple",
        'hoehe': 924
    },
    'Muenchen': {
        'lat': 48.149723,
        'lon': 11.540523,
        'color': "orange",
        'hoehe': 521,
        'csv': munchen_zamg  # Replace 'munchen_zamg' with the actual path to your CSV file if you have data for München
    }
}

# information about MOMMA stations
MOMMA_stations = {
    "0": "Völs",
    "1": "Innsbruck_Bergisel",
    "2": "Patsch_Pfaffenbichl",
    "3": "Innsbruck_Ölberg",
    "4": "Innsbruck_Hotel Hilton",
    "5": "Innsbruck_Saggen_Kettenbrücke",
    "6": "Volders",
    "7": "Unterperfuss",
    "8": "Inzing_Zirl_Modellflugplatz"
}

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
             "height": 597, "key": 8}
}

# units of lidar observations
vars_lidar = {
    'u': 'm/s',
    'v': 'm/s',
    'w': 'm/s',
    'ff': 'm/s',
    'dd': 'degree'}

# hatpro height information
hatpro_vertical_levels = {
    "height_name": [
        "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29", "V30",
        "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38", "V39"
    ],
    "height": [
        "0", "10", "30", "50", "75", "100", "125", "150", "200", "250",
        "325", "400", "475", "550", "625", "700", "800", "900", "1000", "1150",
        "1300", "1450", "1600", "1800", "2000", "2200", "2500", "2800", "3100", "3500",
        "3900", "4400", "5000", "5600", "6200", "7000", "8000", "9000", "10000"
    ]
}
