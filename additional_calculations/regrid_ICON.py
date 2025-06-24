
import xarray as xr
import read_icon_model_3D
import sys
# sys.path.append("D:/MSc_Arbeit/model_comparison_codes/calculations_and_plots")
# from plot_topography import calculate_lon_extent_for_km, calculate_km_for_lon_extent

# icon_filename = read_icon_model_3D.generate_icon_filenames(16, 10, variant="ICON2")  # generate filenames for ICON2TE
# icon_hex = read_icon_model_3D.read_icon_fixed_time(day=16, hour=10, variant="ICON")

inc = 0.005 # increment in degrees for the wanted ICON grid

lat_diff = 49.728592 - 42.67218
lon_diff = 16.333878 - 0.9697978
nr_points = 449235  # number of points in the ICON grid, which is shown in CDO gendis command
# inc = ((lat_diff * lon_diff) / nr_points)**(1/2)

(lat_diff * lon_diff) / inc
print(7)