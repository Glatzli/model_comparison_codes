"""
short script written by Chat to find lat/lon coords of HOBO from HOBO ID
"""
import fix_win_DLL_loading_issue

import xarray as xr
import confg

# Load HOBO dataset
ds = xr.open_dataset(confg.hobos_file)

# Find H38
hobo_ids = ds['hobo_id'].values[:, 0]
h38_idx = None

for i, hobo_id in enumerate(hobo_ids):
    if hobo_id == 'H38':
        h38_idx = i
        break

if h38_idx is not None:
    lat_h38 = float(ds.lat[h38_idx].values)
    lon_h38 = float(ds.lon[h38_idx].values)
    zsl_h38 = float(ds.zsl[h38_idx].values)
    
    print(f"HOBO H38 gefunden bei Index: {h38_idx}")
    print(f"Latitude:  {lat_h38:.6f}")
    print(f"Longitude: {lon_h38:.6f}")
    print(f"Height:    {zsl_h38:.1f} m")
else:
    print("H38 nicht gefunden")

