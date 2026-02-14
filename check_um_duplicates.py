import xarray as xr
import confg
import os
import pandas as pd

vhd_um = xr.open_dataset(os.path.join(confg.ukmo_folder, 'UM_vhd_full_domain_full_time.nc'))
print("UM VHD Dataset time coordinate:")
print(vhd_um.time)
print(f"\nTotal number of times: {len(vhd_um.time)}")

times = pd.to_datetime(vhd_um.time.values)
print(f"Has duplicates: {times.duplicated().any()}")
print(f"Number of unique times: {times.nunique()}")

if times.duplicated().any():
    print("\nDuplicated times:")
    duplicated_times = times[times.duplicated(keep=False)].sort_values()
    print(duplicated_times)
    print(f"\nNumber of duplicated timestamps: {len(duplicated_times)}")

