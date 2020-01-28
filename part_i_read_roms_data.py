import os
import re
import glob
from pandas import DataFrame
import datetime
import pandas as pd
import pyroms
import pyroms_toolbox
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma


# Function to extract a certain property from the netCDF4
def get_property(filepath, prop='temp'):
    if os.path.exists(filepath):
        print('True')
    else:
        return False


# Function to average the bottom layer over a couple of days (2)
def mean_o2():
    filepath_1 = '/Volumes/P10/ROMS/NWA/NWA-SZ.HCob05T/1980/NWA-SZ.HCob05T_avg_1980-06-18T01:00:00.nc'
    filepath_2 = '/Volumes/P10/ROMS/NWA/NWA-SZ.HCob05T/1980/NWA-SZ.HCob05T_avg_1980-06-19T01:00:00.nc'
    o2_day_i = pyroms.utility.get_nc_var('o2', filepath_1)[0]
    o2_day_ii = pyroms.utility.get_nc_var('o2', filepath_2)[0]
    # How to access the layers -> o2_day_i[0] is the bottom, [-1] is the surface
    # np.mean( np.array([ old_set, new_set ]), axis=0 )
    # Remember that these are Masked Arrays
    o2_mean = np.mean(ma.array([o2_day_i, o2_day_ii]), axis=0)

    grd = pyroms.grid.get_ROMS_grid('NWA')
    lon = grd.hgrid.lon_rho
    lat = grd.hgrid.lat_rho
    # Plot 2-D view of [O2] at the bottom
    plt.figure()
    plt.pcolor(lon, lat, o2_mean[-1])
    plt.colorbar()
    plt.axis('image')

    # Plot coastline
    pyroms_toolbox.plot_coast_line(grd)

    # Save plot
    outfile = 'o2_mean_surface.png'
    plt.savefig(outfile, dpi=300, orientation='portrait')

    # Plot single
    plt.figure()
    plt.pcolor(lon, lat, o2_day_ii[-1])
    plt.colorbar()
    plt.axis('image')

    # Plot coastline
    pyroms_toolbox.plot_coast_line(grd)

    # Save plot
    outfile = 'o2_surface_06_19.png'
    plt.savefig(outfile, dpi=300, orientation='portrait')


if __name__ == '__main__':
    folder_path = '/Volumes/P10/ROMS/NWA/NWA-SZ.HCob05T/1980/'
    # Checking for specific files
    if os.path.exists(folder_path):
        print('Folder exists')
    # Now we should get all the netCDF files with daily averages
    # We'll use glob for this
    search_string_glob = folder_path + '*avg*'
    daily_average_files = glob.glob(search_string_glob)
    print(daily_average_files)
    # Number of netCDF files
    print(len(daily_average_files))
    # Now get a test file and do some regex wizardry on it
    test_file_str = daily_average_files[0]
    print(test_file_str)
    date_regex = re.compile('(\d{4})-(\d{2})-(\d{2})')
    date_str = date_regex.search(test_file_str)
    print(date_str.group(0))
    # Make a DataFrame of the file names with dates
    # Need to make a list of dictionaries
    date_list = []
    # Now the dictionaries
    # Loop through the list of files
    for avg_file in daily_average_files:
        file_str = date_regex.search(avg_file)
        # Extract just the year, month and date
        date_str = file_str.group(0)
        # Create a datetime object from 'date_str' string
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        print(date_obj)
        # Create a dictionary and then append it to the list
        temp_dict = {'file_date': date_obj, 'file_path': avg_file}
        date_list.append(temp_dict)
    # Create the DataFrame from the list of dictionaries
    df_daily = DataFrame(date_list)
    # Check if the dataframe is as intended
    print(df_daily.head(10))
    # Select a certain month
    print(df_daily[df_daily['file_date'] == '1980-02-20'])
    df_daily['file_date_formatted'] = pd.to_datetime(df_daily['file_date'])
    # print(df_daily.head(10))
    print(df_daily['file_date'].dt.month != 6)
    # Now try this for the normal column. Should work. Right?
    # print(df_daily['file_date'].dt.month == 6)
    # Spoiler: It works
    # Next steps
    # String together several arrays and make monthly averages for a certain layer

    # Read in the Hauls file. Then start matching up the haul dates with the
    mean_o2()
