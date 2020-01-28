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


# Here we will also debut the method of making the file index for all
# the data
# Return a DataFrame object that is searchable by year, month, date
def get_file_index():
    daily_df = None
    drive_path = '/Volumes/P10/ROMS/NWA/NWA-SZ.HCob05T/'
    start_year = 1980
    end_year = 2010
    # List of dictionaries to store the date and file path
    file_list = []
    # Regex to extract the date
    date_regex = re.compile('(\d{4})-(\d{2})-(\d{2})')
    for year in range(start_year, end_year + 1):
        folder_path = drive_path + str(year) + '/'
        if os.path.exists(folder_path):
            glob_string = folder_path + '*avg*'
            daily_average_files = glob.glob(glob_string)
            for average_file in daily_average_files:
                file_str = date_regex.search(average_file)
                # Extract just the year, month and date
                date_str = file_str.group(0)
                # Create a datetime object from 'date_str' string
                # date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                # Create a dictionary and then append it to the list
                temp_dict = {'file_date': date_str, 'file_path': average_file}
                file_list.append(temp_dict)
    daily_df = DataFrame(file_list)
    daily_df['file_date'] = pd.to_datetime(daily_df['file_date'])
    daily_df.set_index('file_date', inplace=True)
    return daily_df


# Do this later
# Convenient method to extract data from the DataFrame
def get_o2(file_index, start_layer=0, end_layer=None):
    return None


# Our goal here is to get a monthly average for a certain spot from
# which a trawl haul was done
if __name__ == '__main__':
    daily_index = get_file_index()
    # Now let's do a daily average for the month of June
    # Just as a test
    # Folder to store the images and the gif
    image_folder = '/Users/jeewantha/Code/images/1982_06/'
    filtered_range = daily_index[(daily_index['file_date'].dt.year == 1982) & (daily_index['file_date'].dt.month == 6)]
    print(filtered_range.shape)
    print(filtered_range.head(5))
    # This seems to work
    # Set up the grid for later plotting
    grd = pyroms.grid.get_ROMS_grid('NWA')
    lon = grd.hgrid.lon_rho
    lat = grd.hgrid.lat_rho
    # Get a list from the 'file_path' column
    june_list = filtered_range['file_path'].tolist()
    # We first need to figure out the maxes and minimums in the arrays
    surface_max_list = []
    surface_min_list = []
    bottom_max_list = []
    bottom_min_list = []
    for daily_file in june_list:
        # print(daily_file)
        # Get the maximum and minimum from both surface and bottom layers
        # Open the file
        daily_o2 = pyroms.utility.get_nc_var('o2', daily_file)[0]
        # Surface is [-1] and bottom [40]
        surface_max_list.append(np.max(daily_o2[-1]))
        # print(np.max(daily_o2[-1]))
        surface_min_list.append(np.min(daily_o2[-1]))
        bottom_max_list.append(np.max(daily_o2[0]))
        bottom_min_list.append(np.min(daily_o2[0]))
    print("Surface Max: {0} mol/kg".format(max(surface_max_list)))
    print("Surface Min: {0} mol/kg".format(min(surface_min_list)))
    print("Bottom Max: {0} mol/kg".format(max(bottom_max_list)))
    print("Bottom Min: {0} mol/kg".format(min(bottom_min_list)))

    surface_max = max(surface_max_list)
    surface_min = min(surface_min_list)
    bottom_max = max(bottom_max_list)
    bottom_min = min(bottom_min_list)
    # Plot everything
    i = 1
    for daily_file in june_list:
        daily_o2 = pyroms.utility.get_nc_var('o2', daily_file)[0]
        plt.figure()
        # plt.pcolor(lon, lat, daily_o2[0], vmin=bottom_min, vmax=bottom_max)
        plt.pcolor(lon, lat, daily_o2[-1], vmin=surface_min, vmax=surface_max, cmap='RdYlGn')
        plt.colorbar()
        plt.axis('image')
        plt.title('{0}'.format(i))
        # Plot coastline
        pyroms_toolbox.plot_coast_line(grd)
        # Save plot
        outfile = '/Users/jeewantha/Code/images/1982_06/o2_surface_{0:03d}.png'.format(i)
        plt.savefig(outfile, dpi=300, orientation='portrait')
        plt.close()
        i = i + 1
