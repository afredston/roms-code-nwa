from part_iii_mean_o2 import get_file_index
import os
import re
import glob
from pandas import DataFrame
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import pyroms
import pyroms_toolbox
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

parameters_file_path = '/Users/jeewantha/Code/data/monthly_means/'
nwa_grd = pyroms.grid.get_ROMS_grid('NWA')


# Create plot of whatever grid you want
def create_monthly_mean_plot(data_array, plot_date, parameter):
    grd = pyroms.grid.get_ROMS_grid('NWA')
    lon = grd.hgrid.lon_rho
    lat = grd.hgrid.lat_rho
    plt.figure()
    # plt.pcolor(lon, lat, daily_o2[0], vmin=bottom_min, vmax=bottom_max)
    plt.pcolor(lon, lat, data_array, cmap='RdYlGn')
    plt.colorbar()
    plt.axis('image')
    plt.title('{0} for {1}'.format(parameter, plot_date))
    # Plot coastline
    pyroms_toolbox.plot_coast_line(grd)
    # Save plot
    outfile = '/Users/jeewantha/Code/images/monthly_means/{0}_bottom_ver_1.png'.format(parameter)
    plt.savefig(outfile, dpi=300, orientation='portrait')
    plt.close()
    return True


# But would the netCDF4 file be the best cause of action?
# Just writing text files. Would that work?
# Create a netCDF4 file with monthly averages for a certain date
# Haul date is a datetime object
def create_monthly_mean(haul_date):
    print(type(haul_date))
    haul_date = pd.to_datetime(haul_date)
    print(haul_date.day)
    print(haul_date.month)
    print(haul_date.year)
    print(type(haul_date))
    daily_index = get_file_index()
    start_date = haul_date - relativedelta(months=1)
    end_date = haul_date + relativedelta(months=2)
    print(start_date)
    print(end_date)
    file_paths = daily_index.loc[start_date:end_date, 'file_path'].tolist()
    # For O2, Large zooplankton, Medium Zooplankton, Small zooplankton
    o2_list = []
    lg_zplk = []
    me_zplk = []
    sm_zplk = []
    for file_path in file_paths:
        daily_o2 = pyroms.utility.get_nc_var('o2', file_path)[0]
        daily_lg_zplk = pyroms.utility.get_nc_var('nlgz', file_path)[0]
        daily_me_zplk = pyroms.utility.get_nc_var('nmdz', file_path)[0]
        daily_sm_zplk = pyroms.utility.get_nc_var('nsmz', file_path)[0]
        # Append the values to
        o2_list.append(daily_o2[0])
        lg_zplk.append(daily_lg_zplk[0])
        me_zplk.append(daily_me_zplk[0])
        sm_zplk.append(daily_sm_zplk[0])
    # Calculate the mean across axis=0
    mean_o2 = ma.mean(ma.array(o2_list), axis=0)
    mean_lg_zplk = ma.mean(ma.array(lg_zplk), axis=0)
    mean_me_zplk = ma.mean(ma.array(me_zplk), axis=0)
    mean_sm_zplk = ma.mean(ma.array(sm_zplk), axis=0)
    print(mean_o2.shape)
    print(mean_o2)
    # Write this file as 'o2_bottom_monthly_average'
    # np.savetxt('/Users/jeewantha/Code/data/monthly_means/o2_monthly_mean.out', mean_o2)
    # Writing the four parameter files to their location
    storage_location = parameters_file_path + '{0}/{1}/'.format(haul_date.year, haul_date.month)
    # If the path does not exist
    if not os.path.exists(storage_location):
        os.makedirs(storage_location)
        print('Path newly created')
    else:
        print('Path exists')
    files_to_write = [mean_o2, mean_lg_zplk, mean_me_zplk, mean_sm_zplk]
    file_names_to_write = ['o2_data.out', 'lg_zplk_data.out', 'me_zplk_data.out', 'sm_zplk_data.out']
    for file_name, file_to_write in zip(file_names_to_write, files_to_write):
        file_name = storage_location + file_name
        # np.savetxt(file_name, file_to_write)
        file_to_write.dump(file_name)
        print('Saved {0}'.format(file_name))
    # Create the plots for the 4 parameters
    """
    create_monthly_mean_plot(mean_o2, haul_date, 'O2')
    create_monthly_mean_plot(mean_lg_zplk, haul_date, 'lg_zplk')
    create_monthly_mean_plot(mean_me_zplk, haul_date, 'me_zplk')
    create_monthly_mean_plot(mean_sm_zplk, haul_date, 'sm_zplk')
    """
    print('Success')


# A method where we enter a date, and a pair of coordinates
# then we return the mean [O2] for the sea bottom and surface
def get_o2_mean(row):
    # Timedelta object of 30 days
    # Haul date is a string that we convert to a datetime object
    # haul_date_dt = datetime.datetime.strptime(haul_date, '%Y-%m-%d')
    # haul_result = haul_year * haul_month
    # print(row['year'])
    # print(row['month'])
    year = int(row['year'])
    month = int(row['month'])
    o2_file_loc = '{0}{1}/{2}/o2_data.out'.format(parameters_file_path,
                                                  year, month)
    print(o2_file_loc)
    print(row['lon'])
    print(row['lat'])
    if os.path.exists(o2_file_loc):
        print('Path exists')
        # Get the grid coordinates
        i_j = pyroms.utility.get_ij(row['lon'], row['lat'], nwa_grd)
        print(i_j)
        return i_j
    # print("Path doesn't exist")
    return 250


# A method where we enter a date, and a pair of coordinates
# then we return the mean [O2] for the sea bottom and surface
def get_small_zplk_mean(haul_date, haul_lon, haul_lat):
    return None


if __name__ == '__main__':
    daily_index = get_file_index()
    # Set up the grid for later plotting
    # lon = grd.hgrid.lon_rho
    # lat = grd.hgrid.lat_rho
    # Call 'get_o2_mean'
    # get_o2_mean('2016-05-25', 97.5, 82.3)
    # Read the data file containing the hauls
    catch_hauls_df = pd.read_csv('data/catch_data_hauls_merge.csv')
    print(catch_hauls_df.head(10))
    # Add a day column with default set to 1
    catch_hauls_df['day'] = 1
    # Create a 'haul_date' column by using the 'year', 'month', and 'day' columns
    catch_hauls_df['haul_date'] = pd.to_datetime(catch_hauls_df[['year', 'month', 'day']])
    print('How does the haul date column look')
    # Print out the relevant columns
    print(catch_hauls_df[['haul_date', 'year', 'month', 'day']].head(10))
    # Filter the data from or after 1980-01-01
    mask_post_1980 = catch_hauls_df['haul_date'] >= '1980-01-01'
    catch_hauls_df = catch_hauls_df[mask_post_1980]
    print("Print out the final dataframe")
    print(catch_hauls_df)
    # Sort the dataframe by 'haul_date'
    catch_hauls_df = catch_hauls_df.sort_values(by='haul_date')
    print(catch_hauls_df)
    # OK. So this works
    # Now on to the fun stuff
    catch_hauls_df.reset_index(inplace=True)
    print(catch_hauls_df)
    # Let's just get the seasonal range for a certain date
    sample_haul_date = catch_hauls_df.loc[0, 'haul_date']
    # print(sample_haul_date)
    sample_start_date = sample_haul_date - relativedelta(months=1)
    sample_end_date = sample_haul_date + relativedelta(months=2)
    # print(sample_haul_date - relativedelta(months=1))
    # print(sample_haul_date + relativedelta(months=2))
    # Testing if the file index is working
    # print(daily_index.head(10))
    # print(daily_index['2004-01-01']['file_path'])
    print(daily_index.loc[sample_start_date: sample_end_date, 'file_path'])
    print(daily_index.loc[sample_start_date: sample_end_date, 'file_path'].tolist())
    # create_monthly_mean()
    # First a list of unique dates in the 'haul_date' column
    print(type(catch_hauls_df['haul_date'].unique()[0]))
    print(len(catch_hauls_df['haul_date'].unique()))
    # Loop through
    """
    for run_date in catch_hauls_df['haul_date'].unique()[0:10]:
        create_monthly_mean(run_date)
        print('One done')
    print('All done')
    # OK. Let's take the 10th unique value and stick it in there
    """
    # Creating a new column
    # catch_hauls_df['o2_mean'] = get_o2_mean(catch_hauls_df['year'], catch_hauls_df['month'],
    #                                        catch_hauls_df['lon'], catch_hauls_df['lat'], grd)
    # print(catch_hauls_df['o2_mean'])
    """
    test_date_1 = catch_hauls_df['haul_date'].unique()[10]
    create_monthly_mean(test_date_1)
    # Let's just test the plot real quick. To see if it's saved correctly
    data_path = '/Users/jeewantha/Code/data/monthly_means/1981/11/o2_data.out'
    # data_np = np.loadtxt(data_path)
    data_np = np.load(data_path)
    print(data_np)
    test_datetime = datetime.datetime.now()
    create_monthly_mean_plot(data_np, plot_date=test_datetime, parameter='something_o2_ver_2')
    print('Yaaaaas')
    """
    # Now we will start assigning some values
    # Using the 10 unique values
    haul_date_list = catch_hauls_df['haul_date'].unique()[0:10]
    filter_key = catch_hauls_df['haul_date'].isin(haul_date_list)
    test_df = catch_hauls_df[filter_key]
    print(test_df)
    test_df.reset_index(inplace=True)
    # Disable chained assignment warnings. This is such a pain
    pd.options.mode.chained_assignment = None
    blah = test_df.apply(get_o2_mean, axis=1)
    test_df['blah'] = blah
    print(test_df[['year', 'month', 'blah']])
    test_df.to_csv('/Users/jeewantha/Code/data/current_points.csv')
    # print(blah)
    # print(test_df)
    # test_df.loc[: 'blah_blah'] = test_df.apply(get_o2_mean, axis=1)
