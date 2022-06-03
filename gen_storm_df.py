import os
import enum
from socket import AF_NETROM
import sys
import datetime
import random
from os import environ
from this import d
import pandas as pd
import subprocess as sp
import time
import calendar
import gzip
import tempfile
import shutil
import os
import numpy as np
import netCDF4
import glob
from os import walk
import xarray as xr
import util
from netCDF4 import Dataset
from scipy import ndimage
from modify_df import switch_off_storms
import settings as s

lon_NW, lat_NW, lon_SE, lat_SE = -130.005, 55.005, - \
    59.995, 19.995 

def gen_storms_df_for_date(date):
    # Given a date, return the dataframe of storms in date_dir
    # where time, lat, lon, and max MESH are recorded 
    targets = glob.glob('{}/20*/{}/stor*/tar*/ME*/0*/*000.netcdf'.format(s.shave_path, date))
    # open target
    storms_on_date_df = pd.DataFrame(
        columns={
            "timedate",
            "Latitude",
            "Longitude",
            "MESH_max"})
    for target in targets:
        print(target)
        nc = Dataset(target) # open ncdatafile
        var = nc.variables['MESH_Max_30min'][:,:] # convert to npy
        MESH_max = max(0,var.max()) # get max unless its anomolous than fill 100
        print(MESH_max)
        netcdf_file = target.split('/')[-1] # get time from netcdffile
        row = {
            'timedate': netcdf_file.split('.')[0],
            "Latitude": nc.Latitude,
            "Longitude": nc.Longitude,
            "MESH_max": MESH_max} # load up a row as a dict
        storms_on_date_df.loc[len(storms_on_date_df.index)] = row # append row to the end of df
    return storms_on_date_df

class Storm():
    # Storm object containing relevant info about the storm
    def __init__(self, dataset_name='SHAVE'):
        self.dataset_name = dataset_name
    
    def add_basic_info(self, time, lat, lon, MESH_max):
        self.time = time
        self.lat = lat
        self.lon = lon
        self.MESH_max = MESH_max
    
    def get_basic_info(self):
        return self.time, self.lat, self.lon, self.MESH_max
    # one section to collect the basic information about the storm

    def get_statistical_info(self):
        pass

    # another section to collect more advanced info like 'is_Storm'
    # could contain relevant info gleaned from the inputs like mean reflectivity 

def main():
    dates = glob.glob('{}/20*/20*'.format(s.shave_path))
    for date in dates:
        df = gen_storms_df_for_date(date)
        csv_path = '{}/csv'.format(date)
        df.reset_index().to_feather('{}/storm_details.feather'.format(csv_path))

if __name__ == '__main__':
    main()