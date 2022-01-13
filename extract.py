############################################
#
# Modified by: Michael Montalbano
#
# Purpose: Extract data fields from tars 
# to compose database of netcdfs organized  by date
#
#############################################

import os, sys, datetime
import pandas as pd
import subprocess as sp
import time, calendar, gzip
import numpy as np
#import netCDF4

DATA_HOME = '/condo/swatcommon/common/myrorss'
OUT_HOME = '/scratch/mcmontalbano/myrorss'
#lon_NW, lat_NW, lon_SE, lat_SE = # search documents 
# fields to extract/manipulate
fields = ['MESH']

####################################
# Functions 

def get_cases(year = '1998'):
    # given a year, return the cases
    # step through directory 
    cases = []
    for subdir, dirs, files in os.walk('{}/{}'.format(DATA_HOME,year)):
        for file in sorted(files):
            cases.append(file[:8])    
    return cases

def extract(day):
    # extract processed tars for fields saved as .netcdf.gzs 
    year = day[:4]
    os.system('mkdir {}/{}/{}'.format(OUT_HOME,year,day))
    # iterate over fields

    for field in fields:
        os.system('tar -xvf {}/{}/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field))

    for subdir, dirs, files in os.walk('{}/{}/{}/{}'.format(OUT_HOME,year,day,field)): # loop through extracted .netcdf.gz's
        for subdir1, dirs1, files1 in os.walk('{}/{}/{}/{}'.format(OUT_HOME,year,day,field)): # get into tilt folders
            print(subdir1) # troubleshooting
            for f in files1:                                                                     # loop through files
                with gzip.open('{}/{}/{}/{}/{}/{}'.format(OUT_HOME,year,day,field,subdir1,f)) as gz:  # open 
                    with netCDF4.Dataset('dummy',mode='r', memory=gz.read()) as nc:              # convert to netcdf
                        nc.to_necdf(path='{}/{}/{}/{}/{}/{}'.format(OUT_HOME,year,day,field,subdir,f)) # save as netcdf
                        print(nc.variables) 
                        sys.exit()
    # field = 'MESH'
    #os.chdir(inloc  + startdate[0:4])
    # for i in range(startepoch, endepoch, 86400):

        

#def localmax(day):

#def get_storm_info(day):
    # Given a day, return dataframe case_df which contains identifying information about each storm
    # and its intensity, done using MergedReflectivityFeatureTable

#def accumulate(startdate, enddate, inloc, outloc, interval):




def main():

    process = 'extract'
    #year = sys.argv[1]
    # extract first 10 cases of the year (b1 to b2)
    b1 = 0
    b2 = 2
    days = get_cases()[b1:b2]

    inloc = '/condo/swatcommon/common/myrorss/' 
    outloc = '/scratch/mcmontalbano/myrorss/'

    for day in days:
        extract(day)
   
if __name__ == "__main__":
    main()
