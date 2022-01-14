############################################
#
# Author: Michael Montalbano
#
# Purpose: Extract data fields from tars 
# to compose database of netcdfs organized  by date
# and create training samples for use in 
# image-to-image translation models (UNet)
#
#############################################

import os, sys, datetime
#import pandas as pd
import subprocess as sp
import time, calendar, gzip
import numpy as np
import netCDF4

DATA_HOME = '/condo/swatcommon/common/myrorss'
OUT_HOME = '/scratch/mcmontalbano/myrorss'
CONV_HOME = OUT_HOME # may play with names in future
lon_NW, lat_NW, lon_SE, lat_SE = 1, 2, 3, 4 # search documents
# fields to extract/manipulate
fields = ['MESH']

####################################
# Troubleshooting file
trouble = "trouble.txt"
with open(trouble,"w") as f:
    pass

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
    '''
    extract processed tars for fields saved as .netcdf.gzs 
    Note: this loop could perhaps be improved using dir_list = os.listdir(path) rather than looping with os.walk
    see: https://www.geeksforgeeks.org/create-an-empty-file-using-python/
    '''
    year = day[:4]
    os.system('mkdir {}/{}/{}'.format(OUT_HOME,year,day))
    # iterate over fields
    field_path = '{}/{}/{}/{}'.format(OUT_HOME,year,day,field)
    print(os.listdir(field_path))
    sys.exit()
    for field in fields:
        os.system('tar -xvf {}/{}/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field))
    field = ['MESH']
    # loop through files and convert to netcdf format
    for subdir, dirs, files in os.walk('{}/{}/{}/{}'.format(OUT_HOME,year,day,field)):        # loop through extracted .netcdf.gz's
        print(subdir)
        sys.exit() 
        for subdir1, dirs1, files1 in os.walk('{}/{}/{}/{}'.format(OUT_HOME,year,day,field)): # get into tilt folders
            sys.stdout = open(trouble,"a")
            print(subdir,'\n',subdir1,'\n')
            for f in files1:                                                                     # loop through files
                with gzip.open('{}/{}/{}/{}/{}/{}'.format(OUT_HOME,year,day,field,subdir1,f)) as gz:  # open 
                    with netCDF4.Dataset('dummy',mode='r', memory=gz.read()) as nc:              # convert to netcdf
                        nc.to_necdf(path='{}/{}/{}/{}/{}/{}'.format(OUT_HOME,year,day,field,subdir,f)) # save as netcdf
                        print(nc.variables) 
        # REMOVE THE TARS and GZs FROM SCRATCH ONLY
        os.system('rm {}/{}/{}/{}/{}/*.tar'.format(OUT_HOME,year,day,field,subdir1))
        os.system('rm {}/{}/{}/{}/{}/*.gz'.format(OUT_HOME,year,day,field,subdir1))        

def localmax(day, multi_n):
    '''
    This is the old script
    To do:
    - change path to OUT_HOME, which is converted MYRORSS storage
    - pick path to 
    - check field name 'MergedReflectivityQCComposite'
    '''
    year = day[:4]
    myrorss_path = '{}/{}/'.format(OUT_PATH,year)
    os.system('makeIndex.pl /mnt/data/SHAVE_cases/{}/multi{} code_index.xml'.format(date,multi_n))
    os.system('w2localmax -i /mnt/data/SHAVE_cases/{}/multi{}/code_index.xml -I MergedReflectivityQCComposite -o /mnt/data/michaelm/practicum/cases/{}/DATE -s -d "40 60 5"'.format(date,multi_n,date,multi_n))
    os.system('makeIndex.pl /mnt/data/michaelm/practicum/cases/{}/multi{} code_index.xml'.format(date,multi_n))
    os.system('w2table2csv -i {}/{}/multi{}/code_index.xml -T MergedReflectivityQCCompositeMaxFeatureTable -o {}/{}/DATE/csv -h'.format(OUT_HOME,date,multi_n,OUT_HOME,date,multi_n))



def cropconv(case_df, date, nse_fields, fields_accum, multi_n):
    # this also needs reform    
    os.system('makeIndex.pl {}/{}/NSE code_index.xml'.format(DATA_HOME,date))
    for idx, row in case_df.iterrows():
        # if idx <= 200:
        #     continue
        multi = '{}/{}/multi{}'.format(OUT_HOME,date,multi_n)
        lon = row['Longitude']
        lat = row['Latitude']
        delta = 0.15

        lonNW = lon - delta
        latNW = lat + delta
        lonSE = lon + delta
        latSE = lat - delta
        
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=1)).strftime('%Y%m%d-%H%M%S')

        # crop input
        #########################
        os.system("makeIndex.pl {}/{}/multi{}/uncropped code_index.xml {} {}".format(OUT_HOME,date,multi_n, time1, time2)) # make index for uncropped
        for field in fields_accum:
           os.system('w2cropconv -i {}/{}/multi{}/uncropped/code_index.xml -I {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,date, multi_n, field, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
        #########################

        # crop target -
        #########################
        time1 = (date_1+datetime.timedelta(minutes=30)).strftime('%Y%m%d-%H%M%S')
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=1)).strftime('%Y%m%d-%H%M%S')        
        os.system("makeIndex.pl {}/{}/multi{}/uncropped code_index.xml {} {}".format(OUT_HOME,date,multi_n, time1, time2))
        os.system('w2cropconv -i {}/{}/multi{}/uncropped/code_index.xml -I MESH_Max_30min -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d}/target_MESH_Max_30min -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,date, multi_n, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
        # ########################

        # NSE 
        # crop for 30 min prior to 30 min ahead
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time1 = (date_1+datetime.timedelta(minutes=-30)).strftime('%Y%m%d-%H%M%S')
        time2 = (date_1+datetime.timedelta(minutes=30)).strftime('%Y%m%d-%H%M%S')

        os.system("makeIndex.pl {}/{}/NSE code_index.xml {} {}".format(DATA_HOME,date, time1, time2))
        for field in nse_fields:
            os.system('w2cropconv -i {}/{}/NSE/code_index.xml -I {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d}/NSE -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(DATA_HOME,date, field, date, multi_n, idx, latNW, lonNW,latSE,lonSE))

        # commenting this out for repair
        os.system('w2cropconv -i {}/{}/multi{}/code_index.xml -I  MergedReflectivityQC -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(DATA_HOME,date, multi_n, date, multi_n, idx, latNW, lonNW,latSE,lonSE))




#def accumulate(startdate, enddate, inloc, outloc, interval):




def main():

    process = 'extract'
    #year = sys.argv[1]
    # extract first 10 cases of the year (b1 to b2)
    b1 = 0
    b2 = 2
    days = get_cases()[b1:b2]
    year = '1999'
    inloc = '/condo/swatcommon/common/myrorss/' 
    outloc = '/scratch/mcmontalbano/myrorss/'

    for day in days:
        extract(day)
        localmax(day)
        get_cases_info(day) # use mergedtable from localmax to store locations
        
   
if __name__ == "__main__":
    main()
