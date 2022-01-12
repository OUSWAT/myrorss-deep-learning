############################################
#
# Author: Skylar S Williams
# Modified by: Michael Montalbano
#
# Purpose: Extract data fields from tars 
# to compose database of netcdfs organized  by date
#
#############################################


import os, sys, datetime
import pandas as pd
import subprocess as sp
import time
import calendar
import numpy as np
#import netCDF4

DATA_HOME = '/condo/swatcommon/common/myrorss/'
OUT_HOME = '/lscratch/mcmontalbano/myrorss/'

# fields to extract/manipulate
fields = ['MESH']

def get_cases(year = '1998'):
    # given a year, return the cases
    # step through directory 
    cases = []
    for subdir, dirs, files in os.walk('{}/{}'.formaty(DATA_HOME,year):
        for file in files:
            cases.append(file[:8])    
    return cases

def extract(startdate, enddate, inloc, outloc):
    print('hello')
    startepoch = calendar.timegm((int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), 12, 00, 00))
    endepoch = calendar.timegm((int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), 12, 00, 00))
    field = 'MESH'
    #os.chdir(inloc  + startdate[0:4])
    for i in range(startepoch, endepoch, 86400):

        date = time.strftime('%Y%m%d', time.gmtime(i))
        os.system('tar -xvf {}.tar -C {} --wildcards "{}"'.format(date, outloc,field))
        cmd = 'tar -xvf {}.tar -C {} --wildcards "{}"'.format(date, outloc,field)
        # cmd = 'tar -xvf ' + date + '.tar -C ' + outloc + ' --wildcards "MESH"'
        with open('readme.txt', 'w') as f:
            f.write('cmd')
            f.write(startepoch,endepoch)
        # p = sp.Popen(cmd, shell=True)
        # p.wait()

def extract_py(days):
    # extract loop
    # feed it a list of days to extract
    for day in days:
        # iterate over fields
        for field in fields:
            os.system('tar -xvf {}.tar -C {} --wildcards "{}"'.format(day, OUT_HOME,field))

def accumulate(startdate, enddate, inloc, outloc, interval):

    # FILL IN WITH PSEUDO RADAR DATA TO GET EVEN TIMING
    # cmd = 'makeMissingRadar something something something'
    # p = sp.Popen(cmd, shell=True)
    # p.wait()

    # ACCUMULATE EVERY HOUR ON THE HOUR (allows for cleaner accumulations on the yearly scale)
    cmd = 'w2accumulator -g MESHMhtpost -C 1 -t 60 -m MESH -O MESH_Max_'
    p = sp.Popen(cmd, shell=True)
    p.wait()

def shell_loop():
    # loop for doing jobs using the two shell scripts shipMulti.sh and shipExtract.sh
    startdate = sys.argv[1] # dates are input from shipExtract.sh
    enddate = sys.argv[2]

    proc = 'EXTRACT'

    inloc = '/condo/swatcommon/common/myrorss/' 
    outloc = '/scratch/mcmontalbano/myrorss/'

    if proc == 'EXTRACT':
        extract(startdate, enddate, inloc, outloc)


def main():

    process = 'extract'
    year = sys.argv[1]

    # extract first 10 cases of the year (b1 to b2)
    b1 = 0
    b2 = 10
    days = get_cases()[b1:b2]

    inloc = '/condo/swatcommon/common/myrorss/' 
    outloc = '/scratch/mcmontalbano/myrorss/'

    if process == 'extract':
        extract_py(days)

    if proc == 'ACCUMULATE':
        accumulate(startdate, enddate, inloc, outloc)

if __name__ == "__main__":
    main()
