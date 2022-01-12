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
import netCDF4

DATA_HOME = '/condo/swatcommon/common/myrorss'

date1 = sys.argv[1]
date2 = sys.argv[2]

def extract(startdate, enddate, inloc, outloc):
    print('hello')
    startepoch = calendar.timegm((int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), 12, 00, 00))
    endepoch = calendar.timegm((int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), 12, 00, 00))
    field = 'MESH'
    
    #os.chdir(inloc  + startdate[0:4])

    # ACTUAL LOOPING
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


def accumulate(startdate, enddate, inloc, outloc, interval):

    # FILL IN WITH PSEUDO RADAR DATA TO GET EVEN TIMING
    # cmd = 'makeMissingRadar something something something'
    # p = sp.Popen(cmd, shell=True)
    # p.wait()

    # ACCUMULATE EVERY HOUR ON THE HOUR (allows for cleaner accumulations on the yearly scale)
    cmd = 'w2accumulator -g MESHMhtpost -C 1 -t 60 -m MESH -O MESH_Max_'
    p = sp.Popen(cmd, shell=True)
    p.wait()


def main():

    startdate = date1 # dates are input from shipExtract.sh
    enddate = date2

    proc = 'EXTRACT'

    inloc = '/condo/swatcommon/common/myrorss/' 
    outloc = '/scratch/mcmontalbano/myrorss/'

    if proc == 'EXTRACT':
        extract(startdate, enddate, inloc, outloc)

    if proc == 'ACCUMULATE':
        accumulate(startdate, enddate, inloc, outloc)

if __name__ == "__main__":
    main()
