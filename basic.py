# Author: Michael Montalbano
# Title: basic.py
#
# Purpose: Extract data fields from tars
# to compose database of netcdfs organized  by date
# and create training samples stored in directories
# of form YYYY/YYMMDD/stormXX/

import os
import sys
import datetime
import random
from os import environ
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

lscratch = os.environ.get('LSCRATCH')
inmost_path = '/condo/swatcommon/common/myrorss'
lon_NW, lat_NW, lon_SE, lat_SE = -130.005, 55.005, - \
    59.995, 19.995  # see MYRORSS readme at https://osf.io/kbyf7/
# don't include reflectivity (20 levels), only include Ref-0C

def get_iterfields(data_path, final_desired_fields):
    # given teh final desired fields
    # if present, skip extraction of them (as they are already present)
    # otherwise, return unaccumulated version of field strings in list
    iterfields = []
    for field in final_desired_fields:
        files = glob.glob(f'{data_path}/{field}/**/*netcdf')
        if files == []: # if empty
            iterfields.append(field.replace('_Max_30min',''))
    # check for MergedQCReflectivity
    ref_files = glob.glob(f'{data_path}/MergedReflectivityQC/**/*netcdf')
    if ref_files == []:
        iterfields.append('MergedReflectivityQC')
    # check for target
    target_files = glob.glob(f'{data_path}/target*/**/**/*netcdf')
    if target_files == [] and 'MESH' not in iterfields:
        iterfields.append('MESH')
    return iterfields

def extract(date, in_path, out_path, iterfields):
    '''
    extract to lscratch; (then it is localmaxed and accumulated and cropped into train_home by other functions) 
    '''
    year = date[:4]
    cmd3 = f'mkdir -p {out_path}/csv'
    p = sp.Popen(cmd3, shell=True)
    p.wait()
    # iterate over fields
    final_path = f'{s.data_path}/{year}/{date}'
    for field in iterfields:
        files_of_field_in_date = glob.glob(f'{final_path}/storm*/{field}/**/**/*netcdf')
        if files_of_field_in_date: 
            continue
        # 1999 to 2003 saved in Azshear dir in stead of azimuthal_shear_only, so try both
        if field == 'MergedLLShear' or field == 'MergedMLShear':
            if int(year) >= 2002:
                cmd = f"tar -xf {in_path}/{year}/azimuthal_shear_only/{date}.tar -C {out_path} --wildcards '{field}' --exclude='MergerInputRadarsTable'"
                p = sp.Popen(cmd, shell=True)
                p.wait()
            else:
                cmd = f'tar -xf {in_path}/{year}/Azshear/{date}.tar -C {out_path} --wildcards "{field}"'
                p = sp.Popen(cmd, shell=True)
                p.wait()
        else:
            cmd = f"tar -xf {in_path}/{year}/{date}.tar -C {out_path} --wildcards '{field}' --exclude='MergerInputRadarsTable'"
            p = sp.Popen(cmd, shell=True)
            p.wait()
        # pointing to the field in the lscratch directory
        field_path = f'{out_path}/{field}'
        #  list the direotories within the field
        all_files = glob.glob(f'{field_path}/**/*.gz')
        for full_file_path in all_files:
            pieces = full_file_path.split('/')
            nc_file = pieces[-1][:-3]
            # convert from .gz to .netcdf
            gz = gzip.open(full_file_path)
            data = netCDF4.Dataset('dummy', mode='r', memory=gz.read())  # open using netCDF4
            # use xarray backend to convert correctly (idk)
            dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
            dataset.to_netcdf(f'{out_path}/{field}/{nc_file}')  # write
    return None

def localmax(in_path, out_path):
    '''
    Given a day in YYYYMMDD, run w2localmax
    '''
    cmd1 = f'mkdir -p {out_path}'
    cmd2 = f'mkdir -p {out_path}/csv'
    p = sp.Popen(cmd1, shell=True)
    p.wait()
    p = sp.Popen(cmd2, shell=True)
    p.wait()
    cmd1 = f'makeIndex.pl {in_path} code_index.xml'
    cmd2 = f'w2localmax -i {in_path}/code_index.xml -I MergedReflectivityQCComposite -o {in_path} -s -d "40 60 5"'
    cmd3 = f'makeIndex.pl {in_path} code_index.xml'
    cmd4 = f'w2table2csv -i {in_path}/code_index.xml -T MergedReflectivityQCCompositeMaxFeatureTable -o {out_path}/csv -h'
    p = sp.Popen(cmd1, shell=True)
    p.wait()
    p = sp.Popen(cmd2, shell=True)
    p.wait()
    p = sp.Popen(cmd3, shell=True)
    p.wait()
    p = sp.Popen(cmd4, shell=True)
    p.wait()
    return None


def get_storm_info(out_path):
    case_df = pd.DataFrame(
        columns={
            "timedate",
            "Latitude",
            "Longitude",
            "Storm",
            "Reflectivity",
            "is_Storm"})
    i = +1
    delta = 0.15
    # builds dataframe of case centers
    files = sorted(
        glob.glob(
            f'{out_path}/csv/MergedReflectivityQCCompositeMaxFeature**.csv',
            recursive=True))
    for idx, f in enumerate(files):
        timedate = f[-19:-4]
        minutes = timedate[-4:-2]
        if (int(minutes) >= 28 and int(minutes) <= 32) or (int(minutes) >= 58 and int(
                minutes) <= 60) or (int(minutes) >= 0 and int(minutes) <= 2):
            # if (minutes == '30' or minutes =='00') and idx != 0 and idx !=
            # length:
            df = pd.read_csv(
                f'{out_path}/csv/MergedReflectivityQCCompositeMaxFeatureTable_{timedate}.csv')
            if df.empty:
                pass
            else:
                # List of valid clusters
                valid_clusters = {}
                keys = range(df.shape[0])
                for i in keys:
                    valid_clusters[i] = True  # initialize valid with Trues
                # find max
                for idx, val in enumerate(
                        df["MergedReflectivityQCCompositeMax"]):
                    if not valid_clusters[idx]:
                        continue
                    if val < 40 or df['Size'].iloc[idx] < 20:
                        valid_clusters[idx] = False
                        continue
                    lat = df['#Latitude'].iloc[idx]
                    lon = df['Longitude'].iloc[idx]
                    latN = lat + delta
                    latS = lat - delta
                    lonW = lon - delta
                    lonE = lon + delta
                    # Don't include clusters too close to domain edge
                    if latN > (
                        lat_NW -
                        0.16) or latS <= (
                        lat_SE +
                        0.16) or lonW < (
                        lon_NW +
                        0.16) or lonE >= (
                        lon_SE -
                            0.16):
                        valid_clusters[idx] = False
                        continue
                    for idx2, val2 in enumerate(
                            df["MergedReflectivityQCCompositeMax"]):
                        if idx2 == idx or valid_clusters[idx2] == False:
                            continue
                        if df['Size'].iloc[idx2] < 20 or val2 < 40:
                            valid_clusters[idx2] = False
                            continue
                        lat2 = df['#Latitude'].iloc[idx2]
                        lon2 = df['Longitude'].iloc[idx2]
                        if lat2 < latN and lat2 > latS and lon2 > lonW and lon2 < lonE:
                            if val2 > val:
                                valid_clusters[idx] = False
                            else:
                                valid_clusters[idx2] = False
                # valid_clusters is complete
                # # add valid rows to case_dfS
                for key in valid_clusters.keys():
                    if not valid_clusters[key]:
                        continue
                    else:
                        row_idx = key
                        try:
                            row = {
                                "timedate": timedate,
                                "Latitude": df['#Latitude'].iloc[row_idx],
                                "Longitude": df['Longitude'].iloc[row_idx],
                                'Storm': df['RowName'].iloc[row_idx],
                                'Reflectivity': df['MergedReflectivityQCCompositeMax'].iloc[row_idx],
                                'is_Storm': False}
                        except BaseException:
                            pass
                        case_df.loc[len(case_df.index)] = row
    case_df = case_df.sort_values(['timedate'])
    case_df.reset_index().to_feather(f'{out_path}/csv/storms.feather')
    return case_df

def accumulate(iterfields, in_path, out_path, MHT=True):
    '''
    Accumulates (uncropped) fields and stores them in lscratch
    '''
    cmd = 'makeIndex.pl {} code_index.xml'.format(in_path)
    p = sp.Popen(cmd, shell=True)
    p.wait()
    # dont accumulate reflectivity
    # shallow copy
    fields = iterfields.copy()
    try: 
        fields.remove('MergedReflectivityQC')
    except:
        pass 
    for field in fields:
        print(f'accumulating {field}')
        if field == 'MESH' and MHT == True:
            print('accumulating MESH')
            cmd =  f'w2accumulator -i {in_path}/code_index.xml -g {field} -o {out_path} -C 1 -t 30 -m {field} --verbose=4 -Q ablob:7:10:10,mht:3:2:1500:5:5'
        else:
            cmd = f'w2accumulator -i {in_path}/code_index.xml -g {field} -o {out_path} -C 1 -t 30 --verbose="severe"'
        p = sp.Popen(cmd, shell=True)
        p.wait()
        findMin = True  # I don't care about the min shear, for now
        if field[8:] == 'Shear' and findMin:  # so skip this loop
            cmd = f'w2accumulator -i {in_path}/code_index.xml -g {field} -o {in_path} -C 3 -t 30 --verbose="severe"'
            p = sp.Popen(cmd, shell=True)
            p.wait()
    return None

def cropconv(case_df, iterfields, in_path, out_path):
    '''
    Crops samples to 60x60 domain for input&target
    '''
    cmd = f'mkdir {out_path}'
    p = sp.Popen(cmd,shell=True)
    p.wait()
    list_of_products = [f + '_Max_30min' for f in iterfields if f != 'MergedReflectivityQC'] # set up the final products to be cropped
    print('entering crop conv loop')
    print(f' and this is the df {case_df}')
    for idx, row in case_df.iterrows():
        storm_path = f'{out_path}/storm{str(idx).zfill(4)}'
        print(f'in the loop with {storm_path} and products {list_of_products}')
        if row['is_Storm'] == False:
            print('the storm is false so I skip it')
            continue
        lon = row['Longitude']
        lat = row['Latitude']
        delta = 0.15
        lonNW = lon - delta
        latNW = lat + delta
        lonSE = lon + delta
        latSE = lat - delta
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time1 = (
            date_1 -
            datetime.timedelta(
                minutes=3,
                seconds=30)).strftime('%Y%m%d-%H%M%S') # time1 is date1 - 4 minutes
        time2 = (
            date_1 +
            datetime.timedelta(
                minutes=3,
                seconds=30)).strftime('%Y%m%d-%H%M%S')
        # crop input
        cmd = f"makeIndex.pl {in_path} code_index.xml {time1} {time2}"
        p = sp.Popen(cmd, shell=True)
        p.wait()
        for p in list_of_products:
            product_path = f'{storm_path}/{p}'
            netcdfs_in_path = glob.glob(f'{product_path}/**/*netcdf')
            if netcdfs_in_path: 
                print(f'skipping as there are netcdfs in for {p}')
                continue
            cmd = f'w2cropconv -i {in_path}/code_index.xml -I {p} -o /{out_path}/storm{str(idx).zfill(4)} -t "{latNW} {lonNW}" -b "{latSE} {lonSE}" -s "0.005 0.005" -R -n --verbose="severe"'
            print(f'cropping product {p} with command {cmd}')
            p = sp.Popen(cmd, shell=True)
            p.wait()
        # get starting time and ending time for target -
        time1 = (
            date_1 +
            datetime.timedelta(
                minutes=25,
                seconds=30)).strftime('%Y%m%d-%H%M%S')
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (
            date_1 +
            datetime.timedelta(
                minutes=8,
                seconds=30)).strftime('%Y%m%d-%H%M%S')
        # CROP TARGET
        cmd1 = f'makeIndex.pl {in_path} code_index.xml {time1} {time2}'
        cmd2 = f'w2cropconv -i {in_path}/code_index.xml -I MESH_Max_30min -o {out_path}/storm{str(idx).zfill(4)}/target_MESH_Max_30min -t "{latNW} {lonNW}" -b "{latSE} {lonSE}" -s "0.005 0.005" -R -n --verbose="severe"'
        print(f'this is cmd2 to crop the target {cmd2}')
        p = sp.Popen(cmd1, shell=True) # make index for target (+25 to +33 min)
        p.wait()
        p = sp.Popen(cmd2, shell=True) # crop target to storm dir in out_path
        p.wait()
        # NSE
        # crop for 30 min prior to 30 min ahead
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time1 = (
            date_1 +
            datetime.timedelta(
                minutes=-
                40,
                seconds=00)).strftime('%Y%m%d-%H%M%S')
        time2 = (
            date_1 +
            datetime.timedelta(
                minutes=5,
                seconds=00)).strftime('%Y%m%d-%H%M%S')
        cmd3 = f'makeIndex.pl {in_path} code_index.xml {time1[:11]}0000 {time2[:11]}0000'
        p = sp.Popen(cmd3, shell=True)
        p.wait()
        for nse_field in s.NSE_fields:
            #cmd1 = f'makeIndex.pl {OUT_HOME}/NSE code_index.xml {time1} {time2}'
            cmd = f'w2cropconv -i {in_path}/code_index.xml -I {nse_field} -o {out_path}/storm{str(idx).zfill(4)} -t "{latNW} {lonNW}" -b "{latSE} {lonSE}" -s "0.005 0.005" -R -n --verbose="severe"'
            p = sp.Popen(cmd, shell=True)
            p.wait()
    return None 

def extract_NSE(date, out_path):
    # grab netcdf.gz files from NSE_home
    for field in s.NSE_fields: # loop through list of str NSEfields
        cmd = f'mkdir -p {out_path}/{field}'
        p = sp.Popen(cmd, shell=True)
        p.wait()
        nse_files = glob.glob(f'{s.nse_path}/{date[:4]}/{date}/NSE/{field}/**/*.gz') # grab gz files from storage
        # unzip into netcdf and place in lscratch 
        for nse_file in nse_files: # loop through files
            nse_file_without_the_gz = nse_file[:-3] # remove the .gz
            pieces = nse_file_without_the_gz.split('/')
            field = pieces[7]
            nc_file = pieces[-1]
            os.system(f'mkdir -p {out_path}/{field}/nseanalysis')
            gz = gzip.open(nse_file)
            data = netCDF4.Dataset('dummy', mode='r', memory=gz.read())
            dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
            dataset.to_netcdf(f'{out_path}/{field}/nseanalysis/{nc_file}')
            gz.close()
    return None

def test_MHT_settings(date, out_path, MHT_settings = 'blob:10:30:,mht:1:2:168:5:5:l', final_dir='MHT', in_path=lscratch, base_accum=False):
    iterfields = ['MESH'] 
    extract(date, s.inmost_path, lscratch, iterfields = iterfields)
    cmd = 'makeIndex.pl {} code_index.xml'.format(in_path)
    p = sp.Popen(cmd, shell=True)
    p.wait()
    # save MHT settings in txt file
    cmd = f'mkdir -p {out_path}/{final_dir}'
    p = sp.Popen(cmd, shell=True)
    p.wait()
    with open(f'{out_path}/{final_dir}/settings.txt','w') as config_file:
        config_file.write(MHT_settings)
    for field in iterfields:
        cmd =  f'w2accumulator -i {in_path}/code_index.xml -g {field} -o {out_path}/{final_dir} -C 1 -t 30 -Q {MHT_settings} --verbose="severe"'
        if base_accum:
            cmd = f'w2accumulator -i {in_path}/code_index.xml -g {field} -o {out_path} -C 1 -t 30 --verbose'
        p = sp.Popen(cmd, shell=True)
        p.wait()
    # open storms.feather for day
    storms_df = pd.read_csv(f'{s.data_path}/{date[:4]}/{date}/csv/storms.csv')
    # crop MESH 
    products = ['MESH_Max_30min']
    for idx, row in storms_df.iterrows():
        if row['is_Storm'] == False:
            continue
        crop(row, in_path, out_path, products, idx)

    
def crop(row, in_path, out_path, products, idx):
    # portable crop function, to use when testing MHT settings, etc.
    lon = row['Longitude']
    lat = row['Latitude']
    delta = 0.15
    lonNW = lon - delta
    latNW = lat + delta
    lonSE = lon + delta
    latSE = lat - delta
    time1 = row['timedate']
    date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
    time1 = (
        date_1 -
        datetime.timedelta(
            minutes=3,
            seconds=30)).strftime('%Y%m%d-%H%M%S') # time1 is date1 - 4 minutes
    time2 = (
        date_1 +
        datetime.timedelta(
            minutes=3,
            seconds=30)).strftime('%Y%m%d-%H%M%S')
    # crop input
    cmd = f"makeIndex.pl {in_path} code_index.xml {time1} {time2}"
    p = sp.Popen(cmd, shell=True)
    p.wait()
    for p in products:
        cmd = f'w2cropconv -i {in_path}/code_index.xml -I {p} -o /{out_path}/storm{str(idx).zfill(4)} -t "{latNW} {lonNW}" -b "{latSE} {lonSE}" -s "0.005 0.005" -R -n '
        p = sp.Popen(cmd, shell=True)
        p.wait()
    return None

def main():
    date = sys.argv[1]
    year = date[:4] 
    out_path = f'/condo/swatwork/mcmontalbano/MYRORSS/data/{year}/{date}'
    alt_path = f'/condo/swatwork/mcmontalbano/MYRORSS/data/{year}/{date}/MESH'
    top_events_df = pd.read_feather(f'csv/top_events_{year}.feather') # load top events df with lats/lons/mags/times for filtering
    df_hail = pd.read_csv(f'csv/{year}_hail_events.csv') # load hail events with n_hail for day for filtering
    try: 
        df_hail = df_hail[df_hail['day'] == int(date)]
        hail_events = int(df_hail['hail_events'])
    except:
        print('exception')
        sys.exit()
    if hail_events < 15:
        print('too few hail events; exiting')
        sys.exit()    
    iterfields = get_iterfields(data_path=out_path, final_desired_fields=s.fields_accum) 
    #iterfields = ['MESH']
    extract(date, s.inmost_path, lscratch, iterfields = iterfields) # extract froom storage to lscratch
    #extract(date, s.inmost_path, lscratch, ['MESH'])
    # check if storms.feather exists
    # if not, localmax and make storms.feather
    try:  
        storms_before = pd.read_feather(f'{out_path}/csv/storms.csv') # see if feather exists already
        print(f'this is the df {storms_before}')
    except:
        localmax(lscratch, out_path) # if not localmax and make feather
        storms_before = get_storm_info(out_path)    
    date_str = str(date)
    print(storms_before)
    storms_after = switch_off_storms(storms_before, top_events_df, date_str) # filter 
    print(storms_after)
    extract_NSE(date, out_path = lscratch) # extract NSE, only 24 a day and no accum required
    accumulate(iterfields=iterfields, in_path=lscratch, out_path=lscratch,MHT=True) 
    cropconv(storms_after, iterfields, in_path=lscratch, out_path=out_path)
    
if __name__ == "__main__":
    main()
