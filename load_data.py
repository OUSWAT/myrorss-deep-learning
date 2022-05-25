############################################
# Author: Michael Montalbano
#
# Purpose: Load data from the MYRORSS directories on OSCER
#          into .npys.
#
# Returns: 
# @param ins - the inputs for the dataset
# @param outs - the outputs for the dataset 
#
# Functions: get_cases, get_training_cases, load_while() 

import datetime, sys, os, random
from os import walk
import numpy as np
import pandas as pd
from ast import literal_eval
from netCDF4 import Dataset
import glob, util
import util, pyarrow
from ast import literal_eval
import settings as s
import shutil
from sklearn.model_selection import train_test_split
# load new datasets

def make_dict(keys):
    valid_fields = {}
    for key in keys:
        valid_fields[key] = True
    return valid_fields

def load_data_in_year(year='2011',month=None):
    ins_full = [] # instantiate
    outs_full = []
    df_hail = pd.read_csv(f'csv/{year}_hail_events.csv')
    # get list of storm_dfs
    if month:
        list_of_paths_to_storm_dfs_in_year = glob.glob(f'{s.data_path}/{year}/{year}{month}*/csv/storms.feather')
    else:
        list_of_paths_to_storm_dfs_in_year = glob.glob(f'{s.data_path}/{year}/20*/csv/storms.feather')
    # get list of days in year with more than 10 hail events
    df_hail = df_hail[df_hail['hail_events'] >= 10]
    # get list of days with more than 10 hail events
    dates_with_sig_hail = [str(date) for date in df_hail['day'].tolist()]
    #print(type(dates_with_sig_hail[0]))
    # get list of paths to storm_dfs with more than 10 hail events
    list_of_paths_to_storm_dfs_with_sig_hail = [f for f in list_of_paths_to_storm_dfs_in_year if f.split('/')[7] in dates_with_sig_hail]
    print(list_of_paths_to_storm_dfs_with_sig_hail)
    for path_to_storm_df in list_of_paths_to_storm_dfs_with_sig_hail:
        try:
            df = pd.read_feather(path_to_storm_df)
        except Exception as e:
            print(f'0 {e}')
            continue
        pieces = path_to_storm_df.split('/')
        print(pieces)
        date_path = '/'.join(pieces[:-3])
        date = int(pieces[-3])
        print(f'this is the {date_path}')
        print(f'the date is {date}')
        df_True = df[df['is_Storm']==True]
        indices = [x for x in df_True['index'].tolist()]
        print(f'indices: ',indices)
        for idx in indices:
            # get all netcdf files in storm directory
            path_to_storm = f'{date_path}/{date}/storm{str(idx).zfill(4)}'
            pattern_to_inputs = f'{path_to_storm}/**/**/*.netcdf'
            pattern_to_target = f'{path_to_storm}/target*/**/**/*.netcdf'
            patterns = (pattern_to_inputs, pattern_to_target)
            files_to_netcdfs = []
            for pattern in patterns:
                files_to_netcdfs.extend(glob.glob(pattern))
            if not files_to_netcdfs:
                continue
            try:
                ins, outs = load_data_with_netcdf_list(files_to_netcdfs)
                if not ins or not outs:
                    continue
                ins_full.append(ins)
                outs_full.append(outs)
            except Exception as e:
                print(e)
                print(f'Error loading {path_to_storm}')
                continue
    return ins_full, outs_full

def load_data_with_netcdf_list(list_of_netcdf_files):
    ins = []
    outs = []
    if not list_of_netcdf_files:
        print(f'No files')
        return False   
    valid_fields = make_dict(s.multi_fields + s.targets + s.degrees) # valid fields is True if the field has not been added to inputs or outputs
    # step through valid_fields, then find the netcdfs that have the field, then add to inputs or outputs
    for field in valid_fields:
        # get all files with field in name
        list_of_netcdf_files_with_field = [f for f in list_of_netcdf_files if field in f]
        # if empty, continue
        if len(list_of_netcdf_files_with_field) == 0:
            return [], [] 
        # if more than 1 file, check if field is NSE_field
        if len(list_of_netcdf_files_with_field) > 1:
            if field in s.NSE_fields:
                    # if so, add the file with the earliest timedate
                list_of_netcdf_files_with_field.sort(key=lambda x: x.split('/')[-1].split('.')[-2])
                list_of_netcdf_files_with_field = list_of_netcdf_files_with_field[:1]
        # get filename from list_of_netcdf_files_with_field
        file_path = list_of_netcdf_files_with_field[0]
        if valid_fields[field] and field == 'MESH_Max_30min': # check if field has been loaded and that product is MESH (i.e. not the target)
            nc = Dataset(file_path)
            var = nc.variables[field][:,:]
            ins.append(var)
            valid_fields[field] = False
        elif valid_fields[field] and field == 'target_MESH_Max_30min':
            nc = Dataset(file_path)
            var = nc.variables['MESH_Max_30min'][:,:]
            outs.append(var)
            valid_fields[field] = False
        elif valid_fields[field] and file_path.split('/')[-3] == 'MergedReflectivityQC': #  check for valid_fields[degree], and if so, check that
            # Improve: make dict linking valid_fields keys to variable of key,
            # i.e. 01:00 : MergedRefQC, MESH_Max_30min : MESH_Max_30min
            nc = Dataset(file_path)
            var = nc.variables["MergedReflectivityQC"][:,:]
            ins.append(var)
            valid_fields[field] = False
        else:
            nc = Dataset(file_path)
            var = nc.variables[field][:,:]
            ins.append(var)
            valid_fields[field] = False
    if all(value == False for value in valid_fields.values()):
        print(f'returned True for') 
        return ins, outs
    else:
        true_keys = [x for x in valid_fields.keys() if valid_fields[x] == True]
        print(f'Missing the following keys: {true_keys}')
        return [], []
    
def check_files_for_missing_fields(files, fields):
    valid_fields = make_dict(fields) # valid fields is True if the field has not been added to inputs or outputs
    if not files:
        print('F:0')
        return False
    storm_path = files[0]
    for field in valid_fields:
        # get all files with field in name
        list_of_netcdf_files_with_field = [f for f in files if field in f]
        # if empty, continue
        if len(list_of_netcdf_files_with_field) == 0:
            continue
        # if more than 1 file, check if field is NSE_field
        if len(list_of_netcdf_files_with_field) > 1:
            if field in s.NSE_fields:
                # if so, add the file with the earliest timedate 
                list_of_netcdf_files_with_field.sort(key=lambda x: x.split('/')[-1].split('.')[-2])
                list_of_netcdf_files_with_field = list_of_netcdf_files_with_field[:1]
                # delete the other files from the directory
                for f in list_of_netcdf_files_with_field[1:]:
                    os.remove(f)
        # get filename from list_of_netcdf_files_with_field
        file_path = list_of_netcdf_files_with_field[0]
        if valid_fields[field] and field == 'MESH_Max_30min': # check if field has been loaded and that product is MESH (i.e. not the target)
            valid_fields[field] = False  
        elif valid_fields[field] and field == 'target_MESH_Max_30min':
            valid_fields[field] = False
        elif valid_fields[field] and file_path.split('/')[-3] == 'MergedReflectivityQC': #  check for valid_fields[degree], and if so, check that
            valid_fields[field] = False   # note that MergedRefQC is not a valid field, as the degrees stand in as fields of MRQC
        else:
            valid_fields[field] = False # otherwise, handle in the basic way for most common file pattern (i.e. NSE fields and most swaths)
    if all(value == False for value in valid_fields.values()):
        print('T0')
        return True
    else:
        # print the storm path and the dict of valid_fields
        print('F1')
        return False

def main():
    year = sys.argv[1]
    month = sys.argv[2]
    choice = sys.argv[3]
    print(f'year {year} with type {type(year)} and choice {choice} with {type(choice)}')
    if choice == 'load':
        ins, outs = load_data_in_year(year,month=month)
        ins_train, ins_test, outs_train, outs_test = train_test_split(
                                        ins_train, outs_train, test_size=0.16, random_state=3)
        if month is not None:
            filename = f'dataset_{year}{month}.npz'
        else: filename = f'dataset_{year}.npz'
 
        np.savez(f'{filename}', x_train=ins_train, x_test=ins_test,
                                        y_train=outs_train, y_test=outs_test)
    
if __name__ == '__main__':
    main()
