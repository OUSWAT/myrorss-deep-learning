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
import  util
import util, pyarrow
from ast import literal_eval
import settings as s
import shutil
from sklearn.model_selection import train_test_split
# load new datasets
import random
from sys import exit
import argparse
import time
import csv
from glob import glob
from target import Target

def return_targets_in_year(year):
    df = pd.read_feather(f'feather/df_all_{year}_storms_filtered_1.0_deg_0616.feather')
    timedate = npa(df['timedate'].tolist())
    storm_id = npa(df['level_0'].tolist())
    target_paths = [glob(f"{s.data_path}/{x[:4]}/{x.split('-')[0]}/storm{str(y).zfill(4)}/ta*/**/**/*netcdf") for x,y in zip(timedate,storm_id)]
    target_paths = [x for x in target_paths if x != []]
    mesh_list = [read_netcdf(f[0]).max() for f in target_paths]
    file_mesh = [(x[0],y) for x,y in zip(target_paths, mesh_list) if y > 20]
    return file_mesh 

def get_year(year,minMESH):
    t0 = time.time()
    target_mesh_tuples = return_all_targets_in_year(year)  # returns a list of tupled (target_path, mesh_max)
    ins_full = []
    outs_full = []
    for idx, tup in enumerate(target_mesh_tuples):
        target = tup[0]
        mesh = tup[1]
        if mesh < minMESH:
            continue
        storm = '/'.join(target.split('/')[:-4])
        print(f'examining {storm}')
        
        ignore_fields =  s.ignore_fields
        # ah, so the problem was that netcdfs was being reset to the ins netcdfs, which have the storm at -3, whereas the target has it at -4 
        outs=[]
        ins=[]
        # get the mesh first and check if it's non null
        if not isinstance(target, str):
            try:
                target = target[0]
            except:
                print('cant get target')
                continue
        var = read_netcdf(target)
        outs.append(var)
        ############## CHECK & ADD INS ####################
        ins_files = glob(f'{storm}/**/**/*.netcdf') # grab the ins
        ins_files = [x for x in ins_files if x.split('/')[-3] not in ignore_fields]
        # add ignore NSE to create a set without NSE
        ignore_inner_fields = s.ignore_inner_fields #+ [f'{x}/nseanalysis' for x in s.NSE_fields]
        fields = ['/'.join(x.split('/')[-3:-1]) for x in ins_files]
        fields = np.asarray([x for x in fields if x not in ignore_inner_fields]) # sort out the fields we ignore
        if np.unique(fields).shape[0] != len(s.multi_fields+s.degrees): # requires all fields present
            print('skip')
            continue
        ins_files = np.sort(ins_files) # force consistent order
        finished_files = []
        finished_fields = []
        for f in ins_files: # is there a way to do this without a for loop (slowest append method)  
            f_to_inner_field = '/'.join(f.split('/')[:-1])
            if f_to_inner_field not in finished_fields:
                finished_files.append(f)
                finished_fields.append(f_to_inner_field)
        ins = [read_netcdf(f) for f in finished_files]
        outs_full.append(outs)
        ins_full.append(ins)
    ins_full, outs_full = np.asarray(ins_full), np.asarray(outs_full)
    print(ins_full.shape)
    print(outs_full.shape)
    return ins_full, outs_full

def load_date_with_Target(date):
    # gather every target in year using glob'
    date = str(date)
    targets = glob(f'{s.data_path}/{date[:4]}/{date}/**/targ*/**/**/*netcdf')
    if targets == []:
        return None
    target_objects = []
    for target in targets:
        if isinstance(target, list):
            for t in target:
                target_objects.append(Target(t))
        else:
            target_objects.append(Target(target))
    print(len(target_objects))
    target_objects = [x for x in target_objects if x.check_MESH() and x.gather_ins()]
    print(len(target_objects))
    ins_list = [x.ins_list for x in target_objects]
    outs_list = [x.outs_list for x in target_objects]
    ins_list = np.asarray(ins_list)
    outs_list = np.asarray(outs_list)
    np.savez(f'{s.data_path}/{date[:4]}/{date}/sample_mid_intensity.npz',ins=ins_list,outs=outs_list)
    return None

def load_year_with_time_checks(year):
    df = pd.read_feather(f'feather/df_all_{year}_storms_filtered_1.0_deg_0616.feather') 
    timedate = npa(df['timedate'].tolist())
    storm_index = npa(df['level_0'])
    target_paths = [(glob(f"{s.data_path}/{x[:4]}/{x.split('-')[0]}/storm{str(y).zfill(4)}/ta*/**/**/*netcdf"),x) for x,y in zip(timedate,index)] # list of tuples of (timedate, [glob(path)])
    target_paths = [(x,y) for x,y in target_paths if x != []] # remove dirs with empty target
    mesh_list = [read_netcdf(f[0]).max() for f,y in target_paths]
    file_mesh = [(x[0],y) for x,y in zip(target_paths, mesh_list) if y > 20]
    return file_mesh

def load_and_save_npys(year):
    # load a npy for every directory containing a netcdf, save it 
    # then you can just load it into the array when you're making a dataset
    pass

def read_netcdf(f):
    try:
        var = Dataset(f)
    except:
        return np.asarray([-9999]) # use -9999 fill value to indicate missing target 
    var = var.variables[var.TypeName][:,:]
    var = np.where(var<0,0,var)
    return var

def make_dict(keys):
    valid_fields = {}
    for key in keys:
        valid_fields[key] = True
    return valid_fields

def get_storms():
    storms = glob(f'{s.data_path}/20*/20*/csv/storms.feather')
    return storms

def get_date_paths():
    return glob(f'{s.data_path}/20*/20*')

def check_df_with_dict(index, D):
    # Unsure of the purpose of this script
    for field in D.keys():
        # get all files with field in name
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


def load_data_in_year(year='2011',delta=1):
    ins_full = [] # instantiate
    outs_full = []
    df_hail = pd.read_csv(f'csv/{year}_hail_events.csv')
    # get list of storm_dfs
    list_of_paths_to_storm_dfs_in_year = glob(f'{s.data_path}/{year}/20*/feather/storms_filtered_{delta}_deg_0616.feather')
    # get list of days in year with more than 10 hail events
    df_hail = df_hail[df_hail['hail_events'] >= 5]
    # get list of days with more than 10 hail events
    dates_with_sig_hail = [str(date) for date in df_hail['day'].tolist()]
    #print(type(dates_with_sig_hail[0]))
    # get list of paths to storm_dfs with more than 10 hail events
    list_of_paths_to_storm_dfs_with_sig_hail = [f for f in list_of_paths_to_storm_dfs_in_year if f.split('/')[7] in dates_with_sig_hail]
    if list_of_paths_to_storm_dfs_with_sig_hail == []:
        print('no files in list of paths to sig hail')
        return [], [] 
    for path_to_storm_df in list_of_paths_to_storm_dfs_with_sig_hail:
        try:
            df = pd.read_feather(path_to_storm_df)
        except Exception as e:
            print(f'0 {e}')
            continue
        pieces = path_to_storm_df.split('/')
        date_path = '/'.join(pieces[:-3])
        date = int(pieces[-3])
        indices = [x for x in df['level_0'].tolist()]
        for idx in indices:
            # get all netcdf files in storm directory
            path_to_storm = f'{date_path}/{date}/storm{str(idx).zfill(4)}'
            pattern_to_inputs = f'{path_to_storm}/**/**/*.netcdf'
            pattern_to_target = f'{path_to_storm}/target*/**/**/*.netcdf'

            patterns = (pattern_to_inputs, pattern_to_target)
            files_to_netcdfs = []
            for pattern in patterns:
                files_to_netcdfs.extend(glob(pattern))
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

def load_outs_only(year,df_file_ending='_2_deg_0613'): # 0613 refers to the date filter-strat was conceived
    #  IN FUTURE MODULARIZE if you want ppl to think u elegant  
    # in this function we load only the outs
    # saves time, allows quicker data-exploration of
    # data made with new filtering strategies
    outs_full = []
    storm_df_paths = glob(f'{s.data_path}/{year}/20*/feather/storms{df_file_ending}.feather')
    df_hail = pd.read_csv(f'csv/{year}_hail_events.csv')
    df_hail = df_hail[df_hail['hail_events'] >= 2]
    dates_with_sig_hail = [str(date) for date in df_hail['day'].tolist()]
    list_of_paths_to_storm_dfs_with_sig_hail = [f for f in storm_df_paths if f.split('/')[7] in dates_with_sig_hail]
    print(f'this many paths in year:{year}, {len(list_of_paths_to_storm_dfs_with_sig_hail)}')
    for path in storm_df_paths:
        print('in the loop')
        try:
            df = pd.read_feather(path)
        except Exception as e:
            print(f'0 {e}')
            continue
        pieces = path.split('/')
        date_path = '/'.join(pieces[:-3])
        date = int(pieces[-3])
        indices = df['level_0'].tolist()
        for i, idx in enumerate(indices):
            if i > 1000:
                break
            # get all netcdf files in storm directory
            path_to_storm = f'{date_path}/{date}/storm{str(idx).zfill(4)}'
            pattern_to_target = f'{path_to_storm}/target*/**/**/*.netcdf'
            files_to_netcdfs = []
            files_to_netcdfs = glob(pattern_to_target)
            if not files_to_netcdfs:
                print('no files_to_netcdfs')
                continue
            try:
                ins, outs = load_data_with_netcdf_list(files_to_netcdfs, fields=s.targets) # fields controls the fields returned. Here, this will return [], [targets]
                if not outs:
                    continue
                outs_full.append(outs)
            except Exception as e:
                print(e)
                print(f'Error loading {path_to_storm}')
                continue
    return outs_full

def load_data_with_netcdf_list(list_of_netcdf_files,fields=s.multi_fields+s.targets+s.degrees):
    ins = []
    outs = []
    if not list_of_netcdf_files:
        print(f'No files')
        return False   
    valid_fields = make_dict(fields) # valid fields is True if the field has not been added to inputs or outputs
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
   
def load_field_on_day(field='target_MESH_Max_30min',date='20110409'):
    # Given a field and day, load into an npy 
    var_list = []
    storm_df = pandas.read_feather(f'{s.data_path}/{date[:4]}/{date}/csv/storms.feather') 
    df_True = df[df['is_Storm']==True]
    indices = [x for x in df_True['index'].tolist()]
    for idx in indices:
        path_to_storm = f'{date_path}/{date}/storm{str(idx).zfill(4)}'
        pattern_to_var = f'{path_to_storm}/{field}/**/*.netcdf'
        patterns = (pattern_to_var)
        files_to_netcdfs = []
        for pattern in patterns:
            files_to_netcdfs.extend(glob(pattern))
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

def dirty_load(date='20110409',file_pattern='storm*/target*/MESH_Max_30min',field='MESH_Max_30min'):
    var_list = []
    netcdfs = glob(f'{s.data_path}/{date[:4]}/{date}/{file_pattern}/**/*.netcdf')
    for netcdf in netcdfs:
        nc = Dataset(netcdf)
        var = nc.variables[field][:,:]
        var = np.where(var<0,0,var)
        var_list.append(var)
    return np.asarray(var_list)

def random_script_to_check_if_all_the_necessary_fields_are_present_for_a_given_date():
    necessary_fields = s.multi_fields + s.targets + s.degrees + s.NSE_fields
    

def load_year(year):
    #var = dirty_load(file_pattern='MESH/storm*/target*/MESH_Max_30min')
    
    ins, outs = load_data_in_year(year,file_ending='_filtered_2_deg') # given a year, returns full ins/outs
    ins = np.moveaxis(ins, 1, 3) # for some reason we have to switch channels to axis 3
    outs = np.moveaxis(outs, 1, 3) 
    np.save('tmp/ins.npy',ins) # i'm paranoid
    np.save('tmp/outs.npy',outs)
    ins_train, ins_test, outs_train, outs_test = train_test_split(
                                    ins, outs, test_size=0.16, random_state=3)
    ins_train, ins_val, outs_train, outs_val = train_test_split(
                                    ins_train, outs_train, test_size=0.2, random_state=3)
        
    filename = f'../data/npy/dataset_{year}.npz'
    np.savez(f'{filename}', x_train = ins_train, y_train=outs_train, x_val=ins_val, y_val=outs_val, x_test=ins_test, y_test=outs_test)


def get_cats(pivot):
    bins = np.arange(0,50,5)
    new_cats = [pivot]
    for i in range(len(bins)-1):
        new_cat = new_cats[i]*1.1
        new_cats.append(new_cat)
    return new_cats[::-1], bins


def mutate_dataset(ins,outs, cat_maxes):
    max_cats, bins = get_cats(170) # user chosen pivot point (sets a linear with mid point near pivot)
    master_indices = np.asarray([])
    y_maxes = np.asarray([x.max() for x in outs])
    for idx, max_cat in enumerate(max_cats):
        if bins[idx] < 6:
            continue # don't add anything below 6 mm
        low = bins[idx-1]
        high = bins[idx] # mm
        indices = np.where(np.logical_and(y_maxes>low,y_maxes<high)) # remove indices from tuple
        if isinstance(indices, tuple):
            indices = indices[0]
        # randomly select
        k = min(len(indices),int(max_cat))
        return indices, k
        print(f'this is k {k}')
        indices_from_rand = np.asarray(random.choices(indices, k=k)).flatten()      
        # add samples from those indices
        master_indices = np.concatenate((master_indices, indices))
        print(f'and this is the length of master indices {len(master_indices)}')
    max_b = bins[-1]
    print(max_b)
    # add all elements with MESH higher than max_b
    rest = np.where(y_maxes>max_b)
    if isinstance(rest, tuple):
        rest = rest[0]
    rest = rest.flatten()
    master_indices = np.concatenate((master_indices, rest))
    master_indices = master_indices.astype(int)
    print(len(master_indices))
    print(f'length of the rest: {rest}')
    return [],[] 
 #   print(len(np.asarray(master_indices)))
 #   print(master_indices)
 #   return ins[[master_indices]], outs[[master_indices]]

def tr_t_split(ins,outs):
    xtr, xt, ytr, yt = train_test_split(
                                    ins, outs, test_size=0.16, random_state=3)
    xtr, xv, ytr, yv = train_test_split(
                                    xtr, ytr, test_size=0.2, random_state=3)
    datas = [npa(xtr), npa(ytr), npa(xv), npa(yv), npa(xt), npa(yt)]
    for data in datas:
        try:
            print(data.shape)
        except:
            continue
    return xtr, ytr, xv, yv, xt, yt

def npa(x):
    return np.asarray(x)

def get_bins(maxes, bins):
    cats = np.digitize(maxes,bins=bins)
    category = []
    for i in np.arange(len(bins)):
        category.append(len(cats[cats==i]))
    return category

def create_parser():
    parser = argparse.ArgumentParser(description='Load')
    parser.add_argument(
        '-year',
        type=int,
        default=2011,
        help='Are you using the tuner, or just messing around?')
    parser.add_argument(
        '-delta',
        type=float,
        default=0.5,
        help='Degree by which cases were filtered')
    parser.add_argument(
        '-minMESH',
        type=int,
        default=30,
        help='Min MESH needed to add a sample.')
    parser.add_argument(
        '-date',
        type=int,
        default=20110419,
        help='Are you using the tuner, or just messing around?')
    return parser
    

def main():
    parser = create_parser()
    args = parser.parse_args()
    #t0 = time.time()
    #print('getting the year')
    load_date_with_Target(args.date)    
    #ins, outs = get_year(args.year, args.minMESH)
    #print(f'took {time.time()-t0} to load {args.year} to idx 1000')
    #np.save(f'{s.data_path}/npy/shavelike/ins_above_{args.minMESH}_{args.year}.npy',ins)
    #np.save(f'{s.data_path}/npy/shavelike/outs_above_{args.minMESH}_{args.year}.npy',outs)
    #x_new, y_new = load_data_in_year(args.year, args.delta)
    #np.save(f'{s.data_path}/npy/ins_{args.year}.npy',ins)
    #np.save(f'{s.data_path}/npy/outs_{args.year}.npy',outs)
    #xtr, ytr, xv, yv, xt, yt = tr_t_split(x_new, y_new)
    #np.savez(f'../data/datasets/dataset_{args.year}_{args.delta}.npz',
             #x_train=xtr,y_train=ytr,x_val=xv,y_val=yv,x_test=xt,y_test=yt)

if __name__ == '__main__':
    main()
