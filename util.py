import stats
import pickle
import sys
import os
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from os import walk
from collections import Counter
import datetime
import matplotlib.pyplot as plt
from numpy.random import default_rng
#rng = default_rng()
from itertools import product

def cartesian_product(list_of_nums):
    pass    

def clear():
    # clear screen
    os.system('clear')

def move_netcdfs_back_one_dir(year):
    # for each storm, move the netcdfs in target_MESH_Max_30min back one directory
    list_of_paths_to_storms_in_year = glob.glob(f'{s.data_path}/{year}/20*/storm*') 
    for path in list_of_paths_to_storms_in_year:
        list_of_paths_to_netcdfs_in_storm = glob.glob(f'{path}/target_MESH_Max_30min/**/**/*.netcdf')
        for path_to_netcdf in list_of_paths_to_netcdfs_in_storm:
            # get path to target_MESH_Max_30min
            path_to_target_MESH_Max_30min = path_to_netcdf.split('/')[:-1]
            # join path to target_MESH_Max_30min on '/'
            path_to_target_MESH_Max_30min_subdir = '/'.join(path_to_target_MESH_Max_30min)
            # get netcdf file 
            netcdf_file = path_to_netcdf.split('/')[-1]
            # save netcdf_file in path_to_target_MESH_Max_30min_subdir
            os.rename(path_to_netcdf, f'{path_to_target_MESH_Max_30min_subdir}/{netcdf_file}')
        # do the same for NSE files
        list_of_paths_to_netcdfs_in_storm = glob.glob(f'{path}/NSE/**/**/*.netcdf')
        for path_to_netcdf in list_of_paths_to_netcdfs_in_storm:
            # get path to NSE
            path_to_NSE = path_to_netcdf.split('/')[:-1]
            # join path to NSE on '/'
            path_to_NSE_subdir = '/'.join(path_to_NSE)
            # get netcdf file 
            netcdf_file = path_to_netcdf.split('/')[-1]
            # save netcdf_file in path_to_NSE_subdir
            os.rename(path_to_netcdf, f'{path_to_NSE_subdir}/{netcdf_file}')
    return None


def load():
    # return ins, outs
    return np.load(
        'datasets/ins_2011_qc.npy'), np.load('datasets/outs_2011_qc.npy')

def compare_maxes(ins, outs, intersect0, intersect1):
    # Assume that input MESH is ins[,,,-1]
    x = ins[intersect0:intersect1, :, :, -1].squeeze()
    y = outs[intersect0:intersect1, :, :, :].squeeze()
    for idx, val in enumerate(x):
        print(val.max(), y[idx].max())
    return

def load(name='outs_2006'):
    return np.load('datasets/{}.npy'.format(name))

def my_filter(ins, outs, max_val=30, min_pixels=50, ID=None):
    # given max_val, filter out ins and outs for only those where
    # MESH_t-30.max() > max_val
    new_ins = []
    new_outs = []
    ins = ins[:, :, :, -1].squeeze()
    outs = outs[:, :, :, -1].squeeze()
    for idx, img in enumerate(ins):
        img = np.where(img < max_val, 0, img)
        img_out = outs[idx]
        count_in = np.count_nonzero(img)
        if count_in > min_pixels:
            new_ins.append(img)
            new_outs.append(img_out)
    ''' FILTER BY MAX - probably not good
    for idx, val in enumerate(ins):
        if val[:,:,-1].max()>max_val:
            print(val[:,:,-1].max())
            new_ins.append(val)
            new_outs.append(outs[idx])
    '''
    new_ins = np.asarray(new_ins)  # use np.copyto instead
    new_outs = np.asarray(new_outs)  # ^
    if ID:
        np.save('datasets/ins_{}'.format(ID), new_ins)
        np.save('datasets/outs_{}'.format(ID), new_outs)
    print(new_ins.shape, new_outs.shape)
    return new_ins, new_outs


def get_cases(year):
    cases = []
    path = '{}/{}'.format(TRAINING_HOME, year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases


def random_sample_npy(ins, outs, train_size=3000, seed=3):
    '''
    Randomly samples a np array.

    @param ins Full set of training set inputs (examples x row x col x chan)
    @param outs Corresponding set of sample (examples x nclasses)
    '''

    # Randomly select a set of example indices using random module
    indices = random.choices(range(ins.shape[0]), k=train_size)

    # The generator will produce a pair of return values: one for inputs and
    # one for outputs
    return ins[indices, :, :, :], outs[indices, :, :, :]

# make hist of maxes


def max_hist(images, title='Max MESH'):
    maxes = []
    for img in images:
        maxes.append(img.max())
    plt.hist(maxes)
    plt.xlim([0, 140])
    plt.ylabel('Number of Images')
    plt.xlabel('MESH (mm)')
    plt.savefig('{}.png'.format(title))
    return None


def check_year(year='2011'):
    '''
    check a year of training data for missing storms
    writes a dataframe to each day in the year,
    returning 'stormID', missing (bool),
    and fields, the array of the common fields (subtract from features to get missing fields)
    '''
    days = get_days(year)
    for day in days:
        check_missing(day)
    return None

# Get the cases in year


def get_days(year):
    cases = []
    path = '{}/{}'.format(TRAINING_HOME, year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases

def open_pickle(file, dir=None):
    # Open pickle (shortened to op for convenience)
    if dir is None:
        return pd.read_pickle(file)
    else:
        return pd.read_pickle('{}/{}'.format(file, dir))


def get(ID):
    # given an ID, return y_true, y_pred, and the scaler for the TESTING set
    return
# check if any of files in storm directory are missing using glob


def check_day(date='20110409'):
    """
    Purpose: check if storm is missing
    Returns:
        @param dataframe for each date
        df attributes: stormID (str), missing (boolean), missing_fields (list)
    Example use:
        df = check_missing('20110409')
    """
    storms_dir = '{}/{}/{}'.format(TRAINING_HOME,
                                   date[:4], date)  # path to storm directory
    storms = sorted(os.listdir(storms_dir))  # list of storms

    # move all csvs into a csv folder
    os.system('mv {}/*.csv /{}/csv'.format(storms_dir, storms_dir))
    # removes dirs like code_index.fam
    df = pd.DataFrame(columns={'storm', 'missing', 'miss_fields'})
    dirs = sorted(glob.glob(
        '{}/{}/{}/*'.format(TRAINING_HOME, date[:4], date)), reverse=True)
    for storm_path in dirs:
        storm = storm_path.split('/')[-1]  # grab last element, the stormID
        if storm[:5] != 'storm':
            continue  # if it's not a storm directory, skip
        missing, fields = check_storm(storm_path)
        diff = []
        res = [element for element in products]
        for f in fields:
            if f in products:
                res.remove(f)
        else:
            fields = fields
        df = df.append({'storm': storm, 'missing': missing,
                       'miss_fields': fields}, ignore_index=True)
    df.to_csv('{}/missingness.csv'.format(storms_dir))  # save
    return df


def check_storm(storm):
    # given a storm path, check if any of the files are missing
    # find all files in storm path
    files = []
    fields = []
    f_times = []
    target = glob.glob(
        '{}/target_MESH_Max_30min/MESH_Max_30min/00.25/*netcdf'.format(storm))
    if target == []:
        return False  # , [] # if no target, reject
    target = target[0]
    # grab the timestamp from the file name
    target_time = str(target.split('/')[-1]).split('.')[0]
    target_time = datetime.datetime.strptime(target_time, "%Y%m%d-%H%M%S")
    f_times.append(target_time)
    swath_files = glob.glob('{}/**/**/*.netcdf'.format(storm))
    for fname in swath_files:
        files.append(fname)
        field = fname.split('/')[-3]  # grab field
        if field not in fields and field in multi_fields:  # collect each field once
            # check that the time is different from the target (i.e. 30 min
            # early)
            ftime = get_time_from_fname(fname)
            f_times.append(ftime)
            if ftime != target_time and fname not in files:
                fields.append(field)
        # NSE data
        NSE_files = glob.glob(
            '{}/NSE/**/**/*.netcdf'.format(storm),
            recursive=True)
        for fname in NSE_files:
            field = fname.split('/')[-3]  # grab field
            if fname not in files:
                files.append(fname)
    if len(files) != 50:
        return True  # , files
    return False  # , files


def check_day_for_missing(day):
    year = day[:4]
    storms = glob.glob('{}/{}/{}/storm*'.format(TRAINING_HOME, year, day))
    missing_list = []
    for storm in storms:
        missing_list.append(check_storm(storm))
    df = pd.DataFrame(missing_list)
    if not os.path.isdir('{}/{}/{}/csv/'.format(TRAINING_HOME, year, day)):
        os.system('mkdir {}/{}/{}/csv'.format(TRAINING_HOME, year, day))
    df.to_csv('{}/{}/csv/missing_{}.csv'.format(TRAINING_HOME, year, day))
    return df


def get_time_from_fname(fname):
    # returns the time form the fnam
    ftime = str(fname.split('/')[-1]).split('.')[0]
    ret_time = datetime.datetime.strptime(ftime, "%Y%m%d-%H%M%S")
    return ret_time


def get_storms(year):
    storms = []
    path = '{}/{}'.format(TRAINING_HOME, year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            storms.append(storm[:8])
    return storms


def load_npy(prefix='outs'):
    '''
    Purpose: load all npys in a directory with the same prefix
             and return the combined npy
     @param prefix - string prefix of .npy files
    '''
    # load npy with prefix and return as single np array (or npy)
    files = glob.glob('{}/datasets/prep/{}*npy'.format(HOME_HOME, prefix))
    names = []
    
    # data is a list containing each npy
    data = [np.load('{}'.format(x)) for x in files]
    # correct :the shape check
    for idx, d in enumerate(data):
        if d.shape[1:3] != (60, 60):
            d = np.reshape(d, (d.shape[0], 60, 60, d.shape[1]))
            data[idx] = d
    # connect npys in a single npy
    data = np.concatenate(data, axis=0)
    return data
    
def remove_missing(year='2011'):
    # simple function to remove storms that are missing
    days = get_days(year)  # retrieve days in training_data
    for day in days:
        check_day(day)
        missing_df_path = '{}/{}/{}/missingness.csv'.format(
            TRAINING_HOME, day[:4], day)
        missing_df = pd.read_csv(missing_df_path)

        for idx, row in missing_df.iterrows():
            stormID = row['storm']
            missing = row['missing']
            if missing:
                os.system('rm -r {}/{}/{}/{}'.format(TRAINING_HOME,
                          day[:4], day, stormID))  # remove missing
                print(' rm -r {}/{}/{}/{}'.format(TRAINING_HOME,
                      day[:4], day, stormID))

def main():
    #ins, outs = load()
    #new_ins, new_outs = my_filter(ins,outs,max_val=40,min_pixels=50,ID='2011_thres_40')
    # print(new_ins.shape,new_outs.shape)
    data = load_npy(prefix='outs')
    np.save('datasets/outs_20220419.npy',data)
    data = load_npy(prefix='ins')
    np.save('datasets/ins_20220419.npy',data)

if __name__ == "__main__":
    main()
