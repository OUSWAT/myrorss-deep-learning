import pickle, sys, os, glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from os import walk
from collections import Counter
import datetime
import matplotlib.pyplot as plt

multi_fields = ['MergedLLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Max_30min','MergedMLShear_Min_30min','MergedReflectivityQC','MergedReflectivityQCComposite_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min','target_MESH_Max_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
products = multi_fields + NSE_fields
degrees = ['00.50','  01.00','  01.50','  02.00','  03.00','  04.50','  06.00','  07.50','  09.00 ',' 11.00','  13.00','  15.00 ',' 17.00', ' 00.75', ' 01.25' ,' 01.75', ' 02.75' ,' 04.00 ',' 05.00 ',' 07.00  ','08.50  ','10.00', ' 12.00' ,' 14.00', ' 16.00'  ,'20.00']
features = products + degrees
print(len(degrees) + len(products))

DATA_HOME = '/condo/swatcommon/common/myrorss'
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
HOME_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning'

def clear():
    # clear screen
    os.system('clear')

def get_cases(year):
    cases = []
    path = '{}/{}'.format(TRAINING_HOME,year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases

# make hist of maxes
def max_hist(images, title='Max MESH'):
    maxes = []
    for img in images:
        maxes.append(img.max())
    plt.hist(maxes)
    plt.xlim([0,140])
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
    path = '{}/{}'.format(TRAINING_HOME,year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases

#check if any of files in storm directory are missing using glob
def check_day(date='20110409'):
    """
    Purpose: check if storm is missing
    Returns:
        @param dataframe for each date
        df attributes: stormID (str), missing (boolean), missing_fields (list)
    Example use:
        df = check_missing('20110409')
    """
    storms_dir = '{}/{}/{}'.format(TRAINING_HOME,date[:4],date) # path to storm directory
    storms = sorted(os.listdir(storms_dir)) # list of storms

    os.system('mv {}/*.csv /{}/csv'.format(storms_dir, storms_dir)) # move all csvs into a csv folder
    # removes dirs like code_index.fam
    df = pd.DataFrame(columns={'storm','missing','miss_fields'})
    dirs = sorted(glob.glob('{}/{}/{}/*'.format(TRAINING_HOME, date[:4], date)),reverse = True)    
    for storm_path in dirs:
        storm = storm_path.split('/')[-1] # grab last element, the stormID
        if storm[:5] != 'storm':
            continue # if it's not a storm directory, skip
        missing, fields = check_storm(storm_path)
        diff = []
        res = [ element for element in products]
        for f in fields:
            if f in products:
                res.remove(f)
        else:
            fields = fields
        df = df.append({'storm':storm,'missing':missing,'miss_fields':fields}, ignore_index=True)
    df.to_csv('{}/missingness.csv'.format(storms_dir))# save
    return df

def check_storm(storm):
    # given a storm path, check if any of the files are missing
    # find all files in storm path
    files = []
    fields = []
    f_times = []
    target = glob.glob('{}/target_MESH_Max_30min/MESH_Max_30min/00.25/*netcdf'.format(storm))
    if target == []:
        return False # , [] # if no target, reject 
    target = target[0]
    target_time = str(target.split('/')[-1]).split('.')[0] # grab the timestamp from the file name
    target_time = datetime.datetime.strptime(target_time,"%Y%m%d-%H%M%S")
    f_times.append(target_time)
    swath_files = glob.glob('{}/**/**/*.netcdf'.format(storm))
    for fname in swath_files:
        files.append(fname)
        field = fname.split('/')[-3] # grab field
        if field not in fields and field in multi_fields: # collect each field once
        # check that the time is different from the target (i.e. 30 min early)
            ftime = get_time_from_fname(fname)
            f_times.append(ftime)
            if ftime != target_time and fname not in files:
                fields.append(field)
        # NSE data
        NSE_files = glob.glob('{}/NSE/**/**/*.netcdf'.format(storm), recursive=True)
        for fname in NSE_files:
            field = fname.split('/')[-3] # grab field
            if fname not in files:
                files.append(fname)
    if len(files) != 50:
        return True #, files
    return False #, files

def check_day_for_missing(day):
    year = day[:4]
    storms = glob.glob('{}/{}/{}/storm*'.format(TRAINING_HOME,year,day))
    missing_list = []
    for storm in storms:
        missing_list.append(check_storm(storm))
    df = pd.DataFrame(missing_list)
    if not os.path.isdir('{}/{}/{}/csv/'.format(TRAINING_HOME,year,day)):
        os.system('mkdir {}/{}/{}/csv'.format(TRAINING_HOME,year,day))
    df.to_csv('{}/{}/csv/missing_{}.csv'.format(TRAINING_HOME,year,day))
    return df
 
def get_time_from_fname(fname):
    # returns the time form the fnam
    ftime = str(fname.split('/')[-1]).split('.')[0]
    ret_time = datetime.datetime.strptime(ftime,"%Y%m%d-%H%M%S")
    return ret_time

def get_storms(year):
    storms = []
    path = '{}/{}'.format(TRAINING_HOME,year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            storms.append(storm[:8])
    return storms

def load_npy(prefix='outs'):
    # load npy with prefix and return as single np array (or npy)
    files = os.listdir('{}/{}'.format(HOME_HOME,'datasets'))
    names = []
    for idx, f in enumerate(files[:-1]): # collect all npys fname prefix
        if f[:2] == prefix[:2]:
            names.append(f)
    # data is a list containing each npy
    data = [np.load('{}/datasets/{}'.format(HOME_HOME,x)) for x in names]
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
    days = get_days(year) # retrieve days in training_data
    for day in days:
        check_day(day)
        missing_df_path = '{}/{}/{}/missingness.csv'.format(TRAINING_HOME, day[:4], day)
        missing_df = pd.read_csv(missing_df_path)
        print(missing_df)

        for idx, row in missing_df.iterrows():
            stormID = row['storm']
            missing = row['missing']
            if missing == True:
                os.system('rm -r {}/{}/{}/{}'.format(TRAINING_HOME, day[:4], day, stormID)) # remove missing
                print(' rm -r {}/{}/{}/{}'.format(TRAINING_HOME, day[:4], day, stormID)) 
def main():
    days = get_cases('2011')
    print('days',days)
    old_df = []
    for day in days:
        print(day)
        df = check_day_for_missing(day)  
        if old_df == []:
            old_df = df
        else:
            df = df.append(old_df, ignore_index=True)
            old_df = df
    df.to_csv('/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning/missing.csv')
if __name__ == "__main__":
    main()
