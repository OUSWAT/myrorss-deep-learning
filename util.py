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

def check_storm(storm_path):
    # given a storm path, check if any of the files are missing
    storm_path = storm_path
    # find all files in storm path
    files = []
    fields = []
    files = glob.glob('{}/target*/**/*.netcdf'.format(storm_path),recursive = True)
    if len(files) > 1:
        print(sorted(files))
        f = files[0] # grab the first file. Maybe sort and grab the latest
    else:
        try:
            f = files[0]
        except:
            missing = True # target is missing 
            fields = ['target_MESH_Max_30min'] # missing field = target
            return missing, fields
    target_time = str(f.split('/')[-1]).split('.')[0] # grab the timestamp from the file name
    target_time = datetime.datetime.strptime(target_time,"%Y%m%d-%H%M%S")#
    input_time = (target_time+datetime.timedelta(minutes=-30)).strftime('%Y%m%d-%H%M%S')
#    print(target_time)
    print("'\n\n\'")
    for fname in glob.glob('{}/**/*.netcdf'.format(storm_path),recursive = True): # recursively returns all files at any depth ending in .netcdf
        field = fname.split('/')[-3] # grab field
        if field in fields:
            fields.append(field) # append all fields. (len = 51)
        # get the input time 
        if field in products and field != 'target_MESH_Max_30min' and field not in NSE_fields:
            f_time = get_time_from_fname(fname)
            if f_time != target_time and fname not in files: # check whether the input time = target time
                files.append(fname)       # if it does, append it
    # get NSE files
    for fname in glob.glob('{}/NSE/**/*.netcdf'.format(storm_path),recursive=True):
        f_time = get_time_from_fname(fname)
        if fname not in files:
            files.append(fname)
        
    if len(files) != len(features)-1: # subtract one because MergedReflectivtyQC is represented by the degrees
        missing = True
    else:
        missing = False
    return missing, fields

def get_time_from_fname(fname):
    # returns the time form the fnam
    ftime = str(fname.split('/')[-1]).split('.')[0]
    ret_time = datetime.datetime.strptime(ftime,"%Y%m%d-%H%M%S")
    return ret_time

def get_storms(date):
    storms_dir = '{}/{}/{}'.format(TRAINING_HOME,date[:4],date) # path to storm directory
    storms = sorted(os.listdir(storms_dir)) # list of storms
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
    remove_missing()

if __name__ == "__main__":
    main()
