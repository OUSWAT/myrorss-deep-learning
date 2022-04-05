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
import util

home = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning'
DATA_HOME = '/condo/swatcommon/common/myrorss'
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
multi_fields = ['MergedLLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Max_30min','MergedMLShear_Min_30min','MergedReflectivityQC','MergedReflectivityQCComposite_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min']
swath_fields = ['MergedLLShear_Max_30min','MergedMLShear_Max_30min','MergedReflectivityQCComposite_Max_30min','MESH_Max_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
degrees = ['06.50', '02.50', '05.50', '01.50', '08.00', '19.00', '00.25', '00.50', '09.00', '18.00', '01.25', '20.00', '04.50', '03.50', '02.25', '07.50', '07.00', '16.00', '02.75', '12.00', '03.00', '04.00', '15.00', '11.00', '01.75', '10.00', '00.75', '08.50', '01.00', '05.00', '14.00', '13.00', '02.00', '06.00', '17.00']

targets = ['target_MESH_Max_30min']
products = multi_fields  + targets
field_list =  swath_fields
# INPUT VARIABLES HERE
# make this more elegant, to input from the shell. check e0xtract.py
#year=str(sys.argv[1])
#b0=int(sys.argv[2])
#b1=int(sys.argv[3])
year='2011'

# Get the cases in year
def get_cases(year):
    cases = []
    path = '{}/{}'.format(TRAINING_HOME,year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases

def info(df,i):
    # returns storm path for an index i in dataframe df
    for idx, row in df.iterrows():
        if idx == i:
            return row['storm_path'], row['features']
       
def load_data_from_df(df):
    df_qc = df[df['features']>37]
    df_qc = df_qc[df_qc['features']<50]
    ins_full = np.asarray([])
    outs_full = np.asarray([])
    
    for idx, row in df_qc.iterrows():
        n = int(row['features'])
        storm = row['storm_path']
        if n < 36:
            continue # skip
        ins = np.asarray([]) 
    #    outs_full = np.asarray([])        
        nc_files = []
        missing_fields = []
        #outs = []
        # grab reflcectivity first        
        field = 'MergedReflectivityQC'
        for deg in degrees:
            nc_file = glob.glob('{}/{}/{}/*netcdf'.format(storm,field,deg))
            if nc_file == []:
                missing_fields.append(deg)
            else:
                nc_files.append(nc_file[0])
            # check for multiple files or 
        for field in field_list:
            missing_fields = []
            nc_file = glob.glob('{}/{}/**/*netcdf'.format(storm,field))
            if nc_file == []:
                nc_file = glob.glob('{}/NSE/{}/**/*netcdf'.format(storm,field))
                if nc_file == []: 
                    missing_fields.append(field)
                    continue
                else:
                    nc_files.append(nc_file[0])
            nc_files.append(nc_file[0])
        if missing_fields != []:
            break
        for f in nc_files:
            nc = Dataset(f)
            var = nc.variables[f.split('/')[-3]][:,:]
            var = np.where(var<-20,0,var)
            np.copyto(ins,var)
        np.copyto(ins_full,ins,cast='same_kind')
       # ins_full.append(np.asarray(ins))
        # Append out to outs
        target_file = glob.glob('{}/target_MESH_Max_30min/MESH_Max_30min/**/*.netcdf'.format(storm))
        f = target_file[0]        
        if f == []:
            ins_full.pop()
            break
        nc = Dataset(f)
        var = nc.variables['MESH_Max_30min'][:,:]
        var = np.where(var<0,0,var)
        np.copyto(outs_full, var,cast='same_kind')
       # outs_full.append(var)
    return ins_full, outs_full

def get_df_shapes(year='2011'):
    # check shape of ins for each storm_path
    # return df of shape and path
    shape_list = []
    i=0
    days = get_cases(year)
    df = pd.DataFrame(columns={'storm_path','features','feature_list'})
    for idx, day in enumerate(days):
        year = day[:4]
        fields = []
        storms = glob.glob('{}/{}/{}/storm*'.format(TRAINING_HOME,year,day))
        for storm in storms:
            ins = []
            files = get_storm_files(storm)
            if files == []:
                row = {'storm_path':storm, 'features':0, 'feature_list':[]}
                df.loc[i] = row
                i+=1
                continue
            for f_name in files:
                fields.append(f_name.split('/')[3])
            row = {'storm_path':storm,'features':len(files),'feature_list':files,'fields':fields}
            df.loc[i] = row
            i+=1
    df.to_csv('{}/csv/{}_missing_fields.csv'.format(home,year))
    return df

def get_storm_files(storm):
    # given a storm path, return the files as a list of strings
    f_times = [] 
    files = []
    fields = []
    target = glob.glob('{}/target_MESH_Max_30min/MESH_Max_30min/00.25/*netcdf'.format(storm))
    if target == []:
        return [] # return empty if no target
    target = target[0]
    target_time = str(target.split('/')[-1]).split('.')[0] # grab the timestamp from the file name
    target_time = datetime.datetime.strptime(target_time,"%Y%m%d-%H%M%S")
    f_times.append(target_time)
    files.append(target) # append the target as the first file
    swath_files = glob.glob('{}/**/**/*.netcdf'.format(storm))
    for fname in swath_files:
        files.append(fname)
        field = fname.split('/')[-3] # grab field
        if field not in fields and field in multi_fields: # collect each field once and treat multi_fields different by:
            # Check that the time is different from the target (i.e. 30 min early)
            ftime = util.get_time_from_fname(fname)
            f_times.append(ftime)
            if ftime != target_time and fname not in files: # target time is different, and the file is new
                fields.append(field) # add to fields
                files.append(fname)  # add to files     
    # NSE data
    NSE_files = sorted(glob.glob('{}/NSE/**/**/*.netcdf'.format(storm), recursive=True))
    for fname in NSE_files:
        field = fname.split('/')[-3] # grab field
        if field not in fields:
            fields.append(field)
            files.append(fname)    
    return files

# Build pandas dataframe of days and the number of storms in each day
def build_df(cases):
    days = cases
    storm_count = []
    for day in days: 
        storm_count = 0
        year = day[:4]
        storm_path = '{}/{}/{}'.format(TRAINING_HOME,year,day)
        subdirs = sorted(os.listdir(storm_path))
        for subdir in subdirs: # for storm in storms
            if subdir[:5] == 'storm' and subdir[:6] != 'storms':
                storm_count+=1
    df = pd.DataFrame(data={'days':days,'storms':storm_count})
    return df

def modify_ins(ins,indices):
    """
    Purpose: deletes rows of ins to make new datasets
     @param - ins : 4D myrorss ndarray
     @param - indices: list of integers, the indices to delete (0->N-1)
    """
    for idx in indices:
        ins = ins[:,:,:,idx]*0
    return ins

def main():
   # year = '2011'
   # df = get_df_shapes(year='2011')
   # print(df)
   # df.to_csv('csv/{}_missing_fields.csv'.format(year))
    df = pd.read_csv('csv/2011_missing_fields.csv')
    ins, outs = load_data_from_df(df)
    ins = np.asarray(ins)
    outs = np.asarray(outs)
  #  ins = np.reshape(ins, (ins.shape[0],60,60,ins.shape[1]))
  #  outs = np.reshape(outs, (outs.shape[0],60,60,outs.shape[1]))

    np.save('datasets/ins_2011_fixed.npy',ins)
    np.save('datasets/outs_2011.npy',outs)
        
    ins = np.reshape(ins, (ins.shape[0],60,60,ins.shape[1]))
    outs = np.reshape(outs, (outs.shape[0],60,60,outs.shape[1]))
    np.save('datasets/reshaped_ins_2011.npy',ins)
    np.save('datasets/reshaped_outs_2011.npy',outs)
    
if __name__ == '__main__':
    main()

