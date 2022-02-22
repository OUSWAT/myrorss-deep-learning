############################################

# Author: Michael Montalbano
#
# Purpose: Load data from the MYRORSS directories on OSCER
#          into .npys.
#
# Returns: 
# @param ins - the inputs for the dataset
# @param outs - the outputs for the dataset 
# TODO: 
# 1. edit load_while() with just shape echekck
#
# Functions: get_cases, get_training_cases, load_while() 

import datetime, sys, os, random
from os import walk
import numpy as np
import pandas as pd
from ast import literal_eval
from netCDF4 import Dataset
import util

DATA_HOME = '/condo/swatcommon/common/myrorss'
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
multi_fields = ['MergedLLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Max_30min','MergedMLShear_Min_30min','MergedReflectivityQC','MergedReflectivityQCComposite_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min','target_MESH_Max_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
products = multi_fields + NSE_fields

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

def load_while(year):
    '''
    Given a year and bounds for days, this will return ins and outs
    @param year - the chosen year
    @param b0 - lower-bound (i.e. b0 = 0 is day 1 of the year)
    @param b1 - upper-bound 
    '''
    ins_full = []
    outs_full = []
    days = util.get_days(year)
    print(days)
    days = ['20110409']
    for day in days:
        year = day[:4]
        storm_path = '{}/{}/{}'.format(TRAINING_HOME,year,day)
        subdirs = sorted(os.listdir(storm_path))
        for subdir in subdirs:
            if subdir[:5] == 'storm' and subdir[:6] != 'storms':
                missing=False # reset missing
                ins = []
                outs = []
                while not missing:
                    storm_dir = '{}/{}'.format(storm_path,subdir)
                    field_dirs = os.listdir(storm_dir)
                    for field in field_dirs:
                        if field in NSE_fields:
                            field_path = '{}/NSE/{}'.format(storm_dir,field)
                            files = sorted(os.listdir(field_path))
                            if len(files) != 0 and files != []:
                                try:
                                    nc = Dataset('{}/{}'.format(field_path,files[0]))
                                    ins.append(np.array(nc.variables[field][:]))
                                except:
                                    print('{}/{}/{}/{}/{}'.format(storm_path,subdir,field,field_path,files[0]))
                                    missing = True 
                                    break
                                var = nc.variables[field][:,:]
                                var = np.where(var<-100,0,var)
                                print(var)
                                ins.append(var)
                        if field in multi_fields and field != 'target_MESH_Max_30min':
                            try:
                                subdirs2 = os.listdir('{}/{}/{}'.format(storm_path,subdir,field))
                            except:
                                print('{}/{}/{}'.format(storm_path,subdir,field))
                                missing = True 
                                break
                            for subdir2 in subdirs2:
                                if subdir2[2] == '.':
                                    files = next(walk('{}/{}/{}/{}'.format(storm_path,subdir,field,subdir2)), (None, None, []))[2]
                                    if len(files) != 1 and files != []: # if there not a  file, then missing
                                        f = sorted(files)[0]
                                    elif len(files)==1 and files != []:
                                        f = files[0]
                                    else:
                                        missing=True 
                                        print('there was a missing file with the following path: {}/{}/{}/{}/{}'.format(storm_path,subdir,field,subdir2,f))
                                        break
                                    file_path = '{}/{}/{}/{}/{}'.format(storm_path,subdir,field,subdir2,f)
                                    try:
                                        nc = Dataset(file_path, mode='r')
                                    except:
                                        print('missing',file_path)
                                        missing=True 
                                        break
                                    var = nc.variables[field][:,:]
                                    var = np.where(var<-50,0,var)
                                    temp_array = []
                                    temp_array.append(var)
                                    if len(temp_array[0]) == 0:
                                        print('empty array with path',file_path)
                                        missing=True
                                        break
                                    ins.append(var)
                        if field == 'target_MESH_Max_30min':
                            files = next(walk('{}/{}/{}/MESH_Max_30min/00.25'.format(storm_path, subdir, field)), (None, None, []))[2]
                            if len(files) > 1 and files != []: # if there is more than 1 file, use the latest one (timestep) since this is the target (t+ 30 min)
                                f = sorted(files)[-1]
                            elif len(files)==1 and files != []: # if there is only one file, then use that
                                f = files[0]
                            else:
                                print('target missing')
                                missing=True # if no files, then missing
                                break
                            file_path='{}/{}/{}/MESH_Max_30min/00.25/{}'.format(storm_path, subdir, field, f)
                            try:
                                nc = Dataset(file_path,mode='r')
                            except:
                                print('missing',file_path)
                                missing = True# if no files, then missing
                                break
                            var = nc.variables['MESH_Max_30min'][:,:]
                            var = np.where(var<-10,0,var)
                            temp_array = []
                            temp_array.append(var)
                            if len(temp_array[0]) == 0:
                                print('empty array with path',file_path)
                                missing=True
                                break
                            outs.append(var)
                   #if not missing:
                    ins = np.asarray(ins)
                    outs = np.asarray(outs)
                    print(ins.shape)
                    if np.asarray(outs).shape == (1, 60, 60):
                        print('success')
                        ins_full.append(ins)
                        outs_full.append(outs)
                        missing=True # reset missing
                        break # reset, exit loop, move to new storm directory
                    else:
                        #print('MISSING, ins.shape: {}, {}'.format(np.asarray(ins).shape,file_path))
                        missing = True
                        break
    return ins_full, outs_full

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
    """
    for idx in indices:
        ins = ins[:,:,:,idx]*0
    return ins

#def main():
ins, outs = load_while(year=year) # load
#print(ins,outs)
#ins = np.asarray(ins)
outs = np.asarray(outs)
#ins = np.reshape(ins, (ins.shape[0],60,60,ins.shape[1]))
outs = np.reshape(outs, (outs.shape[0],60,60,outs.shape[1]))

#np.save('datasets/ins_june_18_19.npy',ins)
np.save('datasets/outs_20110409.npy',outs)

#if __name__ == '__main__':
#    main()

