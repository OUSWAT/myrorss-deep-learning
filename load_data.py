############################################
# Author: Michael Montalbano
#
# Purpose: Load data from the MYRORSS directories on OSCER
#          into .npys.
#
# Returns: 
# @param ins - the inputs for the dataset
# @param outs - the outputs for the dataset 

import datetime, sys, os, random
from os import walk
import numpy as np
import pandas as pd
from ast import literal_eval
from netCDF4 import Dataset

DATA_HOME = '/condo/swatcommon/common/myrorss'
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
fields = ['MergedLLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Max_30min','MergedMLShear_Min_30min','MergedReflectivityQC','MergedReflectivityQCComposite_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min','target_MESH_Max_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
products = fields + NSE_fields
print(products[-14])

# INPUT VARIABLES HERE
# make this more elegant, to input from the shell. check extract.py
year = '1999'
num_cases=8

# Get the cases in year
def get_cases(year = '1999'):
    cases = []
    path = '{}/{}'.format(TRAINING_HOME,year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases


#def load_data(cases):
def load():
    ins_full = []
    outs_full = []
    days = np.random.choice(get_cases('1999'),10,replace=False)

    for day in days:
        year = day[:4]
        storm_path = '{}/{}/{}'.format(TRAINING_HOME,year,day)
        #print(storm_path)
        subdirs = sorted(os.listdir(storm_path))
        #print(subdirs)
 #       except:
  #          print('subdirs problem with path ',storm_path)
   #         break
        missing=False
        # COULD HAVE DONE A (while missing == False loop)
        for subdir in subdirs: # for storm in storms
           # print('subdir',subdir
            ins = []
            outs = []
            if subdir[:5] == 'storm' and subdir[:6] != 'storms':
               # print('inside storm',subdir)
                subdir1 = os.listdir('{}/{}'.format(storm_path,subdir))
                # now check cycle through the fields
                #  if any field fails, break
                for product in products:
                    if missing == True: # check if any field is missing
                        break
                   # print('field',field)
                    if product == 'target_MESH_Max_30min':
                        files = next(walk('{}/{}/{}/MESH_Max_30min/00.25'.format(storm_path, subdir, product)), (None, None, []))[2]
                        if len(files) > 1:
                            f = sorted(files)[-1]
                        elif len(files)==1:
                            f = files[0] 
                        else:
                            missing=True
                            break
                        file_path='{}/{}/{}/MESH_Max_30min/00.25/{}'.format(storm_path, subdir, product, f)
                        try:
                            nc = Dataset(file_path,mode='r')
                        except:
                            print('missing',file_path)
                            missing = True
                            break
                        var = nc.variables['MESH_Max_30min'][:,:]
                        var = np.where(var<-50,0,var)
                        temp_array = []
                        temp_array.append(var)
                        if len(temp_array[0]) == 0:
                            print('empty array with path',file_path)
                            missing=True
                            break
                        outs.append(var)
                   # print('field',field)
                        #print(outs.shape)
                      #  except:
                      #      print('There is a missing target file with the following path: {}'.format(file_path))
                      #      missing = True
                      #      break
                    elif product in NSE_fields:
                        files = next(walk('{}/{}/NSE/{}'.format(storm_path, subdir,product)), (None, None, []))[2]
                        if len(files) >= 0 and files != []:
                            print(files)
                            f = sorted(files)[0]
                        else:
                            missing=True
                            break
                        file_path = '{}/{}/NSE/{}/{}'.format(storm_path, subdir, product, f)
                        try:
                            nc = Dataset(file_path,mode='r')
                        except:
                            print('missing',file_path)
                            missing=True
                            break    
                        var = nc.variables[product][:,:]
                        var = np.where(var<-50,0,var)
                        temp_array = []
                        temp_array.append(var)
                        if len(temp_array[0]) == 0:
                            print('empty array with path',file_path)
                            missing=True
                            break
                        ins.append(var)                        
                    else:
                        try:
                            subdirs2 = os.listdir('{}/{}/{}'.format(storm_path,subdir,product))
                        except:
                            print('directory missing for {} {}'.format(subdir,product))
                            missing=True
                            break
                        for subdir2 in subdirs2:
                            if len(subdir2) > 4:
                                files = next(walk('{}/{}/{}/{}'.format(storm_path,subdir,product,subdir2)), (None, None, []))[2]
                                if len(files) != 1:
                                    f = sorted(files)[0]
                                elif len(files)==1:
                                    f = files[0]
                                else:
                                    missing=True
                                    break
                                file_path = '{}/{}/{}/{}/{}'.format(storm_path,subdir,product,subdir2,f)
                                try:
                                    nc = Dataset(file_path, mode='r')
                                except:
                                    print('missing',file_path)
                                    missing=True
                                    break
                                var = nc.variables[product][:,:]
                                var = np.where(var<-50,0,var)
                                temp_array = []
                                temp_array.append(var)
                                if len(temp_array[0]) == 0:
                                    print('empty array with path',file_path)
                                    missing=True
                                    break
                                ins.append(var)
            if missing == False and subdir[:5] == 'storm' and subdir[:6] != 'storms': # if missing is still false, append the ins and outs
                print('success',subdir)
                ins_full.append(ins)
                outs_full.append(outs)
            missing = False # reset
    return ins_full, outs_full
#return urn ins_full, outs_full

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

#cases = get_cases(1999) # these are the cases which have already been run.
#insfull, outsfull = load_data(cases)
#print(ins_full)
ins_full, outs_full = load()
ins = np.asarray(ins_full)
outs = np.asarray(outs_full)

ins = np.squeeze(np.reshape(ins, (ins.shape[0],60,60,ins.shape[1])))
outs = np.squeeze(np.reshape(outs, (outs.shape[0],60,60,outs.shape[1])))

np.save('ins.npy',ins)
np.save('outs.npy',outs)
#print(insfull.shape,outsfull.shape)

