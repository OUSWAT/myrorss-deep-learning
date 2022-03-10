########################################
# Functions for stats and model analysis
# Author: Michael Montalbano
#
# TODO:
#       1. stats() funct that returns POD, FAR, CSI, etc (Roebber 2008)
#       2. Taylor (2001) performance graph
#	2. MED() for mean_error_distance (use networkx) 
#       3. Check for dependence on lat,lon

import pickle, sys, os, glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from os import walk
from collections import Counter

multi_fields = ['MergedLLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Max_30min','MergedMLShear_Min_30min','MergedReflectivityQC','MergedReflectivityQCComposite_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min','target_MESH_Max_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
products = multi_fields + NSE_fields

DATA_HOME = '/condo/swatcommon/common/myrorss'
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
HOME_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning'

def stats(y_true, y_pred,scaler):
    # return metrics for finding POD, FAR, CSI, etc (Roebber 2008)
    container, c = binary_accuracy(y_true, y_pred, scaler)
    correct, total, TP, FP, TN, FN, events = [x for x in container]   
    POD = TP/(TP + FN)
    FAR = FN/(TP + FP) # SR = 1 - FAR 
    bias = (TP + FP)/(TP + FN)
    CSI = TP / (TP + FP + FN)
    return [POD, FAR, bias, CSI]

def binary_accuracy(y_true,y_pred,scaler):
    # return metrics for finding POD, FAR, CSI, etc
    y_true = y_true
    y_pred = y_pred

    # scale data back to mm
    correct = 0
    FP = 0
    total = 0
    TP = 0
    TN = 0
    FN = 0
    events = 0
    for idx, ex in enumerate(y_true):
        ex = ex.reshape(1, 60*60)
        ex_pred = y_pred[idx].reshape(1,60*60)
        ex = scaler.inverse_transform(ex)
        ex_pred = scaler.inverse_transform(ex_pred)

        # Count Area Shared Above threshold (20 mm)
        thres = 20
        true_area = 0
        pred_area = 0
        common_area = 0
        for idx, row in enumerate(ex):
            for idx2, pixel in enumerate(row):
                if pixel >= thres:
                    true_area+=1
                    if ex_pred[idx][idx2] >= thres:
                        common_area+=1
                if ex_pred[idx][idx2] >= thres:
                    pred_area+=1
        
        # binarize the data
        ex_pred = np.where(ex_pred < 20, 0, 1) 
        ex = np.where(ex < 20, 0, 1)

        # Compute positives, negatives, etc
        for idx, row in enumerate(ex):
            for idx2, pixel in enumerate(row):
                pred_pix = ex_pred[idx][idx2]
                if pixel == pred_pix:
                    correct+=1
                    if pixel == 1:
                        TP+=1
                if pixel == 1:
                    events+=1
                if pixel != pred_pix and pred_pix == 1:
                    FP +=1
                if pixel == pred_pix and pred_pix ==0:
                    TN+=1
                if pixel != pred_pix and pred_pix == 0:
                    FN+=1
                total+=1
    print('correct total TP FP TN FN events, true area    predicted area    commmon area')
    return [correct, total, TP, FP, TN, FN, events], [true_area, pred_area, common_area]

# Build dataframe of days and storm count 
def build_df(cases):
    days = cases
    storms = []
    for day in days:
        storm_count = 0
        year = day[:4]
        storm_path = '{}/{}/{}'.format(TRAINING_HOME,year,day)
        subdirs = sorted(os.listdir(storm_path))
        print(subdirs)
        for subdir in subdirs: # for storm in storms
            if subdir[:5] == 'storm' and subdir[:6] != 'storms':
                storm_count+=1
        storms.append(storm_count)
        print(storms)
    df = pd.DataFrame(data={'days':days,'storms':storms})
    return df

def open_pickle(file):
    r = pd.read_pickle(file)
    return r
# Get the cases in year
def get_cases(year = '1999'):
    cases = []
    path = '{}/{}'.format(TRAINING_HOME,year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases

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

# check if any of files in storm directory are missing using glob
def check_missing(date):
    """
    Purpose: check if storm is missing
    Returns:
        @param dataframe for each date
        df attributes: stormID (str), missing (boolean), missing_fields (list)
    """
    storms_dir = '{}/{}/{}'.format(TRAINING_HOME,date[:4],date) # path to storm directory
    storms = sorted(os.listdir(storms_dir)) # list of storms
    df = pd.DataFrame(columns=['storm','missing','miss_fields'])
    for storm in storms:
        # use glob to check if any of the files are missing
        dirs = sorted(glob.glob('{}/{}/{}/*'.format(TRAINING_HOME,date[:4],date)), reverse = True) # sort the dirs in reverse order
        for d in dirs:
            if int(d[-1]): # check if the last char is a number
                missing, fields = check_storm(d)
                try:
                    missing_fields = list((Counter(fields) - Counter(products).elements()))
                except:
                    missing_fields = fields
                df = df.append({'storm':storm,'missing':missing,'miss_fields':fields}, ignore_index=True)
    return df

def check_storm(storm_path):
    # given a storm path, check if any of the files are missing
    storm_path = storm_path
    # find all files in storm path
    files = []
    fields = []
    for f in glob.glob('{}/**/*.netcdf'.format(storm_path),recursive = True): # recursively returns all files at any depth ending in .netcdf
        field = f.split('/')[-3]
        if field in fields:
            continue # if the field is already in the list, skip
        fields.append(field)
        files.append(f)
    if len(files) != 44:
        missing = True
    else:
        missing = False
    return missing, fields 
