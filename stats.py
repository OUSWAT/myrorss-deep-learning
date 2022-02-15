########################################
# Functions for stats and model analysis
# Author: Michael Montalbano
#
# TODO:
#       1. stats() funct that returns POD, FAR, CSI, etc (Roebber 2008)
#       2. Taylor (2001) performance graph
#	2. MED() for mean_error_distance (use networkx) 
#       3. Check for dependence on lat,lon

import pickle, sys, os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from os import walk

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

# check if any of files in storm directory are missing
#
def check_storm(date):
    """
    Purpose: check if storm is missing
    Returns:
        @param number of missing cases
        @param number of total cases
    """
    storms_dir = '{}/{}/{}'.format(TRAINING_HOME,date[:4],date)
    storms = sorted(os.listdir(storms_dir))
    df = pd.DataFrame(columns=['storm','missing'])
    for storm in storms:
        if storm[:5] == 'storm' and storm[:6] != 'storms':
            missing = False
            n_missing = 0
            for p in products:
                if p in NSE_fields:
                    # check if the NSE/product/.netcdf file is missing
                    file_list = os.listdir('{}/{}/NSE/{}'.format(storms_dir,storm,p))
                    if len(file_list) == 0:
                        missing = True
                        # note that p is missing in a pandas database
                        n_missing+=1
                        break
                if p == 'MergedReflectivityQC':
                    # check if any of the MergedReflectivityQC files are missing
                    degrees = os.listdir('{}/{}/MergedReflectivityQC/'.format(storms_dir, storm))
                    for degree in degrees: # check 
                        if len(os.listdir('{}/{}/MergedReflectivityQC/{}'.format(storms_dir, storm, degree))) == 0:
                            missing = True
                            n_missing+=1
                            break
                if p == 'target_MESH_Max_30min':
                    # check if any of the target_MESH_Max_30min files are missing
                    targ_dirs = os.listdir('{}/{}/target_MESH_Max_30min/'.format(storms_dir, storm))
                    if 'MESH_Max_30min' not in targ_dirs:
                        missing = True
                        n_missing+=1
                        break
                    else:
                        if len(os.listdir('{}/{}/target_MESH_Max_30min/MESH_Max_30min'.format(storms_dir, storm))) == 0:
                            missing = True
                            n_missing+=1
                            break
                if p in multi_fields and p != 'target_MESH_Max_30min':
                    dummy_dir = os.listdir('{}/{}/{}'.format(storms_dir, storm, p))[0]
                    if len(os.listdir('{}/{}/{}/{}'.format(storms_dir, storm, p, dummy_dir))) == 0:
                        missing = True
                        n_missing+=1
                        break
                n_missing+=1
            df = df.append({'storm':storm, 'missing': n_missing}, ignore_index=True)
                    # check if any of the multi_fields files are missing                 
    return df
