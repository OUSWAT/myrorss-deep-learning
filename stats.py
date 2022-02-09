########################################
# Functions for stats and model analysis
# Author: Michael Montalbano

import pickle, sys, os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from os import walk



DATA_HOME = '/condo/swatcommon/common/myrorss'
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'

def pod(file,scalers):
    # return metrics for finding POD, FAR, CSI, etc
    r = pd.read_pickle(file)
    scaler = scalers[0] # the np scaler is within a list

def binary_accuracy(r,scaler,group='testing'):
    # return metrics for finding POD, FAR, CSI, etc
    y_true = r['true_testing']
    y_pred = r['predict_testing']

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
    pickle_data = pd.read_pickle(file

# Get the cases in year
def get_cases(year = '1999'):
    cases = []
    path = '{}/{}'.format(TRAINING_HOME,year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases


