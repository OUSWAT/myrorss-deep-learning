########################################
# Functions for stats and model analysis
# Author: Michael Montalbano
#
# stats.py

import pickle, random
import sys
import os
import glob
import numpy as np
import pandas as pd
from os import walk
from collections import Counter
import util

multi_fields = [
    'MergedLLShear_Max_30min',
    'MergedLLShear_Min_30min',
    'MergedMLShear_Max_30min',
    'MergedMLShear_Min_30min',
    'MergedReflectivityQC',
    'MergedReflectivityQCComposite_Max_30min',
    'Reflectivity_0C_Max_30min',
    'Reflectivity_-10C_Max_30min',
    'Reflectivity_-20C_Max_30min',
    'target_MESH_Max_30min']
NSE_fields = [
    'MeanShear_0-6km',
    'MUCAPE',
    'ShearVectorMag_0-1km',
    'ShearVectorMag_0-3km',
    'ShearVectorMag_0-6km',
    'SRFlow_0-2kmAGL',
    'SRFlow_4-6kmAGL',
    'SRHelicity0-1km',
    'SRHelicity0-2km',
    'SRHelicity0-3km',
    'UWindMean0-6km',
    'VWindMean0-6km',
    'Heightof0C',
    'Heightof-20C',
    'Heightof-40C']
products = multi_fields + NSE_fields

DATA_HOME = '/condo/swatcommon/common/myrorss'
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
HOME_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning'


def mse_binned(r):
    y_true = r['true_testing']
    y_pred = r['predict_testing']

    sectors = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,70)] # MESH bins
    image_loss = []
    for idx, image in enumerate(y_true):
        sector_loss = []
        for s in sectors:
            # make mask
            l = s[0] # upper bound
            u = s[1] # lower bound
            y_true = np.where(l<y_true<u,0,y_true) # mask
            y_pred = np.where(l<y_pred<u,0,y_pred) # mask
            sector_loss.append(np.sqrt(np.mean((y_pred-y_true)**2))) #append rmse
        image_loss.append(sector_loss)
    return image_loss # returns an array of arrays.
        
def stats(r, threshold=20):
    # return metrics for finding POD, FAR, CSI, etc (Roebber 2008)
    container = binary_accuracy(r, threshold)
    correct, total, TP, FP, TN, FN, events = [x for x in container]
    POD = TP / (TP + FN)
    FAR = FP / (TP + FP)  # SR = 1 - FAR
    bias = (TP + FP) / (TP + FN)
    CSI = TP / (TP + FP + FN)
    return [POD, FAR, bias, CSI]

def get_quantiles(image):
    # return the quartiles
    #image = np.flatten(image)
    q = np.quantile(image,[0.25,0.5,0.75,0.9,0.95,1])
    return q
    
def binary_accuracy(r, threshold):
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
        # Don't scale, just keep y_true and y_pred unchanged
       # ex = ex.reshape(1, 60 * 60)
        #ex_pred = y_pred[idx].reshape(1, 60 * 60)
       # ex = scaler.inverse_transform(ex)
       # ex_pred = scaler.inverse_transform(ex_pred)
        
        # Count Area Shared Above threshold (20 mm)
        thres = 20

        # binarize the data
        ex_pred = np.where(y_pred[idx] < 20, 0, 1)
        ex = np.where(ex < 20, 0, 1)

        # Compute positives, negatives, etc
        for idx, row in enumerate(ex):
            for idx2, pixel in enumerate(row):
                pred_pix = ex_pred[idx][idx2]
                if pixel == pred_pix:
                    correct += 1
                    if pixel == 1:
                        TP += 1
                if pixel == 1:
                    events += 1
                if pixel != pred_pix and pred_pix == 1:
                    FP += 1
                if pixel == pred_pix and pred_pix == 0:
                    TN += 1
                if pixel != pred_pix and pred_pix == 0:
                    FN += 1
                total += 1
    return [correct, total, TP, FP, TN, FN, events]
# Build dataframe of days and storm count


def build_df(cases):
    days = cases
    storms = []
    for day in days:
        storm_count = 0
        year = day[:4]
        storm_path = '{}/{}/{}'.format(TRAINING_HOME, year, day)
        subdirs = sorted(os.listdir(storm_path))
        print(subdirs)
        for subdir in subdirs:  # for storm in storms
            if subdir[:5] == 'storm' and subdir[:6] != 'storms':
                storm_count += 1
        storms.append(storm_count)
        print(storms)
    df = pd.DataFrame(data={'days': days, 'storms': storms})
    return df

# Get the cases in year


def get_cases(year='1999'):
    cases = []
    path = '{}/{}'.format(TRAINING_HOME, year)
    possible_storms = os.listdir(path)
    for storm in possible_storms:
        if storm[:4] == year:
            cases.append(storm[:8])
    return cases


def load_npy(prefix='outs'):
    # load npy with prefix and return as single np array (or npy)
    files = os.listdir('{}/{}'.format(HOME_HOME, 'datasets'))
    names = []
    for idx, f in enumerate(files[:-1]):  # collect all npys fname prefix
        if f[:2] == prefix[:2]:
            names.append(f)
    # data is a list containing each npy
    data = [np.load('{}/datasets/{}'.format(HOME_HOME, x)) for x in names]
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
    storms_dir = '{}/{}/{}'.format(TRAINING_HOME,
                                   date[:4], date)  # path to storm directory
    storms = sorted(os.listdir(storms_dir))  # list of storms
    df = pd.DataFrame(columns=['storm', 'missing', 'miss_fields'])
    for storm in storms:
        # use glob to check if any of the files are missing
        # sort the dirs in reverse order
        dirs = sorted(glob.glob(
            '{}/{}/{}/*'.format(TRAINING_HOME, date[:4], date)), reverse=True)
        for d in dirs:
            if int(d[-1]):  # check if the last char is a number
                missing, fields = check_storm(d)
                try:
                    missing_fields = list(
                        (Counter(fields) - Counter(products).elements()))
                except BaseException:
                    missing_fields = fields
                df = df.append({'storm': storm,
                                'missing': missing,
                                'miss_fields': fields},
                               ignore_index=True)
    return df


def check_storm(storm_path):
    # given a storm path, check if any of the files are missing
    storm_path = storm_path
    # find all files in storm path
    files = []
    fields = []
    for f in glob.glob(
        '{}/**/*.netcdf'.format(storm_path),
            recursive=True):  # recursively returns all files at any depth ending in .netcdf
        field = f.split('/')[-3]
        if field in fields:
            continue  # if the field is already in the list, skip
        fields.append(field)
        files.append(f)
    if len(files) != 44:
        missing = True
    else:
        missing = False
    return missing, fields


def main():
     # comment this out when you want to predict on shave
#    y_true = np.load('datasets/outs_raw.npy')
#    ins = np.load('datasets/ins_raw.npy')
#    y_pred = model.predict(ins)
    
    r = util.open_pickle('results/2008_64a128_filters_100e_20b_Nonel2_10s_results.pkl')
    image_loss = mse_binned(r)
    print(image_loss)
'''
    outs1 = np.load('datasets/outs_2011.npy')
    ins2 = np.load('datasets/ins_raw.npy')
    outs2 = np.load('datasets/outs_raw.npy')
    quants1 = np.zeros((6))
    quants2 = np.zeros((6))

    for idx, MESH in enumerate(outs1):
        quants = get_quantiles(MESH)
        print(quants)
        quants1 = np.mean(np.array([quants,quants1]))        
    for idx, MESH in enumerate(outs2):
        quants = get_quantiles(MESH)
        quants2 = np.mean(np.array([quants,quants2]))
    print(quants1)
    print(quants2)
    indices = []
    new_ins=[]
    new_outs=[]
    for idx, MESH in enumerate(outs1):
        quants = get_quantiles(MESH)
        diff = quants1 > quants
        p = .1*(np.count_nonzero(diff==True))
        if random.uniform(0,1) > p:
            indices.append(idx)
    for idx in indices:
        new_ins.append(ins1[idx,:,:,:])
        new_outs.append(outs1[idx,:,:,:])
    new_ins = np.asarray(new_ins)
    new_outs = np.asarray(new_outs)
    np.save('datasets/ins_2011.npy',new_ins)
    np.save('datasets/outs_2011.npy',new_outs)
'''
if __name__ == "__main__":
    main()
