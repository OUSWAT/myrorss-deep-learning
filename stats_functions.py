########################################
# Functions for stats and model analysis
# Author: Michael Montalbano

import pickle, sys, os
import numpy as np
import pandas as pd

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