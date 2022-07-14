import numpy as np
import pandas as pd
import argparse
from glob import glob

def rescale(y, scaler):
    # go to original scale for test data
    y = y.squeeze()
    yf = np.reshape(y, newshape=(y.shape[0], 3600)) 
    print(type(yf))
    print(yf.shape)
    y = scaler.inverse_transform(yf)
    return np.reshape(y, newshape=(y.shape[0], 60, 60, 1))

def create_parser():
    parser = argparse.ArgumentParser(description='Load')
    parser.add_argument(
        '-filename',
        type=str,
        default='',
        help='Are you using the tuner, or just messing around?')
    parser.add_argument(
        '-act',
        type=str,
        default='linear',
        help='Activation function of pkls to analyze')
    return parser

def stats(y_true, y_pred, cutoff):
    # get FP
    y_true_ones = np.where(y_true > cutoff, 1, 0)
    y_true_reversed = np.where(y_true > cutoff, 0, 1)
    y_pred_ones = np.where(y_pred > cutoff, 1, 0)
    y_pred_negatives = np.where(y_pred > cutoff, 0, 1)
    # get TP
    TP = np.sum(np.multiply(y_true_ones, y_pred_ones)) # pointwise multiplication 
    FP = np.sum(np.multiply(y_pred_ones, y_true_reversed))
    FN = np.sum(np.multiply(y_true_ones, y_pred_negatives))
    POD = np.divide(TP,(0.1+np.add(TP,FN)))
    FAR = np.divide(FP,(0.1+np.add(TP,FP)))
    return POD, FAR 

def main():
    parser = create_parser()
    args = parser.parse_args()
    files = glob(f'results/*{args.act}*')
    o = pd.read_pickle('ytr_scaler.pkl')
    sc = o['scaler'][0]
    for filename in files:
        r = pd.read_pickle(f'{filename}')
        y_true = r['true_testing']
        y_pred = r['predict_testing']
        print(f'model: {filename}')
        print(f'POD/FAR with thres 20: {stats(y_true,y_pred, 20)}')
        print(f'POD/FAR with thres 30: {stats(y_true,y_pred, 30)}')
        print(f'POD/FAR with thres 50: {stats(y_true,y_pred, 50)}')
        print(f'POD/FAR with thres 80: {stats(y_true,y_pred, 80)}')

        print('\n\n')
    return None

if __name__ == '__main__':
    main()

