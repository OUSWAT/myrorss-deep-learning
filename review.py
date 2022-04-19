# # # # # # # #
# Author: Michael Montalbano
# purpose: Review trained models, usually using the model's pickle
#          or the y_true and y_pred in the test set, usually
# # # # # # # #

import pickle
import sys
import os
import glob
import plotlib
import numpy as np
import pandas as pd
from collections import Counter
import util
import stats
from netCDF4 import Dataset
import matplotlib.pyplot as plt

DATA_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data/'


def get_images(fname, step='testing'):
    # given an fname, open the pickle and return the images of interest
    # @param step - choose the step ('training' or 'testing')
    pik = open_pickle(fname)
    # return the images from the selected step
    return pik['true_{}'.format(step)], pik['predict_{}'.format(step)]


def quick_check(y_true, y_pred, scaler):
    # check the difference in maxes
    for idx, yt in enumerate(y_true):
        ex = yt.reshape(1, 3600)
        ex = scaler.inverse_transform(ex)
        yp = y_pred[idx]
        yp = yp.reshape(1, 3600)
        yp = scaler.inverse_transform(yp)
        print(ex.max(), yp.max())

def get_interesting_pickles():
    return glob.glob('results/interest/*.pkl')

# examine the outs

def plot_interesting_pickles():
    pickles = get_interesting_pickles()
    for fname in pickles:
        r = util.open_pickle(fname)
        plotlib.plot_loss(r['history'],ID='{}_loss'.format(fname)) #plot the loss
        plt.savefig('most_recent_{}.png'.format(fname.split('/')[-1]))   


def summarize(outs):
    # given an np.array of images, return relevant information
    dist = []
    i = 0
    for image in outs:
        i = 0
        for row in image:
            for pixel in row:
                if pixel > 15:
                    i += 1
        dist.append([image.max(), image.mean(), i])
    return dist


def percent_above(field, thres, min_pixels):
    '''
    Return the percent of samples with field above val for min_pixels
    '''
    if field == 'target_MESH_Max_30min':
        files = glob.glob(
            '{}/2011/**/**/target_MESH_Max_30min/**/**/*netcdf'.format(DATA_HOME))
        field = 'MESH_Max_30min'
    else:
        files = glob.glob('{}/2011/**/{}/**/*netcdf'.format(DATA_HOME, field))
    num_above = 0
    # open
    for f in files:
        nc = Dataset(f)
        var = nc.variables[field][:, :]
        var = np.where(var < thres, 0, var)
        count = np.count_nonzero(var)
        if count >= min_pixels:
            num_above += 1
    return num_above / len(files)


def main():
    fname_list = get_interesting_pickles()
 #   plot_interesting_pickles()
 #   scaler = util.op('scalers/scaler_outs_2011_80.pkl')
    for fname in fname_list:
        r = util.open_pickle(fname)
        try:
            print(fname)
            print(stats.stats(r,threshold=25))
        except: 
            print('divide by zero')
    # skip the intersting loop writing for later
    #fname = fname_list[0]
    #res = util.open_pickle(fname)
    #y_true, y_pred = get_images(fname)
    #scaler = op('scaler_MSE_1.pkl')[0]  # replace with fname
    # returns a list of [POD, FAR, bias, and CSI
    #container = stat(y_true, y_pred, scaler)
    #print(container)

if __name__ == '__main__':
    main()
