# # # # # # # #
# Author: Michael Montalbano
# purpose: Review trained models, usually using the model's pickle 
#          or the y_true and y_pred in the test set, usually
# # # # # # # # 

import pickle, sys, os, glob
import numpy as np
import pandas as pd
from collections import Counter
from util import open_pickle as op 
import util
from stats import stats

def get_images(fname,step='testing'):
    # given an fname, open the pickle and return the images of interest
    # @param step - choose the step ('training' or 'testing')
    pik = op(fname)
    return pik['true_{}'.format(step)], pik['predict_{}'.format(step)] # return the images from the selected step 
   
def quick_check(y_true, y_pred,scaler):
    # check the difference in maxes
    for idx, yt in enumerate(y_true):
        ex = yt.reshape(1,3600)
        ex = scaler.inverse_transform(ex)
        yp = y_pred[idx]
        yp = yp.reshape(1,3600)
        yp = scaler.inverse_transform(yp)
        print(ex.max(),yp.max())
 
def get_interesting_pickles():
    return glob('results/interest/*.pkl')

# examine the outs 
def summarize(outs):
    # given an np.array of images, return relevant information 
    dist = []
    i=0
    for image in outs:
        i=0
        for row in image:
            for pixel in row:
                if pixel > 15:
                   i+=1
        dist.append([image.max(),image.mean(),i])
    return dist

def main():
    fname_list = get_interesting_pickles()
    # skip the intersting loop writing for later
    fname = fname_list[0]
    res = op(fname)
    y_true, y_pred = get_images(fname)
    scaler = op('scaler_MSE_1.pkl')[0] # replace with fname 
    container = stat(y_true, y_pred, scaler) # returns a list of [POD, FAR, bias, and CSI
    print(container)

if __name__ == '__main__':
    main() 
