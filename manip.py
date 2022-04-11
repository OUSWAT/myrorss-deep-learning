#
# Author: Michael Montalbano
# Purpose: Maniputate datasets to form new datasets

import numpy as np
import pandas as pd
import util
import sys
import matplotlib.pyplot as plt
from collections import Counter
#from util import open_pickle as op
ins = np.load('datasets/ins_2011_qc.npy')  # load full ins
mesh = ins[:, -1, :, :]  # grab input mesh only
outs = np.load('datasets/outs_2011_qc.npy')


def get_pixel_list(outs, thres=20, ID='most_recent'):
    # returns the number of pixels above threshold for each image, as a list
    n_list = []
    for idx, out in enumerate(outs):
        output = np.where(out < thres, 0, out)
        n = np.count_nonzero(output)
        n_list.append(n)
    df = pd.DataFrame(n_list)
    df.to_csv('n_list_{}.csv'.format(ID))
    return n_list

#def num_above(n):
    # simple function to return 

def proportions(arr):
    total = len(arr)
    # Alternate method
    bins = np.arange(0,3600,3600/10) 
    bin_indices = np.digitize(arr,bins)
    return Counter(bin_indices) 

def plot_hist(numbers, threshold, ID='most_recent'):
    plt.hist(numbers)
    plt.title(ID)
    plt.ylabel('Number of pixels above {} MM'.format(threshold))
    plt.xlabel('Cases')
    plt.savefig('{}.png'.format(ID))
    return None

def main():
    threshold = 20  # use 25 MM threshold
    n_list1 = get_pixel_list(outs, threshold, '2011_qc')
    outs2 = np.load('datasets/outs_shave.npy')
    # get the list of pixels above threshold for each image in outs)
    n_list2 = get_pixel_list(outs2, threshold, 'raw')

    #plot_hist(n_list1, threshold, '2011_qc')

    Counter_2011 = proportions(n_list1)
    
    Counter_raw = proportions(n_list2)    
    print(Counter_2011)
    print(Counter_raw)

if __name__ == "__main__":
    main()
