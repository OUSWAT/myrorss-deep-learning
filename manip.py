#
# Author: Michael Montalbano
# Purpose: Maniputate datasets to form new datasets

import numpy as np
import pandas as pd
import util
import sys, stats
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

def build_dataset(ins,outs,n):
    # simple function to 
    outs_indices = np.arange(0,len(outs),1)
    # choose n random indices
    choices = np.random.choice(outs_indices, n)
    #new_ins = np.zeros(shape=(choices,ins.shape[1], 60,60)) 
    new_ins, new_outs = [], []
    for i in choices:
        new_ins.append(ins[i,:,:,:])
        new_outs.append(outs[i,:,:,:])
    new_ins = np.asarray(new_ins)
    new_outs = np.asarray(new_outs)
    return new_ins, new_outs        

def form_dataset_from_quants(ins1,outs1,ins2,outs2):
    # Make two datasets similar by removing images semi-randomly based on image quantiles
    quants1 = np.zeros((6))
    quants2 = np.zeros((6))
    for idx, MESH in enumerate(outs1):
        quants = stats.get_quantiles(MESH)
        print(quants)
        quants1 = np.mean(np.array([quants,quants1])) # add then take mean, iteratively
    for idx, MESH in enumerate(outs2):
        quants = stats.get_quantiles(MESH)
        quants2 = np.mean(np.array([quants,quants2])) # add, then mean
    print(quants1) # display quants
    print(quants2)
    indices = []
    new_ins=[]
    new_outs=[]
    # check outs1 to build returning nwe outs and ins
    for idx, MESH in enumerate(outs1):
        quants = stats.get_quantiles(MESH)
        diff = quants1 > quants # return boolean list
        p = .1*(np.count_nonzero(diff==True)) # for each True in list, increase p by 0.1
        if random.uniform(0,1) > p: # if greater than p, then keep sample 
            indices.append(idx)     # record keep_index
    for idx in indices:
        new_ins.append(ins1[idx,:,:,:]) 
        new_outs.append(outs1[idx,:,:,:])
    new_ins = np.asarray(new_ins)
    new_outs = np.asarray(new_outs)
    np.save('datasets/ins_2011.npy',new_ins)
    np.save('datasets/outs_2011.npy',new_outs)
    return new_ins, new_outs
 
def get_maxes():
    '''
    I need a function to basically get cat=whisker plots of each images MESH values 
    for target or input MESH
    '''
    pass

#def num_above(n):
    # simple function to return 

def proportions(arr,div=20):
    total = len(arr)
    # Alternate method
    bins = np.arange(0,3600,3600/div) 
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
