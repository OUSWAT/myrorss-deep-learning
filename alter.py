#
# Author: Michael Montalbano
# Purpose: Alter datasets to form 'new' datasets (subsets with different statistics)
# title: alter.py

import numpy as np
import pandas as pd
import util, random
import sys, stats
import matplotlib.pyplot as plt
from collections import Counter
'''
Sample method for deleting elements of a dataset:
bin_indices, ratios, shave_ratios = my_hist(outs_shave, outs_2008)
print(ratios,shave_ratios) # pick group with very different proportion

outs = delete_some(outs_2008,bindices, index=1,percent=10)
# Repeat until satisfactory 
'''

def get_hist(outs):
    # given proportions, alter a dataset to conform to these proportions
    maxes = [x.max() for x in outs]
    bins = [0,15,30,45,65,100] # np.arange(0,100,15)
    hist, edges = np.histogram(maxes,bins)
    bin_ind = np.digitize(maxes,bins)    
    return hist, bin_ind

def my_hist(outs,outs1):
    # @param outs - dataset to become more like
    # @param outs1 - dataset to alter
    # @param index - group to delete from
    # given a dataset outs, remove samples so that its distribution 
    # is more like SHAVE (outs_raw.npy)
    hist, edges = get_hist(outs) # get hist for outs
    n = len(outs)
    proportions = [x/n for x in hist]
    hist1, bin_indices = get_hist(outs1)
    n1 = len(outs1)
    ratios = [x/n1 for x in hist1]
    max_index = np.argmax(ratios)
#    delete_some(var,indices,percent=10)
#    mask = np.where(bin_indices == max_index, 1, 0) # set all 0, but 1 if max_index = group
    return bin_indices, ratios, proportions

def delete_some(var,index,bindices,percent=10):
    # var - 4D np array
    # index - int index of group to remove from
    # bindices - bin_indices from my-hist()
    # percent - percent of group to remove
    indices = np.where(bindices == index) # get indices of index
    del_ind = np.random.choice(bindices, size=int(len(bindices)/percent)) # delete 10 percent
    del_ind = np.flip(np.sort(del_ind)) # sort so that elements dont change order as they priors are deleted
    for ind in del_ind:
        var = np.delete(var,ind,0) # delete ind element along axis 0 (whole thing)
    return var

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

def rank_images(images, threshold):
    # 
    ret_images = [] # array to hold return images
    indices = np.arange(0,len(images),1) # indices
    count_list = []      
    for image in images:
        image = np.where(image<threshold,0,image)
        count = np.count_nonzero(image)
        count_list.append(count)
    ret_arr = [x for _, x in sorted(zip(count_list, indices))]
    return ret_arr[::-1] # reverse it while keeping list (reverse(list) returns not-list

def get_probability(image,outs_shave):
    # given an image, determine the probability of removal
    # compare with N from SHAVE
    # each comparison can add max .15 p 
    indices = np.random.choice(np.arange(0,len(outs_shave),1),5) # get 5 indices 
    p = 0 # probability
    p_max = .1 
    for index in indices:
        shave_image = outs_shave[index,:,:,:]
        # filter out zeroes
        shave_image = np.where(shave_image>10,0,shave_image)
        image = np.where(image>10,0,image)
        top = abs(np.mean(shave_image)-np.mean(image))
        bottom = max(np.mean(shave_image),np.mean(image))
    return None   

def new_dataset(ins_prev,outs_prev,ins_standard,outs_standard,ID='most_recent'):
    # Make two datasets similar by removing images semi-randomly based on image quantiles
    '''
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
    new_ins = []
    new_outs = []
    indices = []
    # check outs1 to build returning nwe outs and ins
    for idx, MESH in enumerate(outs_prev):
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
    np.save('datasets/ins_2011_a.npy',new_ins)
    np.save('datasets/outs_2011_a.npy',new_outs)
    '''
    new_ins = []
    new_outs = []
    indices = []
    # check outs1 to build returning nwe outs and ins
    for idx, image in enumerate(outs_prev):
        p = get_probability(image, outs_standard) # can't we just have outs_shave already asssigned? 
        if random.uniform(0,1) > p: # if greater than p, then keep sample 
            indices.append(idx)     # record keep_index
    for idx in indices:
        new_ins.append(ins_prev[idx,:,:,:]) 
        new_outs.append(outs_prev[idx,:,:,:])
    new_ins = np.asarray(new_ins)
    new_outs = np.asarray(new_outs)
    np.save('datasets/ins_year_{}.npy'.format(ID),new_ins)
    np.save('datasets/outs_2011_{}.npy'.format(ID),new_outs)
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
    '''
    threshold = 20  # use 25 MM threshold
    n_list1 = get_pixel_list(outs, threshold, '2011')
    outs2 = np.load('datasets/outs_shave.npy')
    # get the list of pixels above threshold for each image in outs)
    n_list2 = get_pixel_list(outs2, threshold, 'raw')
    #plot_hist(n_list1, threshold, '2011_qc')
    Counter_2011 = proportions(n_list1)
    '''
    year = '2011'
   # ins = np.load('datasets/ins_{}.npy'.format(year))
    outs11 = np.load('datasets/outs_2011.npy') 
    year = '2005'
    outs05 = np.load('datasets/outs_2005.npy')
    year = '2006'
    outs06 = np.load('datasets/outs_2006.npy')
    #year = '2008'
    #outs08 = np.load('datasets/outs_2009.npy')
    h05 = get_hist(outs05)
    h06 = get_hist(outs06)
    #h08 = get_hist(outs08)
    h11 = get_hist(outs11)
    print('bins')
    print(bins)
    print('2005 {}'.format(h05))
    print('2006 {}'.format(h06))
    #print('2008 {}'.format(h08))
    print('2011 {}'.format(h11)) 
   # ins_shave = np.load('datasets/ins_raw.npy')
   # outs_shave = np.load('datasets/outs_raw.npy')
   # ins_new, outs_new = new_dataset(ins, outs, ins_shave, outs_shave, ID='abs_of_mean')

if __name__ == "__main__":
    main()
