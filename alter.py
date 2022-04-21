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
    print('Desired ratios: {}'.format(proportions))
    print('Dataset ratios: {}'.format(ratios))
#    delete_some(var,indices,percent=10)
#    mask = np.where(bin_indices == max_index, 1, 0) # set all 0, but 1 if max_index = group
    return bin_indices

def delete_images(var,bindices,index,N=10):
    '''
    Needs troubleshooting
    # var - 4D np array
    # index - int index of group to remove from
    # bindices - bin_indices from my-hist()
    # percent - percent of group to remove
    '''
    indices_tuple = np.where(bindices == index) # get indices of index
    indices = list(indices_tuple[0]) # np where returnss a tuple for some reason
    del_ind = np.random.choice(indices, size=int(len(bindices)/percent)) # delete 10 percent
    del_ind = np.flip(np.sort(del_ind)) # sort and flip to reverse order for deletion
    for ind in del_ind:
        var = np.delete(var,ind,0) # delete ind element along axis 0 (whole thing)
    return var

def add_images(ind, ins_to_take_from, outs_to_take_from, ins, outs, N, index = 1):
    indices_tuple = np.where(ind == index) # get indices of index
    indices = list(indices_tuple[0]) # np where returnss a tuple for some reason
    indices = np.random.choice(indices, size=N)
    i = 0
    for ind in indices:
        print(ind, i)
        i+=1
        image = outs_to_take_from[ind]
        image = np.expand_dims(image,0) # keep outs and image same shape 
        outs = np.concatenate((outs, image))
        images = np.expand_dims(ins_to_take_from[ind], 0)          
        ins = np.concatenate((ins, images))
    return ins, outs
    
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
   
   # shave = np.load('datasets/outs_raw.npy')
    ins06 = np.load('datasets/ins_2006.npy')
    outs06 = np.load('datasets/outs_2006.npy')
    ins08 = np.load('datasets/ins_2008.npy')
    outs08 = np.load('datasets/outs_2008.npy')

    hist, ind06 = get_hist(outs06)
    print('hist06 (before) {}'.format(hist))
    hist, ind08 = get_hist(outs08)
    print(outs06.shape)
    ins, outs = add_images(ind08, ins08, outs08, ins06, outs06, N=1800, index=1)
    print(outs.shape)
    np.save('ins06.npy', ins)
    np.save('outs06.npy', outs)

   # ins_shave = np.load('datasets/ins_raw.npy')
   # outs_shave = np.load('datasets/outs_raw.npy')
   # ins_new, outs_new = new_dataset(ins, outs, ins_shave, outs_shave, ID='abs_of_mean')

if __name__ == "__main__":
    main()
