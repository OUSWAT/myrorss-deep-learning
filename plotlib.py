#####################
# Author : Michael Montalbano
# Purpose: provides various plotting functions for analyzzing MYRORSS models and datasets
#####################

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, pickle, sys, datetime, glob, util, random
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import figure
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.colors import rgb2hex
from util import open_picke


multi_fields = ['MergedLLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Max_30min','MergedMLShear_Min_30min','MergedReflectivityQC','MergedReflectivityQCComposite_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
targets = ['target_MESH_Max_30min']
products = multi_fields + NSE_fields + targets
degrees = ['06.50', '02.50', '05.50', '01.50', '08.00', '19.00', '00.25', '00.50', '09.00', '18.00', '01.25', '20.00', '04.50', '03.50', '02.25', '07.50', '07.00', '16.00', '02.75', '12.00', '03.00', '04.00', '15.00', '11.00', '01.75', '10.00', '00.75', '08.50', '01.00', '05.00', '14.00', '13.00', '02.00', '06.00', '17.00']

def plot_predict(r,idx, scaler,group='testing'):
    f, axs = plt.subplots(1,2,figsize=(15,15))
    
    scalers = open_pickle('scaler_raw.pkl')
    scaler = scalers[0]
    plt.subplot(121)
    true = r['true_{}'.format(group)][idx]
    print(true.mean())
    slc = true.reshape(1, 60*60)
    transformed = scaler.inverse_transform(slc)
    transformed = transformed.reshape(60, 60, 1) # reshape it back to tiles
    print(transformed.mean())
    ax = plt.gca()
#    im = ax.imshow(transformed)
    im = ax.imshow(transformed,cmap=cm.nipy_spectral)
    plt.xlabel('Degrees Longitude')
    plt.ylabel('Degrees Latitude')
    plt.title('True MESH {} Set {}'.format(group,idx))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(122)
    pred = r['predict_{}'.format(group)][idx]
    slc = pred.reshape(1, 60*60)
    transformed = scaler.inverse_transform(slc)
    transformed = transformed.reshape(60, 60, 1) # reshape it back to tiles
    ax = plt.gca()
    im = ax.imshow(transformed,cmap=cm.nipy_spectral)
    plt.title('Predicted MESH {} Set {}'.format(group,idx))
    plt.xlabel('Degrees Longitude')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig('001.png')
 
