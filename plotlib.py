#####################
# Author : Michael Montalbano
# Purpose: provides various plotting functions for analyzzing MYRORSS models and datasets
#####################

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
from util import open_pickle

TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'

NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
targets = ['target_MESH_Max_30min']
degrees = ['06.50', '02.50', '05.50', '01.50', '08.00', '19.00', '00.25', '00.50', '09.00', '18.00', '01.25', '20.00', '04.50', '03.50', '02.25', '07.50', '07.00', '16.00', '02.75', '12.00', '03.00', '04.00', '15.00', '11.00', '01.75', '10.00', '00.75', '08.50', '01.00', '05.00', '14.00', '13.00', '02.00', '06.00', '17.00']
shear_colors = ['#202020','#808080','#4d4d00','#636300','#bbbb00','#dddd00','#ffff00','#770000','#990000','#bb0000','#dd0000','#ff0000','#ffcccc']
shear_bounds = [0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]
MESH_colors = ['#aaaaaa','#00ffff','#0080ff','#0000ff','#007f00','#00bf00','#00ff00','#ffff00','#bfbf00','#ff9900','#ff0000','#bf0000','#7f0000','#ff1fff']
MESH_bounds = [9.525,15.875,22.225,28.575,34.925,41.275,47.625,53.975,60.325,65,70,75,80,85]
uwind_r = ['0x00','0x00','0x00','0x00','0x00','0xbf','0xff','0xff','0xff','0xbf','0x7f','0xff']
uwind_g = ['0x80','0x00','0x7f','0xbf','0xff','0xff','0xff','0xbf','0x99','0x00','0x00','0x33']
uwind_b = ['0xff','0xff','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0xff']
ref_r = [0,115,120,148,2,17,199,184,199,199,153,196,122,199]
ref_g = [0,98,120,164,199,121,196,143,113,0,0,0,69,199]
ref_b = [0,130,120,148,2,1,2,0,0,0,16,199,161,199]
ref_levels = [-10,10,13,18,28,33,38,43,48,53,63,68,73,77]
wind_levels = [-30,-25,-20,-15,-10,-5,-1,1,5,10,15,20,25,30]

wind_colors = []
ref_colors = []
for idx, color in enumerate(uwind_r):
    wind_colors.append(str('#%02x%02x%02x' % (int(color,16),int(uwind_g[idx],16),int(uwind_b[idx],16))))
    ref_colors.append(str('#%02x%02x%02x' % (ref_r[idx],ref_g[idx],ref_b[idx])))


def plot_in(storm_path):
    # Given path to storm, plot relevant ins for verification
    # return 2x2 plot of in_fields and target 
    fields = ['MESH_Max_30min', 'MergedLLShear_Max_30min','MergedMLShear_Max_30min']
    # load up files   
    files = []
    var_list = []
    for field in fields:
        field_path = '{}/{}'.format(storm_path, field)
        f = glob.glob('{}/**/*netcdf'.format(field_path))
        files.append(f) # open 
    target = glob.glob('{}/target_MESH_Max_30min/MESH_Max_30min/**/*netcdf'.format(storm_path))
    files.append(target)
    for f in files:       
        f = f[0] # unwrap from list
        nc = Dataset(f)
        print(f)
        var = nc.variables[f.split('/')[-3]][:,:]
        var = np.where(var<-20,0,var)
        var_list.append(var)
    f, axs = plt.subplots(2,2,figsize=(15,15))
    ax = plt.gca()
    plt.subplot(121)
    plt.imshow(var_list[0]) # input MESH
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.subplot(122)
    plt.imshow(var_list[3]) # output MESH
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    return files, var_list
        
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
 
