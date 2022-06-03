
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle, sys, glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import figure
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.colors import rgb2hex
from netCDF4 import Dataset



NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km']
multi_fields = ['MergedReflectivityQCComposite_Max_30min','MergedLLShear_Max_30min','MergedMLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min']
products = NSE_fields + multi_fields

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

def open_pickle(file):
    fp = open(file,'rb')
    r = pickle.load(fp)
    fp.close()
    return r

def plot_predict(r,idx,group='testing'):
    f, axs = plt.subplots(1,2,figsize=(15,15))
    
    plt.subplot(121)
    true = r['true_{}'.format(group)][idx]
    ax = plt.gca()
    im = ax.imshow(true,cmap=cm.nipy_spectral)
    plt.xlabel('Degrees Longitude')
    plt.ylabel('Degrees Latitude')
    plt.title('True MESH {} Set {}'.format(group,idx))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(122)
    pred = r['predict_{}'.format(group)][idx]
    ax = plt.gca()
    im = ax.imshow(pred,cmap=cm.nipy_spectral)
    plt.title('Predicted MESH {} Set {}'.format(group,idx))
    plt.xlabel('Degrees Longitude')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def plot_extra(r,idx,other_idx, group='testing'): 

    mse, indices = get_mse(r,'testing')
    f, axs = plt.subplots(2,4,figsize=(10,14))

    plt.subplot(421)
    ax = plt.gca()
    bounds = MESH_bounds
    true = r['true_{}'.format(group)][idx]
    x = np.squeeze(true)
    cs = plt.contourf(x, levels=bounds, colors=MESH_colors, extend='both',orientation='horizontal', shrink=0.5, spacing='proportional')   

    plt.colorbar(cs, ticks=bounds)
    f.tight_layout(pad=3.0)
    plt.ylabel('y (1/2 km)')
    plt.xlim([0,60])
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])
    plt.title('True MESH {} Set #{} (mm)'.format(group,idx))


########################################################################
    plt.subplot(422)
    pred = r['predict_{}'.format(group)][idx]

    x = np.squeeze(pred)
    bounds = MESH_bounds
    cs = plt.contourf(x, levels=bounds, colors=MESH_colors, extend='both')    
    plt.colorbar(cs, ticks=bounds)
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])
    plt.title('Predicted MESH {} Set #{} (mm)'.format(group,idx))
    plt.ylabel('y (1/2 km)')

def plot_predictions(prediction_pairs, title, netcdf=False):
    # prediction_pairs should be of form zip(trues,preds)
    max_images = 128
    grid_width = 16
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for i, (true, pred) in enumerate(prediction_pairs):
        ax = axs[int(i / grid_width), i % grid_width
        ax.contourf(true,levels=MESH_bounds,colors=MESH_colors, extend='both')   
        ax.contour(pred, [20,35,50])
        #plt.colorbar(cs,ticks=bounds)
        ax.axis('off')
    fig.savefig(f'{title}.png')
    return fig
