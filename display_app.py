# Michael Montalbano
# Streamlit app for comparing images





import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex

MESH_colors = ['#aaaaaa','#00ffff','#0080ff','#0000ff','#007f00','#00bf00','#00ff00','#ffff00','#bfbf00','#ff9900','#ff0000','#bf0000','#7f0000','#ff1fff']
MESH_bounds = [9.525,15.875,22.225,28.575,34.925,41.275,47.625,53.975,60.325,65,70,75,80,85]

st.write("""
# Data Exploration
Compare MESH images.
""")

loss_functions = ['mse', 'mae']

loss = st.radio("Pick a loss function", loss_functions)

#if loss == 'mse':
#    y_pred = load('data/y_pred_mse_app.npy')
#else:
#    y_pred = load('data/y_pred_{}_app.npy'.format(loss))
y_1 = load('toy_datasets/MESH_subset_2011.npy')
y_2 = load('toy_datasets/MESH_subset_shave.npy'

number = st.number_input("Pick a sample number (0-939)",0,939)

y_1 = np.squeeze(y_1[number])
y_2 = np.squeeze(y_2[number])

f, axs = plt.subplots(1,2,figsize=(16,8))

plt.subplot(121)
ax = plt.gca()
cs = plt.contourf(y_1,levels=MESH_bounds,colors=MESH_colors, extend='both',orientation='horizontal', shrink=0.5, spacing='proportional')   
plt.colorbar(cs, ticks=MESH_bounds)
f.tight_layout(pad=3.0)
plt.ylabel('y (1/2 km)')
plt.xlabel('x (1/2 km)')
plt.xlim([0,60])
plt.xticks([0,10,20,30,40,50,60])
plt.yticks([0,10,20,30,40,50,60])
plt.title('True MESH  #{} (mm)'.format(number))

plt.subplot(122)
ax = plt.gca()
cs = plt.contourf(y_1,levels=MESH_bounds,colors=MESH_colors, extend='both',orientation='horizontal', shrink=0.5, spacing='proportional')   
plt.colorbar(cs, ticks=MESH_bounds)
f.tight_layout(pad=3.0)
plt.ylabel('y (1/2 km)')
plt.xlabel('x (1/2 km)')
plt.xlim([0,60])
plt.xticks([0,10,20,30,40,50,60])
plt.yticks([0,10,20,30,40,50,60])
plt.title('Predicted MESH with {} #{} (mm)'.format(loss,number))

st.pyplot(f)

ax = plt.gca()

