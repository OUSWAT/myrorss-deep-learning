#######################################################
# Author: Michael Montalbano
# Date: 01/13/2021
# Purpose: build U-Net for training by basic.py
# Rather than unet.py, the network is built
# using loops that make hyperparameter-search easier
#######################################################

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
# import layers from keras like RandomContrast
from tensorflow.keras.layers import RandomContrast, RandomRotation, RandomZoom, RandomFlip, CenterCrop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import GaussianNoise, AveragePooling2D, Dropout, BatchNormalization, SpatialDropout2D, RandomTranslation
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Concatenate, Input, UpSampling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error
from tensorflow import keras
#from customs import *
#from helper_functions import mean_iou, dice_coef, dice_coef,loss, compute_iou
binary=False
mse=True

tf.config.run_functions_eagerly(True)

def UNet(input_shape,loss, batch_size=512, lrate=0.0003,dropout=None, lambda_regularization=None, filters=[16, 32],
         junction='Add', nclasses=1, constant_loss=None,activation='relu', 
         preprocessing=[True,False,False,False,False,False,False,False,False,True]):
    '''
    Builds a UNet using for loops.

    @param lambda_reg: l1 or l2 regularization for overfitting
    @param dropout: randomly turns off p % of outputs in a layer, for overfitting
    @param filters: the number of filters per layer. It also controls the number of steps
                    Consider changing this. Can this loop reproduce typical filter-progression for UNets?
    @param activation: pick any activation function within the keras API
    '''
    if activation == 'lrelu':
        def lrelu(x): return tf.keras.activations.relu(x, alpha=0.1)
        activation = lrelu
           
    
    # input images from sample as a tensor
    input_tensor = Input(shape=input_shape, name="input")
    # Preprocessing here 
    # use boolean list to turn on/off preprocessing layers like RandomContrast, RandomRotation, etc.
    factor = 0.3
    if preprocessing[0]:
        input_tensor = RandomTranslation(height_factor=factor,width_factor=factor,fill_mode='constant') (input_tensor)
    if preprocessing[1]:
        input_tensor = RandomContrast(factor,factor) (input_tensor)
    if preprocessing[2]:
        input_tensor = RandomRotation(factor,factor) (input_tensor)
    if preprocessing[3]:
        input_tensor = RandomZoom(factor,factor) (input_tensor)
    if preprocessing[4]:
        input_tensor = RandomSizedCrop(factor,factor) (input_tensor)
    if preprocessing[5]:
        input_tensor = RandomFlip(factor) (input_tensor)
    if preprocessing[6]:
        input_tensor = GaussianNoise(factor) (input_tensor)

    # of samples
    tensor = BatchNormalization()(input_tensor)
    tensor_list = []
        # downsampling loop
    for idx, f in enumerate(filters[:-1]):
        tensor = Convolution2D(f,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(lambda_regularization),
                               activation=activation)(tensor)
        if dropout is not None:
            tensor = SpatialDropout2D(dropout)(tensor)
        tensor = BatchNormalization()(tensor)
    # with downsampling swing finisor = BatchNormalization()(tensor)
        tensor = Convolution2D(filters[idx + 1],
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(lambda_regularization),
                               activation=activation)(tensor)
        if dropout is not None:
            tensor = SpatialDropout2D(dropout)(tensor)
        tensor = BatchNormalization()(tensor)
        tensor_list.append(tensor)  # for use in skip
        tensor = AveragePooling2D(
            pool_size=(
                2, 2), strides=(
                2, 2), padding='same')(tensor)
    tensor = Convolution2D(filters[-1],  # grab last filter-count
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=tf.keras.regularizers.l2(lambda_regularization),
                           activation=activation)(tensor)
    if dropout is not None:
        tensor = SpatialDropout2D(dropout)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Convolution2D(filters[-1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=tf.keras.regularizers.l2(lambda_regularization),
                           activation=activation)(tensor)
    if dropout is not None:
        tensor = SpatialDropout2D(dropout)(tensor)
    tensor = BatchNormalization()(tensor)
    # upsampling loop
    for idx, f in enumerate(list(reversed(filters[:-1]))):
        tensor = UpSampling2D(size=2)(tensor)  # increase dimension
        if junction == 'Add':
            tensor = Add()([tensor, tensor_list.pop()])  # skip connection
        else:
            tensor = Concatenate()([tensor, tensor_list.pop()])
        tensor = Convolution2D(f,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(lambda_regularization),
                               activation=activation)(tensor)
        if dropout is not None:
            tensor = SpatialDropout2D(dropout)(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Convolution2D(f,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(lambda_regularization),
                               activation=activation)(tensor)
        if dropout is not None:
            tensor = SpatialDropout2D(dropout)(tensor)
        tensor = BatchNormalization()(tensor)
    tensor = Convolution2D(filters[0],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=tf.keras.regularizers.l2(lambda_regularization),
                           activation=activation)(tensor)

    logits = Convolution2D(1,
                           kernel_size=(1, 1),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=None,
                           activation=activation, name='output')(tensor)
    
    targets = keras.Input(shape=(60,60,1), name='targets')
   
    if loss != 'mse':
        predictions = G_Beta(name='predictions')(logits, targets)
    else:
        predictions = MSE_layer(name='predictions')(logits, targets)

    opt = tf.keras.optimizers.Adam(
        lr=lrate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False)

    model = Model(inputs=[input_tensor,targets], outputs=predictions)
    model.compile(optimizer=opt)    

    return model

def create_UNetPlusPlus():
    # TBD
    pass

def my_custom_MSE():
    def inner(y_true, y_pred):
        return tf.math.reduce_mean(tf.square(y_true - y_pred))
    return inner

def my_POD(cutoff=25.4):
    # computes the probability of detection
    # POD = PT / TP + FN
    def pod(y_true, y_pred):
        y_true = tf.where(y_true>cutoff,1,0)
        y_pred = tf.where(y_pred>cutoff,1,0)
     
        TP = tf.math.reduce_sum(tf.where(((y_true-1)+y_pred)<1,0,1))
        FN = tf.math.reduce_sum(tf.where(y_true-y_pred<1,0,1))
        FP = tf.math.reduce_sum(tf.where(y_pred-y_true<1,0,1))
        return TP/(TP+FN)
    return pod

class MSE_layer(keras.layers.Layer):
    def __init__(self, name='Gbeta_loss'):
        super(MSE_layer, self).__init__(name=name)
        self.loss_fn = my_custom_MSE()
        self.POD_fn = my_POD(20)

    def call(self, y_true, y_pred, sample_weights=None):
        loss = self.loss_fn(y_true, y_pred)
        self.add_loss(loss)

        # log the metric
        POD = self.POD_fn(y_true, y_pred)
        return tf.keras.activations.relu(y_pred,alpha=0.1)