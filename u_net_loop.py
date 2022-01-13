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
from tensorflow.keras.layers import GaussianNoise, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Concatenate, Input, UpSampling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error


def UNet(input_shape, nclasses=2, filters=[12,24], 
            lambda_regularization = None, dropout = None,
            activation='relu')
            '''
            purpose: function allowing for hyperparemters in the number of filters per "step"
            lambda_reg: l1 or l2 regularization for overfitting
            dropout: randomly turns off p % of outputs in a layer, for overfitting
            '''

    tensor_list = [] # used to store high-res tensors for skip connections to upsampled tensors (The Strings in the Net)
    input_tensor = Input(shape=input_shape, name="input") # input images from sample as a tensor
    tensor = BatchNormalization(input_tensor)(tensor)     # prevents overfitting by normalizing for each batch, i.e. for each batch of samples
    
    # Add gaussian noise 

    # downsampling loop
    for idx, f in enumerate(filters):
        tensor = Convolution2D(f,
                            kernel_size=(3,3),
                            use_bias=True,
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            kernel_regularizer=lambda_regularization,
                            activation=activation)(tensor)

        tensor = BatchNormalization()(tensor)
    
        tensor = Convolution2D(f,
                            kernel_size=(3,3),
                            use_bias=True,
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            kernel_regularizer=lambda_regularization,
                            activation=activation)(tensor)
    
        tensor = BatchNormalization()(tensor)

        tensor_list.append(tensor) # for use in skip

        tensor = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')(tensor)

    # with downsampling swing finished, lowest res convolutions
    tensor = Convolution2D(filters[-1],  # grab last filter-count
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)  
    tensor = BatchNormalization()(tensor)

    tensor = UpSampling2D(size=2) (tensor) # increase dimension


    # upsampling loop
    for idx, f in enumerate(filters):
        
        # Skip connection, Add()
        tensor = Add()([tensor, tensor_list.pop()])

        tensor = Convolution2D(f,
                            kernel_size=(3,3),
                            use_bias=True,
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            kernel_regularizer=lambda_regularization,
                            activation=activation)(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Convolution2D(f,
                            kernel_size=(3,3),
                            use_bias=True,
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            kernel_regularizer=lambda_regularization,
                            activation=activation)(tensor)
        tensor = BatchNormalization()(tensor)

        tensor = UpSampling2D(size=2) (tensor) # increase dimension

    output_tensor = Convolution2D(1,
                        kernel_size=(1,1),
                        padding='same', 
                        use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=lambda_regularization,
                        activation=activation,name='output')(tensor)
    
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                      metrics=[tf.keras.metrics.MeanSquaredError()])

    # model.compile(loss=custom,optimizer=opt,metrics=['mse'])
    return model



    

