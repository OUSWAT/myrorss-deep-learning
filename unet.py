# Author: Michael Montalbano
# Create U-Network or Autoencoder

from turtle import xcor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import GaussianNoise, AveragePooling2D, SpatialDropout2D, BatchNormalization, Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Concatenate, Input, UpSampling2D, Add
from tensorflow.keras.layers import Conv2DTranspose, RandomTranslation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error
import sys
import numpy as np
import metrics
import loss_functions
import random
import keras_tuner as kt
from blocks import resnet_block

class UNet(object):
    def __init__(self, args, dataset, input_shape=(60,60,41)):
        self.args = args # pass it around
        print(self.args) 
        self.dataset = dataset # 
        self.input_shape = self.dataset.xtr.shape[1:] 
        self.dropout = args.dropout 
        self.depth = args.depth
        self.l2 = args.L2
        self.junction = args.junction
        self.tensor_list = []
        self.filters = args.filters
        self.custom_object_names = []
        self.custom_objects = {}
        self.metrics_fncts = [] 
        self.use_pooling = False
        if self.dropout > 0:
            self.use_dropout = False
        else: self.use_dropout = True 
        self.use_transpose = bool(args.use_transpose)
        self.transpose_params = dict(kernel_size=(2,2), padding='same', strides=(2,2),use_bias=False)
        self.paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
        self.model = None
        self.translation_factor = self.args.t_fac
        self.gauss_factor = self.args.g_fac
        self.gain = args.gain
        self.bias = args.bias 
        self.std_gain = args.std_gain
        self.init_kind = args.init_kind
        self.initializer = tf.keras.initializers.RandomNormal()
        self.conv_params =dict(kernel_size=(3,3), padding='same',kernel_initializer=self.initializer, activation='relu',use_bias=False)
        self.use_resnet = True

    def double_conv_block(self,tensor,filters,use_resnet=True,kernel=(3,3)):
        x = Convolution2D(filters,  # grab last filter-count
                           kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                           **self.conv_params)(tensor)
        tensor = BatchNormalization()(x)
        tensor = Convolution2D(filters,  # grab last filter-count
                           kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                           **self.conv_params)(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Convolution2D(filters,  # grab last filter-count
                           kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                           **self.conv_params)(tensor)
        if self.use_dropout:
            tensor = SpatialDropout2D(self.dropout)(tensor)
        tensor = BatchNormalization()(tensor)
        if use_resnet:
            tensor = layers.add([x, tensor])
        return tensor

    def output_block(self,tensor, nclasses, activation=None,kernel=(1,1), name='output'):
        if activation is None:
            activation = self.activation
        tensor = Convolution2D(nclasses,  # grab last filter-count
                           kernel_size=kernel,
                           padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                           activation=activation,name=name)(tensor)
        return tensor

    def conv_chain_down(self,tensor):
        filters = self.filters[:-1]
        for idx, f in enumerate(filters):
            if idx == 0:
                self.use_resnet = False
            else:
                self.use_resnet = True
            tensor = self.double_conv_block(tensor,f,self.use_resnet)
            self.tensor_list.append(tensor)  # for use in skip
            # use convolution to stride down and reduce dimension by 2
            if not self.use_pooling:
                tensor = Convolution2D(f, strides=(2,2), **self.conv_params)(tensor)
            else:
                tensor = AveragePooling2D(
                    pool_size=(
                        2, 2), strides=(
                        2, 2), padding='same')(tensor)
            tensor = BatchNormalization()(tensor)
        return tensor

    def conv_chain_up(self,tensor):
        filters = list(reversed(self.filters[:-1]))        
        for idx, f in enumerate(filters):
            if self.use_transpose:
                tensor = Conv2DTranspose(f, **self.transpose_params)(tensor)
            else:
                tensor = UpSampling2D(size=2)(tensor)  # increase dimension
                tensor = Convolution2D(f, **self.conv_params)(tensor)
            if self.junction == 'Add':
                print(self.tensor_list)
                popped_tensor = self.tensor_list.pop()
                tensor = Add()([tensor, popped_tensor])  # skip connection
            else:
                tensor = Concatenate()([tensor, self.tensor_list.pop()])
            tensor = self.double_conv_block(tensor,f)
            if self.use_dropout:
                tensor = SpatialDropout2D(self.dropout)(tensor)
            tensor = BatchNormalization()(tensor)
        return tensor
    
    def preprocess(self, tensor):
        tensor = RandomTranslation(height_factor=self.translation_factor, width_factor=self.translation_factor)
        tensor = GaussianNoise(self.gauss_factor)(tensor)
        return tensor # you could just keep an updating self.tensor object as you go
        # if p > 0.5:
        #     return RandomTranslation(height_factor=self.trans_factor)(tensor)
        # else:
        #     return RandomTranslation(height_factor=0,width_factor=self.trans_factor)
        
    def build_model(self):
        print('building model')
        input_tensor = Input(self.input_shape, name='input')
        tensor = BatchNormalization()(input_tensor)
        tensor = GaussianNoise(0.1)(tensor)
        tensor = RandomTranslation(self.args.t_fac, self.args.t_fac)(tensor)
        self.set_conv_params()
        tensor = self.conv_chain_down(tensor)
        tensor = self.double_conv_block(tensor, self.filters[-1])
        tensor = self.double_conv_block(tensor, self.filters[-1])
        tensor = self.conv_chain_up(tensor)
        output_tensor = self.output_block(tensor,nclasses=1)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        self.model = model # save model as attribute of UNet
        print('built the model',model.summary())
        print(self.gain,self.std_gain,self.bias)
        return model
    
    def build_tuning_model(self, hp):
        self.set_tuning_parameters(hp)
        self.set_conv_params()
        print('building tuning model')
        self.build_model()
        self.compile_model()
        return self.model
        
    def set_tuning_parameters(self, hp):
        self.translation_factor = hp.Float('translation_factor', min_value=0.0, max_value=0.6, step=0.2) # will it just
        self.gauss_factor = hp.Float('gauss_factor', min_value=0.0, max_value=1.0, step=0.25)
        self.depth = hp.Int('depth',min_value=2,max_value=3,step=1)
        self.junction =  hp.Choice('junction', ['Add', 'Concatenate'])
       # self.gain = hp.Choice('initializer_gain', [0.05,0.1,0.5,0.7,0.9])
       # self.std_gain = hp.Choice('multiple_std', [1,2,4,5,7,9])
       # self.bias = hp.Choice('bias_value', [0.05,0.1,0.5])
        self.dropout = hp.Float('dropout', min_value=0.0, max_value=0.6, step=0.2)
        self.L2 = hp.Float('L2', min_value=0.0, max_value=0.6, step=0.2)
        self.filters = []
        for idx in np.arange(self.depth):
            imin =idx+1
            imax=idx+3
            f = hp.Int(f'layer_{idx}_filters', min_value=16*imin, max_value=32*imax, step=16)
            self.filters.append(f)
        #activation = hp.Choice('activation', ['swish','lrelu','linear'])

    def set_conv_params(self):
        if self.init_kind == 'Normal':
            self.initializer = tf.keras.initializers.RandomNormal(mean=self.gain, stddev=0.05*self.std_gain)
        elif self.init_kind == 'Uniform':
            self.initializer = tf.keras.initializers.RandomUniform(minval=-0.1*self.gain, maxval=0.1*self.gain)
        self.conv_params = dict(kernel_size=(3,3), padding='same',kernel_initializer=self.initializer, activation=self.activation, bias_initializer=tf.constant_initializer(self.bias))

    def compile_model(self):
        self.set_optimizer()
        print('metrics follow',self.metrics_fncts)
        print(self.loss_function)
        self.model.compile(loss=self.loss_function,optimizer=self.optimizer,metrics=self.metrics_fncts, run_eagerly=True)
        #loaded_model = keras.models.load_model('my_model')
        #self.model = loaded_model.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics_fncts, run_eagerly=True)
        return self.model

    def set_loss(self,loss_fn='mse'):
        self.loss_fn = loss_fn
        if loss_fn != 'mse' and loss_fn != 'bce':
            print('inside the set_loss method',loss_fn)
            function = getattr(loss_functions, str(loss_fn))
            fnct = function(scaler=self.dataset.ytr_scalers[0])
            self.custom_objects[loss_fn] = fnct
            self.loss_function = fnct
            print(self.custom_objects)
        elif loss_fn == 'bce':
            self.dataset.binarize_y()
            self.loss_function = tf.keras.losses.BinaryCrossentropy()

        else:
            self.loss_function=loss_fn

    def set_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    def set_metrics(self,metrics_names=['POD','FAR']):
        print('Setting metrics')
        if self.loss_fn != 'bce':
            for metric in metrics_names:
                function = getattr(metrics, metric)
                fnct = function()
                self.metrics_fncts.append(fnct)
                self.custom_objects[metric] = fnct
        else:
            self.metrics_fncts.append('binary_accuracy')

    def set_activation(self,name='lrelu'):
        def lrelu(x): return tf.keras.activations.relu(x, alpha = 0.1)
        def linear(x): return tf.keras.activations.linear(x)
        def swish(x): return tf.keras.activations.swish(x)
        def sigmoid(x): return tf.keras.activations.sigmoid(x)
        activations = dict(lrelu=lrelu, linear=linear, swish=swish, sigmoid=sigmoid)
        self.activation = activations[name]
        self.custom_objects[name] = self.activation

