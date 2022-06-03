##############################################
# Author: Michael Montalbano
# Date: 1/26/22

# Purpose: This brings loading data and training the models into two functinos: data() and mode()
#          This is a constraint hyperas, which utilizes hyperopt to optimize over chosen options
#          @param: filters [x,x,x]
#          etc

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random
import argparse
import numpy as np
from tensorflow.python.ops.numpy_ops.np_math_ops import true_divide
import pickle
from sklearn.preprocessing import StandardScaler
from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from tensorflow import keras
from tensorflow.keras.layers import GaussianNoise, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Concatenate, Input, UpSampling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import time
import os
import stats
K.set_image_data_format('channels_last')

supercomputer = True
swatml = False
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

# set constants
RESULTS_PATH = '/condo/swatcommon/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning/results'
data_path = r'C:\\Users\\User\\deep_learning\\data'
data_path = '/condo/swatwork/mcmontalbano/SHAVE/data'
#results_path = r'C:\\Users\\User\\deep_learning\\data'
#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

start_time = time.time()
twice = False

def create_parser():
    parser = argparse.ArgumentParser(description='Hail Swath Learner')
    parser.add_argument(
        '-ID',
        type=str,
        default='raw',
        help='ID of dataset (2005, 2006, 2007)')
    parser.add_argument(
        '-exp_type',
        type=str,
        default='l1',
        help='How to name this model?')
    parser.add_argument(
        '-batch_size',
        type=int,
        default=256,
        help='Enter the batch size.')
    parser.add_argument(
        '-dropout',
        type=float,
        default=0.1,
        help='Enter the dropout rate (0<p<1)')
    parser.add_argument(
        '-lambda_regularization',
        type=float,
        default=0.2,
        help='Enter l1, l2, or none.')
    parser.add_argument(
        '-epochs',
        type=int,
        default=100,
        help='Training epochs')
    parser.add_argument(
        '-steps',
        type=int,
        default=10,
        help='Steps per epoch')
    parser.add_argument(
        '-filters',
        type=int,
        default=[
            12,
            12,
            12
            ],
        help='Enter the number of filters for convolutional network')
    parser.add_argument(
        '-loss',
        type=str,
        default='MSE',
        help='Enter a loss function (MSE, MED, etc')
    parser.add_argument(
        '-unet_type',
        type=str,
        default='add',
        help='Enter whether to concatenate or add during skips in unet')
    parser.add_argument(
        '-results_path',
        type=str,
        default=RESULTS_PATH,
        help='Results directory')
    parser.add_argument(
        '-lrate',
        type=float,
        default=0.001,
        help="Learning rate")
    parser.add_argument(
        '-patience',
        type=int,
        default=75,
        help="Patience for early termination")
    parser.add_argument(
        '-network',
        type=str,
        default='unet',
        help='Enter u-net.')
    parser.add_argument(
        '-activation',
        type=str,
        default='relu',
        help='Enter the activation function.')
    parser.add_argument(
        '-optimizer',
        type=str,
        default='adam',
        help='Enter the optimizer.')
    parser.add_argument(
        '-exp_index',
        nargs='+',
        type=int,
        help='Array of integers')
    parser.add_argument(
        '-type',
        type=str,
        default='regression',
        help='How type')
    parser.add_argument(
        '-error',
        type=str,
        default='mse',
        help="What type of error?")
    parser.add_argument(
        '-hyperparameters_string',
        type=str,
        default='',
        help="What are the hyperparameters?")
    parser.add_argument('-thres', type=float, default=None,
                        help='Threshold (for what purpose?).')
    return parser


def data():
    """
    Returns the dataset:
    @xtrain, ytrain, xtest, ytest
    """
    #num_id = random.randint(0, 10000)  # random ID for each model
    #input_shape = (15930, 60, 60, 39)  # hard-code this for the time being
    ID = '2005'
    data_path = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning/datasets/'

    def transform(var):
        '''
        Purpose: scale data b/w 0 and 1
        input:
        @param var - 4D numpy array, channels last
        returns
        @aram tdata_transformed - scaled data between 0 and 1
        @channel_scalers - the scalers used to convert each channel back to its original (60 x 60 form)
        '''
        n_channels = var.shape[3]
        tdata_transformed = np.zeros_like(var)
        channel_scalers = []

        for i in range(n_channels):
            mmx = StandardScaler()
            # make it a bunch of row vectors
            slc = var[:, :, :, i].reshape(var.shape[0], 60 * 60)
            transformed = mmx.fit_transform(slc)
            transformed = transformed.reshape(
                var.shape[0], 60, 60)  # reshape it back to tiles
            # put it in the transformed array
            tdata_transformed[:, :, :, i] = transformed
            channel_scalers.append(mmx)  # store the transform

        return tdata_transformed, channel_scalers

    ins = np.load('{}/ins_{}.npy'.format(data_path, ID))
    outs = np.load('{}/outs_{}.npy'.format(data_path, ID))
    indices = np.asarray(range(ins.shape[0]))

    # train test split 75 25
    rand = 3
    ins_train, ins_test, outs_train, outs_test = train_test_split(
        ins, outs, test_size=0.16, random_state=rand)
    # ins_train_indices, ins_test_indices, outs_train_indices, outs_test_indices = train_test_split(
    #     indices, indices, test_size=0.25, random_state=rand)

    ins_train, scalers = transform(ins_train)
    ins_test, scalers = transform(ins_test)
    return ins_train, outs_train, ins_test, outs_test


def model(x_train, y_train, x_test, y_test, dropout=None):
    '''
    Builds a UNet using for loops.

    @param lambda_reg: l1 or l2 regularization for overfitting
    @param dropout: randomly turns off p % of outputs in a layer, for overfitting
    @param filters: the number of filters per layer. It also controls the number of steps
                    Consider changing this. Can this loop reproduce typical filter-progression for UNets?
    @param activation: pick any activation function within the keras API
    '''
    dropout = None  # figure out why dropout can't be set from within the model call? maybe call it in the main loop?
    batch_size = 256

    def training_set_generator_images(ins, outs, batch_size=50,
                                      input_name='input',
                                      output_name='output'):
        '''
        Generator for producing random minibatches of image training samples.

        @param ins Full set of training set inputs (examples x row x col x chan)
        @param outs Corresponding set of sample (examples x nclasses)
        @param batch_size Number of samples for each minibatch
        @param input_name Name of the model layer that is used for the input of the model
        @param output_name Name of the model layer that is used for the output of the model
        '''

        while True:
            # Randomly select a set of example indices
            example_indices = random.choices(range(ins.shape[0]), k=batch_size)

            # The generator will produce a pair of return values: one for
            # inputs and one for outputs
            yield({input_name: ins[example_indices, :, :, :]},
                  {output_name: outs[example_indices, :, :, :]})

    def lrelu(x): return tf.keras.activations.relu(x, alpha=0.1)
    activation = lrelu

    in_shape = (10642, 60, 39)
    patience = 200
    # used to store high-res tensors for skip connections to upsampled tensors
    # (The Strings in the Net)
    tensor_list = []
    # input images from sample as a tensor
    input_tensor = Input(shape=in_shape, name="input")
    # prevents overfitting by normalizing for each batch, i.e. for each batch
    # of samples
    tensor = BatchNormalization()(input_tensor)
    tensor = GaussianNoise(0.1)(tensor)
    
    # downsampling loop
    filters = [32, 64, 128]
    l2 = {{choice((0.1,0.3,0.5))}}
    for idx, f in enumerate(filters[:-1]):
        tensor = Convolution2D(f,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(l2),
                               activation=activation)(tensor)
   #     if dropout is not None:
   # tensor = Dropout({{uniform(0,1)}})(tensor) # parameter search between
   # p_d = 0 to p_d = 1

        tensor = BatchNormalization()(tensor)
        tensor = Convolution2D(filters[idx + 1],
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(l2),
                               activation=activation)(tensor)
        tensor = BatchNormalization()(tensor)

        tensor_list.append(tensor)  # for use in skip

        tensor = AveragePooling2D(
            pool_size=(
                2, 2), strides=(
                2, 2), padding='same')(tensor)

        # if dropout is not None:
        #    tensor = Dropout({{uniform(0,1)}})(tensor)
    # with downsampling swing finished, lowest res convolutions
    tensor = Convolution2D(filters[-1],  # grab last filter-count
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=tf.keras.regularizers.l2(l2),
                           activation=activation)(tensor)
    if dropout is not None:
        tensor = Dropout(dropout)(tensor)
   
    tensor = BatchNormalization()(tensor)
    tensor = Convolution2D(filters[-1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=tf.keras.regularizers.l2(l2),
                           activation=activation)(tensor)
    tensor = BatchNormalization()(tensor)

    # upsampling loop
    for idx, f in enumerate(list(reversed(filters[:-1]))):
        tensor = UpSampling2D(size=2)(tensor)  # increase dimension

        # Skip connection, Add() or Concatenate()
        tensor = Add()([tensor, tensor_list.pop()])

        tensor = Convolution2D(f,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(l2),
                               activation=activation)(tensor)

#        if dropout is not None:
#            tensor = Dropout({{uniform(0,1)}})(tensor)

        tensor = BatchNormalization()(tensor)
        tensor = Convolution2D(f,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(l2),
                               activation=activation)(tensor)

#        if dropout is not None:
#            tensor = Dropout({{uniform(0,1)}})(tensor)

        tensor = BatchNormalization()(tensor)

    tensor = Convolution2D(filters[0],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=tf.keras.regularizers.l2(l2),
                           activation=activation)(tensor)

    output_tensor = Convolution2D(1,
                                  kernel_size=(1, 1),
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer='random_uniform',
                                  bias_initializer='zeros',
                                  kernel_regularizer=None,
                                  activation=activation, name='output')(tensor)

    opt = keras.optimizers.Adam(
        lr=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    generator = training_set_generator_images(
        ins_train,
        outs_train,
        batch_size=batch_size,
        input_name='input',
        output_name='output')

    # early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,
    #                                                 restore_best_weights=True,
    #                                                 min_delta=0.0)

    # =< n_train // batch_size - artificially creates 'slow cooking' approximately 44 for training
    history = model.fit(x=generator, epochs=250, steps_per_epoch=10)
    # can decay learning rate within a single epoch, if your data is huge
    # https://datascience.stackexchange.com/questions/47405/what-to-set-in-steps-per-epoch-in-keras-fit-generator
    test_mse = history.history['mean_squared_error']
    mse1, mse2 = model.evaluate(ins_test, outs_test)
    # are you sure this is - mse???
    return {'loss': mse1, 'status': STATUS_OK, 'model': model}

#########################################
# set args and train
if __name__ == '__main__':
    info = 'testing'
    start = time.time()
    exp_type = 'dropout_search'
    ID = '2011_qc'
    ins_train, outs_train, ins_test, outs_test = data()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,  # run TPE algorithm from hyperopt for optimization
                                          max_evals=5,        # at most 10 evaluation runs
                                          trials=Trials())
    model = best_model
    with open('model_files/{}_{}.txt'.format(exp_type, ID), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    model.summary()

    results = {}
    #results['args'] = args
    #results['true_outs'] = outs
    results['predict_training'] = model.predict(ins_train)
    results['predict_training_eval'] = model.evaluate(ins_train, outs_train)
    results['true_training'] = outs_train
    results['true_testing'] = outs_test
    results['predict_testing'] = model.predict(ins_test)
    results['predict_testing_eval'] = model.evaluate(ins_test, outs_test)

    # Save results
    fbase = r"results/{}_{}_{}_{}e_{}b_{}l2_{}s".format(args.loss, args.ID, args.exp_type, args.epochs, args.batch_size, args.lambda_regularization, args.steps)
    results['fname_base'] = fbase
    fp = open("%s_results.pkl" % (fbase), "wb")
    pickle.dump(results, fp)
    fp.close()

    # Model
    model.save("%s_model" % (fbase))
    end = time.time()
    print(fbase)
    print('time:',end-start)

# save model in txt
