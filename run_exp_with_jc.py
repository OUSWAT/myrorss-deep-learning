##############################################
# Author: Michael Montalbano
# Date: 1/20/22
#
# RUN EXP WITH Job Control 
# tasks: - fix saving results in a df (no unnamed column creation)
#        - add validation set and read during training

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random
import argparse
import time
import numpy as np
# from tensorflow.python.ops.numpy_ops.np_math_ops import true_divide
from u_net_loop import *
from job_control import *
import pickle
import datetime
from sklearn.preprocessing import StandardScaler
#from stats import *
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')

supercomputer = True  # See line 138 where paths are set
swatml = False

# set constants
twice = False

# fix

def create_parser():
    parser = argparse.ArgumentParser(description='Hail Swath Learner')
    parser.add_argument(
        '-ID',
        type=str,
        default='2006',
        help='ID of dataset')
    parser.add_argument(
        '-exp_type',
        type=str,
        default='64a128_filters',
        help='How to name this model?')
    parser.add_argument(
        '-batch_size',
        type=int,
        default=200,
        help='Enter the batch size.')
    parser.add_argument(
        '-dropout',
        type=float,
        default=None,
        help='Enter the dropout rate (0<p<1)')
    parser.add_argument(
        '-lambda_regularization',
        type=float,
        default=None,
        help='Enter l1, l2, or none.')
    parser.add_argument(
        '-epochs',
        type=int,
        default=500,
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
            12],
        help='Enter the number of filters for convolutional network')
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
        default=0.0001,
        help="Learning rate")
    parser.add_argument(
        '-patience',
        type=int,
        default=20,
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
    parser.add_argument('-exp_index', 
        nargs='+',
        type=int,
        default=0,
        help="Used in job_control.")
    parser.add_argument('-thres', 
        type=float,
        default=None,
        help='Threshold (for what purpose?).')
    return parser


def augment_args(args):
    # if you specify exp index, it translates that into argument values that
    # you're overiding
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.
    @return A string representing the selection of parameters to be used in the file name
    '''
    index = args.exp_index
    if(index == -1):
        return ""

    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    # Overides Ntraining and rotation
    if args.lambda_regularization is not None:
        p = {'lambda_regularization': [0.0001, 0.005, 0.01]}
    else:
        p = {'lambda_regularization': [0.01, 0.1],
             'dropout': [0.1,0.5],
             'ID': ['2005','2006','2011']} # test case
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())

    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index <
            ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print('parameters specific to index {}'.format(args.exp_index),ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    # destructively modifies the args
    # string encodes info about the arguments that have been overwritten
    return ji.set_attributes_by_index(args.exp_index, args)


def transform(var):
    print(var.shape)
    n_channels = var.shape[3]
    print(n_channels)
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


def transform_test(var, scalers):
    n_channels = var.shape[3]
    tdata_transformed = np.zeros_like(var)
    channel_scalers = []
    for i in range(n_channels):
        mmx = StandardScaler()
        slc = var[:, :, :, i].reshape(var.shape[0], 60 * 60)
        transformed = mmx.fit_transform(slc)
        transformed = transformed.reshape(
            var.shape[0], 60, 60)  # reshape it back to tiles
        # put it in the transformed array
        tdata_transformed[:, :, :, i] = transformed
        channel_scalers.append(mmx)
    return tdata_transformed, channel_scalers


def training_set_generator_images(ins, outs, batch_size=10,
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

        # The generator will produce a pair of return values: one for inputs
        # and one for outputs
        yield({input_name: ins[example_indices, :, :, :]},
              {output_name: outs[example_indices, :, :, :]})

def generate_fname():
    pass

if(swatml):
    strategy = tf.distribute.MirroredStrategy()
    HOME_PATH = '/home/michaelm/'
    RESULTS_PATH = '/home/michaelm/results'
    DATA_HOME = '/home/michaelm/data/'
elif(supercomputer):
    HOME_PATH = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning'
    RESULTS_PATH = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning/results'
    DATA_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning/datasets'

def load_dataset():
    ins = np.load('datasets/ins_{}.npy'.format(args.ID))
    outs = np.load('datasets/outs_{}.npy'.format(args.ID))
    
    ins_train, ins_val, outs_train, outs_val = train_test_split(
    ins, outs, test_size=0.15, random_state=3)
    ins_train, ins_test, outs_train, outs_test = train_test_split(
    ins_train, outs_train, test_size=0.15, random_state=3)
    ins_train, scalers = transform(ins_train)
    ins_val, scalers = transform(ins_val)
    ins_test, scalers = transform(ins_test)
    return ins_train, outs_train, ins_val, outs_val, ins_test, outs_test

#########################################
# set args and train
def execute_exp(args=None):
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
    # Check the arguments
    params_str = augment_args(args)
    print("after augmenting",args)
    ins_train, outs_train, ins_val, outs_val, ins_test, outs_test = load_dataset()
    start = time.time()
    supercomputer = True
    if not supercomputer: 
        with strategy.scope():
            model = UNet(ins_train.shape[1:], nclasses=1)
    else:
        model = UNet(ins_train.shape[1:], nclasses=1,filters=args.filters, lambda_regularization=args.lambda_regularization, dropout=args.dropout)
    
    with open('model_files/model_{}.txt'.format(args.ID), 'w') as f:  # save model architecture
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    model.summary()  # print model architecture
    # experiment with smaller batch sizes, as large batches have smaller variance
    generator = training_set_generator_images(
                                          ins_train,
                                          outs_train,
                                          batch_size=args.batch_size,
                                          input_name='input',
                                          output_name='output')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                  monitor='mean_squared_error',
                                                  restore_best_weights=True,
                                                  min_delta=0.0)
    # Fit the model
    history = model.fit(x=generator,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps,
                        use_multiprocessing=False,
                        validation_data=(ins_val, outs_val),
                        verbose=True,
                        callbacks=[early_stopping_cb])

    results = {}
    results['time'] = time.time()-start
    results['true_outs'] = outs
    results['predict_training'] = model.predict(ins_train)
    results['predict_training_eval'] = model.evaluate(ins_train, outs_train)
    #results['predict_validation'] = model.predict(ins_val)
    #results['predict_validation_eval'] = model.evaluate(ins_val, outs_val)
    #results['true_validation'] = outs_val
    results['true_testing'] = outs_test
    results['predict_testing'] = model.predict(ins_test)
    results['predict_testing_eval'] = model.evaluate(ins_test, outs_test)
    #results['outs_test_indices'] = outs_test_indices
    results['history'] = history.history
    # Save results
    fbase = r"results/{}_{}e_{}b_{}l2_{}s".format(args.ID, args.epochs, args.batch_size, args.steps)
    results['fname_base'] = fbase
    fp = open("%s_results.pkl" % (fbase), "wb")
    pickle.dump(results, fp)
    fp.close()

    # Model
    model.save("%s_model" % (fbase))
    end = time.time()
    print(fbase)
    print('time:',end-start)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print('before',args)
#    check_args(args) - function to check args are within bounds 
    execute_exp(args)


