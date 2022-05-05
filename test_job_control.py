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

#################################################################
def check_args(args):
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.lambda_regularization is None or (args.lambda_regularization > 0.0 and args.lambda_regularization < 1)), "L2_regularizer must be between 0 and 1"
    

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
    p = {'lambda_regularization': [0.01, 0.1],
         'dropout': [0.1,0.5]} # test case
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


if(swatml):
    strategy = tf.distribute.MirroredStrategy()
    HOME_PATH = '/home/michaelm/'
    RESULTS_PATH = '/home/michaelm/results'
    DATA_HOME = '/home/michaelm/data/'
elif(supercomputer):
    HOME_PATH = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning'
    RESULTS_PATH = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning/results'
    DATA_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning/datasets'


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
    print(args.lambda_regularization)
    print(args.dropout)
    print('experimental index',args.exp_index)
    check_args(args)
    print(args.lambda_regularization)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print('before',args)
#    check_args(args) - function to check args are within bounds 
    execute_exp(args)


