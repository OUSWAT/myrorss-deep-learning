##############################################
# Author: Michael Montalbano
# Date: 1/20/22
# 
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
from stats import *
import pickle
import datetime
from sklearn.preprocessing import StandardScaler
#from stats import *
import tensorflow.keras.backend as K
#from unet import create_uNet
import stats
K.set_image_data_format('channels_last')

supercomputer = True  # See line 138 where paths are set
swatml = False

# set constants
twice = False


def create_parser():
    parser = argparse.ArgumentParser(description='Hail Swath Learner')
    parser.add_argument(
        '-ID',
        type=str,
        default='raw',
        help='ID of dataset (2005, 2006, 2007, shavelike)')
    parser.add_argument(
        '-exp_type',
        type=str,
        default='l2',
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
        default=0.1,
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
            32,
            64,
            128
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
        p = {'lambda_regularization: [0.0001, 0.005, 0.01]',
             'activation: ["elu","sigmoid","tanh","relu"]',
             'optimizer: ["adam","RMSProp","SGD-momentum"]'}

    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())

    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index <
            ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))

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

    for i in range(n_channels-1): # don't tranform Input MESH
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

# Standard MSE loss function plus term penalizing only misses

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
parser = create_parser()
args = parser.parse_args()

ins = np.load('datasets/ins_{}.npy'.format(args.ID))
outs = np.load('datasets/outs_{}.npy'.format(args.ID))
indices = np.asarray(range(ins.shape[0]))

ins_train, ins_val, outs_train, outs_val = train_test_split(
    ins, outs, test_size=0.16, random_state=3)
ins_train, ins_test, outs_train, outs_test = train_test_split(
    ins_train, outs_train, test_size=0.16, random_state=3)
#ins_train_indices, ins_test_indices, outs_train_indices, outs_test_indices = train_test_split(
#    indices, indices, test_size=0.25, random_state=3)
# scaling
ins_train, scalers = transform(ins_train)
ins_val, scalers = transform(ins_val)
#pickle.dump(scalers, open('scalers/scaler_{}.pkl'.format(args.ID), 'wb'))
ins_test, scalers = transform(ins_test)


start = time.time()
if swatml:
    with strategy.scope():
        model = UNet(ins_train.shape[1:], nclasses=1)
elif supercomputer:
    model = UNet(ins_train.shape[1:], args.loss, nclasses=1,filters=args.filters, lambda_regularization=args.lambda_regularization, dropout=args.dropout)  # create model
    '''
    model = create_uNet(ins_train.shape[1:], nclasses=5,lambda_regularization=args.lambda_regularization,
                        activation=args.activation, dropout=args.dropout,
                        type=args.type, optimizer=args.optimizer,threshold=args.thres)
    '''
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
                                                  monitor='val_loss',
                                                  restore_best_weights=True,
                                                  min_delta=0.0)

checkpoint_cb = keras.callbacks.ModelCheckpoint("results/chkpt_model.h5")
# Fit the model
history = model.fit(x=generator,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps,
                    use_multiprocessing=False,
                    validation_data=(ins_val, outs_val),
                    verbose=True,
                    callbacks=[early_stopping_cb,checkpoint_cb])

results = {}
results['true_outs'] = outs
results['predict_training'] = model.predict(ins_train)
results['predict_training_eval'] = model.evaluate(ins_train, outs_train)
results['true_training'] = outs_train
#results['predict_validation'] = model.predict(ins_val)
#results['predict_validation_eval'] = model.evaluate(ins_val, outs_val)
#results['true_validation'] = outs_val
results['true_testing'] = outs_test
results['predict_testing'] = model.predict(ins_test)
results['predict_testing_eval'] = model.evaluate(ins_test, outs_test)
#results['outs_test_indices'] = outs_test_indices
#results['folds'] = folds
results['history'] = history.history

# Save results
fbase = r"results/{}_{}_{}_{}e_{}b_{}l2_{}s".format(args.loss, args.ID, args.exp_type, args.epochs, args.batch_size, args.lambda_regularization, args.steps)
results['fname_base'] = fbase
fp = open("{}_results.pkl".format(fbase), "wb")
pickle.dump(results, fp)
fp.close()

# Model
model.save("{}_model.h5".format(fbase)) # necessary if using custom metrics
end = time.time()
print(fbase)
print('time:',end-start)

ins = np.load('datasets/ins_shavelike.npy')
outs = np.load('datasets/outs_raw.npy')

print(model.evaluate(ins,outs))
