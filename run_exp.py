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
from u_net_loop import *
import pickle
import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K
from job_control import *
K.set_image_data_format('channels_last')
from customs import *
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
        default='random_translation',
        help='How to name this model?')
    parser.add_argument(
        '-preprocess',
        default='1000001',
        help='Binary string instructions for preprocessing layers to use in UNet.')
    parser.add_argument(
        '-batch_size',
        type=int,
        default=256,
        help='Enter the batch size.')
    parser.add_argument(
        '-loss2',
        type=str,
        default=None,
        help='Enter the second loss function')
    parser.add_argument(
        '-dropout',
        type=float,
        default=None,
        help='Enter the dropout rate (0<p<1)')
    parser.add_argument(
        '-L2',
        type=float,
        default=None,
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
        '-epochs2',
        type=int,
        default=5,
        help='Enter number of epochs for second training.')
    parser.add_argument(
        '-filters',
        type=int,
        default=[
            32,
            64],
        help='Enter the number of filters for convolutional network')
    parser.add_argument(
        '-factor',
        type=float,
        default='0.1',
        help='Enter factor to be used in preprocessing (i.e. GaussianNoise, RandomTranslation, etc).')
    parser.add_argument(
        '-junction',
        type=str,
        default='Add',
        help='Enter either Add or Concat to specify skip connection junction')
    parser.add_argument(
        '-loss',
        type=str,
        default='MSE',
        help='Enter a loss function (MSE, MED, etc')
    parser.add_argument(
        '-results_path',
        type=str,
        default='results',
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
    if not isinstance(index, int):
        index = 0

    p = {'L2': [0.01, 0.1, 0.3, 0.5],
         'dropout': [0.1, 0.3, 0.5],
         'junction': ['Add', 'Concat'],
         'filters': [[32, 64], [64, 128], [32, 64, 128], [16, 32, 64]]}
 
    cartesian_product = list(dict(zip(p,x)) for x in product(*p.values()))
    element = cartesian_product[index]
    for k, v in element.items():
        setattr(args, k, v)
    return args


def transform(var):
    n_channels = var.shape[3]
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
    while True:
        # Randomly select a set of example indices
        example_indices = random.choices(range(ins.shape[0]), k=batch_size)

        # The generator will produce a pair of return values: one for inputs
        # and one for outputs
        yield({input_name: ins[example_indices, :, :, :]},
              {output_name: outs[example_indices, :, :, :]})

def main():
    # set args and train
    parser = create_parser()
    args = parser.parse_args()
    args = augment_args(args)

    dataset = np.load('datasets/dataset_{}.npz'.format(args.ID))
    #train_dataset = tf.data.Dataset.from_tensor_slices((dataset['x_train'], dataset['y_train']))
    #val_dataset = tf.data.Dataset.from_tensor_slices((dataset['x_val'], dataset['y_val']))
    #test_dataset = tf.data.Dataset.from_tensor_slices((dataset['x_test'], dataset['y_test']))
    ins_train, outs_train = dataset['x_train'], dataset['y_train']
    ins_val, outs_val = dataset['x_val'], dataset['y_val']
    ins_test, outs_test = dataset['x_test'], dataset['y_test']

    # scaling
    ins_train, scalers = transform(ins_train)
    ins_val, scalers = transform(ins_val)
    #pickle.dump(scalers, open('scalers/scaler_{}.pkl'.format(args.ID), 'wb'))
    ins_test, scalers = transform(ins_test)

    start = time.time()

    # rewrite model call in shortest form on multiple lines
    model = UNet(ins_train.shape[1:], args.loss, args.batch_size, args.lrate, args.dropout, args.L2, args.filters, 
                args.junction, args.factor, args.preprocess)  # create model

    model_prefix = f'loss-{args.loss}_dataset-{args.ID}_l2-{args.L2}_dropout-{args.dropout}_factor-{args.factor}_prefix-{args.preprocess}_junction-{args.junction}'

    with open(f'model_files/{model_prefix}.txt'.format(args.ID), 'w') as f:  # save model architecture
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    model.summary()  # print model architecture

    # experiment with smaller batch sizes, as large batches have smaller variance
    generator = training_set_generator_images(
        ins_train,
        outs_train,
        batch_size=args.batch_size,
        input_name='input_1',
        output_name='output')

    callbacks_list = [keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=20,
                verbose=1),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_delta=0,
                min_lr=0.0001),
            keras.callbacks.BackupAndRestore(
                backup_dir='tmp/backup'), # saves weights to retrain model if interrupted
            keras.callbacks.CSVLogger(
                filename=f'csv/{model_prefix}.csv'),
            keras.callbacks.ModelCheckpoint(
                filepath=f"ckpt/{model_prefix}.h5",
                save_best_only=True,  # Only save a model if `val_loss` has improved.
                monitor="val_loss",
                save_freq=25,),
            keras.callbacks.TensorBoard(
                log_dir='log',
                histogram_freq=10,
                update_freq='epoch')]

    #checkpoint_cb = keras.callbacks.ModelCheckpoint("results/.h5")
    # Fit the model
    history = model.fit(x=generator,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps,
                        use_multiprocessing=False,
                        validation_data=(ins_val, outs_val),
                        verbose=True,
                        callbacks=callbacks_list)

    np.savez(f'results/predictions/{model_prefix}.npz', targets = outs_test, predictions = model.predict(ins_test))

    results = {}
    #results['true_outs'] = outs
    results['predict_training_eval'] = model.evaluate(ins_train, outs_train)
    results['predict_testing_eval'] = model.evaluate(ins_test, outs_test)
    #results['outs_test_indices'] = outs_test_indices
    results['history'] = history.history

    # Save results

    results['fname_base'] = model_prefix
    fp = open(f"results/{model_prefix}_results.pkl", "wb")
    pickle.dump(results, fp)

if __name__ == '__main__':
    main()
