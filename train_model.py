from inspect import getmodule
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import pickle
import random
import argparse
from sklearn.preprocessing import StandardScaler
import time
from u_net_loop import *
import datetime
from itertools import product
from customs import *
tf.config.run_functions_eagerly(True) # best for experimental, but slower than graph-mode

def create_parser():
    parser = argparse.ArgumentParser(description='Hail Swath Learner')
    parser.add_argument(
        '-batch_size',
        type=int,
        default=512,
        help='Enter the batch size.')
    parser.add_argument(
        '-custom_training',
        type=bool,
        default=False,
        help='Enter boolean (T if custom_training, F if model.fit()')
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
        '-loss',
        type=str,
        default='G_beta_IL',
        help='Enter a loss function (MSE, MED, etc')
    parser.add_argument(
        '-exp_index',
        nargs='+',
        type=int,
        help='Array of integers')
    parser.add_argument(
        '-load_model',
        type=bool,
        default=False,
        help='Enter boolean (T if loading model from memory, F if creating model.')
    return parser

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
    
    p = {'epochs': [1,2,4,6,10],
         'loss': ['G_beta_IL', 'mse']}
    p = {'epochs': [1,2,4,6,10]}
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

def preprocess(dataset, factor = 0.1):
    # takes tf.Data object as dataset
    dataset = dataset.map(lambda x, y: (RandomTranslation(factor), y))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_loop(train_dataset, val_dataset, model, loss, opt, epochs, steps):
    # Overriding train function to make custom training loop
    # train_dataset is a tf.data.Dataset object
    # model is a keras model object
    # loss is a string representing the loss function to use
    # opt is a keras optimizer object
    # epochs is the number of epochs to train for
    # steps is the number of steps per epoch
    # returns the trained model

    # initialize loss function
    if loss == 'MSE':
        loss_fn = my_custom_MSE()
    if loss == 'G_beta_IL':
        loss_fn = G_beta_IL()

    training_POD_metric = my_POD(cutoff=15) # looser POD for training
    val_POD_metric = my_POD(cutoff=25.4)    # tighter POD for validation
    for epoch in range(epochs):
        for step, (ins_batch, outs_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                outs_pred = model(ins_batch, training=True)
                # compute loss using G_beta_IL
                loss_value = loss_fn(outs_batch, outs_pred)
            # apply gradient to trainable weights
            grads = tape.gradient(loss_value, model.trainable_weights)
            # apply gradient to the optimizer
            opt.apply_gradients(zip(grads, model.trainable_weights))
            
            if step % 100 == 0:
                print(
                     f'Training loss (for a single batch) at step {step}: {float(loss_value)}'
                     )

        training_POD_metric.update_state(outs_batch, outs_pred)
        training_POD = training_POD_metric.result()
        print(
             f'POD {training_POD} at epoch {epoch}'
             )
        training_POD_metric.reset_states()    
        
        for x_batch_val, y_batch_val in val_dataset:
            val_pred = model(x_batch_val, training=False)
            val_POD_metric.update_state(y_batch_val, val_pred)
            # update val metrics
        val_POD = val_POD_metric.result()
        val_POD_metric.reset_states()
        print(f'Validation POD: {val_POD}')
        
def main():
    # Get arguments, and augment if using job array
    parser = create_parser()
    args = parser.parse_args()
    args = augment_args(args)

    # Load dataset  
    dataset = np.load('datasets/dataset_{year}.npz') # keys along pattern x_train, y_train, x_val, etc (7to1 split twice)
    train_dataset = tf.data.Dataset.from_tensor_slices((dataset['x_train'], dataset['y_train']))
    val_dataset = tf.data.Dataset.from_tensor_slices((dataset['x_val'], dataset['y_val']))
    test_dataset = tf.data.Dataset.from_tensor_slices((dataset['x_test'], dataset['y_test']))   
    # randomize order of training data
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)
    val_dataset = val_dataset.batch(args.batch_size)
    # separate ins_train into equally sized batches     # (batch_size, row, col, chan)
 
    custom_objects={"G_beta":G_beta,"my_custom_MSE": my_custom_MSE, "my_POD": my_POD}
    if args.load_model == False:
        model = UNet(dataset['x_train'].shape[1:], loss='mse')
        config = model.get_config()
        with keras.util.custom_object_scope(custom_objects):
            model = keras.models.model_from_config(config)
    else: 
        # load partially trained model
        model_name = 'MSE_epochs_100_dataset_2006_model.h5'
        G_beta = G_beta_IL() # initialize loss function (required for closure)
        model = keras.models.load_model(f"results/{model_name}", custom_objects=custom_objects)
    
    model_prefix = f'{args.loss}_epochs_{args.epochs}_data_{args.data_ID}'
    data = {'input' : train_dataset[0],
            'targets' : train_dataset[1]}

    callbacks_list = [keras.callbacks.EarlyStopping(
              monitor='val_loss',
              min_delta=1e-2,
              patience=5,
              verbose=1),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ReduceLROnPlateau(
              monitor='val_loss',
              factor=0.3,
              patience=5,
              min_delta=0.1, 
              min_lr=0.0001),
        keras.callbacks.BackupAndRestore(
              backup_dir='tmp/backup'), # saves weights to retrain model if interrupted 
        keras.callbacks.CSVLogger(
              filename=f'csv/{model_prefix}.csv'),
        keras.callbacks.ModelCheckpoint(
              filepath=f"ckpt/{model_prefix}",
              save_best_only=True,  # Only save a model if `val_loss` has improved.
              monitor="val_loss",
              save_freq=100,),
        keras.callbacks.TensorBoard(
              log_dir='log',
              histogram_freq=10,
              update_freq='epoch')]

    callbacks = keras.callbacks.CallbackList(callbacks_list, model=model)
    
    opt = tf.keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False)
    
    if args.custom_train == True:
        model.compile(optimizer=opt)
        train_loop(train_dataset, model, args.loss, opt, args.epochs, args.steps, callbacks)
    else:
        history = model.fit(data,
                    epochs=args.epochs,
                    validation_data=val_dataset,
                    validation_steps=10,
                    callbacks=callbacks)

    model.save(f'results/{model_name}')
    print(f'Saved model to {model_name}')
    # evaluate model on test data
    print('Evaluating model on test data')

    print('\n')

    results = {}
    results['history'] = history
    results['predict_training_eval'] = model.evaluate(train_dataset[0], train_dataset[1])
    results['predict_testing'] = model.predict(test_dataset[0])
    results['predict_testing_eval'] = model.evaluate(test_dataset[0], test_dataset[1])

    # Save results
    fbase = f"results/{args.loss}_epochs_{args.epochs}_dataset_2006"
    fp = open(f"{fbase}_results.pkl", "wb")
    pickle.dump(results, fp)
    fp.close()
    model.save("{}_model.h5".format(fbase)) # necessary if using custom metrics
    
if __name__ == '__main__':
    main()
