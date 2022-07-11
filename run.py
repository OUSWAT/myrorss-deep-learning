from unet import UNet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse
from keras import backend as K
K.set_image_data_format('channels_last')
from dataset import Dataset
import settings as s
import keras_tuner as kt
from loss_functions import *
from exp import Experiment
import pickle

#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
class Run(object):

    def __init__(self):
    # set args and train
        parser = create_parser()
        self.args = parser.parse_args()
        self.print_args = False
        if self.print_args is True:
            print(self.args)
        self.data_filename = f'{s.data_path}/datasets/dataset_{self.args.ID}.npz'
        self.dataset = Dataset(self.args, self.data_filename, self.args.activation)

    def run_experiment(self):
        self.dataset.setup_dataset()
        scaler = {}
        scaler['outs_scaler'] = self.dataset.ytr_scalers
        filename = 'outs_scaler.pkl'
        o = open(filename, 'wb')
        pickle.dump(scaler, o) 
        print('dumped pickle')
        Exp = Experiment(self.args, self.dataset) # make experiment object
        self.exp = Exp
        Exp.set_callbacks()
        self.model = Exp.train()       
        self.history = Exp.history
        
    def save_results(self):
        if self.args.model_name != 'None':
            prefix = f'{self.args.model_name}_{args.loss}-{self.args.omega}'
        else:
            prefix = self.exp.gen_filename()
        self.exp.model.save(f'models/{prefix}.h5')
        np.savez(f'results/predictions/{prefix}.npz', targets = self.dataset.yt, predictions = self.exp.model.predict(self.dataset.xt))
        results = {}
        #results['true_outs'] = outs
        results['predict_training_eval'] = self.exp.model.evaluate(self.dataset.xtr, self.dataset.ytr)
        results['predict_testing_eval'] = self.exp.model.evaluate(self.dataset.xt, self.dataset.yt)
        if self.args.activation != 'linear':
            results['true_testing'] = self.dataset.rescale(self.dataset.yt)
            results['predict_testing'] = self.dataset.rescale(self.exp.model.predict(self.dataset.xt))
        else:
            results['true_testing'] = self.dataset.yt
            results['predict_testing'] = self.exp.model.predict(self.dataset.xt)
        results['ins_scaler'] = self.dataset.xtr_scalers
        results['outs_scaler'] = self.dataset.ytr_scalers
        #results['outs_test_indices'] = outs_test_indices
        results['history'] = self.history.history
        prefix = f'{prefix}_gain-{self.args.gain}_bias-{self.args.bias}_init-{self.args.init_kind}'
        # Save results
        results['fname_base'] = prefix
        fp = open(f"results/{prefix}_results.pkl", "wb")
        pickle.dump(results, fp)
        print(f"results/{prefix}_results.pkl")
    

def main():
    run = Run()
    print(run.args)
    run.run_experiment()
    run.save_results()
    return None

def create_parser():
    parser = argparse.ArgumentParser(description='Hail Swath Learner')
    parser.add_argument(
        '-model_name',
        type=str,
        default='None')
    parser.add_argument(
        '-use_resnet',
        type=int,
        default=1)
    parser.add_argument(
        '-tuner',
        type=int,
        default=0,
        help='Are you using the tuner, or just messing around?')
    parser.add_argument(
        '-save_dir',
        type=str,
        default='default',
        help='ID of dataset (2005, 2006, 2007, shavelik)')
    parser.add_argument(
        '-ID',
        type=str,
        default='testset',
        help='ID of dataset (2005, 2006, 2007, shavelike)')
    parser.add_argument(
        '-std_gain',
        type=int,
        default=1)
    parser.add_argument(
        '-depth',
        type=int,
        default=2)
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
        default=50,
        help='Enter the batch size.')
    parser.add_argument(
        '-dropout',
        type=float,
        default=0.0,
        help='Enter the dropout rate (0<p<1)')
    parser.add_argument(
        '-note',
         type=str,
         default='None',
         help='Optional note for filename')
    parser.add_argument(
        '-L2',
        type=float,
        default=None,
        help='Enter l1, l2, or none.')
    parser.add_argument(
        '-epochs',
        type=int,
        default=3,
        help='Training epochs')
    parser.add_argument(
        '-steps',
        type=int,
        default=10,
        help='Steps per epoch')
    parser.add_argument(
        '-cutoff',
        type=int,
        default=30,
        help='Steps per epoch')
    parser.add_argument(
        '-filters',
        type=int,
        default=[
            32,
            64],
        nargs='+',
        help='Enter the number of filters for convolutional network')
    parser.add_argument(
        '-init_kind',
        type=str,
        default='uniform')
    parser.add_argument(
        '-gain',
        type=float,
        default='1')
    parser.add_argument(
        '-bias',
        type=float,
        default='0.2')
    parser.add_argument(
        '-use_transpose',
        type=int,
        default=1)
    parser.add_argument(
        '-omega',
        type=float,
        default=0.2)
    parser.add_argument(
        '-t_fac',
        type=float,
        default='0.1',
        help='Enter factor to be used in RandomTranslation).')
    parser.add_argument(
        '-g_fac',
        type=float,
        default='0.1',
        help='Enter factor to be used in RandomTranslation).')
    parser.add_argument(
        '-junction',
        type=str,
        default='Add',
        help='Enter either Add or Concat to specify skip connection junction')
    parser.add_argument(
        '-loss',
        type=str,
        default='mse',
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
        default='lrelu',
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

if __name__ == '__main__':
    main()

