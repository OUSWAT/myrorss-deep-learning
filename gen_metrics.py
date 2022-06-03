import pandas as pd
import numpy as np
import settings as s
from metrics import *
from metrics import G_beta_RMSE, G_beta, Delta, PHD_k, Hausdorff, POD, FAR, MSE
import metrics
import matplotlib.pyplot as plt

# This code-base will provide functions to compute the metrics of a
# given test-set, and place them into a dataframe with columns
# prediction_index, MSE, G_beta_IL, etc.

def initialize_df(metrics=s.metrics):
    # metrics is an iterable of strings
    df = pd.DataFrame(metrics)
    return df

def compute_metrics(true, pred, metrics_list=s.metrics):
    # given a list of metrics and prediction pair,
    # return row of scores for pair
    scores = []
    for metric in metrics_list:
        function = getattr(metrics, metric)
        fnct = function()
        score = fnct(true, pred)
        print(f'{metric}: {score}')
        scores.append(score)
    row = dict(zip(metrics_list, scores))
    return row

def get_df(filename):
    # load dataframe from file
    dataset = np.load(f'results/predictions/{filename}.npz')
    
    pieces = filename.split('_')
    # use the pieces to make a dataframe

    hyperparameters = s.index + s.one_hot_preprocessing + s.one_hot_loss
    df_hyperparameters = pd.DataFrame(hyperparameters)
    df_metrics = initialize_df()

    # get the hyperparameters from the filename
    pieces = filename.split('.np')[0].split('_')
    keys = []
    values = []
    for piece in pieces:
        try:
            key = piece.split('-')[0]
            value = piece.split('-')[1]
        except:
            continue
        keys.append(key)
        values.append(value)
        if key == 'prefix':
            # convert binary sting to list of ints
            string = list(map(int, value))
            values = values + string   
            keys = keys + s.one_hot_preprocessing
    dictionary = dict(zip(keys,values))
    return dictionary
    df_hyperparameters = df_hyperparameters.append(dictionary, ignore_index=True)
    print(df_hyperparameters)   
    
    # debugging to check line 57
    print(f'df_hyperparameters: {df_hyperparameters}')
    # one-hot encode any string values in the hyperparameters
    trues = dataset['targets']
    preds = dataset['predictions']

    # compute metrics for each prediction pair 
    # iteratively build df_metrics, which contains metrics for each pair 
    for idx, (true, pred) in enumerate(zip(trues,preds)):
        if idx > 100: # testing 
            break
        row = compute_metrics(true, pred)
        df_metrics.loc[len(df_metrics.index)] = row

    df_test = df_hyperparameters
    # find mean of each metric across all predictions and add to df_hyperparameters
    for metric in s.metrics:
        mean = df_metrics[metric].mean()
        df_test.loc[len(df_hyperparameters.index)] = [mean]
    return df_hyperparameters, df_metrics

def get_test_sample():
    # filename = 'MSE_2006_pred_pairs'
    # dataset = np.load(f'results/predictions/{filename}.npz')
    # trues = dataset['targets']
    # preds = dataset['predictions']
    target = np.load('data/target.npy')
    prediction = np.load('data/prediction.npy')
    return target, prediction

def main():
    t, p = get_test_sample()
    # imshow t and p as 2 by 1 subplot with colorbars and save as png/pred_pair.png
    # set one colorbar for both subplots
    row = compute_metrics(t, p)
    print(row)    

if __name__ == '__main__':
    main()
