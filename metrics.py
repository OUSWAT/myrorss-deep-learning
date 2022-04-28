import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense

def myMED(y_true, y_pred):
    # Sample specific MED (i.e. not whole batch at once)
    target_tensor = tf.cast(tf.where(y_true>cutoff,0.0,1.0),tf.float32)
    prediction_tensor = tf.cast(tf.where(y_pred>cutoff,0.0,1.0),tf.float32)
    dist = tf.norm(y_true_discrete - y_pred_discrete, axis=-1) # axis = -1 means along the last dimension (channel)
    dist = tf.reduce_mean(dist)
    count = tf.math.count_nonzero(target_tensor) # we use
    MED = dist/count
    return MED

def csi(hard_discretization_threshold=20):
    # From CIRA guide to loss functions, with slight differences        
    # Does not train well off thebat, but could be used as second phase for MSE model
    def loss(target_tensor, prediction_tensor):
        target_tensor = tf.cast(tf.where(y_true>cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred>cutoff,0.0,1.0),tf.float32)

        num_true_positives = K.sum(target_tensor * prediction_tensor)
        num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
        num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))
            
        denominator = (
            num_true_positives + num_false_positives + num_false_negatives +
            K.epsilon()
            )
        csi_value = num_true_positives / denominator
        
        if use_as_loss_function:
            return 1. - csi_value
        else:
            return csi_value
    
    return loss

def POD(hard_discretization_threshold=20):
    # From CIRA guide to loss functions, with slight differences
    # Does not train well off thebat, but could be used as second phase for MSE model
    def loss(target_tensor, prediction_tensor):
        target_tensor = tf.cast(tf.where(y_true>cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred>cutoff,0.0,1.0),tf.float32)

        TP = K.sum(target_tensor * prediction_tensor)
        FP = K.sum((1 - target_tensor) * prediction_tensor)
        FN = K.sum(target_tensor * (1 - prediction_tensor))

        denominator = (
            TP + FN +
            K.epsilon() # dont div by 0
            )
        csi_value = TP / denominator

        if use_as_loss_function:
            return 1. - csi_value
        else:
            return csi_value

    return loss


def FAR(hard_discretization_threshold=20):
    # From CIRA guide to loss functions, with slight differences
    # Does not train well off thebat, but could be used as second phase for MSE model
    def loss(target_tensor, prediction_tensor):
        target_tensor = tf.cast(tf.where(y_true>cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred>cutoff,0.0,1.0),tf.float32)

        TP = K.sum(target_tensor * prediction_tensor)
        FP = K.sum((1 - target_tensor) * prediction_tensor)
        FN = K.sum(target_tensor * (1 - prediction_tensor))

        denominator = (
            TP + FP +
            K.epsilon() # dont div by 0
            )
        csi_value = TP / denominator

        if use_as_loss_function:
            return 1. - csi_value
        else:
            return csi_value

    return loss


