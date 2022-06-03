import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense  
import math

def to_tensor(a):
    return tf.convert_to_tensor(a, dtype=tf.float32)

def find_coordinates_numpy(points):
    # Take in tensor of points (flattened)
    # Return list of 2D coords
    if tf.equal(tf.size(points), 0):
        return 0
    coord = []
    length = int(math.sqrt(points.size)) # to the square root of the second dimension of shape
    print("length",length)
    for idx in points:
        i = idx//length
        j = idx % length 
        coord.append([i,j])
    return np.asarray(coord)

def find_mean_distance(coord_A, coord_B):
    n_coord = len(coord_A)
    if n_coord < 10:
        return 'too few'
    for point in coord_A:
        distances = []
        min_dist = 1000
        for coord in coord_B:
            diff = np.subtract(coord,point)
            dist = math.sqrt(diff[0]**2 + diff[1]**2)
            if dist < min_dist:
                min_dist = dist
        distances.append(min_dist)
    distances = np.asarray(distances)
    return np.mean(distances)

def find_coordinates_numpy(points):
    # Take in tensor of points (flattened)
    # Return list of 2D coords
    if points.sum() < 20:
        return 'too few'
    coord = []
    length = 60 # to the square root of the second dimension of shape
    for idx, point in enumerate(points):
        if point > 0:
            i = idx//length
            j = idx % length
            coord.append([i,j])
    return np.asarray(coord)

def MSE_plus_MED(cutoff=25.4,constant_loss=1):
    # medMiss gives mean distance of misses in A, by B
    def loss_function(y_true, y_pred):
        MSE = tf.math.reduce_mean(tf.square(y_true - y_pred))
        target_tensor = tf.reshape(y_true, (tf.shape(y_true)[0], -1)).numpy() # reshape for sample-wise
        prediction_tensor = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1)).numpy()
        # hard discretization
        target_tensor = tf.cast(tf.where(target_tensor<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(prediction_tensor<cutoff,0.0,1.0),tf.float32)
        # Now calculate MED
        MED = 0 # initialize MED
        for true, pred in zip(target_tensor, prediction_tensor):
            points_A = tf.where(true).numpy() # get indices of nonzero elements and convert to numpy
            points_B = tf.where(pred).numpy()
            coord_A = find_coordinates_numpy(points_A) # get coordinates of nonzero elements on 2D grid
            coord_B = find_coordinates_numpy(points_B)
            if not isinstance(coord_A, str) and not isinstance(coord_B, str): # There are points in both A and B
                MED = MED + find_mean_distance(coord_A,coord_B)
            if not isinstance(coord_A, str) and isinstance(B, str): # There are points in A but not in B
                MED = MED + constant_loss # for now, just add a constant if B == 0
            else: # A is completely below threshold
                MED = MED + 0 # if A is below threshold, just use MSE (i.e. MED = 0)
        return MSE + MED
    return loss_function

def MSE_plus_medMiss(cutoff=25.4, constant_loss=1):
    def medMiss(y_true, y_pred):
        MSE = tf.math.reduce_mean(tf.square(y_true - y_pred))
        print(MSE)
        target_tensor = tf.reshape(y_true, (tf.shape(y_true)[0], -1)).numpy() # reshape for sample-wise
        prediction_tensor = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1)).numpy()
       # MSE = K.sqrt(tf.math.reduce_mean(tf.square(target_tensor - prediction_tensor), axis = [1])) # calculate sample-wise MSE
        # hard discretization
        target_tensor = tf.cast(tf.where(target_tensor<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(prediction_tensor<cutoff,0.0,1.0),tf.float32)
        # Now calculate MED
        MED = 0 # initialize MED
        for idx, sample in enumerate(target_tensor):
            points_A = tf.where(sample).numpy() # get indices of nonzero elements and convert to numpy
            points_B = tf.where(prediction_tensor[idx]).numpy()
            coord_A = find_coordinates_numpy(points_A) # get coordinates of nonzero elements on 2D grid
            coord_B = find_coordinates_numpy(points_B)
            if coord_A != 0 and coord_B != 0:
                MED = MED + find_mean_distance(coord_B,coord_A)
            if coord_A != 0 and coord_B == 0: # check for all FNs in B
                # mupltiply by constant loss * number of FNs
                MED = MED + constant_loss # for now, just add a constant if B == 0
            else:
                MED = MED + 0 # if A is below threshold, just use MSE (i.e. MED = 0)
            print(MED)
        return MSE + MED
    return medMiss

def myMED(cutoff=20):
    def mean_error_distance(y_true, y_pred):
        # Sample specific MED (i.e. not whole batch at once)
        # Either maxpool or limit domain size or threshold by high val to reduce computation
        target_tensor = tf.cast(tf.where(y_true<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred<cutoff,0.0,1.0),tf.float32)
        target_tensor = tf.reshape(target_tensor, (tf.shape(target_tensor)[0], -1)).numpy()
        prediction_tensor = tf.reshape(prediction_tensor, (tf.shape(prediction_tensor)[0], -1)).numpy()
        MED = 0
        for idx, sample in enumerate(target_tensor):
            points_A = tf.where(sample).numpy() # get indices of nonzero elements and convert to numpy
            points_B = tf.where(prediction_tensor[idx]).numpy()
            coord_A = find_coordinates_numpy(points_A) # get coordinates of nonzero elements on 2D grid
            coord_B = find_coordinates_numpy(points_B)
            if coord_A != 0 and coord_B != 0:
                print('Coordinates are present, calculating MED')
                MED = MED + find_mean_distance(coord_A,coord_B)
                print('MED {}'.format(MED))
            elif coord_A == 0 and coord_B != 0:
                MED = MED + K.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis = [1]))
                print('A completely below threshold with {}'.format(MED))
            elif coord_A != 0 and coord_B == 0:
                MED = MED + K.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis = [1]))
                print('B completely below threshold with {}'.format(MED))
            else:
                MED = MED + K.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis = [1]))
                print('Triggered else clause with {}'.format(MED))
            #except IndexError:
            #    print('index errror')
            #    MED = MED + 0
        return MED
    return mean_error_distance


def samplewise_RMSE(y_true, y_pred):
    # from CIRA guide
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_true)[0], -1])
    return K.sqrt( tf.math.reduce_mean(tf.square(y_true - y_pred), axis=[1]))        


#def hausdorf(y_true, y_pred):
    # get max distance between A and B


def CSI(cutoff=20):
    # From CIRA guide to loss functions, with slight differences        
    # Does not train well off thebat, but could be used as second phase for MSE model
    def loss(y_true, y_pred):
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
        return csi_value
        #return 1. - csi_value # subtract from 1 so we minimize i.e. increase CSI
    return loss

def POD(cutoff=20):
    # From CIRA guide to loss functions, with slight differences
    # Does not train well off thebat, but could be used as second phase for MSE model
    def pod(y_true, y_pred):
        target_tensor = tf.cast(tf.where(y_true>cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred>cutoff,0.0,1.0),tf.float32)

        TP = K.sum(target_tensor * prediction_tensor)
        FP = K.sum((1 - target_tensor) * prediction_tensor)
        FN = K.sum(target_tensor * (1 - prediction_tensor))

        denominator = (
            TP + FN +
            K.epsilon() # dont div by 0
            )
        POD  = TP / denominator

        return POD

    return pod


def FAR(cutoff=20):
    # From CIRA guide to loss functions, with slight differences
    # Does not train well off thebat, but could be used as second phase for MSE model
    def far(y_true, y_pred):
        target_tensor = tf.cast(tf.where(y_true>cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred>cutoff,0.0,1.0),tf.float32)

        TP = K.sum(target_tensor * prediction_tensor)
        FP = K.sum((1 - target_tensor) * prediction_tensor)
        FN = K.sum(target_tensor * (1 - prediction_tensor))

        denominator = (
            TP + FP +
            K.epsilon() # dont div by 0
            )
        FAR = TP / denominator

        return FAR

    return far

# find the shortest distance between two set of points as numpy arrays
def find_coordinates(A):
    # returns a tensor of coordinates of nonzero elements in A
    coord_A = []
    length = A.shape[0]
    for idx, row in enumerate(A):
        for idx2, pixel in enumerate(row):
            if pixel == 1: 
                i = idx%length
                j = idx2
                coord_A.append([i,j])  
    return coord_A

def find_shortest_distance(A,B):
    # returns a tensor of coordinates of nonzero elements in A
    coord_A = find_coordinates(A)
    coord_B = find_coordinates(B)

    point = coord_A[0]
    distances = []
    for coord in coord_B:
        distances.append(np.linalg.norm(np.subtract(coord,point)))
    return np.min(distances)    
