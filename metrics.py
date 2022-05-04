import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense  

def to_tensor(a):
    return tf.convert_to_tensor(a, dtype=tf.float32)

def find_coordinates_tensor(A):
    # returns a tensor of coordinates of nonzero elements in A
    coord_A = []
    length = 60 # or set equal to the square root of the second dimension of shape
    A = np.reshape(A,(length,length))
    for i in range(length):
        for j in range(length):
            # make tensor of ones of length length
            row = tf.ones(length)
            # set all but index i of row to 0
            row = tf.where(tf.equal(row,i),1.0,0.0)
            col = tf.ones(length)
            col = tf.where(tf.equal(col,j),1.0,0.0)
            # multiply row by A by col to get a scalar value
            val = tf.reduce_sum(row*A*col)

    return coord_A

def find_coordinates_numpy(A):
    coord_A = []
    length = 8# or set equal to the square root of the second dimension of shape
    for idx in A:
        i = idx//length
        j = idx % length 
        coord_A.append([i,j])
    return coord_A
    
def find_mean_distance(coord_A, coord_B):
    # returns a tensor of coordinates of nonzero elements in A
    coord_A = find_coordinates_tensor(A)
    coord_B = find_coordinates_tensor(B)
    point = coord_A[0] 
    distances = []
    for coord in coord_B:
        distances.append(np.linalg.norm(np.subtract(coord,point)))
    return np.mean(distances)

def myMED(cutoff=20):
    def mean_error_distance(y_true, y_pred):
        # Sample specific MED (i.e. not whole batch at once)
        # Either maxpool or limit domain size or threshold by high val to reduce computation
        target_tensor = tf.cast(tf.where(y_true<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred<cutoff,0.0,1.0),tf.float32)
        target_tensor = tf.reshape(target_tensor, (tf.shape(target_tensor)[0], -1))
        prediction_tensor = tf.reshape(prediction_tensor, (tf.shape(prediction_tensor)[0], -1))
        MED = 0
        for idx, sample in enumerate(target_tensor):
            points_A = tf.where(sample).numpy() # get indices of nonzero elements and convert to numpy
            points_B = tf.where(prediction_tensor[idx]).numpy()
            coord_A = find_coordinates_numpy(points_A) # get coordinates of nonzero elements on 2D grid
            coord_B = find_coordinates_numpy(points_B)
            try:
                MED = MED + find_mean_distance(coord_A,coord_B)
            except IndexError:
                MED = MED + 0
        return MED
    return mean_error_distance

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
