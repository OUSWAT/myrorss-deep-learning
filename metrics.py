import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense  

def to_tensor(a):
    return tf.convert_to_tensor(a, dtype=tf.float32)

def find_coordinates_numpy(points):
    # Take in tensor of points (flattened)
    # Return list of 2D coords
    if tf.equal(tf.size(points), 0):
        return 0
    coord = []
    length = 60# or set equal to the square root of the second dimension of shape
    for idx in points:
        i = idx//length
        j = idx % length 
        coord.append([i,j])
    return coord
    
def find_mean_distance(coord_A, coord_B):
    # returns a tensor of coordinates of nonzero elements in A
    # Calculates MED(A,B)
    try:
        point = coord_A[0] 
    except:
        return 0
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

def MSE_plus_MED(cutoff=25.4):
    def loss_function(y_true, y_pred):
        MSE = tf.math.reduce_mean(tf.square(y_true - y_pred))
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
                print('Coordinates are present, calculating MED')
                MED = MED + find_mean_distance(coord_A,coord_B)
                print('MED {}'.format(MED))
            if coord_A != 0 and coord_B == 0: # check for all FNs in B
                # mupltiply by constant loss * number of FNs 
                MED = MED + 5 # for now, just add a constant if B == 0
            else:
                MED = MED + 0 # if A is below threshold, just use MSE (i.e. MED = 0)
        return MSE + MED
    return loss_function

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
