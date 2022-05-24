from cmath import inf
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense, Lambda  
import math
import sys
from skimage import measure
np.set_printoptions(threshold=sys.maxsize)
import keras 
import tensorflow_probability as tfp
import scipy.interpolate as interp

def to_tensor(a):
    return tf.convert_to_tensor(a, dtype=tf.float32)
# write custom model to make use of gradient tape
class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        i = 0
        with tf.GradientTape() as tape:
            print(i)
            i+=1
            y_pred = self(x, training=True)
            loss_value = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            print(loss_value)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        # return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

class custom_model(keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = G_beta_IL()
            loss_value = loss(y, y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        # return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

def find_mean_distance(coord_A, coord_B): 
    n_coord = len(coord_A)
    if n_coord < 1:
        return 0 
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

def find_min_distances(coord_A, coord_B):
    distances = []
    min_dist = 1000
    for coord in coord_A:
        for coord2 in coord_B:
            diff = np.subtract(coord,coord2)
            dist = math.sqrt(diff[0]**2 + diff[1]**2)
            if dist < min_dist:
                min_dist = dist
    distances.append(min_dist)
    return np.asarray(distances)

def find_coordinates(points, length=60):
    # Take in tensor of points (flattened)
    # Return list of 2D coords
    try:
        points = points[0] # get the points out of the tuple
    except:
        return np.asarray([])
    if points.sum() < 2:
        return np.asarray([])
    coord = []
    length = length # to the square root of the second dimension of shape
    for point in points:
        i = point//length
        j = point % length
        coord.append([i,j])
    return np.asarray(coord)

def get_y(A,B,cutoff=25,length=60):
    n_A = np.count_nonzero(A)
    n_B = np.count_nonzero(B)
    n_AB = np.count_nonzero(A*B)
    y1 = n_A + n_B + n_AB
    coord_A, coord_B = get_coordinates(A,B)
    med_AB = find_mean_distance(coord_A,coord_B)
    med_BA = find_mean_distance(coord_B,coord_A)
    y2 = med_AB*n_B + med_BA*n_A
    y = y1*y2 # size of the area by ABc and AcB magnified by distance between A and B.
#    print(f'coordA {coord_A} coordB {coord_B}')
    return y

def G_beta(A, B, beta=12960000):
    y = get_y(A,B)
    G = y**(1/3)
    const = 1- (y/beta)
    G_beta_loss_value = max(1-y/beta, 0.0)
    return -G_beta_loss_value

def G_beta_loss(y_true, y_pred):
    # make cutoff low at low epoch and then increase over time?
    # uses MSE as base loss, then adds (negative) gbeta
    cutoff=20
    beta =12960000 # N^2
    y_true = tf.reshape(y_true, (tf.shape(y_true)[0],-1))
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
    loss_value = tf.math.reduce_mean(tf.square(y_true - y_pred))
    gbeta_list = []
    intensity_list = []
    #loss_value = tf.constant([0])
    #l = tf.cast(0, dtype=tf.float32)
    for true, pred in zip(y_true, y_pred): 
        # Interpolate and compute sorted correlation coefficient
        true_binary, pred_binary = Lambda(binarize(true, pred))
        gbeta = tf.cast(G_beta(true, pred, beta=beta), dtype=tf.float32)
        gbeta = -tf.cast(gbeta, dtype=tf.float32)
        loss_value+= gbeta
    return loss_value

def flatten_tensor(y_true, y_pred):
    return tf.stop_gradient(tf.cast(tf.reshape(y_true, (tf.shape(y_true)[0], -1)), dtype=tf.float32).numpy()), tf.stop_gradient(tf.cast(tf.reshape(y_pred, (tf.shape(y_pred)[0], -1)),dtype=tf.float32).numpy())

def scheduler(epoch, lr):
    pass
 
def my_POD(cutoff=25.4):
    # computes the probability of detection
    # POD = PT / TP + FN
    def pod(y_true, y_pred):
        y_true, y_pred = binarize(y_true, y_pred, cutoff=cutoff)
        TP = tf.math.reduce_sum(tf.where(((y_true-1)+y_pred)<1,0,1))
        FN = tf.math.reduce_sum(tf.where(y_true-y_pred<1,0,1))
        FP = tf.math.reduce_sum(tf.where(y_pred-y_true<1,0,1))
        return TP/(TP+FN)
    return pod

def my_FAR(cutoff=25.4):
    def far(y_true, y_pred, cutoff=cutoff):
        y_true, y_pred = binarize(y_true, y_pred, cutoff=cutoff)
        TP = tf.math.reduce_sum(tf.where(((y_true-1)+y_pred)<1,0,1))
        FN = tf.math.reduce_sum(tf.where(y_true-y_pred<1,0,1))
        FP = tf.math.reduce_sum(tf.where(y_pred-y_true<1,0,1))
        return FP/(TP+FP)
    return far

def my_custom_MSE(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def Delta_with_contours(y_true, y_pred):
    # Find baddeley's delta metric with contours
    # probably other ways to speed it up. Like if 
    # index is above 0, maybe cut all but top lines? 
    p=2
    y_true, y_pred = flatten_tensor(y_true, y_pred) # returns np form
    raster = np.ones(3600)
    coord_raster = find_coordinates(raster, length=60)
    delta = 0
    for true, pred in zip(y_true, y_pred):    
        objects_in_A = measure.find_contours(np.reshape(true, (60,60)))
        objects_in_B = measure.find_contours(np.reshape(pred, (60,60)))
        sums=0
        for idx, coord in enumerate(coord_raster):
            mindist_A, mindist_B = 1000, 1000
            for obj_A in objects_in_A:
                dist = find_min_distances(coord, obj_A)
                if dist < mindist_A:
                    mindist_A = dist
            for obj_B in objects_in_B:
                dist = find_min_distances(coord, obj_B)
                if dist < mindist_B:
                    mindist_B = dist
            w_x_A = w(mindist_A) # cutoff
            w_x_B = w(mindist_B)
            sums = sums + abs(w_x_A - w_x_B)
        delta+=sums/3600
    return tf.cast(delta, tf.float32)

def get_coordinates(true, pred):
    points_A = tf.where(true).numpy() # get indices of nonzero elements and convert to numpy
    points_B = tf.where(pred).numpy()
    coord_A = find_coordinates(points_A,length=60) # get coordinates of nonzero elements on 2D grid
    coord_B = find_coordinates(points_B,length=60)
    return coord_A, coord_B

def binarize(true, pred, cutoff=25):
    true = tf.where(true<cutoff,0.0,1.0)
    pred = tf.where(pred<cutoff,0.0,1.0)
    return true, pred

def Hausdorff(cutoff=25.4, constant_loss=1):
    # get max distance between A and B
    def mse_plus_maxdist(y_true, y_pred):
        MSE = tf.math.reduce_mean(tf.square(y_true - y_pred)) # calculate the MSE for entire batch
        target_tensor = tf.reshape(y_true, (tf.shape(y_true)[0], -1)).numpy() # reshape for sample-wise
        prediction_tensor = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1)).numpy()
        # hard discretization
        target_tensor = tf.cast(tf.where(target_tensor<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(prediction_tensor<cutoff,0.0,1.0),tf.float32)
        # Calculate max distance between A and B
        for true, pred in zip(target_tensor, prediction_tensor):
            coord_A = find_coordinates(tf.where(true).numpy()) # convert to numpy then find coord of nonzero values
            coord_B = find_coordinates(tf.where(pred).numpy())
            if np.any(coord_A) and np.any(coord_B): # There are points in both A and B
                max_dist_B_to_A = np.max(find_min_distances(coord_A,coord_B))
                max_dist_A_to_B = np.max(find_min_distances(coord_B,coord_A))
                max_dist = max(max_dist_B_to_A, max_dist_A_to_B)
            if np.any(coord_A) and not np.any(coord_B): # There are points in A but not in B
                max_dist = constant_loss # for now, just add a constant if B == 0
            else:
                max_dist = 0
            loss = MSE + max_dist
        print(type(loss))
        return loss
    return mse_plus_maxdist


def PHD_k(cutoff=25.4, constant_loss=20, k=10):
    # Partial Hausdorff Distance (kth)
    # get the kth largest distance between A and B
    def mse_plus_kth_largest_dist(y_true, y_pred):
        MSE = tf.math.reduce_mean(tf.square(y_true - y_pred))
        target_tensor = tf.reshape(y_true, (tf.shape(y_true)[0], -1)).numpy()
        prediction_tensor = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1)).numpy()
        # hard discretization
        target_tensor = tf.cast(tf.where(target_tensor<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(prediction_tensor<cutoff,0.0,1.0),tf.float32)
        # Calculate kth largest distance between A and B
        for true, pred in zip(target_tensor, prediction_tensor):
            points_true = tf.where(true).numpy()
            points_pred = tf.where(pred).numpy()
            coord_A = find_coordinates(points_true)
            coord_B = find_coordinates(points_pred)
            if np.any(coord_A) and np.any(coord_B): # There are points in both A and B              
                distances_A_to_B = find_min_distances(coord_A,coord_B)
                distances_A_to_B = np.sort(distances_A_to_B)
                kth_largest_dist_A_to_B = distances_A_to_B[-k]
                distances_B_to_A = find_min_distances(coord_B,coord_A)
                distances_B_to_A = np.sort(distances_B_to_A)
                kth_largest_dist_B_to_A = distances_B_to_A[-k]
            else:
                kth_largest_dist_A_to_B = 0
                kth_largest_dist_B_to_A = 0
            kth_largest_dist = max(kth_largest_dist_A_to_B, kth_largest_dist_B_to_A)
            loss = MSE + kth_largest_dist
        return loss
    return mse_plus_kth_largest_dist

def w(distance,cutoff=30):
    # cutoff transformation
    return min(distance, cutoff)

def delta(p, A, B, skip=10):
    # Baddeley's delta metric
    # goes over every point in X, A, and B
    # can be tuned with skip to only find A.size/skip points
    raster = np.ones_like(A).flatten()
    A = A.flatten()
    B = B.flatten()
    coord_raster = find_coordinates(raster)
    coord_A = find_coordinates(A)
    coord_B = find_coordinates(B)
    sums = 0
    for idx, coord in enumerate(coord_raster):
        if idx % skip != 0:
            continue
        if A[idx] == 1 or B[idx] == 1:
            continue 
        dist_x_A = find_min_distances(coord, coord_A)
        dist_x_B = find_min_distances(coord, coord_B)
        w_x_A = w(dist_x_A)
        w_x_B = w(dist_x_B)
        sums = sums + abs(w_x_A - w_x_B)
    return (sums/len(A))**(1/p)

def samplewise_RMSE(y_true, y_pred):
    # from CIRA guide
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_true)[0], -1])
    return K.sqrt( tf.math.reduce_mean(tf.square(y_true - y_pred), axis=[1]))        


def get_model():
    # simple model
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def main():
    a = [0,3,6,9,12,15,18,21]
    b = [4,8,12,16,20]
    #f = interp1d(a,b)
    #f2 = interp1d(a,b,kind='cubic')
    #print(f)
    #print(f2)
    model = get_model()
 
    loaded_1 = keras.models.load_model("my_model", custom_objects={"Custom_loss": G_beta_loss})

    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    
    model.fit(test_input, test_target)


def G_beta_IL(batch_size=512, omega=0.2):
    #rewrite inner_G_beta_IL to handle non-differentiable functions
    def inner_G_beta_IL(y_true, y_pred):
        cutoff=20
        beta =12960000
        y_true, y_pred = flatten_tensor(y_true, y_pred) # returns np form
        intensity_loss = 0
        loss_value = 0
        for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
            # get the number of nonzero elements in true
            zeros = true.numpy().shape[-1]
            # get nonzero numpy array of true
            true_nonzero = tf.stop_gradient(true.numpy()[true.numpy()>5])
            pred_nonzero = tf.stop_gradient(pred.numpy()[pred.numpy()>5])
            true_nz_np = np.array(true_nonzero.numpy())
            pred_nz_np = np.array(pred_nonzero.numpy())
            print(repr(true_nz_np)) 
            print('\n\n\n\n\n\n\n\n <')
            print(repr(pred_nz_np))
            N_true = tf.stop_gradient(np.count_nonzero(true_nonzero))
            N_pred = tf.stop_gradient(np.count_nonzero(pred_nonzero))
            if np.greater(N_true,N_pred): min_N = N_pred
            else: min_N = np.maximum(0,N_true)
            if min_N < 20: # Calculate RMSE for pair and use as loss
                loss_value+=tf.cast(math.sqrt(np.square(np.subtract(true, pred)).mean()/batch_size), dtype=tf.float32) 
            else:
                # get binary versions of true and pred
                true_binary, pred_binary = tf.stop_gradient(binarize(true, pred))
                # get the gbeta loss
                gbeta = tf.stop_gradient(G_beta(true_binary, pred_binary))
                # get the interpolated versions of true and pred, using lambda layers
                true_interp_func = interp.interp1d(np.arange(N_true), true_nonzero, bounds_error=False)
                pred_interp_func = interp.interp1d(np.arange(N_pred), pred_nonzero, bounds_error=False)
                true_interp = tf.stop_gradient(np.sort(true_interp_func(np.linspace(0,tf.math.subtract(N_true,1),min_N))))# interpolate to smaller
                pred_interp = tf.stop_gradient(np.sort(pred_interp_func(np.linspace(0,tf.math.subtract(N_pred,1),min_N))))
                print(f'{true_interp} with shape true {true_interp.shape}')
                print(f'{pred_interp} with shape pred {pred_interp.shape}')
                # get the covariance of the interpolated versions using Lambda layer
                covar = tf.negative(tf.stop_gradient(np.cov(true_interp, pred_interp)[0][1])) # make it negative so that high covariance lessens the loss value
                # calculate the intensity loss
                print(covar)
                intensity_value = tf.stop_gradient(covar/np.dot(np.std(true_interp),np.std(pred_interp)))
                # calculate the final loss
                print(gbeta)
                print(intensity_value)
                if idx >= 1:
                    break
                loss_value += tf.cast(tf.cast(omega*gbeta,dtype=tf.float32) + tf.cast((1 - omega)*intensity_value,dtype=tf.float32),dtype=tf.float32) # cast as float 32s then cast again as float32s before casting one last time as float32s :check:
        print(loss_value)
        return loss_value
    return inner_G_beta_IL

def get_nonzero_and_min(true, pred,cutoff=25):
    true = true.numpy()
    pred = pred.numpy()
    N_true = tf.stop_gradient(np.count_nonzero(np.where(true<cutoff,int(0),true)))
    N_pred = tf.stop_gradient(np.count_nonzero(np.where(pred<cutoff,int(0),pred)))
    true_nz = tf.stop_gradient(np.sort(true[np.greater(true, 0)]))
    pred_nz = tf.stop_gradient(np.sort(pred[np.greater(pred, 0)]))
    if N_true > N_pred: min_N = N_pred
    else: min_N = max(0,N_true)
    [tf.ragged.constant(tf.cast(x, dtype=tf.float32)).numpy() for x in [true_nz, pred_nz, min_N]]

if __name__ == '__main__':
    main()
