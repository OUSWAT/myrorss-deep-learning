from cmath import inf
import numpy as np
import math
import sys
from skimage import measure
np.set_printoptions(threshold=sys.maxsize)
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import tensorflow as tf
# Numpy versions of metrics, for computing on the fly
import tensorflow.keras.backend as K

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
    return y

def neg_G_beta(A, B, beta=12960000):
    # returns negative G_beta for loss value 
    y = get_y(A,B)
    G = y**(1/3) # why isn't this accessed? 
    const = 1- (y/beta)
    G_beta_loss_value = max(const, 0.0)
    return -G_beta_loss_value

def G_beta(batch_size=512, omega=0.2, cutoff=15,metric=True):
    #rewrite inner_G_beta_IL to handle non-differentiable functions
    def inner_G_beta_IL(y_true, y_pred):
        beta =12960000
        loss_value = 0
        y_true, y_pred = y_true.flatten(), y_pred.flatten() # returns np form
        shape = y_true.shape
        dummy = np.zeros((shape))
        # concatenate y_true and dummy so the shape is (2,3600)
        y_true = np.asarray([y_true, dummy])
        y_pred = np.asarray([y_pred, dummy])
        for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
            # check if true is all zero, and if so break
            if np.all(true == 0):
                break
            # get the number of nonzero elements in true
            zeros = true.shape[-1]
            # get nonzero numpy array of true
            true_nonzero = (true[true>cutoff])
            pred_nonzero = (pred[pred>cutoff])
            N_true = (np.count_nonzero(true_nonzero))
            N_pred = (np.count_nonzero(pred_nonzero))
            if np.greater(N_true,N_pred): min_N = N_pred
            else: min_N = np.maximum(0,N_true)
            if min_N < 5: # Calculate RMSE for pair and use as loss
                if metric == True:
                    # return nan
                    loss_value = np.nan
            else:
                # get binary versions of true and pred
                true_binary, pred_binary = np.where(true>5,1,0), np.where(pred>5,1,0)
                # get the gbeta loss
                gbeta = (neg_G_beta(true_binary, pred_binary))
                # get the interpolated versions of true and pred, using lambda layers
                loss_value += gbeta
        return loss_value
    return inner_G_beta_IL

def flatten_tensor(y_true, y_pred):
    return np.reshape(y_true, (np.shape(y_true)[0], -1)), np.reshape(y_pred, (np.shape(y_pred)[0], -1))
 
def POD(cutoff=0.2):
    # computes the probability of detection
    # POD = PT / TP + FN
    def pod(y_true, y_pred):
        y_true = tf.where(y_true>cutoff,1.0,0.0)
        y_pred = tf.where(y_pred>cutoff,1.0,0.0)

        TP = tf.math.reduce_sum(tf.where(((y_true-1)+y_pred)<1.0,0.0,1.0))
        FN = tf.math.reduce_sum(tf.where(y_true-y_pred<1.0,0.0,1.0))
        return TP/(TP+FN+K.epsilon())
    return pod

def FAR(cutoff=0.2):
    def far(y_true, y_pred):
        y_true = tf.where(y_true>cutoff,1.0,0.0)
        y_pred = tf.where(y_pred>cutoff,1.0,0.0)

        TP = tf.math.reduce_sum(tf.where(((y_true-1.0)+y_pred)<1.0,0.0,1.0))
        FP = tf.math.reduce_sum(tf.where(y_pred-y_true<1.0,0.0,1.0))
        return FP/(TP+FP+K.epsilon())
    return far

def MSE():
    def inner(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    return inner

def Delta(p=2,cutoff=15):
    # Find baddeley's delta metric with contours
    # probably other ways to speed it up. Like if 
    # index is above 0, maybe cut all but top lines? 
    def inner(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        y_true = np.where(y_true<cutoff,0,1)
        y_pred = np.where(y_pred<cutoff,0,1)
        # attach dummy var of shape y_true to y_true
        y_true = np.asarray([y_true, np.zeros(np.shape(y_true))])
        y_pred = np.asarray([y_pred, np.zeros(np.shape(y_pred))])
        raster = np.ones(3600)
        coord_raster = find_coordinates(raster, length=60)
        delta = 0
        for true, pred in zip(y_true, y_pred):   
            if np.all(true == 0):
                print('breaking')
                break 
            objects_in_A = measure.find_contours(np.reshape(true, (60,60)))
            objects_in_B = measure.find_contours(np.reshape(pred, (60,60)))
            print(objects_in_A)
            print(objects_in_B)
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
            delta+=sums/3600 # 3600 is the number of pixels in the raster
        return delta
    return inner

def get_coordinates(true, pred):
    points_A = np.where(true) # get indices of nonzero elements and convert to numpy
    points_B = np.where(pred)
    coord_A = find_coordinates(points_A,length=60) # get coordinates of nonzero elements on 2D grid
    coord_B = find_coordinates(points_B,length=60)
    return coord_A, coord_B

def Hausdorff(cutoff=25.4, constant_loss=1):
    # get max distance between A and B
    def inner_haus(y_true, y_pred):
        mse = MSE()
        y_true = np.reshape(y_true, (np.shape(y_true)[0], -1)) # reshape for sample-wise
        y_pred = np.reshape(y_pred, (np.shape(y_pred)[0], -1))
        # hard discretization
        y_true = np.where(y_true<cutoff,0.0,1.0)
        y_pred = np.where(y_pred<cutoff,0.0,1.0)
        # Calculate max distance between A and B
        for true, pred in zip(y_true, y_pred):
            coord_A = find_coordinates(np.where(true)) # convert to numpy then find coord of nonzero values
            coord_B = find_coordinates(np.where(pred))
            if np.any(coord_A) and np.any(coord_B): # There are points in both A and B
                max_dist_B_to_A = np.max(find_min_distances(coord_A,coord_B))
                max_dist_A_to_B = np.max(find_min_distances(coord_B,coord_A))
                max_dist = max(max_dist_B_to_A, max_dist_A_to_B)
            if np.any(coord_A) and not np.any(coord_B): # There are points in A but not in B
                max_dist = constant_loss # for now, just add a constant if B == 0
            else:
                max_dist = 0
            loss = mse(true,pred) + max_dist
        return loss
    return inner_haus

def PHD_k(cutoff=25.4, constant_loss=20, k=10):
    # Partial Hausdorff Distance (kth)
    # get the kth largest distance between A and B
    def mse_plus_kth_largest_dist(y_true, y_pred):
        MSE = np.mean(np.square(y_true - y_pred))
        target_tensor = np.reshape(y_true, (np.shape(y_true)[0], -1))
        prediction_tensor = np.reshape(y_pred, (np.shape(y_pred)[0], -1))
        # hard discretization
        target_tensor = np.where(target_tensor<cutoff,0.0,1.0)
        prediction_tensor = np.where(prediction_tensor<cutoff,0.0,1.0)
        # Calculate kth largest distance between A and B
        for true, pred in zip(target_tensor, prediction_tensor):
            points_true = np.where(true)
            points_pred = np.where(pred)
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
    y_true = np.reshape(y_true, [np.shape(y_true)[0], -1])
    y_pred = np.reshape(y_pred, [np.shape(y_true)[0], -1])
    return K.sqrt( np.mean(np.square(y_true - y_pred), axis=[1]))        

def G_beta_RMSE(batch_size=512, omega=0.2, cutoff=15,metric=True):
    #rewrite inner_G_beta_IL to handle non-differentiable functions
    def inner_G_beta_IL(y_true, y_pred):
        beta =12960000 # why is beta this number? 
        y_true, y_pred = y_true.flatten(), y_pred.flatten() # returns np form
        intensity_loss = 0
        loss_value = 0
        shape = y_true.shape
        dummy = np.zeros((shape))
        # concatenate y_true and dummy so the shape is (2,3600)
        y_true = np.asarray([y_true, dummy])
        y_pred = np.asarray([y_pred, dummy])
        for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
            # check if true is all zero, and if so break
            if np.all(true == 0):
                break
            # get the number of nonzero elements in true
            zeros = true.shape[-1]
            # get nonzero numpy array of true
            true_nonzero = (true[true>cutoff])
            pred_nonzero = (pred[pred>cutoff])
            N_true = (np.count_nonzero(true_nonzero))
            N_pred = (np.count_nonzero(pred_nonzero))
            if np.greater(N_true,N_pred): min_N = N_pred
            else: min_N = np.maximum(0,N_true)
            if min_N < 5: # Calculate RMSE for pair and use as loss
                if metric == True:
                    # return nan
                    loss_value = np.nan
                # get RMSE_max as the RMSE between true and zeros((true.shape))
                RMSE_max = np.sqrt(np.mean(np.square(true - np.zeros((true.shape)))))
                RMSE = np.sqrt(np.mean(np.square(true - pred)))
                # normalize RMSE by RMSE_max
                RMSE = RMSE/RMSE_max
                loss_value += RMSE
            else:
                # get binary versions of true and pred
                true_binary, pred_binary = np.where(true>5,1,0), np.where(pred>5,1,0)
                # get the gbeta loss
                gbeta = (neg_G_beta(true_binary, pred_binary))
                # get the interpolated versions of true and pred, using lambda layers
                true_interp_func = interp.interp1d(np.arange(N_true), true_nonzero, bounds_error=False)
                pred_interp_func = interp.interp1d(np.arange(N_pred), pred_nonzero, bounds_error=False)
                true_interp = (np.sort(true_interp_func(np.linspace(0,np.subtract(N_true,1),min_N))))# interpolate to smaller
                pred_interp = (np.sort(pred_interp_func(np.linspace(0,np.subtract(N_pred,1),min_N))))
                # get the covariance of the interpolated versions using Lambda layer
                # plot true_interp against pred_interp
                # calculate RMSE between true_interp and pred_interp
                RMSE = math.sqrt(np.square(np.subtract(true_interp, pred_interp)).mean())
                # make array of length min_N with all values equal to cutoff
                cutoff_array = np.full(min_N,cutoff)
                max_RMSE = math.sqrt(np.square(np.subtract(true_interp, cutoff_array)).mean())
                # normalize RMSE by max_RMSE
                RMSE_norm = RMSE/max_RMSE
                # multiply RMSE by gbeta
                loss_value+=RMSE_norm*omega + (1-omega)*gbeta
                # divide loss_value by twice RMSE
                return loss_value
        return loss_value
    return inner_G_beta_IL

def IL():
    def covar():
        pass
        # covar = (np.cov(true_interp, pred_interp)[0][1]) # make it negative so that high covariance lessens the loss value
        # intensity_value = (covar/np.dot(np.std(true_interp),np.std(pred_interp)))
        # # calculate the final loss
        # print(gbeta)
        # print(intensity_value)
        # loss_value += omega*gbeta + (1 - omega)*intensity_value 
    pass


def get_nonzero_and_min(true, pred,cutoff=25):
    true = true
    pred = pred
    N_true = (np.count_nonzero(np.where(true<cutoff,int(0),true)))
    N_pred = (np.count_nonzero(np.where(pred<cutoff,int(0),pred)))
    true_nz = (np.sort(true[np.greater(true, 0)]))
    pred_nz = (np.sort(pred[np.greater(pred, 0)]))
    if N_true > N_pred: min_N = N_pred
    else: min_N = max(0,N_true)
    return None
    #[np.ragged.constant(x) for x in [true_nz, pred_nz, min_N]]
