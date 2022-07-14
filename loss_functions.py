import tensorflow as tf
import numpy as np
import scipy.interpolate as interp
import math
tf.executing_eagerly()
import tensorflow.experimental.numpy as tnp
import tensorflow.math as tfm
import tensorflow.keras.backend as K
from scipy.interpolate import interp1d
from itertools import product
from skimage import measure
# Disable
# all distance functions use euclidean distance

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    pass

def swish(x):
    return tf.keras.activations.swish(x)

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    pass

def GBETA(scaler, omega=0.2,cutoff=30, batch_size=2,beta=64800000, min_N=6):
    # test each of the functions before using them in big G_Beta
    # get your coordinates using find_coordinates, then compute a dummy loss 
    # just to check that the loss function can use the functino
    def gbeta(y_true, y_pred):
        const = tf.constant([omega],tf.float32)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        #MSE = tf.reduce_mean(tf.square(y_true-y_pred))
        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1))
        y_true = y_true*scaler.scale_
        y_true = y_true+scaler.mean_
        y_pred = y_pred*scaler.scale_
        y_pred = y_pred+scaler.mean_
        #y_true = tf.stop_gradient(tf.cast(scaler.inverse_transform(y_true), tf.float32))
        #y_pred = tf.stop_gradient(tf.cast(scaler.inverse_transform(y_pred), tf.float32))
        loss = tf.cast([0],dtype=tf.float32)
        trues = tf.map_fn(fn = lambda x: x, elems=y_true)
        preds = tf.map_fn(fn = lambda x: x, elems=y_pred)
        min_nonzero_pixels = tf.reduce_sum(tf.constant(min_N, dtype=tf.float32))
        for idx in tf.range(batch_size):
            idx = tf.cast(idx, tf.int32)
            true = trues[idx]
            pred = preds[idx]
            MSE = tf.reduce_mean(tfm.square(tfm.subtract(true,pred)))
            y_true = tf.where(true<cutoff,0.0,1.0)
            y_pred = tf.where(pred<cutoff,0.0,1.0)
            n_true = tf.reduce_sum(y_true)
            n_pred = tf.reduce_sum(y_pred)
            switch_true = tf.cond(n_true < min_nonzero_pixels, lambda: tf.zeros_like(true), lambda: tf.ones_like(true)) 
            switch_pred = tf.cond(n_pred < min_nonzero_pixels, lambda: tf.zeros_like(pred), lambda: tf.ones_like(pred)) 
            switch = tfm.multiply_no_nan(switch_true,switch_pred)
            gbeta_TA = get_neg_G_beta(y_true,y_pred,beta)
            gbeta = gbeta_TA.read(0)
            print(gbeta)
            loss_g = tfm.reduce_mean(tfm.multiply_no_nan(gbeta,switch))
            loss_g_MSE = tfm.multiply(loss_g, MSE)
            #gbeta = tf.constant(0,dtype=tf.float32)
            # set gbeta to 0 if either switch is flipped 
            # hopefully the NaNs are resulting from empty 
            #omega_g = tf.stop_gradient(tf.constant(omega))
            #MSE_comp = tfm.multiply_no_nan((1-omega),MSE)
            #GBETA_comp = tfm.multiply_no_nan(omega_g,loss_g_MSE)
            tf.print(gbeta)
            tf.print(loss_g_MSE)
            loss+= tfm.multiply_no_nan(loss_g_MSE,const) + MSE
        return loss 
    return gbeta 

def DELTA(scaler, cutoff=30, min_pixels=5, p=2):
    # Find baddeley's delta metric with contours
    # probably other ways to speed it up. Like if 
    # index is above 0, maybe cut all but top lines? 
    @tf.function
    def delta(y_true, y_pred):
        loss = tf.cast(0,dtype=tf.float32)
        h_pixels = 30
        n_pixels = tf.cast(900.0, tf.float32)
        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1)) # do batch_n x (everything flattened)
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1))
        y_true = y_true*scaler.scale_
        y_true = y_true+scaler.mean_
        y_pred = y_pred*scaler.scale_
        y_pred = y_pred+scaler.mean_
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        ones = tf.where(tf.ones((h_pixels, h_pixels)))
        raster_x_TA, raster_y_TA = find_coord(ones)
        delta = 0
        trues = tf.map_fn(fn = lambda x: x, elems=y_true)
        preds = tf.map_fn(fn = lambda x: x, elems=y_pred)
        for idx in tf.range(batch_size):
            idx = tf.cast(idx,tf.int32)
            true = trues[idx]
            pred = preds[idx]
            # max pool true and pred
            MSE = tf.reduce_mean(tfm.square(tfm.subtract(true,pred)))
            y_true = tf.where(true<cutoff,0.0,1.0)
            y_pred = tf.where(pred<cutoff,0.0,1.0)
            n_true = tf.reduce_sum(y_true)
            n_pred = tf.reduce_sum(y_pred)
            switch_true = tf.cond(n_true < min_pixels, lambda: tf.zeros_like(true), lambda: tf.ones_like(true)) # if false, no effect on gbeta
            switch_pred = tf.cond(n_pred < min_pixels, lambda: tf.zeros_like(pred), lambda: tf.ones_like(pred)) # if true, 
            indices_true = tf.where(y_true>0.0)[0]
            indices_pred = tf.where(y_pred>0.0)[0]
            coord_x_A_TA, coord_y_A_TA = find_coord(indices_true) # feed nonzero elements into find_coord
            coord_x_B_TA, coord_y_B_TA = find_coord(indices_pred) # 
            mindists_to_A_TA = find_min_distances(raster_x_TA, raster_y_TA, coord_x_A_TA, coord_y_A_TA) # shape should be 3600 
            mindists_to_B_TA = find_min_distances(raster_x_TA, raster_y_TA, coord_x_B_TA, coord_y_B_TA) # one for every pixel in the raster
            mindists_to_A = mindists_to_A_TA.read(0)
            mindists_to_B = mindists_to_B_TA.read(0)
            mindists_A = tf.map_fn(fn = lambda x: x, elems=mindists_to_A) # if A is empty, mindists_A is of length 1
            mindists_B = tf.map_fn(fn = lambda x: x, elems=mindists_to_B)
            # switch off if the first element of mindists_A or mindists_B is 2.701
            switch = tf.cond(tf.equal(mindists_A[0],2.701), lambda: tf.zeros_like(y_true), lambda: tf.ones_like(true_binary))
            sums = tf.cast(0,tf.float32)
            try:
                for idx in tf.range(n_pixels):
                    idx = tf.cast(idx,tf.int32)
                    print(idx)
                    subject = tfm.square(tfm.subtract(w(mindists_A[idx]),w(mindists_B[idx]))) # p=2
                    sums += subject
            except:
                sums = tf.cast(0,tf.float32)
            switch = tfm.multiply_no_nan(switch,switch_true)
            switch = tfm.multiply_no_nan(switch,switch_pred)
            delta = tfm.sqrt(tfm.divide(sums,n_pixels)) # 3600 is the number of pixels in the raster
            delta = tfm.reduce_mean(tfm.multiply_no_nan(delta,switch))
            loss += MSE+delta
        return loss
    return delta

def w(distance,cutoff=15):
    # cutoff transformation
    return tf.cast(tfm.minimum(distance, cutoff), tf.float32)

def PHDK(scaler,cutoff=20, cutoff_distance=10, reward=10, k=10):
# d = distance
# if d is less than cutoff_distance, subtract f(d) from MSE 
# it operates as a reward for low PHDK
# the reward (negative number) grows more negative the closer to 0 it is
# but FQ is worse as PHD_k increases
# so f(d) should be most negative at F(0) and least negative at F(cutoff_distance), and F(>cutoff_distance) = 0 (or switched off)
# a cos wave with a period equal to 1/4 the cutoff distance
# if the cutoff distance is 10 gridpoints, then f(x) = cos(2pi x/40)
# should feed this distance in to the find_min_distances and cause it to stop looking once the distance is above cd
    # Partial Hausdorff Distance (kth)
    # get the kth largest distance between A and B
    def phdk(y_true, y_pred):
        loss = tf.cast(0,dtype=tf.float32)
        cut = tf.constant(cutoff,dtype=tf.float32)
        zero = tf.constant(0,dtype=tf.float32)
        dummy = tf.constant(2.701,dtype=tf.float32)
        h_pixels = 30
        n_pixels = tf.cast(900.0, tf.float32)
        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1)) # do batch_n x (everything flattened)
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1))
        y_true = y_true*scaler.scale_
        y_true = y_true+scaler.mean_
        y_pred = y_pred*scaler.scale_
        y_pred = y_pred+scaler.mean_
        batch_size = tf.shape(y_true)[0]
        trues = tf.map_fn(fn = lambda x: x, elems=y_true)
        preds = tf.map_fn(fn = lambda x: x, elems=y_pred)
        MSE = tf.reduce_mean(tf.square(y_true-y_pred))
        # Calculate kth largest distance between A and B
        for idx in tf.range(batch_size):
            true = trues[idx] # get image
            pred = preds[idx] # get prediction
            MSE = tfm.sqrt(tf.reduce_mean(tfm.square(tfm.subtract(true,pred)))) # calculate RMSE
            true = tf.where(true<cut,0.0,1.0) # if less than cutoff, 0, else 1
            pred = tf.where(pred<cut,0.0,1.0) # binarize true and pred
            n_true = tf.reduce_sum(true) # get number of pixels in true
            n_pred = tf.reduce_sum(pred)   # get number of pixels in pred
            switch_true = tf.cond(n_true < k, lambda: tf.zeros_like(true), lambda: tf.ones_like(true)) # check there are enough pixels
            switch_pred = tf.cond(n_pred < k, lambda: tf.zeros_like(pred), lambda: tf.ones_like(pred)) # if either is 0 the PHD_k comp of loss -> 0
            switch = tfm.multiply(switch_true, switch_pred) # get single switch, shape of true
            indices_true = tf.cast(tf.where(tf.not_equal(true, zero)), tf.float32) # get indices of non-zero elements
            indices_pred = tf.cast(tf.where(tf.not_equal(pred, zero)), tf.float32)
            indices_true = tf.map_fn(fn=lambda x: x, elems=indices_true)
            indices_pred = tf.map_fn(fn=lambda x: x, elems=indices_pred)
            coord_x_A_TA, coord_y_A_TA = find_coord(indices_true) # feed nonzero elements into find_coord
            coord_x_B_TA, coord_y_B_TA = find_coord(indices_pred) # if there are no nonzero elements, this will return two tensors with single scalar of value 100
            distances_A_to_B_TA = find_min_distances(coord_x_A_TA, coord_y_A_TA, coord_x_B_TA, coord_y_B_TA) # if coord empty, returns tensorarray with 2.701
            distances_A_to_B = distances_A_to_B_TA.read(0)
            distances_A_to_B = tf.sort(distances_A_to_B)
            dists_AB = tf.map_fn(fn = lambda x: x, elems=distances_A_to_B)
            kth_AB = get_dist_at_k(dists_AB, k)
            # will need the size of nonzero elements in both dists_AB and dists_BA but how do you get the length of a symbolic tensor?
            distances_B_to_A_TA = find_min_distances(coord_x_B_TA, coord_y_B_TA, coord_x_A_TA, coord_y_A_TA)
            distances_B_to_A = distances_B_to_A_TA.read(0)
            distances_B_to_A = tf.sort(distances_B_to_A)
            dists_BA = tf.map_fn(fn = lambda x: x, elems=distances_B_to_A)
            kth_BA = get_dist_at_k(dists_BA, k)
            # then shut off the PHDK loss if either switch_true or switch_pred is zeros
            # check if either kth_AB or kth_BA is 2.701.0 and if so set switch to 0 conditional
            switch_AB = tf.cond(tf.equal(kth_AB,dummy), lambda: tf.zeros_like(switch_true), lambda: tf.ones_like(pred))
            switch_BA = tf.cond(tf.equal(kth_BA,dummy), lambda: tf.zeros_like(switch_pred), lambda: tf.ones_like(pred))
            switch = tfm.multiply(switch_true,switch_pred)
            switch = tfm.multiply(switch,switch_AB)
            switch = tfm.multiply(switch,switch_BA)
            kth = tfm.reduce_mean(tfm.multiply_no_nan(tfm.maximum(kth_AB, kth_BA), switch)) # if both switchs, this creates a ones at value max(kth_AB, kth_BA), which reduces to max(kth_AB, kth_BA)
            # if K is below the cutoff distance
            switch = tf.cond(tf.less_than(kth, cutoff_distance), lambda: tf.zeros_like(kth), lambda: tf.ones_like(kth))
            kth = tfm.multiply(switch,kth)
            loss -= reward*tfm.sin(tfm.divide(tfm.multiply(2*3.14,kth),4*cutoff_distance)) 
            # if we subtracted, that would incentivise larger kth_distances
        return MSE + loss
    return phdk



def get_dist_at_k(dists, k):
    # return the distance at k if there is one, otherwise return 100
    try: 
        dist = dists[k]
    except:
        dist = tf.cast(2.701, tf.float32)
    return tf.cast(dist, tf.float32)

def get_neg_G_beta(A, B, beta=12960000):
    # returns negative G_beta for loss value 
    y_TA = get_Y(A,B)
    y = y_TA.read(0) 
    const = 1 - (y/beta)
    G_beta_loss_value = tf.cast(tfm.negative(tfm.maximum(const, 0.0)), tf.float32)
    G_beta_TA = tf.TensorArray(tf.float32, size=0, dynamic_size=True).write(0, G_beta_loss_value)
    return G_beta_TA


def BCE(cutoff=0.5):
    def bce(y_true, y_pred):
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1))
        y_true = tf.where(y_true<cutoff,0.0,1.0)
        y_pred = tf.where(y_pred<cutoff,0.0,1.0)
        loss = tfm.add(tfm.multiply_no_nan(y_pred,tfm.log_sigmoid(y_true)), tfm.multiply_no_nan(tfm.subtract(tf.ones_like(y_pred),y_pred),tfm.log_sigmoid(tfm.subtract(tf.ones_like(y_true),y_true))))        
        return loss
    return bce
def get_G(A, B):
    # just return G, which is smaller for better predictions
    y_TA = get_Y(A,B)
    y = y_TA.read(0)
    G = tf.pow(y,tf.constant([0.3333])) # different metric than G_beta
    G_TA = tf.TensorArray(tf.float32, size=0, dynamic_size=True).write(0, G)
    return G_TA

def get_Y(A,B):
    # takes in binary tensors
    n_A = tfm.reduce_sum(A)
    n_B = tfm.reduce_sum(B)
    n_AB = tfm.reduce_sum(tfm.multiply(A,B))
    n_A, n_B, n_AB = tf.cast(n_A, tf.float32), tf.cast(n_B, tf.float32), tf.cast(n_AB, tf.float32)
    y1 = n_A + n_B + n_AB
    indices_A, indices_B = tf.where(A), tf.where(B)
    coordX_A_TA, coordY_A_TA = find_coord(indices_A)
    coordX_B_TA, coordY_B_TA = find_coord(indices_B)
    mindists_AB_TA = find_min_distances(coordX_A_TA, coordY_A_TA, coordX_B_TA, coordY_B_TA) 
    mindists_BA_TA = find_min_distances(coordX_B_TA, coordY_B_TA, coordX_A_TA, coordY_A_TA)
    # MED = mean error distance = 
    med_AB = tf.reduce_mean(mindists_AB_TA.read(0))
    med_BA = tf.reduce_mean(mindists_BA_TA.read(0))
    y2 = tfm.multiply(med_AB,n_B) + tfm.multiply(med_BA,n_A)
    y = tfm.multiply(y1,y2)
    y_TA = tf.TensorArray(tf.float32, size=0, dynamic_size=True).write(0, y)
    return y_TA

def return_num(num):
    return tf.constant(num,dtype=tf.float32)

def DELTA_with_contours(p=2,cutoff=15):
    # Find baddeley's delta metric with contours
    # probably other ways to speed it up. Like if 
    # index is above 0, maybe cut all but top lines? 
    @tf.function
    def inner(y_true, y_pred):
        loss = tf.cast(0,dtype=tf.float32)
        h_pixels = 60
        n_pixels = 3600.0
        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1)) # do batch_n x (everything flattened)
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1))
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        # attach dummy var of shape y_true to y_true
        # coord_raster_np = product(tf.range(h_pixels),tf.range(h_pixels))        
        # coord_raster = tf.convert_to_tensor(coord_raster_np)
        ones = tf.ones((h_pixels, h_pixels))
        coord_raster = find_coord(ones)
        delta = 0
        trues = tf.map_fn(fn = lambda x: x, elems=y_true)
        preds = tf.map_fn(fn = lambda x: x, elems=y_pred)
        for idx in tf.range(batch_size):
            idx = tf.cast(idx,tf.int32)
            true = trues[idx]
            pred = preds[idx]
            y_true = tf.where(true<cutoff,0.0,1.0)
            y_pred = tf.where(pred<cutoff,0.0,1.0)
            print(true.shape)
            print(pred.dtype)
            RMSE = tfm.sqrt(tf.reduce_mean(tfm.square(tfm.subtract(true,pred))))
            true_matrix = tf.reshape(true, (h_pixels,h_pixels))
            print(true_matrix.shape)
            objects_in_A = measure.find_contours(tf.reshape(true, (h_pixels,h_pixels)))
            objects_in_B = measure.find_contours(tf.reshape(pred, (h_pixels,h_pixels)))
            sums=0
            for idx, coord in enumerate(coord_raster):
                mindist_A, mindist_B = 1000, 1000
                for obj_A in objects_in_A: # cycle through list of coordinates for each object
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
            delta+=tfm.divide(sums,n_pixels) # 3600 is the number of pixels in the raster
        return delta
    return inner

def G(scaler, cutoff=15, min_N=6):
    # Lower G ~ better prediction
    # G_beta is transformed version of G to give values between 0 and 1 (both are functions of y)
    # perhaps this version will be more useful, as the window for its usefulness is not determined by arbitrary constant beta
    # but matching it with MSE may prove problematic
    def g(y_true, y_pred):
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1))
        y_true = y_true*scaler.scale_
        y_true = y_true+scaler.mean_
        y_pred = y_pred*scaler.scale_
        y_pred = y_pred+scaler.mean_
        loss = tf.cast(0,dtype=tf.float32)
        trues = tf.map_fn(fn = lambda x: x, elems=y_true)
        preds = tf.map_fn(fn = lambda x: x, elems=y_pred)
        min_nonzero_pixels = tf.reduce_sum(tf.constant(min_N, dtype=tf.float32))
        for idx in tf.range(batch_size):
            idx = tf.cast(idx, tf.int32)
            true = trues[idx]
            pred = preds[idx]
            MSE = tf.reduce_mean(tfm.square(tfm.subtract(true,pred)))
            y_true = tf.where(true<cutoff,0.0,1.0)
            y_pred = tf.where(pred<cutoff,0.0,1.0)
            n_true = tf.reduce_sum(y_true)
            n_pred = tf.reduce_sum(y_pred)
            switch_true = tf.cond(n_true < min_nonzero_pixels, lambda: tf.zeros_like(true), lambda: tf.ones_like(true))
            switch_pred = tf.cond(n_pred < min_nonzero_pixels, lambda: tf.zeros_like(pred), lambda: tf.ones_like(pred))
            switch = tfm.multiply_no_nan(switch_true,switch_pred)
            G_TA = get_G(y_true,y_pred)
            G = G_TA.read(0)
            loss_G = tfm.reduce_mean(tfm.multiply_no_nan(G,switch)) # switch off if cond met
            loss+= MSE + loss_G
        return loss
    return g

def MSE(y_true, y_pred):
    return tf.reduce_mean(tfm.square(tfm.subtract(y_true,y_pred)))

def avgMED(scaler, cutoff=20, min_N=30,c=3):
    def AVGmed(y_true, y_pred):
        const = tf.constant([c],tf.float32) # constant c, multiplied by MED (
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        MSE = tf.reduce_mean(tf.square(y_true-y_pred))
        y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], -1))
        y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], -1))
        loss, loss_med = tf.cast([0],dtype=tf.float32), tf.cast([0],dtype=tf.float32)
        # rescale
        y_true = y_true*scaler.scale_
        y_true = y_true+scaler.mean_
        y_pred = y_pred*scaler.scale_
        y_pred = y_pred+scaler.mean_
        trues = tf.map_fn(fn = lambda x: x, elems=y_true)
        preds = tf.map_fn(fn = lambda x: x, elems=y_pred)
        min_nonzero_pixels = tf.reduce_sum(tf.constant(min_N, dtype=tf.float32))
        for idx in tf.range(batch_size):
            idx = tf.cast(idx, tf.int32)
            true = trues[idx]
            pred = preds[idx]
            MSE = tf.reduce_mean(tfm.square(tfm.subtract(true,pred)))
            true = tf.where(true<cutoff,0.0,1.0)
            pred = tf.where(pred<cutoff,0.0,1.0)
            n_true = tf.reduce_sum(true)
            n_pred = tf.reduce_sum(pred)
            loss_TA = tf.cond(tf.logical_or(n_true < min_nonzero_pixels, n_pred < min_nonzero_pixels), lambda: get_zero(true,pred), lambda: get_MED(true,pred))
            try:
                loss_med += loss_TA.read(0)
            except:
                loss_med += 0
            loss += loss_med + MSE # do we benefit from reducing across the batch dimension? we should be able to look at familiar batches and see the little increase due to the distance component
            tf.print(n_true,n_pred)
            tf.print(loss_med)
        return loss # this is essentially MSE given c ~ 0. Thus, this will show if there are some weird gradients flowing through that are preventing the model from learning
    return AVGmed 

def get_MED(A,B):
    # takes in binary tensors
    tf.print('in MED')
    coordX_A_TA, coordY_A_TA = find_coord(A)
    coordX_B_TA, coordY_B_TA = find_coord(B)
    mindists_AB_TA = find_min_distances(coordX_A_TA, coordY_A_TA, coordX_B_TA, coordY_B_TA)
    mindists_BA_TA = find_min_distances(coordX_B_TA, coordY_B_TA, coordX_A_TA, coordY_A_TA)
    # MED = mean error distance = 
    med_AB = tf.reduce_mean(mindists_AB_TA.read(0))
    med_BA = tf.reduce_mean(mindists_BA_TA.read(0))
    avg_med = tfm.divide(tfm.add(med_AB,med_BA),tf.constant(0.5))
    loss_TA = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    loss_TA = loss_TA.write(loss_TA.size(), avg_med)
    return loss_TA 

def find_coord(A,length=60):
    length = length # to the square root of the second dimension of shape
    # initialize TAs
    indices = tf.where(A)
    indices = tf.map_fn(fn=lambda x: x, elems=indices)
    coord_X_TA = tf.TensorArray(tf.float32, size=0, dynamic_size=True,clear_after_read=False)
    coord_Y_TA = tf.TensorArray(tf.float32, size=0, dynamic_size=True,clear_after_read=False)
    coord_X, coord_Y = [], []
    for idx in tf.range(len(indices)):
        # get the first value of each tensor 
        x = float(indices[idx] % length)
        y = float(indices[idx] // length)
        coord_X_TA = coord_X_TA.write(coord_X_TA.size(), x)
        coord_Y_TA = coord_Y_TA.write(coord_Y_TA.size(), y)
    return coord_X_TA, coord_Y_TA

def find_min_distances(coord_X_A_TA, coord_Y_A_TA, coord_X_B_TA, coord_Y_B_TA, is_delta=False):
    # i.e. MED
    # mindistances from A to B
    X_A = tf.map_fn(fn=lambda x: x, elems=coord_X_A_TA.read(0))
    Y_A = tf.map_fn(fn=lambda x: x, elems=coord_Y_A_TA.read(0))
    X_B = tf.map_fn(fn=lambda x: x, elems=coord_X_B_TA.read(0))
    Y_B = tf.map_fn(fn=lambda x: x, elems=coord_Y_B_TA.read(0))
    # get the first value of each tensor 
    distances_TA = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True) 
    idx2 = 0
    min_dist = 1000.0
    dist = tf.constant(0,tf.float32)
    for idx in tf.range(len(X_A)): # for delta, should be 0 to 59
        x_A = X_A[idx] # 0
        y_A = Y_A[idx] # 0
        mindist=1000.0
        for idx2 in tf.range(X_B.shape[0]): # if nothin in A, this will throw an error
            y_B = X_B[idx2]
            x_B = Y_B[idx2]
            diff_X = tfm.subtract(x_A,x_B)
            diff_Y = tfm.subtract(y_A,y_B)
            dist = tfm.sqrt(tfm.square(diff_X) + tfm.square(diff_Y))
            if dist < min_dist:
                min_dist = dist
        distances_TA = distances_TA.write(distances_TA.size(), dist)
    return distances_TA 

def get_zero(A,B):
    tf.print('in zero')
    loss_TA = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    tf.print(f'loss_TA before writing {loss_TA}')
    loss_TA = loss_TA.write(loss_TA.size(), 0.0) # write a single 0 to loss_TA
    tf.print(f'loss_TA after writing {loss_TA}')
    print(loss_TA.size())
    tf.print(loss_TA.size())
    return loss_TA

def get_n_AB(A,B,cutoff):
    n_A = tf.where(A>cutoff,1.0,0.0)
    n_B = tf.where(B>cutoff,1.0,0.0)
    n_AB = tfm.multiply(n_A,n_B)
    return n_AB

def iterativeMSE(bins=[0,30,45,60,85,115]):
    def MSEbinned(y_true, y_pred):
        # the idea of this MSE is to find it at different levels
        # algorithms will undershoot because double penalties at high magnitudes hurt
        # so subtract off the lower limit for each set of MSEs
        # then add together at the end to get a kind of normalized MSE
        # still will suffer from there naturally being more low values than high values
        pass
    pass
