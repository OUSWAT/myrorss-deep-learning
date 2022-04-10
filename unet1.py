# Author: Michael Montalbano
# Create U-Network or Autoencoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import GaussianNoise, AveragePooling2D, Dropout, BatchNormalization, Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Concatenate, Input, UpSampling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error
import sys
import numpy as np

# channels should increase as you step down the U-Net
# ratio of rows*columns/channels should go down
l = []
for i in np.arange(1, 41, 1):
    l.append(i)


def create_uNet(input_shape, nclasses=2, filters=[64, 128, 256],
                lambda_regularization=None, activation='relu', dropout=None,
                type='binary', optimizer='adam', threshold=0.1):

    global thres
    thres = threshold

    if lambda_regularization is not None:
        lambda_regularization = keras.regularizers.l2(lambda_regularization)

    def lrelu(x): return tf.keras.activations.relu(x, alpha=0.1)
    activation = lrelu

    tensor_list = []
    input_tensor = Input(shape=input_shape, name="input")

    # First Layer is Batch Normalization
    tensor = BatchNormalization()(input_tensor)  # batch1
    tensor = GaussianNoise(0.1)(tensor)

    # conv2d
    tensor = Convolution2D(filters[0],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = BatchNormalization()(tensor)  # batch2

    # conv2d_1
    tensor = Convolution2D(filters[0],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)
    # tensor = BatchNormalization()(tensor) # batch2
    print('first tensor to add', tensor)
    tensor_list.append(tensor)

    if dropout is not None:
        tensor = Dropout(dropout)(tensor)

    #############################

    # batch3
    tensor = AveragePooling2D(pool_size=(2, 2),
                              strides=(2, 2),
                              padding='same')(tensor)
    print('pooled', tensor)
    # batch4
    # 30x30
    tensor = BatchNormalization()(tensor)
    # First step - first conv conv2d_2
    tensor = Convolution2D(filters[1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    # batch5
    tensor = BatchNormalization()(tensor)

    # 2nd conv conv2d_3
    tensor = Convolution2D(filters[1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)
    print('pre-pool, also added', tensor)
    # SKIP CONNECTION 1
    tensor_list.append(tensor)

    tensor = BatchNormalization()(tensor)

    tensor = AveragePooling2D(pool_size=(2, 2),
                              strides=(2, 2),
                              padding='same')(tensor)
    print("Pooled once more", tensor)
    # 15x15
    tensor = BatchNormalization(axis=[1, 2])(tensor)

    tensor = Convolution2D(filters[2],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = BatchNormalization(axis=[1, 2])(tensor)

    tensor = Convolution2D(filters[1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)
    # /upsample and batchnorm
    print('just about to upsample', tensor)
    # take 1 pixel and expand it out to 2 x 2
    tensor = UpSampling2D(size=2)(tensor)
    print('upsampled', tensor)
    tensor = BatchNormalization(axis=[1, 2])(tensor)

    # 3rd conv2d_4
    tensor = Convolution2D(filters[1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = BatchNormalization()(tensor)

    # 3rd conv2d_5
    tensor = Convolution2D(filters[1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    print('the tensor to be connected to the first appended', tensor)
    # SKIP CONNECTION 1
    tensor = Add()([tensor, tensor_list.pop()])  # doesn't like this
    print('Added together', tensor)
    # 60x60
    # take 1 pixel and expand it out to 2 x 2
    tensor = UpSampling2D(size=2)(tensor)

    tensor = BatchNormalization()(tensor)

    tensor = Convolution2D(filters[0],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = BatchNormalization()(tensor)

    tensor = Convolution2D(filters[0],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = Add()([tensor, tensor_list.pop()])

    tensor = BatchNormalization(axis=[1, 2])(tensor)

    #############################
    if type == 'binary':
        output_tensor = Convolution2D(
            1,
            kernel_size=(
                1,
                1),
            padding='same',
            use_bias=True,
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            kernel_regularizer=lambda_regularization,
            activation='sigmoid',
            name='output')(tensor)
    if type == 'regression' or type == 'custom':
        output_tensor = Convolution2D(
            1,
            kernel_size=(
                1,
                1),
            padding='same',
            use_bias=True,
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            kernel_regularizer=lambda_regularization,
            activation=activation,
            name='output')(tensor)

    if type == 'categorical':
        output_tensor = Convolution2D(
            nclasses,
            kernel_size=(
                1,
                1),
            padding='same',
            use_bias=True,
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            kernel_regularizer=lambda_regularization,
            activation='softmax',
            name='output')(tensor)

    if optimizer == 'adam':
        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.0, amsgrad=False)
    if optimizer == 'SGD':
        opt = keras.optimizers.SGD(lr=1e-4, momentum=0.1,
                                   nesterov=False, name='SGD')
    if optimizer == 'RMSprop':
        opt = keras.optimizers.RMSprop(lr=0.00001)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    if type == 'custom':
        model.compile(loss=customMSE, optimizer=opt,
                      metrics=['mse'])

    if type == 'regression':
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                      metrics=[tf.keras.metrics.MeanSquaredError()])
    if type == 'binary':
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                      metrics=[tf.keras.metrics.MeanSquaredError()])
    if type == 'categorical':
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics='accuracy')

    # model.compile(loss='mse',optimizer="adam",
    #             metrics=[MyBinaryAccuracy(),
    #                       MyAUC()])

    return model


def loss_MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1])


def my_MSE_weighted(y_true, y_pred):
    return K.mean(
        tf.multiply(
            tf.exp(tf.multiply(5.0, y_true)),
            tf.square(tf.subtract(y_pred, y_true))

        )
    )


def my_mse_with_sobel(weight=0.0):
    # combines MSE with Sobel edges, which can help produce model predictions
    # with sharper spatial gradients (Uphoff et. al 2021)
    def loss(y_true, y_pred):
        # This function assumes that both y_true and y_pred have no channel dimension.
        # For example, if the images are 2-D, y_true and y_pred have dimensions of
        # batch_size x num_rows x num_columns. tf.expand_dims adds the channel
        # dimensions before applying the Sobel operator.
        edges = tf.image.sobel_edges(tf.expand_dims(y_pred, -1))
        dy_pred = edges[..., 0, 0]
        dx_pred = edges[..., 0, 1]
        edges = tf.image.sobel_edges(tf.expand_dims(y_true, -1))
        dy_true = edges[..., 0, 0]
        dx_true = edges[..., 0, 1]
        return K.mean(
            tf.square(tf.subtract(y_pred, y_true)) +
            weight * tf.square(tf.subtract(dy_pred, dy_true)) +
            weight * tf.square(tf.subtract(dx_pred, dx_true))
        )
    return loss


# Function to calculate "fractions skill score" (FSS).
#
# Function can be used as loss function or metric in neural networks.
#
# Implements FSS formula according to original FSS paper:
# N.M. Roberts and H.W. Lean, "Scale-Selective Verification of
# Rainfall Accumulation from High-Resolution Forecasts of Convective Events",
# Monthly Weather Review, 2008.
# This paper is referred to as [RL08] in the code below.
25


def make_FSS_loss(mask_size):  # choose any mask size for calculating densities
    def my_FSS_loss(y_true, y_pred):
        # First: DISCRETIZE y_true and y_pred to have only binary values 0/1
        # (or close to those for soft discretization)
        want_hard_discretization = False
        # This example assumes that y_true, y_pred have the shape (None, N, N,
        # 1).
        cutoff = 0.7  # choose the cut off value for discretization
        if (want_hard_discretization):
            # Hard discretization:
            # can use that in metric, but not in loss
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)
        else:
            # Soft discretization
            c = 10  # make sigmoid function steep
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))
            # Done with discretization.
        # To calculate densities: apply average pooling to y_true.
        # Result is O(mask_size)(i,j) in Eq. (2) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor
        # 1/n^2 in Eq. (2).
        pool1 = tf.keras.layers.AveragePooling2D(
            pool_size=(
                mask_size, mask_size), strides=(
                1, 1), padding='valid')
        y_true_density = pool1(y_true_binary)
        # Need to know for normalization later how many pixels there are after
        # pooling
        n_density_pixels = tf.cast(
            (tf.shape(y_true_density)[1] *
             tf.shape(y_true_density)[2]),
            tf.float32)
        # To calculate densities: apply average pooling to y_pred.
        # Result is M(mask_size)(i,j) in Eq. (3) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor
        # 1/n^2 in Eq. (3).
        pool2 = tf.keras.layers.AveragePooling2D(
            pool_size=(
                mask_size, mask_size), strides=(
                1, 1), padding='valid')
        y_pred_density = pool2(y_pred_binary)
        # This calculates MSE(n) in Eq. (5) of [RL08].
        # Since we use MSE function, this automatically includes the factor
        # 1/(Nx*Ny) in Eq. (5).
        MSE_n = tf.keras.losses.MeanSquaredError()(y_true_density, y_pred_density)
        # To calculate MSE_n_ref in Eq. (7) of [RL08] efficiently:
        # multiply each image with itself to get square terms, then sum up those terms.
        # Part 1 - calculate sum( O(n)i,j^2
        # Take y_true_densities as image and multiply image by itself.
        O_n_squared_image = tf.keras.layers.Multiply()(
            [y_true_density, y_true_density])
        # Flatten result, to make it easier to sum over it.
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        # Calculate sum over all terms.
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)
        # Same for y_pred densitites:
        # Multiply image by itself
        M_n_squared_image = tf.keras.layers.Multiply()(
            [y_pred_density, y_pred_density])
        # Flatten result, to make it easier to sum over it.
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        # Calculate sum over all terms.

        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)
        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels
        # FSS score according to Eq. (6) of [RL08].
        # FSS = 1 - (MSE_n / MSE_n_ref)
        # FSS is a number between 0 and 1, with maximum of 1 (optimal value).
        # In loss functions: We want to MAXIMIZE FSS (best value is 1),
        # so return only the last term to minimize.
        # Avoid division by zero if MSE_n_ref == 0
        # MSE_n_ref = 0 only if both input images contain only zeros.
        # In that case both images match exactly, i.e. we should return 0.
        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if (want_hard_discretization):
            if MSE_n_ref == 0:
                return(MSE_n)
            else:
                return(MSE_n / MSE_n_ref)
        else:
            return (MSE_n / (MSE_n_ref + my_epsilon))
    return my_FSS_loss


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def customMSE(y_true, y_pred):
    '''
    Correct predictions of 0 do not affect performance.
    Performance is affected only by predictions where y_true > 20
    '''
    loss = tf.square(y_true - y_pred)
    # ignore elements where BOTH y_true & y_pred < 0.1
    mask = tf.cast(tf.logical_or(y_true >= thres, y_pred >= thres), tf.float32)
    loss *= mask
    return loss


def alt_unet(input_shape, nclasses=1, dropout=None, l=None):

    input_tensor = Input(shape=input_shape, name="input")

    x = layers.Conv2D(64, 1, strides=2, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for filters in [128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.AveragePooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(
            filters,
            1,
            strides=2,
            padding="same",
            kernel_regularizer=l2(l),
            bias_regularizer=l2(l))(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [128, 64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(
            filters,
            3,
            padding="same",
            kernel_regularizer=l2(l),
            bias_regularizer=l2(l))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(
            filters,
            3,
            padding="same",
            kernel_regularizer=l2(l),
            bias_regularizer=l2(l))(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(
            filters,
            1,
            padding="same",
            kernel_regularizer=l2(l),
            bias_regularizer=l2(l))(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    output_tensor = layers.Conv2D(
        nclasses,
        1,
        activation="relu",
        padding="same",
        name='output')(x)

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)

    # Define the model
    model = keras.Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer=opt, loss='mse')

    return model


def keras_custom_loss_function(y_true, y_pred):
    # custom loss function - penalize early on, otherwise leave alone
    loss = np.mean(np.sum(np.square((y_true - y_pred) / 10)))
    return loss


def create_seq(input_shape, nclasses, filters=[30, 45, 60],
               lambda_regularization=None, activation='elu'):
    # A sequential model for semantic reasoning

    if lambda_regularization is not None:
        lambda_regularization = keras.regularizers.l2(lambda_regularization)

    tensor_list = []
    input_tensor = Input(shape=input_shape, name="input")

    # 256x256

    tensor = Convolution2D(filters[0],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(input_tensor)

    tensor = Convolution2D(filters[1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = Convolution2D(filters[2],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = Convolution2D(filters[1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = Convolution2D(filters[2],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    tensor = Convolution2D(filters[1],
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=True,
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=lambda_regularization,
                           activation=activation)(tensor)

    #############################
    output_tensor = Convolution2D(nclasses,
                                  kernel_size=(1, 1),
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer='random_uniform',
                                  bias_initializer='zeros',
                                  kernel_regularizer=lambda_regularization,
                                  activation='sigmoid', name='output')(tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt,
                  metrics=["binary_accuracy"])

    return model
