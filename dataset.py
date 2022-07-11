import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import pickle
class Dataset(object):
    def __init__(self, args, filename, activation, batch_size=256):
        self.dataset = np.load(filename)
        self.batch_size = batch_size
        if activation == 'linear' or args.loss == 'bce':
            self.transform_y = False
            self.ytr_scalers = []
        else: self.transform_y = True

    def transform(self):
        self.xtr, self.xtr_scalers = self.transform_var(self.xtr)
        if self.transform_y:
            print('ytr before self.transform',self.ytr.mean(),self.ytr.std())
            self.ytr, self.ytr_scalers = self.transform_var(self.ytr)
            print('ytr after self.transform',self.ytr.mean(),self.ytr.std())
            
            r = {}
            r['scaler'] = self.ytr_scalers
            fp = open('ytr_scaler.pkl','wb')
            pickle.dump(r,fp)
        print('xt before',self.xt.mean(),self.xt.std())
        print('yt before',self.yt.mean(),self.yt.std())
        self.transform_val_test()
        print('yt after',self.yt.mean(),self.yt.std())
        print('xt after',self.xt.mean(),self.xt.std())
    def transform_val_test(self):
        # Using the training scalers,
        # transform x_val and x_test, 
        # optionally y_val and y_test if non-linear activation is used
        xf1 = self.flatten(self.xv)
        xf2 = self.flatten(self.xt)
        
        xfs = [xf1, xf2]
        for idx, xf in enumerate(xfs): # do both xv and xt in a loop (idx-0: xv, idx-1: xt)
            for i in range(self.xtr.shape[3]):
                x = xf[i]
                scaler = self.xtr_scalers[i]
                x = scaler.transform(x)
                x = x.reshape(x.shape[0], 60,60)
                if idx ==0:
                    self.xv[:,:,:,i] = x
                if idx == 1:
                    self.xt[:,:,:,i] = x
        if self.transform_y:
            yf1 = self.flatten(self.yv)
            yf2 = self.flatten(self.yt)
            yfs = [yf1, yf2] # flattened ys
            for idx, yf in enumerate(yfs):
                for i in range(self.ytr.shape[3]):
                    y = yf[i]
                    scaler = self.ytr_scalers[i]
                    y = scaler.transform(y)
                    y = y.reshape(y.shape[0], 60,60)
                    if idx == 0:
                        self.yv[:,:,:,i] = y
                    if idx == 1:
                        self.yt[:,:,:,i] = y

    def rescale(self, y):
        # go to original scale for test data
        yf = self.flatten(y)
        yf = yf[0]
        print(type(yf))
        print(yf.shape)
        scaler = self.ytr_scalers[0]
        y = scaler.inverse_transform(yf)
        return y.reshape(self.yt.shape[0], 60, 60, 1)

    def open_dataset(self):
        self.xtr, self.ytr, self.xv, self.yv, self.xt, self.yt = [self.dataset[x] for x in self.dataset.files]

    def setup_dataset(self):
        self.open_dataset()
        self.transform()
        self.generator = self.training_set_generator_images(self.xtr,self.ytr,input_name='input', output_name='output')
    
    def get_input_shape(self):
        return self.xtr.shape[1:]
    
    def transform_var(self,var):
        self.n_channels = var.shape[3]
        tdata_transformed = np.zeros_like(var)
        channel_scalers = []

        for i in range(self.n_channels): # don't tranform Input MESH
            mmx = StandardScaler()
            # make it a bunch of row vectors
            slc = var[:, :, :, i].reshape(var.shape[0], 60 * 60)
            transformed = mmx.fit_transform(slc)
            transformed = transformed.reshape(
                var.shape[0], 60, 60)  # reshape it back to tiles
            # put it in the transformed array
            tdata_transformed[:, :, :, i] = transformed
            channel_scalers.append(mmx)  # store the transform
        return tdata_transformed, channel_scalers
            
    def flatten(self, var):
        return [var[:,:,:,i].reshape(var.shape[0], 3600) for i in range(var.shape[3])]

    def training_set_generator_images(self, ins, outs,
                                  input_name='input_1',
                                  output_name='output'):
        while True:
            # Randomly select a set of example indices
            example_indices = random.choices(range(ins.shape[0]), k=self.batch_size)

            # The generator will produce a pair of return values: one for inputs
            # and one for outputs
            yield({input_name: ins[example_indices, :, :, :]},
                {output_name: outs[example_indices, :, :, :]})

    def binarize_y(self):
        self.ytr = np.where(self.ytr > 30, 1, 0)
        self.yv = np.where(self.yv > 30, 1, 0)
        self.yt = np.where(self.yt > 30, 1, 0)

def rescale(y, scaler):
    # go to original scale for test data
    yf = y.flatten()
    yf = yf[0]
    print(type(yf))
    print(yf.shape)
    y = scaler.inverse_transform(yf)
    return y.reshape(self.yt.shape[0], 60, 60, 1)

