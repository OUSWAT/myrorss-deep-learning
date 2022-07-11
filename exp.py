import numpy as np
import tensorflow as tf
import keras
from unet import UNet
#from altnet import UNet
import pickle
import kerastuner as kt
import metrics as m
import loss_functions as lf

class Experiment():
    def __init__(self, args, dataset, epochs=150, batch_size=256, steps_per_epoch=10):
        self.use_tuner = bool(args.tuner)
        self.args = args
        self.epochs = args.epochs
        self.steps_per_epoch = args.steps
        self.loss = args.loss
        self.dataset=dataset
        self.prefix = 'default'
        if args.model_name != 'None':
            self.first_pass = False
            def lrelu(x) : return tf.keras.activations.relu(x, alpha=0.1)
            self.pod = m.POD()
            self.far = m.FAR()
            self.loss_fn = getattr(lf, args.loss)
            if self.loss == 'GBETA':
                self.loss_fn = self.loss_fn(self.dataset.ytr_scalers[0],self.args.omega)
            else:
                self.loss_fn = self.loss_fn(self.dataset.ytr_scalers[0], self.args.cutoff)
            self.custom_objects = {'lrelu':lrelu, 'pod':self.pod, self.args.loss:self.loss_fn, 'far':self.far}
        else:
            self.first_pass = True
       # self.prefix = need to generate a prefix based off the args

    def set_callbacks(self):
        self.callbacks_list =[keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.01,
                patience=10,
                verbose=1)]
    
    def train(self):
        if self.use_tuner == False:
            if self.first_pass:
                self.model = self.get_UNet()
            else:
                self.model = tf.keras.models.load_model(f'models/{self.args.model_name}.h5', custom_objects = self.custom_objects)
                self.model.compile(loss=self.loss_fn, optimizer=keras.optimizers.Adam(), metrics=[self.pod,self.far])
            self.train_single_model()
        elif self.use_tuner == True:
            self.tune()

    def train_single_model(self):
        self.history = self.model.fit(x=self.dataset.generator,
                                    epochs=self.epochs,
                                    shuffle=True,
                                    steps_per_epoch=self.args.steps,
                                    use_multiprocessing=False,
                                    validation_data=(self.dataset.xv, self.dataset.yv),
                                    callbacks=self.callbacks_list)
        return self.model
    
    def tune(self):
        network = UNet(self.args, self.dataset)
        network.set_loss(self.args.loss)
        network.set_activation(self.args.activation)
        network.set_metrics()
        self.tuner = kt.Hyperband(
                    network.build_tuning_model,
                    objective='val_loss',
                    max_epochs=self.args.epochs,
                    project_name='none',
                    directory=f'results/{self.args.save_dir}',
                    overwrite=True)
        self.tuner.search(self.dataset.xtr,self.dataset.ytr,
                    epochs=self.args.epochs,
                    steps_per_epoch=self.args.steps,
                    shuffle=True,
                    validation_data=(self.dataset.xv, self.dataset.yv),
                    verbose=True,
                    callbacks=self.callbacks_list,)
        self.tuner.results_summary()
        n_best_hyperparameters = self.tuner.get_best_hyperparameters(num_trials=10)
        self.best_hps = n_best_hyperparameters[0]
        self.model = self.tuner.hypermodel.build(self.best_hps)
        self.train_single_model()
        
    def get_UNet(self):
        network = UNet(self.args, self.dataset)
        UNet.dropout = self.args.dropout
        network.set_loss(self.args.loss)
        network.set_activation(self.args.activation)
        network.set_metrics()
        network.build_model()
        network.compile_model()
        return network.model

    def gen_filename(self):
        if self.use_tuner:
            prefix = self.make_model_prefix_tuner()
        else:
            filters_str = 'f'.join([str(i) for i in self.args.filters])
            cut=self.args.cutoff
            note=self.args.note
            transpose = self.args.use_transpose
            resnet = self.args.use_resnet
            prefix = f'loss-{self.args.loss}_dataset-{self.args.ID}_L2-{self.args.L2}_drop-{self.args.dropout}_junct-{self.args.junction}_filters-{filters_str}_act-{self.args.activation}_cut-{cut}_transpose-{transpose}_resnet-{resnet}_note-{note}'
        self.prefix = prefix
        return prefix

    def make_model_prefix_tuner(self):
        t_fac = self.best_hps.get('t_fac')
        g_fac =  self.best_hps.get('g_fac')
        depth =  self.best_hps.get('depth')
        junct =  self.best_hps.get('junct')
        drop =  self.best_hps.get('drop')
        L2   = self.best_hps.get('L2')
        filters = []
        for d in np.arange(depth):
            filters.append(self.best_hps.get(f'layer_{d}_filters'))
        ID = self.args.ID
        loss = self.args.loss
        epochs = self.args.epochs
        cut = self.args.cutoff
        act = self.args.activation
        filters_str = 'f'.join([str(i) for i in filters])
        prefix = f'loss-{loss}_dataset-{ID}_L2-{L2}_drop-{drop}_tfac-{t_fac}_gfac-{g_fac}_junct-{junct}_filters-{filters_str}_act-{act}_cut-{cut}'
        return prefix
            
