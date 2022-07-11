import pandas as pd
import numpy as np
import datetime as dt
import settings as s
from glob import glob
from netCDF4 import Dataset
import os, sys
# given a target_filepath, 
# check that a. the MESH is greater than 20 mm or that there are N pixels above threshold
# b. gather files

class Target_builder(object):
    def __init__(self, filepath):
        if isinstance(filepath, list):
            targets = [Target(x) for x in filepath]
            return targets# should probably just do the whole thing
        elif isinstance(filepath, str):
            return Target(filepath)	

class Target(object):
    def __init__(self, filepath):
        self.target_path = filepath
        self.storm_dir = '/'.join(filepath.split('/')[:-4])
        self.target_dt = dt.datetime.strptime(filepath.split('/')[-1].split('.')[0], '%Y%m%d-%H%M%S')
        self.fields = ['target_MESH_Max_30min'] + s.shave_fields
        self.filepath = filepath
        self.ins = {}
        self.ins_list = []
        self.outs_list = []
        self.present = True
        self.file_times = []  
 
    def check_MESH(self):
        var = self.read_netcdf(self.target_path)
        cond1 = points_above(var, 30) > 20
        cond2 = points_above(var, 40) > 10
        cond3 = points_above(var, 55) > 5
        if cond1 and cond2 and cond3:
            return True
        else:
            return False

    def gather_ins(self):
        print('beginning to gather')
        # field must be of format '{field}/{subdir}'
        for field in s.fields_10min:
            key = field
            values = glob(f'{self.storm_dir}/{field}/*.netcdf')
            times = [(x,(self.target_dt - dt.datetime.strptime(x.split('/')[-1].split('.')[0], '%Y%m%d-%H%M%S')).total_seconds()/60) for x in values] # get file times
            files = [(x,y) for x,y in times if y > 28 and y < 32] # files must be 30 minutes before target_dt
            if len(files) > 0:
                self.missing=False
                self.ins[key] = files[0]
            else:
                self.missing = True
                break
        if self.missing == True:
            return False
        for field in s.fields_30min:
            key = field
            values = glob(f'{self.storm_dir}/{field}/*.netcdf')
            times = [(x,(self.target_dt - dt.datetime.strptime(x.split('/')[-1].split('.')[0], '%Y%m%d-%H%M%S')).total_seconds()/60) for x in values] # get file times
            files = [x for x,y in times if y>  20 and y < 70]
            if len(files) > 0:
                self.ins[key] = files[0]
                self.missing=False
            else:
                self.missing = True
                break
        if self.missing == True:
            return False
        return True
    
    def read_netcdf(self,f):
        var = Dataset(f)
        var = var.variables[var.TypeName][:,:]
        var = np.where(var<0,0,var)
        return var

    def load_npy(self):
        for key, value in self.ins.items():
            print(key)
            if isinstance(value, tuple):
                value = value[0]
            self.ins_list.append(self.read_netcdf(value))
        sys.exit()
        self.outs_list.append(self.read_netcdf(self.target_path))
        return self.ins_list, self.outs_list

    def save(self):
        if not os.path.isdir(f'{self.storm_dir}/npy'):
            os.makedirs(f'{self.storm_dir}/npy')
        pass

def points_above(image,thres):
    # count pixels
    pixels = np.where(image.squeeze()>thres)[0]
    points_above=len(pixels)
    return points_above
