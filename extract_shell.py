############################################
#
# Author: Michael Montalbano
#
# Purpose: Extract data fields from tars 
# to compose database of netcdfs organized  by date
# and create training samples for use in 
# image-to-image translation models (UNet)
#
#############################################

import os, sys, datetime, random
from os import environ 
import pandas as pd
import subprocess as sp
import time, calendar, gzip, tempfile, shutil, os
import numpy as np
import netCDF4, glob
from os import walk
import xarray as xr
import util 
from netCDF4 import Dataset
from datetime.datetime import strftime

lscratch = os.environ.get('LSCRATCH')

# cases to extract from the year, from b0 to b1
b0 = 120 
b1 = 122

DATA_HOME = '/condo/swatcommon/common/myrorss'
UNCROPPED_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data/uncropped'

TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
NSE_HOME = '/condo/swatcommon/NSE'
scripts = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning'

lon_NW, lat_NW, lon_SE, lat_SE = -130.005, 55.005, -59.995, 19.995 # see MYRORSS readme at https://osf.io/kbyf7/
fields = ['MergedReflectivityQC','MergedReflectivityQCComposite','MESH','MergedLLShear','MergedMLShear','Reflectivity_0C', 'Reflectivity_-10C','Reflectivity_-20C']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
min_accum_fields = ['MergedReflectivityQCComposite','MESH','MergedLLShear','MergedMLShear']
fields_accum = ['MergedLLShear_Max_30min','MergedMLShear_Max_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min', 'MergedReflectivityQCComposite_Max_30min']
accum_fields = fields_accum
all_fields = fields+NSE_fields

####################################
# Functions 

def get_raw_cases(year = '1999'):
    # given a year, return the cases
    # step through directory 
    # @returns - randomized (shuffled) list of dates as strings 
    cases = []
    for subdir, dirs, files in os.walk('{}/{}'.format(DATA_HOME,year)):
        for fname in sorted(files):
            print(fname[:4])
            if fname[:4] == '2011':
                cases.append(fname[:8])    
    return cases

def extract(day,OUTPATH,check=True):
    '''
    extract processed tars for fields saved as .netcdf.gzs 
    Note: this loop could perhaps be improved using dir_list = os.listdir(path) rather than looping with os.walk
    see: https://www.geeksforgeeks.org/create-an-empty-file-using-python/
    '''
    if check==True: # only unzip the composite reflectivity and MESH
        fields = ['MESH','MergedReflectivityQCComposite']   


    OUT_HOME = OUTPATH
    year = day[:4]
    # iterate over fields    
    for field in fields:
        if field == 'MergedLLShear' or field == 'MergedMLShear':
            #ry:
            #   os.system('tar -xvf {}/{}/Azshear/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field))
            #xcept:
            cmd = 'tar -xvf {}/{}/azimuthal_shear_only/{}.tar -C {} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,field)
            p = sp.Popen(cmd, shell=True)
            p.wait()
        else:
            cmd = 'tar -xvf {}/{}/{}.tar -C {} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,field)
            p = sp.Popen(cmd, shell=True)
            p.wait()
        field_path = '{}/{}'.format(OUT_HOME,field) # pointing to the field in the uncropped directory
        subdirs = os.listdir(field_path)                           # list the directories within the field
        for subdir in subdirs:                                     # usually a dummy directory like '00.00' or '00.25'
            files = next(walk('{}/{}'.format(field_path,subdir)), (None, None, []))[2] # only grab files
            for f in files:
                stupid = True
                if stupid == True:
                    # convert from .gz to .netcdf
                    print('trying')
                    filename = '{}/{}/{}'.format(field_path, subdir, f)
                    print('filename',filename)
                    if filename[:-2] != 'gz':
                        pass
                    gz = gzip.open(filename)
                    data = netCDF4.Dataset('dummy',mode='r',memory=gz.read())
                    dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
                    print('assigning')
                    dataset.to_netcdf('{}/{}/{}.netcdf'.format(lscratch,field,f[:15]))
    return None  

def untar(day,field):
    # untars day for field
    if field == 'MergedLLShear' or field == 'MergedMLShear':
        cmd = 'tar -xvf {}/{}/azimuthal_shear_only/{}.tar -C {} --wildcards "{}"'.format(DATA_HOME,day[:4],day,OUT_HOME,field)
        p = sp.Popen(cmd, shell=True)
        p.wait()
    else:
        cmd = 'tar -xvf {}/{}/{}.tar -C {} --wildcards "{}"'.format(DATA_HOME,day[:4],day,OUT_HOME,field)
        p = sp.Popen(cmd, shell=True)
        p.wait()    
   return None 

def get_valid_files(files, valid_times):
    '''
    Given a set of files and a set of valid times,
    Return only those files for which the time is valid
    '''
    # initialize valid_files dictionary
    n = len(files)
    valid_files = {}
    keys = range(n)
    for i in keys:
        valid_files[i] = True
    # check files; upgrade this by sorting and speeding up process
    for idx, f in enumerate(files):
        t = f.split('/')[-1]
        if t not in valid_times:
            valid_files[idx] = False

    # Fill new_files list to contain the valid files
    new_files = []
    for key in valid_files.keys():
        if valid_files[key] == False:
            continue
        else:
            new_files.append(files[key])
    return new_files

def extract_after(day, potential_times,OUTPATH=lscratch, target=False):
    '''
    extracts the data to /lscratch only for given times
    '''
    OUT_HOME = OUTPATH
    valid_times = potential_times # same thing, kind of 
    if target == True:
        fields = ['MESH']
        # extract MESH code needed 
    else:
        fields = all_fields
    # extract the day tarball 
    for field in fields:
        untar(field)
    # Try to extract only the valid .netcdf.gz's 
        if target == True:
            field_path = '{}/target_MESH_Max_30min/{}/**/*.netcdf.gz'.format(OUT_HOME,field)
        else:
            field_path = '{}/{}'.format(OUT_HOME,field)
        files = glob.glob('{}/**/*.netcdf.gz'.format(field_path))
        valid_files = get_valid_files(files, valid_times) # return the valid files from the list of actual files and the valid times
        # extract these files
        for filename in valid_files:
            gz = gzip.open(filename)
            data = netCDF4.Dataset('dummy',mode='r',memory=gz.read())
            dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
            print('assigning')
            dataset.to_netcdf('{}/{}/{}.netcdf'.format(lscratch,field,f[:15]))   
    return None

def full_extract(day, valid_times, OUTPATH):
    ''' 
    given the day, valid times, and OUTPATH, fully extract to /data
    ''' 
    # first extract inputs from t0 to t1, where t1 is the time where decide(MESH) == True
    for t1 in valid_times:
        t1 = time.strftime('%Y%m%d-%H%M%S')
        date1 = datetime.datetime.strptime(t1, '%Y%m%d-%H%M%S')
        t0 = (date1 - datetime.timedelta(minutes=31)).strftime('%Y%m%d-%H%M%S')
        t2 = (date1 + datetime.timedelta(minutes=31)).strftime('%Y%m%d-%H%M%S')
        input_times = [t0]
        while t0 != t1:
            # Advance by 1 second
            t0 = (t0.strftime('%Y%m%d-%H%M%S') + datetime.timedelta(seconds=1)).strftime('%Y%m%d-%H%M%S')
            input_times.append(t0)
        target_times = [t1]
        while t1 != t2:
            # Advance by one second
            t1 = (t1.strftime('%Y%m%d-%H%M%S') + datetime.timedelta(seconds=1)).strftime('%Y%m%d-%H%M%S')
            target_times.append(t1)
    extract_after(day, input_times, target=False)
    extract_after(day, target_times, target=True)
        


def extract_nse(day,OUTPATH):
    year=day[:4]
    OUT_HOME = OUTPATH
    nse_path = '{}/NSE'.format(OUT_HOME)
    cmd = 'mkdir {}'.format(nse_path)
    p = sp.Popen(cmd, shell=True)
    p.wait()
    for field in NSE_fields:
        cmd = 'mkdir {}/{}'.format(nse_path,field)
        p = sp.Popen(cmd, shell=True)
        p.wait()
        field_path = '{}/{}/nseanalysis/'.format(nse_path, field) # i.e. $LSCRATCH/NSE/SRHelicity0-2km/nseanalysis
        files = next(walk(field_path), (None, None, []))[2] # list files in the directory
        for f in files:
            try:
                filename = '{}/{}'.format(field_path, f)
                infile = gzip.open(filename,'rb') # e.g. nseanalysis/20110401-100000.netcdf.gz (unzip using gzip.open)
                tmp = tempfile.NamedTemporaryFile(delete=False)
                shutil.copyfileobj(infile, tmp)
                infile.close()
                tmp.close()
                data = netCDF4.Dataset(tmp.name)
                dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
                dataset.to_netcdf(path='{}/NSE/{}/{}.netcdf'.format(OUT_HOME, field, f[:15]))
            except:
                pass
       # os.system('rm {}/*gz'.format(field_path))
        # REMOVE THE TARS and GZs FROM SCRATCH ONLY
    return None 

def localmax(day,OUTPATH):
    '''
    Given a day in YYYYMMDD, run w2localmax
    To do:
    - change path to OUT_HOME, which is converted MYRORSS storage
    - pick path to 
    - check field name 'MergedReflectivityQCComposite'
    '''
    OUT_HOME = OUTPATH
    year = day[:4]
    myrorss_path = '{}/{}'.format(TRAINING_HOME,year)
    out_path = OUT_HOME
    cmd1 = 'makeIndex.pl {} code_index.xml'.format(out_path)
    cmd2 = 'w2localmax -i {}/code_index.xml -I MergedReflectivityQCComposite -o {}/{} -s -d "40 60 5"'.format(out_path,out_path,day)
    cmd3 = 'makeIndex.pl {}/{} code_index.xml'.format(out_path,day)
    cmd4 = 'w2table2csv -i {}/{}/code_index.xml -T MergedReflectivityQCCompositeMaxFeatureTable -o {}/{}/csv -h'.format(out_path,day,myrorss_path,day)
    cmds = [cmd1,cmd2,cmd3,cmd4]
    for c in cmds:
        print(c)
    p = sp.Popen(cmd1, shell=True)
    p.wait()
    p = sp.Popen(cmd2, shell=True)
    p.wait()
    p = sp.Popen(cmd3, shell=True)
    p.wait()
    p = sp.Popen(cmd4, shell=True)
    p.wait()
    return None

def get_storm_info(day,OUTPATH):
    OUT_HOME = OUTPATH
    year = day[:4] 
    myrorss_path = '{}/{}/'.format(TRAINING_HOME,year)
    LOCALMAX_PATH = '{}/{}/csv'.format(myrorss_path, day)
    case_df = pd.DataFrame(columns={"timedate","Latitude","Longitude","Storm","Reflectivity","is_Storm"})
    i=+1
    delta = 0.15
    # builds dataframe of case centers
    files = sorted(glob.glob('{}/MergedReflectivityQCCompositeMaxFeature**.csv'.format(LOCALMAX_PATH),recursive=True)) 
#    files = sorted(next(walk('{}'.format(LOCALMAX_PATH)), (None, None, []))[2]) # Grab files within localmax directory (csv)
    length=len(files)-1
    for idx, f in enumerate(files):
        timedate = f[-19:-4]
        minutes = timedate[-4:-2]
        if (int(minutes) >= 28 and int(minutes) <= 32) or (int(minutes)>=58 and int(minutes)<=60) or (int(minutes)>=0 and int(minutes)<=2):
        #if (minutes == '30' or minutes =='00') and idx != 0 and idx != length:
            df = pd.read_csv('{}/MergedReflectivityQCCompositeMaxFeatureTable_{}.csv'.format(LOCALMAX_PATH, timedate))
            if df.empty:
                print("Empty dataframe!")  
            else:
                # List of valid clusters
                valid_clusters = {}
                keys = range(df.shape[0])
                for i in keys:
                    valid_clusters[i] = True # initialize valid with Trues
                # find max
                for idx, val in enumerate(df["MergedReflectivityQCCompositeMax"]):
                    if valid_clusters[idx] == False:
                        continue
                    if val < 40 or df['Size'].iloc[idx] < 20:
                        valid_clusters[idx] = False
                        continue
                    lat = df['#Latitude'].iloc[idx]
                    lon = df['Longitude'].iloc[idx]
                    latN = lat + delta
                    latS = lat - delta
                    lonW =  lon - delta
                    lonE =  lon + delta
                    # Don't include clusters too close to domain edge
                    if latN > (lat_NW - 0.16) or latS <= (lat_SE + 0.16) or lonW < (lon_NW + 0.16) or lonE >= (lon_SE-0.16):
                        valid_clusters[idx] = False
                        continue
                    for idx2, val2 in enumerate(df["MergedReflectivityQCCompositeMax"]):
                        if idx2 == idx or valid_clusters[idx2] == False:
                            continue
                        if df['Size'].iloc[idx2] < 20 or val2 < 40:
                            valid_clusters[idx2] = False
                            continue 
                        lat2 = df['#Latitude'].iloc[idx2]
                        lon2 = df['Longitude'].iloc[idx2]
                        if lat2 < latN and lat2 > latS and lon2 > lonW and lon2 < lonE:
                            if val2 > val: 
                                valid_clusters[idx] = False
                            else:
                                valid_clusters[idx2] = False
                # valid_clusters is complete
                # # add valid rows to case_dfS
                for key in valid_clusters.keys():
                    if valid_clusters[key] == False:
                        continue
                    else:
                        row_idx = key
                        try:
                            row = {"timedate":timedate,"Latitude":df['#Latitude'].iloc[row_idx],"Longitude":df['Longitude'].iloc[row_idx],'Storm':df['RowName'].iloc[row_idx],'Reflectivity':df['MergedReflectivityQCCompositeMax'].iloc[row_idx], 'is_Storm':False}
                        except:
                            print(row_idx)
                        case_df.loc[len(case_df.index)] = row
    case_df = case_df.sort_values(['timedate'])
    case_df.to_csv('{}/{}/{}/csv/storms.csv'.format(TRAINING_HOME, year, day))
    return case_df

def check_df(storms_df):
    # take in df and return list of valid times for day
    valid_times = []
    for idx, row in storms_df.iterrows():
        t0 = row['timedate']
        is_Storm = row['is_Storm']
        if is_Storm == True and t0 not in valid_times:
            valid_times.append(t0)
    return valid_times

def check_MESH(storms_df, day,OUTPATH):
    OUT_HOME = OUTPATH
    # Check whether the input MESH swath is above a threshold to ensure that hailstorms are chosen, not fledgeling storms 
    year = day[:4] # whatever
    if not os.path.isdir('{}/MESH_Max_30min'.format(OUT_HOME, year, day)):
        cmd = 'w2accumulator -i {}/code_index.xml -g MESH -o {} -C 1 -t 30 --verbose="severe"'.format(OUT_HOME,OUT_HOME)
        p = sp.Popen(cmd, shell=True)
        p.wait()    
    MESH_dir = '{}/test_MESH/MESH_Max_30min/00.25'.format(OUT_HOME)
    if os.path.isdir('{}'.format(MESH_dir)):
        cmd = 'rm {}/*'.format(MESH_dir)
        p = sp.Popen(cmd, shell=True)
        p.wait()
    keep_list = []
    skip_times = []
    for idx, row in storms_df.iterrows():
        lon = row['Longitude']
        lat = row['Latitude']
        date1 = datetime.datetime.strptime(row['timedate'], "%Y%m%d-%H%M%S")
        delta = 0.15
        lonNW = lon - delta
        latNW = lat + delta
        lonSE = lon + delta
        latSE = lat - delta
        time1 = (date1 - datetime.timedelta(minutes=2, seconds=30)).strftime('%Y%m%d-%H%M%S')
        time2 = (date1 + datetime.timedelta(minutes=2, seconds=30)).strftime('%Y%m%d-%H%M%S')         
        # Check if this time has already been skipped
        if time1 in skip_times:
            keep_list.append(False) # Set false 
            continue # skip
        # crop MESH
        cmd_list = []
        cmd_list.append("makeIndex.pl {} code_index.xml {} {}".format(OUT_HOME, time1, time2))
        cmd_list.append('w2cropconv -i {}/code_index.xml -I MESH_Max_30min -o {}/test_MESH -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n'.format(OUT_HOME, OUT_HOME, latNW, lonNW, latSE, lonSE))
        for cmd in cmd_list:
            p = sp.Popen(cmd, shell=True)
            p.wait()
        # check for sufficient MESH pixels
        files = glob.glob('{}/*netcdf'.format(MESH_dir))     
        if files == []:
            skip_times.append(time1) # if empty, then skip this time in the future
            keep_list.append(False)  
            continue # skip
        nc = Dataset(files[0])
        var = nc.variables['MESH_Max_30min'][:,:] 
        var = np.where(var<-50,0,var) 
       # Decide whether it meets the conditions 
        keep = decide(var)
        # wipe dir
        cmd = 'rm {}/*'.format(MESH_dir)
        p = sp.Popen(cmd,shell=True)
        p.wait() 
        # modify is_Storm column of storms_df
        keep_list.append(keep)
        #storms_df = storms_df.at[idx, 'is_Storm'] = keep # set is_Storm to true or false
    print(keep_list)
    storms_df['is_Storm'] = keep_list
    storms_df.to_csv('{}/{}/{}/csv/storms.csv'.format(TRAINING_HOME, year, day))
    return storms_df 

def decide(image):
    # given a MESH image, decide whether to keep the sample
    # take a subset in the center
    image = np.squeeze(image)
    image = image[10:50,10:50]
    MESH_pixels = 0 
    for row in image:
        for pixel in row:
            if pixel > 20: 
                MESH_pixels+=1
    if MESH_pixels > 25:
        return True
    else:
        return False

def accumulate(day, fields,OUTPATH):
    date = day
    year = date[:4]
    OUT_HOME = OUTPATH
    cmd = 'makeIndex.pl {} code_index.xml'.format(OUT_HOME)
    #if os.path.exists('{}/{}/{}/{}'.format(OUT_HOME,year,date,fields[0])):
    p = sp.Popen(cmd,shell=True)   
    stdout, stderr = p.communicate()
    #if type(stdout) == None:
    #    lines = []
    #for line in stdout:
    #    print(line.decode(),end='')     
    for field in fields:
        folders = glob.glob('{}/{}_*30min'.format(OUT_HOME,field))
        if field in folders:
            continue
        cmd = 'w2accumulator -i {}/code_index.xml -g {} -o {} -C 1 -t 30 --verbose="severe"'.format(OUT_HOME, field,OUT_HOME)
        p = sp.Popen(cmd,shell=True)
        stdout, stderr = p.communicate()
        print(cmd)
        findMin = False # I don't care about the min shear, for now
        if field[8:] == 'Shear' and findMin == True: # so skip this loop
            cmd = 'w2accumulator -i {}/code_index.xml -g {} -o {} -C 3 -t 30 --verbose="severe"'.format(OUT_HOME, field, OUT_HOME)
            p = sp.Popen(cmd,shell=True)
            stdout, stderr = p.communicate()
    #        if type(stdout) == None:
    #            continue
    #        lines = []
    #        for line in stdout:
    #            print(line.decode(), end='')
    return

def cropconv(case_df, date,OUTPATH):
    # this also needs reform    
    year = date[:4]
    OUT_HOME = OUTPATH
    myrorss_path = '{}/{}/'.format(TRAINING_HOME,year)
    #os.system('makeIndex.pl {}/{}/NSE code_index.xml'.format(myrorss_path,date))
    cmd = 'makeIndex.pl {}/NSE code_index.xml'.format(OUT_HOME)
    p = sp.Popen(cmd,shell=True)
    p.wait()
    products = ['MergedReflectivityQC','MergedLLShear_Max_30min','MergedMLShear_Max_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min', 'MergedReflectivityQCComposite_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min']
    #products = ['MergedReflectivityQC']
    for idx, row in case_df.iterrows():
        if row['is_Storm'] == False:
            continue # skip 
        lon = row['Longitude']
        lat = row['Latitude']
        delta = 0.15
        lonNW = lon - delta
        latNW = lat + delta
        lonSE = lon + delta
        latSE = lat - delta
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time1 = (date_1-datetime.timedelta(minutes=3,seconds=30)).strftime('%Y%m%d-%H%M%S')
        time2 = (date_1+datetime.timedelta(minutes=3,seconds=30)).strftime('%Y%m%d-%H%M%S')
        # crop input
        cmd = "makeIndex.pl {} code_index.xml {} {}".format(OUT_HOME, time1, time2)
        p = sp.Popen(cmd,shell=True)
        p.wait()
        for field in products:
            if os.path.isdir('{}/{}/storm{:04d}/{}'.format(myrorss_path, date,idx,field)):
               continue 
            cmd = 'w2cropconv -i {}/code_index.xml -I {} -o /{}/{}/storm{:04d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME, field, myrorss_path, date, idx, latNW, lonNW,latSE,lonSE)
            p = sp.Popen(cmd,shell=True)
            p.wait()

        # crop target -
        time1 = (date_1+datetime.timedelta(minutes=27,seconds=30)).strftime('%Y%m%d-%H%M%S')
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=5,seconds=30)).strftime('%Y%m%d-%H%M%S')     
        
        cmd1 = "makeIndex.pl {} code_index.xml {} {}".format(OUT_HOME, time1, time2)
        cmd2 = 'w2cropconv -i {}/code_index.xml -I MESH_Max_30min -o {}/{}/storm{:04d}/target_MESH_Max_30min -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME, myrorss_path, date, idx, latNW, lonNW,latSE,lonSE)
        cmd3 = "makeIndex.pl {}/NSE code_index.xml {}0000 {}0000".format(OUT_HOME, time1[:11],time2[:11])
        
        p = sp.Popen(cmd1,shell=True)
        p.wait()
        p = sp.Popen(cmd2,shell=True)
        p.wait()

        # NSE 
        # crop for 30 min prior to 30 min ahead
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time1 = (date_1+datetime.timedelta(minutes=-30,seconds=00)).strftime('%Y%m%d-%H%M%S')
        time2 = (date_1+datetime.timedelta(minutes=5,seconds=00)).strftime('%Y%m%d-%H%M%S')
        p = sp.Popen(cmd3,shell=True)
        p.wait()
        for f in NSE_fields:
            if os.path.isdir('{}/{}/storm{:04d}/NSE/{}'.format(myrorss_path,date,idx,f)):
                continue
            cmd = 'w2cropconv -i {}/NSE/code_index.xml -I {} -o {}/{}/storm{:04d}/NSE -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME, f, myrorss_path,date, idx, latNW, lonNW,latSE,lonSE)
            p = sp.Popen(cmd,shell=True)
            p.wait()
    return 'happy meal'

def get_training_cases(year = '1999'):
    # grabs cases from the traininghome
    cases = []
    for subdir, dirs, files in os.walk('{}/{}'.format(TRAINING_HOME,year)):
        for file in sorted(files):
            cases.append(file[:8])
    return cases


def main():
    testing = True
    day = sys.argv[1]
    year = day[:4]
    OUT_PATH = lscratch
    #days = get_raw_cases(year='2011') # get training days
    start = time.time() # start time

    year = day[:4]
    day_path = '{}/{}/{}'.format(TRAINING_HOME,year,day)  
    #if not os.path.isdir('{}/{}/{}/{}'.format(OUT_HOME,year,day,fields[-1])):
    t0 = time.time()
    extract(day,OUT_PATH,check=True) 
    print('extraction;',time.time()-t0)
    t1 = time.time()
#    extract_nse(day,OUT_PATH)
    t2 = time.time()
    print('extraction (NSE);',t2-t1)
    localmax(day,OUT_PATH)
    print(glob.glob('{}/*'.format(lscratch)))
    print(glob.glob('{}/MergedReflectivityQCComposite/**/*'.format(lscratch)))
#    print(glob.glob('{}/20110326/**/'.format(lscratch))) 
    t3=time.time()
    print('localmax;', t3-t2)
    storms = get_storm_info(day,OUT_PATH)
    t4 = time.time()
    print('storm info;', t4-t3)
    samples = check_MESH(storms,day,OUT_PATH)
    t5 = time.time()
    print('check MESH;', t5-t4)
    times = check_df(samples)
    print(times)
#    accumulate(day,accum_fields,OUT_PATH) # accumulate the correct storms
    t6 = time.time()
    print('accumulated;',t6-t5)
#    dummy = cropconv (samples,day,OUT_PATH)
    t7 = time.time()
    print('cropped and converted;',t7-t6)
    
if __name__ == "__main__":
    main()
