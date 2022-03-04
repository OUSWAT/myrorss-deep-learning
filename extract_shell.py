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
import pandas as pd
import subprocess as sp
import time, calendar, gzip, tempfile, shutil, os
import numpy as np
import netCDF4, glob
from os import walk
import xarray as xr
import util 
from netCDF4 import Dataset
import subprocess as sp 

# cases to extract from the year, from b0 to b1
b0 = 120 
b1 = 122

DATA_HOME = '/condo/swatcommon/common/myrorss'
OUT_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data/uncropped'
CONV_HOME = OUT_HOME # may play with names in future
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
NSE_HOME = '/condo/swatcommon/NSE'
fields_accum = ['MESH']


lon_NW, lat_NW, lon_SE, lat_SE = -130.005, 55.005, -59.995, 19.995 # see MYRORSS readme at https://osf.io/kbyf7/
fields = ['MergedReflectivityQC','MergedReflectivityQCComposite','MESH','MergedLLShear','MergedMLShear','Reflectivity_0C', 'Reflectivity_-10C','Reflectivity_-20C']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
min_accum_fields = ['MergedReflectivityQCComposite','MESH','MergedLLShear','MergedMLShear']
fields_accum = ['MergedLLShear_Max_30min','MergedMLShear_Max_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min', 'MergedReflectivityQCComposite_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min']
#fields = ['MergedReflectivityQC']
####################################
# Troubleshooting file
trouble = "trouble.txt"
with open(trouble,"w") as f:
    pass
#os.system('makeIndex.pl /condo/swatwork/mcmontalbano/MYRORSS/data/uncropped/1999/19990103 code_index.xml')
#sys.exit()

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

def extract(day):
    '''
    extract processed tars for fields saved as .netcdf.gzs 
    Note: this loop could perhaps be improved using dir_list = os.listdir(path) rather than looping with os.walk
    see: https://www.geeksforgeeks.org/create-an-empty-file-using-python/
    '''
    year = day[:4]
    os.system('mkdir {}/{}/{}'.format(OUT_HOME,year,day))
    # iterate over fields   
 
    for field in fields:
        time.sleep(5)
        if field == 'MergedLLShear' or field == 'MergedMLShear':
            #ry:
            #   os.system('tar -xvf {}/{}/Azshear/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field))
            #xcept:
            cmd = 'tar -xvf {}/{}/azimuthal_shear_only/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field)
            p = sp.Popen(cmd, shell=True)
            p.wait()
        else:
            cmd = 'tar -xvf {}/{}/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field)
            p = sp.Popen(cmd, shell=True)
            p.wait()
        field_path = '{}/{}/{}/{}'.format(OUT_HOME,year,day,field) # pointing to the field in the uncropped directory
        subdirs = os.listdir(field_path)                           # list the directories within the field
        for subdir in subdirs:                                     # usually a dummy directory like '00.00' or '00.25'
            files = next(walk('{}/{}'.format(field_path,subdir)), (None, None, []))[2] # only grab files
            for f in files:
                try:
                    # convert from .gz to .netcdf
                    filename = '{}/{}/{}'.format(field_path, subdir, f)
                    infile = gzip.open(filename,'rb')
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    shutil.copyfileobj(infile, tmp)
                    infile.close()
                    tmp.close()
                    data = netCDF4.Dataset(tmp.name)
                    dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
                    dataset.to_netcdf(path='{}/{}/{}.netcdf'.format(field_path, subdir, f[:15]))
                except:
                    pass    
    return None  
               # except:
                   #     print('fields, first loop exception: {}/{}/{}'.format(field_path, subdir, f[:15]))
                            
        # REMOVE THE TARS and GZs FROM SCRATCH ONLY
    #        os.system('rm {}/{}/{}/{}/{}/*.gz'.format(OUT_HOME,year,day,field,subdir))        

def extract_NSE(day):
    year=day[:4]
    nse_path = '{}/{}/{}/NSE'.format(OUT_HOME,year,day)
    cmd = 'mkdir {}'.format(nse_path)
    p = sp.Popen(cmd, shell=True)
    p.wait()
    for field in NSE_fields:
        time.sleep(4)
        cmd = 'mkdir {}/{}'.format(nse_path,field)
        p = sp.Popen(cmd, shell=True)
        p.wait()
        field_path = '{}/{}/{}/NSE/{}/nseanalysis/'.format(NSE_HOME, year, day, field) # i.e. /condo/swatcommon/NSE/1999/19990101/NSE/SRHelicity0-2km/nseanalysis
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
                os.system('mkdir {}/{}/{}/NSE/{}'.format(OUT_HOME, year, day, field))
                dataset.to_netcdf(path='{}/{}/{}/NSE/{}/{}.netcdf'.format(OUT_HOME, year, day, field, f[:15]))
            except:
                pass
        # REMOVE THE TARS and GZs FROM SCRATCH ONLY
    return None 
def localmax(day):
    '''
    Given a day in YYYYMMDD, run w2localmax
    To do:
    - change path to OUT_HOME, which is converted MYRORSS storage
    - pick path to 
    - check field name 'MergedReflectivityQCComposite'
    '''
    year = day[:4]
    myrorss_path = '{}/{}'.format(TRAINING_HOME,year)
    out_path = '{}/{}/{}'.format(OUT_HOME,year,day)
    cmd1 = 'makeIndex.pl {} code_index.xml'.format(out_path)
    cmd2 = 'w2localmax -i {}/code_index.xml -I MergedReflectivityQCComposite -o /{}/{} -s -d "40 60 5"'.format(out_path,myrorss_path,day)
    cmd3 = 'makeIndex.pl {}/{} code_index.xml'.format(myrorss_path,day)
    cmd4 = 'w2table2csv -i {}/{}/code_index.xml -T MergedReflectivityQCCompositeMaxFeatureTable -o {}/{}/csv -h'.format(myrorss_path,day,myrorss_path,day)
    p = sp.Popen(cmd1, shell=True)
    p.wait()
    p = sp.Popen(cmd2, shell=True)
    p.wait()
    p = sp.Popen(cmd3, shell=True)
    p.wait()
    p = sp.Popen(cmd4, shell=True)
    p.wait()
    time.sleep(5)
    return None


def get_storm_info(day):
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

def check_MESH(storms_df, day):
    # Check whether the input MESH swath is above a threshold to ensure that hailstorms are chosen, not fledgeling storms 
    year = day[:4] # whatever
    if not os.path.isdir('{}/{}/{}/MESH_Max_30min'.format(OUT_HOME, year, day)):
        cmd = 'w2accumulator -i {}/{}/{}/code_index.xml -g MESH -o {}/{}/{} -C 1 -t 30 --verbose="severe"'.format(OUT_HOME,year, day,OUT_HOME,year,day)
        p = sp.Popen(cmd, shell=True)
        p.wait()    
    MESH_dir = '{}/{}/{}/test_MESH/MESH_Max_30min/00.25'.format(OUT_HOME, year, day)
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
        cmd_list.append("makeIndex.pl {}/{}/{} code_index.xml {} {}".format(OUT_HOME, year, day, time1, time2))
        cmd_list.append('w2cropconv -i {}/{}/{}/code_index.xml -I MESH_Max_30min -o {}/{}/{}/test_MESH -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n'.format(OUT_HOME, year, day, OUT_HOME, year, day, latNW, lonNW, latSE, lonSE))
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
            if pixel > 15: 
                MESH_pixels+=1

    if MESH_pixels > 10:
        return True
    else:
        return False

def accumulate(day, fields):
    date = day
    year = date[:4]
    cmd = 'makeIndex.pl {}/{}/{} code_index.xml'.format(OUT_HOME,year,date)
    #if os.path.exists('{}/{}/{}/{}'.format(OUT_HOME,year,date,fields[0])):
    p = sp.Popen(cmd,shell=True)   
    stdout, stderr = p.communicate()
    #if type(stdout) == None:
    #    lines = []
    #for line in stdout:
    #    print(line.decode(),end='')     
    for field in fields:
        folders = glob.glob('{}/{}/{}/{}_*30min'.format(OUT_HOME,year,day,field))
        if field in folders:
            continue
        cmd = 'w2accumulator -i {}/{}/{}/code_index.xml -g {} -o {}/{}/{} -C 1 -t 30 --verbose="severe"'.format(OUT_HOME,year, date, field,OUT_HOME,year,date)
        p = sp.Popen(cmd,shell=True)
        stdout, stderr = p.communicate()
        print(cmd)
        findMin = False # I don't care about the min shear, for now
        if field[8:] == 'Shear' and findMin == True: # so skip this loop
            cmd = 'w2accumulator -i {}/{}/{}/code_index.xml -g {} -o {}/{}/{} -C 3 -t 30 --verbose="severe"'.format(OUT_HOME,year, date, field, OUT_HOME,year, date)
            p = sp.Popen(cmd,shell=True)
            stdout, stderr = p.communicate()
    #        if type(stdout) == None:
    #            continue
    #        lines = []
    #        for line in stdout:
    #            print(line.decode(), end='')
    return

def cropconv(case_df, date):
    # this also needs reform    
    year = date[:4]

    myrorss_path = '{}/{}/'.format(TRAINING_HOME,year)
    #os.system('makeIndex.pl {}/{}/NSE code_index.xml'.format(myrorss_path,date))
    cmd = 'makeIndex.pl {}/{}/NSE code_index.xml'.format(myrorss_path,date)
    p = sp.Popen(cmd,shell=True)
    p.wait()
    products = ['MergedReflectivityQC','MergedLLShear_Max_30min','MergedMLShear_Max_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min', 'MergedReflectivityQCComposite_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min']
    #products = ['MergedReflectivityQC']
    with open('trouble.txt',"a") as o:
        print(case_df,file=o)
    for idx, row in case_df.iterrows():
        print('Is this a hailstorm?',row['is_Storm'])
        print('the row goes:',row)
        if row['is_Storm'] == False:
            continue # skip 
        # if idx <= 200:
        #     continue
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
        cmd = "makeIndex.pl {}/{}/{} code_index.xml {} {}".format(OUT_HOME, year, date, time1, time2)
        p = sp.Popen(cmd,shell=True)
        p.wait()
        for field in products:
            if os.path.isdir('{}/{}/storm{:04d}/{}'.format(myrorss_path, date,idx,field)):
               continue 
            cmd = 'w2cropconv -i {}/{}/{}/code_index.xml -I {} -o /{}/{}/storm{:04d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,year,date, field, myrorss_path, date, idx, latNW, lonNW,latSE,lonSE)
            p = sp.Popen(cmd,shell=True)
            p.wait()
        with open('trouble.txt','a') as o:
            print('input {} {}'.format(time1,time2),file=o)

        # crop target -
        time1 = (date_1+datetime.timedelta(minutes=27,seconds=30)).strftime('%Y%m%d-%H%M%S')
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=5,seconds=30)).strftime('%Y%m%d-%H%M%S')     
        with open('trouble.txt','a') as o:
            print('target {} {}'.format(time1,time2),file=o)   
        
        cmd1 = "makeIndex.pl {}/{}/{} code_index.xml {} {}".format(OUT_HOME,year,date, time1, time2)
        cmd2 = 'w2cropconv -i {}/{}/{}/code_index.xml -I MESH_Max_30min -o {}/{}/storm{:04d}/target_MESH_Max_30min -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,year,date, myrorss_path, date, idx, latNW, lonNW,latSE,lonSE)
        cmd3 = "makeIndex.pl {}/{}/{}/NSE code_index.xml {}0000 {}0000".format(OUT_HOME, year, date, time1[:11],time2[:11])
        
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
            cmd = 'w2cropconv -i {}/{}/{}/NSE/code_index.xml -I {} -o {}/{}/storm{:04d}/NSE -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME, year, date, f, myrorss_path,date, idx, latNW, lonNW,latSE,lonSE)
            p = sp.Popen(cmd,shell=True)
            p.wait()
    return None


def get_NSE(date,fields):
    year = date[:4]
    for field in fields:
        os.system('mkdir {}/{}/{}/NSE'.format(OUT_HOME,year,date))
        os.sytem('ln -s {}/{}/{}/NSE/{} /{}/{}/{}/NSE'.format(NSE_HOME,year,date,field,OUT_HOME,year,date))

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
    #days = get_raw_cases(year='2011') # get training days
    start = time.time() # start time

    year = day[:4]
    day_path = '{}/{}/{}'.format(TRAINING_HOME,year,day)  
    if not os.path.isdir('{}/{}/{}/{}'.format(OUT_HOME,year,day,fields[-1])):
        extract(day) 
   # if not os.path.isdir('{}/{}/{}/{}'.format(OUT_HOME,year,day,NSE_fields[-1])):
    time.sleep(4)
    if not os.path.isdir('{}/{}/{}/{}'.format(OUT_HOME,year,day,NSE_fields[-1])):
        extract_NSE(day)
    
    # Loop to deciding whether to localmax (true) or not (false) - verified works to generate storms.csv in /csv folder of /data/2011/etc 
    run = True # initially, we expect to run it
    if os.path.isdir('{}/csv'.format(day_path)):
        try:
            storms = pd.read_csv('{}/csv/storms.csv'.format(day_path))
            if len(storms) > 0:
                run = False # set run to False, signaling that localmax has already been run 
            else: # if len == 0, run localmax to fill csv
                run = True
        except: # storms csv not here - run localmax to make csv 
            run = True
    else: # if there's no csv, run localmax() to create the csv
        run = True        
    if run == True:
        localmax(day)
        storms = get_storm_info(day)
    elif run == False:
        storms = pd.read_csv('{}/csv/storms.csv'.format(day_path))  
       
    p = sp.Popen('ls',shell=True)
    p.wait()
    #localmax(day)
    time.sleep(10)
    storms = pd.read_csv('{}/csv/storms.csv'.format(day_path))
    #storms = check_MESH(storms,day)
    p = sp.Popen('ls',shell=True)
    p.wait()
    time.sleep(5)
    #storms_df = pd.read_csv('{}/csv/storms.csv'.format(day_path)) # read in after checking  
    #accumulate(day,min_accum_fields) # accumulate the correct storms
    time.sleep(10)
    cropconv(storms,day) # crop those storms
    print('-----------------------')      
    #util.check_missing(day)
    
    end = time.time() # end time
    with open('info.txt','a') as info:
        print('Elapsed time for 1 day:',end-start)
    print(end-start)
if __name__ == "__main__":
    main()
