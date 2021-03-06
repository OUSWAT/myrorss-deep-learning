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

def get_cases(year = '1999'):
    # given a year, return the cases
    # step through directory 
    cases = []
    for subdir, dirs, files in os.walk('{}/{}'.format(DATA_HOME,year)):
        for file in sorted(files):
            cases.append(file[:8])    
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
        if field == 'MergedLLShear' or field == 'MergedMLShear':
            #ry:
            #   os.system('tar -xvf {}/{}/Azshear/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field))
            #xcept:
            os.system('tar -xvf {}/{}/azimuthal_shear_only/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field))
            print('tar -xvf {}/{}/azimuthal_shear_only/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field))
        else:
            os.system('tar -xvf {}/{}/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field)) #untar it to field path
            print('tar -xvf {}/{}/{}.tar -C {}/{}/{} --wildcards "{}"'.format(DATA_HOME,year,day,OUT_HOME,year,day,field))
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
    return    
               # except:
                   #     print('fields, first loop exception: {}/{}/{}'.format(field_path, subdir, f[:15]))
                            
        # REMOVE THE TARS and GZs FROM SCRATCH ONLY
    #        os.system('rm {}/{}/{}/{}/{}/*.gz'.format(OUT_HOME,year,day,field,subdir))        

def extract_NSE(day):
    year=day[:4]
    os.system('mkdir {}/{}/{}/NSE'.format(OUT_HOME, year, day))
    for field in NSE_fields:
        field_path = '{}/{}/{}/NSE/{}/nseanalysis/'.format(NSE_HOME, year, day, field) # i.e. /condo/swatcommon/NSE/1999/19990101/NSE/SRHelicity0-2km/nseanalysis
        files = next(walk(field_path), (None, None, []))[2] # list files in the directory
        for f in files:
            try:
                filename = '{}/{}'.format(field_path, f)
                infile = gzip.open(filename,'rb')
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
    return   
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
    #sys.stdout(trouble,"w")
    #print(cmd)
    os.system('makeIndex.pl {} code_index.xml'.format(out_path))
    os.system('w2localmax -i {}/code_index.xml -I MergedReflectivityQCComposite -o /{}/{} -s -d "40 60 5"'.format(out_path,myrorss_path,day))
    os.system('makeIndex.pl {}/{} code_index.xml'.format(myrorss_path,day))
    os.system('w2table2csv -i {}/{}/code_index.xml -T MergedReflectivityQCCompositeMaxFeatureTable -o {}/{}/csv -h'.format(myrorss_path,day,myrorss_path,day))
    return


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
        os.system('w2accumulator -i {}/{}/{}/code_index.xml -g MESH -o {}/{}/{} -C 1 -t 30 --verbose="severe"'.format(OUT_HOME,year, day,OUT_HOME,year,day))

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

        # crop MESH
        os.system("makeIndex.pl {}/{}/{} code_index.xml {} {}".format(OUT_HOME, year, day, time1, time2)) # make index for uncropped
        os.system('w2cropconv -i {}/{}/{}/code_index.xml -I MESH_Max_30min -o {}/{}/{}/test_MESH -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME, year, day, OUT_HOME, year, day, latNW, lonNW, latSE, lonSE))
        
        # check for sufficient MESH pixels
        MESH_dir = '{}/{}/{}/test_MESH'.format(OUT_HOME, year, day)
        # open MESH file
        f = glob.glob('{}/**/*netcdf'.format(MESH_dir))
        nc = Dataset(f)
        print(nc.max())
        
        # Decide whether it meets the conditions 
        keep = decide(nc)
        # wipe dir
        os.system('rm {}/*'.format(MESH_dir)) 
        # modify is_Storm column of storms_df
        storms_df[idx, 'is_Storm'] = keep # set is_Storm to true or false
    return None

def decide(image):
    # given a MESH image, decide whether to keep the sample
    # take a subset in the center
    image = np.squeeze(image)
    image = image[15:45,15:45]
    MESH_pixels = 0 
    for row in image:
        for pixel in row:
            if pixel > 10: 
                MESH_pixels+=1

    if MESH_pixels > 10:
        return True
    else:
        return False

def accumulate(day, fields):
    date = day
    year = date[:4]
    os.system('makeIndex.pl {}/{}/{} code_index.xml'.format(OUT_HOME,year,date))
    #if os.path.exists('{}/{}/{}/{}'.format(OUT_HOME,year,date,fields[0])):
            
    for field in fields:
        os.system('w2accumulator -i {}/{}/{}/code_index.xml -g {} -o {}/{}/{} -C 1 -t 30 --verbose="severe"'.format(OUT_HOME,year, date, field,OUT_HOME,year,date))
        if field[8:] == 'Shear':
            os.system('w2accumulator -i {}/{}/{}/code_index.xml -g {} -o {}/{}/{} -C 3 -t 30 --verbose="severe"'.format(OUT_HOME,year, date, field, OUT_HOME,year, date))
    return

def cropconv(case_df, date):
    # this also needs reform    
    year = date[:4]
    myrorss_path = '{}/{}/'.format(TRAINING_HOME,year)
    #os.system('makeIndex.pl {}/{}/NSE code_index.xml'.format(myrorss_path,date))
    
    products = ['MergedReflectivityQC','MergedLLShear_Max_30min','MergedMLShear_Max_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min', 'MergedReflectivityQCComposite_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min']
 
    with open('trouble.txt',"a") as o:
        print(case_df,file=o)
    for idx, row in case_df.iterrows():
        if row['is_Storm'] == False:
            continue # skip
        print(row)
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
        time1 = (date_1-datetime.timedelta(minutes=2,seconds=30)).strftime('%Y%m%d-%H%M%S')
        time2 = (date_1+datetime.timedelta(minutes=2,seconds=20)).strftime('%Y%m%d-%H%M%S')
        
        # check if there is significant MESH 
        # open netcdf
        
        
        # crop input
        #########################
        os.system("makeIndex.pl {}/{}/{} code_index.xml {} {}".format(OUT_HOME, year, date, time1, time2)) # make index for uncropped
        for field in products:
            os.system('w2cropconv -i {}/{}/{}/code_index.xml -I {} -o /{}/{}/storm{:04d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,year,date, field, myrorss_path, date, idx, latNW, lonNW,latSE,lonSE))
        with open('trouble.txt','a') as o:
            print('input {} {}'.format(time1,time2),file=o)
         #########################

        # crop target -
        #########################
        time1 = (date_1+datetime.timedelta(minutes=27,seconds=30)).strftime('%Y%m%d-%H%M%S')
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=5,seconds=30)).strftime('%Y%m%d-%H%M%S')     
        with open('trouble.txt','a') as o:
            print('target {} {}'.format(time1,time2),file=o)   
        os.system("makeIndex.pl {}/{}/{} code_index.xml {} {}".format(OUT_HOME,year,date, time1, time2))
        os.system('w2cropconv -i {}/{}/{}/code_index.xml -I MESH_Max_30min -o {}/{}/storm{:04d}/target_MESH_Max_30min -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,year,date, myrorss_path, date, idx, latNW, lonNW,latSE,lonSE))
        # ########################

        # NSE 
        # crop for 30 min prior to 30 min ahead
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time1 = (date_1+datetime.timedelta(minutes=-30,seconds=00)).strftime('%Y%m%d-%H%M%S')
        time2 = (date_1+datetime.timedelta(minutes=30,seconds=00)).strftime('%Y%m%d-%H%M%S')
        os.system("makeIndex.pl {}/{}/{}/NSE code_index.xml {}0000 {}0000".format(OUT_HOME, year, date, time1[:11],time2[:11]))
        print("makeIndex.pl {}/{}/{}/NSE code_index.xml {} {}".format(OUT_HOME, year, date, time1, time2))
        for f in NSE_fields:
            os.system('w2cropconv -i {}/{}/{}/NSE/code_index.xml -I {} -o {}/{}/storm{:04d}/NSE -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME, year, date, f, myrorss_path,date, idx, latNW, lonNW,latSE,lonSE))
        # commenting this out for repair
        # os.system('w2cropconv -i {}/{}/multi{}/code_index.xml -I  MergedReflectivityQC -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:04d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(DATA_HOME,#date, multi_n, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
    return


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

   #year = sys.argv[1]
    #b0 = sys.argv[1]
    #b1 = sys.argv[2]
    # extract first 10 cases of the year (b1 to b2)
#    days = util.get_cases(year='2011') # get training days
    
    #finished_days = get_training_cases(year='2011')
    #ays = [x for x in days if x not in finished_days] # subtracts the finished days from days.
    #ays = np.random.choice(days, ,replace=False)

    # date loop : https://stackoverflow.com/questions/993358/creating-a-range-of-dates-in-python
    days = ['20110619','20110618','20110617','20110616','20110615','20110614','20110613','20110612','20110611','20110610','20110609']
    start = time.time() # start time
    days = ['20110409','20110619']
    for day in days:
        #year = day[:4]
        #extract(day)
        #print('extracting NSE \n \n \n')
        #extract_NSE(day)
       # localmax(day)
        print('finished localmaxing')
        storms = get_storm_info(day) # use mergedtable from localmax to store locations     
        check_MESH(storms,day)
       # storms = pd.read_csv('{}/{}/{}/csv/storms.csv'.format(TRAINING_HOME, day[:4], day))
       # accumulate(day,fields[1:])
        #cropconv(storms,day)
        #util.check_missing(day)
    end = time.time() # end time
    with open('info.txt','a') as info:
        print('Elapsed time for 1 day:',end-start)
    print(end-start)
if __name__ == "__main__":
    main()
