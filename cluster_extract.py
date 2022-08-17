import os, sys, random, datetime, time, calendar
import gzip, tempfile, shutil, argparse 
from glob import glob
from os import environ
import pandas as pd
import subprocess as sp
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import settings as s 
from load_data import npa, read_netcdf
import matplotlib.pyplot as plt
lscratch = os.environ.get('LSCRATCH')

def check_clusters(date, timestamp):
    groupings = []
    try:
        # make datetime out of timestamp
        timedate = datetime.datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
        timestamps = get_times(timedate)
        cluster_files = []
        for timestamp in timestamps:
            file = glob(f'{s.cluster_path}/{date[:4]}/KMeans/scale_0/{date}/{timestamp[:12]}*.netcdf')
            if len(file) == 1:
                cluster_files.append(file[0])
        kmeans = Dataset(cluster_files[0])['KMeans'][:,:]
        kmeans[kmeans<0] = 0
        kmeans = kmeans.astype(int)
        corners_list = get_corners(kmeans)
        # open MESH
        mesh_files = []
        for timestamp in timestamps:
            mesh_file = glob(f'{lscratch}/MESHMhtpost_Max_30min/**/{timestamp[:12]}*.netcdf')
            if len(mesh_file) == 1:
                mesh_files.append(mesh_file[0])
        mesh = Dataset(mesh_files[0])['MESHMhtpost_Max_30min'][:,:]
        groupings = []
        for idx, corners in enumerate(corners_list):
            swath = mesh[corners[0]:corners[2],corners[3]:corners[1]]
            plt.imshow(swath)
            plt.title(f'{corners[0]}:{corners[2]},{corners[3]}:{corners[1]};{timestamps[3]}')
            plt.savefig(f'{s.scripts}/plt_{idx}_{timestamps[3]}.png')
            grouping = test_conditions(swath) # this returns a 1-hot array signaling the class of swath
            x_center, y_center = int((corners[0]+corners[2])/2), int((corners[3]+corners[1])/2)
            groupings.append([x_center, y_center, grouping])
            print(grouping)
        return groupings
    except Exception as E:
        print(E)
        return -1

def get_corners(kmeans):
    n_graphs = int(kmeans.max())
    graph_corners = []
    for i in range(n_graphs):
        x_k, y_k = np.nonzero(np.where(kmeans==i, 1, 0))
        # get mean of x_k and y_k
        x_mean = int(np.mean(x_k))
        y_mean = int(np.mean(y_k))
        # get the NW and SE corners by adding or subtracting 30 pixels
        x_nw = x_mean - 30
        y_nw = y_mean + 30
        x_se = x_mean + 30
        y_se = y_mean - 30
        graph_corners.append([x_nw, y_nw, x_se, y_se])
    return graph_corners

def extract_storms(date, out_path, df):
    #accumulate_MESH(date,lscratch) # this should still be in 
    accumfields = ['MergedReflectivityQCComposite', 'MergedLLShear', 'MergedMLShear', 'Reflectivity_OC', 'Reflectivity_-10C', 'Reflectivity_-20C']
    timestamp_set = sorted(set(df['timestamp'].tolist()))
    extractfields = accumfields+['MergedReflectivityQC']
    extract(date, s.inmost_path, lscratch, timestamp_set, iterfields = extractfields) # extracts nonNSE fields to lscratch
    extract_NSE(date, lscratch)                                        # extracts NSE fields to lscratch
    date_td = datetime.datetime.strptime(date, "%Y%m%d")                 
    next_day = (date_td + datetime.timedelta(days=1)).strftime('%Y%m%d')
    extract_NSE(next_day, lscratch) # extract next day too
    accumulate(iterfields=accumfields,in_path=lscratch, out_path=lscratch) # accumulates nonNSE fields to lscratch
    accumfields = [f'{x}_Max_30min' for x in accumfields] + ['MergedLLShear_Min_30min', 'MergedMLShear_Min_30min']
    for idx, row in df.iterrows():
        try:
            ID = int(row['index']) # set the ID to a non-taken (i.e. above 5k) unique ID
            lat, lon, t1 = row['lat'], row['lon'], row['timedate']
            stormdir_ID = ID + 700000 # set the ID to a non-taken (i.e. above 5k) unique ID
            storm_path = f'{out_path}/storm{stormdir_ID}'
            crop_path = f'{lscratch}/storm{stormdir_ID}'
            cropconv_field_at_time(lat, lon, t1, 'MESHMhtpost_Max_30min', -3, 3, lscratch, crop_path)
            folders_in_crop = glob(f'{crop_path}/*')
            mesh_file = glob(f'{crop_path}/MESH*/**/*netcdf')
            mesh_file = mesh_file[0]
            pieces = mesh_file.split('/')
            new_mesh_file = '/'.join(pieces[-3:])
            if not os.path.exists(f'{storm_path}/MESHMhtpost_Max_30min/00.25'):
                os.makedirs(f'{storm_path}/MESHMhtpost_Max_30min/00.25')
            shutil.move(mesh_file, f'{storm_path}/{new_mesh_file}')
            for field in accumfields:
                try:
                    cropconv_field_at_time(lat, lon, t1, field, -3, 3,  lscratch, storm_path)
                except Exception as e:
                    print(E)
                    pass
            for field in s.NSE_fields:
                cropconv_field_at_time(lat, lon, t1, field, -40,3, lscratch, storm_path)
            cropconv_field_at_time(lat, lon, t1, 'MergedReflectivityQC', -3, 3 ,lscratch, storm_path)
        except Exception as E:
            print(E)
            pass
    return None

def cropconv_field_at_time(lat, lon, timestamp, field, t1, t2, in_path, storm_path):
    '''
    Given a row from a dataframe with columns: Longitude, Latitude, cropconv MESH_swath to be within 0.15
    '''
    delta = 0.15
    lonNW, latNW, lonSE, latSE = lon - delta, lat + delta, lon + delta, lat - delta
    date_1 = datetime.datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
    time1 = (date_1 + datetime.timedelta(minutes=t1,seconds=00)).strftime('%Y%m%d-%H%M%S') # add t1 minutes to t0
    time2 = (date_1 + datetime.timedelta(minutes=t2,seconds=30)).strftime('%Y%m%d-%H%M%S') # add t2 minutes to t1
    cmd1 = f'makeIndex.pl {in_path} code_index.xml {time1} {time2}'
    cmd2 = f'w2cropconv -i {in_path}/code_index.xml -I {field} -o {storm_path} -t "{latNW} {lonNW}" -b "{latSE} {lonSE}" -s "0.005 0.005" -R -n --verbose="severe"'
    p = sp.Popen(cmd1, shell=True) # make index for target (+25 to +33 min)
    p.wait()
    p = sp.Popen(cmd2, shell=True) # crop target to storm dir in out_path
    p.wait()
    return None


def accumulate(iterfields, in_path, out_path):
    '''
    Accumulates (uncropped) fields
    Fields must be in the in_path in order to be accumulated
    Accumulated fields are stored
    '''
    cmd = f'makeIndex.pl {in_path} code_index.xml'
    p = sp.Popen(cmd, shell=True)
    p.wait()
    # dont accumulate reflectivity
    # shallow copy
    fields = iterfields.copy()
    try:
        fields.remove('MergedReflectivityQC') # remove reflectivity if present (not a field to accumulate)
    except as Exception as E:
        print(E)
        pass
    for field in fields:
        if field == 'MESH':
            cmd =  f'w2accumulator -i {in_path}/code_index.xml -g {field} -o {out_path} -C 1 -t 30 -m {field} --verbose=4 -Q ablob:7:10:10,mht:3:2:1500:5:5'
        p = sp.Popen(cmd, shell=True)
        p.wait()
        findMin = True  # I don't care about the min shear, for now
        if field[8:] == 'Shear' and findMin:  # so skip this loop
            cmd = f'w2accumulator -i {in_path}/code_index.xml -g {field} -o {out_path} -C 3 -t 30 --verbose="severe"'
            p = sp.Popen(cmd, shell=True)
            p.wait()
    return None

def get_wildcards(timestamp_set, field):
    # get min td in td_set, which is already sorted
    wildcards = []
    min_timestamp = timestamp_set[0]
    # use datetime to subtract 1 hour
    min_timedate = datetime.datetime.strptime(min_timestamp, "%Y%m%d-%H%M%S").strftime('%Y%m%d-%H%M%S')
    wildcards.append(f'*{field}*{min_timedate[:11]}*')
    for timestamp in timestamp_set:
        wildcards.append(f'*{field}*{timestamp[:11]}*')
    wildcards_str = ""
    for wildcard in wildcards:
        wildcards_str += f" '{wildcard}' "
    return wildcards_str

def get_times(timedate): 
    # return -2 min, -1 min, 0, +1 minutes, +2 minutes
    add = [-2, -1, 0, 1, 2]
    times = ()
    for i in add:
        times.append(timedate + datetime.timedelta(minutes=i)).strftime("%Y%m%d-%H%M%S")
    return tuple(times)

def test_conditions(swath):
    # check that the swath is greater than 20 mm
    # check that there are N pixels above threshold
    cond1 = points_above(swath, 25) > 10
    cond2 = points_above(swath, 40) > 4
    cond3 = swath.max() > 50
    if cond1 and cond2 and cond3: # sigsevere swath
        return 3
    cond1 = cond1
    cond2 = points_above(swath, 40) > 20
    cond3 = swath.max() <= 50
    if cond1 and cond2 and cond3: # severe swath
        return 2
    cond1 = points_above(swath, 20) > 50
    cond2 = points_above(swath, 20) < 1500
    cond3 = swath.max() < 40
    if cond1 and cond2 and cond3: # subsevere swath
        return 1
    cond1 = swath.max() < 15
    cond2 = points_above(swath, 10) < 50
    if cond1 and cond2: # null swath
        return 0
    return -1

def points_above(image,thres):
    # count pixels
    pixels = np.where(image.squeeze()>thres)[0]
    points_above=len(pixels)
    return points_above

def accumulate_MESH(date, out_path, timedate):
    year = date[:4]
    tar_path = f'{s.target_path}/{year}/{date}'
    cmd1 = f'tar -xf {tar_path}.tar -C {out_path} --exclude="MergerInputRadarsTable"' # untar MESHMht into lscratch
    p = sp.Popen(cmd1, shell=True)
    p.wait()
    field_path = f'{out_path}/{date}/MESHMhtpost'
    if timedate != None:
        timedate = datetime.datetime.strptime(timedate, "%Y%m%d-%H%M%S")
        t1 = (timedate - datetime.timedelta(minutes=35)).strftime('%Y%m%d-%H%M%S')
        timedate = datetime.datetime.strptime(timedate, "%Y%m%d-%H%M%S")
        t2 = (timedate + datetime.timedelta(minutes=35)).strftime('%Y%m%d-%H%M%S')
        cmd2 = f'makeIndex.pl {lscratch}/{date} code_index.xml {t1} {t2}'
    else:
        cmd2 = f'makeIndex.pl {lscratch}/{date} code_index.xml'
    p = sp.Popen(cmd2, shell=True)
    p.wait()
    cmd3 = f'w2accumulator -i {lscratch}/{date}/code_index.xml -g MESHMhtpost -o {lscratch} -C 1 -t 30 --verbose="severe"'  # accumulate MESHMht
    p = sp.Popen(cmd3, shell=True)
    p.wait()
    return None

def extract(date, in_path, out_path, timestamp_set, iterfields):
    '''
    extract to lscratch; (then it is localmaxed and accumulated and cropped into train_home by other functions) 
    '''
    year = date[:4]
    cmd3 = f'mkdir -p {out_path}/csv'
    p = sp.Popen(cmd3, shell=True)
    p.wait()
    final_path = f'{s.data_path}/{year}/{date}'
    for field in iterfields:
        try:
            wildcards = get_wildcards(timestamp_set, field)
            # 1999 to 2003 saved in Azshear dir in stead of azimuthal_shear_only, so try both
            if field == 'MergedLLShear' or field == 'MergedMLShear':
                if int(year) >= 2002:
                    cmd = f"tar -xf {in_path}/{year}/azimuthal_shear_only/{date}.tar -C {out_path} --wildcards {wildcards} --exclude='MergerInputRadarsTable'"
                    p = sp.Popen(cmd, shell=True)
                    p.wait()
                else:
                    cmd = f"tar -xf {in_path}/{year}/Azshear/{date}.tar -C {out_path} --wildcards {wildcards}"
                    p = sp.Popen(cmd, shell=True)
                    p.wait()
            else:
                try:
                    cmd = f"tar -xf {in_path}/{year}/{date}.tar -C {out_path} --wildcards {wildcards} --exclude='MergerInputRadarsTable'"
                    p = sp.Popen(cmd, shell=True)
                    p.wait()
                except Exception as e:
                    print(e)
                    pass
        # pointing to the field in the lscratch directory
            field_path = f'{out_path}/{field}'
            #  list the direotories within the field
            all_files = glob(f'{field_path}/**/*.gz')
            print(f'{field} with \n {all_files}')
            for full_file_path in all_files:
                try:
                    pieces = full_file_path.split('/')
                    nc_file = pieces[-1][:-3]
                    # convert from .gz to .netcdf
                    gz = gzip.open(full_file_path)
                    data = Dataset('dummy', mode='r', memory=gz.read())  # open using netCDF4
                    # use xarray backend to convert correctly (idk)
                    dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
                    dataset.to_netcdf(f'{out_path}/{field}/{nc_file}')  # write
                except Exception as E:
                    print(E)
                    pass
        except Exception as E:
            print(E)
            pass
    print(f'{glob(f"{lscratch}/*")}')
    print(f'{glob(f"{lscratch}/**/*")}')
    print(f'{glob(f"{lscratch}/**/**/*")}')
    print(f'{glob(f"{lscratch}/**/**/**/*")}')
    return None

def extract_NSE(date, out_path):
    # grab netcdf.gz files from NSE_home
    for field in s.NSE_fields: # loop through list of str NSEfields
        try:
            cmd = f'mkdir -p {out_path}/{field}'
            p = sp.Popen(cmd, shell=True)
            p.wait()
            nse_files = glob(f'{s.nse_path}/{date[:4]}/{date}/NSE/{field}/**/*.gz') # grab gz files from storage
        # unzip into netcdf and place in lscratch
            for nse_file in nse_files: # loop through files
                try:
                    nse_file_without_the_gz = nse_file[:-3] # remove the .gz
                    pieces = nse_file_without_the_gz.split('/')
                    field = pieces[7]
                    nc_file = pieces[-1]
                    os.system(f'mkdir -p {out_path}/{field}/nseanalysis')
                    gz = gzip.open(nse_file)
                    data = netCDF4.Dataset('dummy', mode='r', memory=gz.read())
                    dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
                    dataset.to_netcdf(f'{out_path}/{field}/nseanalysis/{nc_file}')
                    gz.close()
                except Exception as E:
                    print(E)
                    pass
        except Exception as E:
            print(E)
            pass
    return None

def convert_to_latlon(x, y):
    Lon = -130.0050048828125
    Lat = 55.005001068115234
    latlon_grid_spacing = 0.009999999776482582
    lat = Lat - (latlon_grid_spacing*x)
    lon = Lon + (latlon_grid_spacing*y)
    return lat, lon

def create_parser():
    parser = argparse.ArgumentParser(description='Load')
    parser.add_argument(
        '-date',
        type=int,
        default=2011,
        help='Are you using the tuner, or just messing around?')
    parser.add_argument(
        '-delta',
        type=float,
        default=0.5,
        help='Degree by which cases were filtered')
    parser.add_argument(
        '-proc_targs',
        type=int,
        default=0,
        help='Degree by which cases were filtered')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    date = str(args.date)
    proc_targs = bool(args.proc_targs)
    year = date[:4]
    print(date)
    out_path = f'/condo/swatwork/mcmontalbano/MYRORSS/data/{year}/{date}'
    if proc_targs:
        return None
    top_events_df = pd.read_feather(f'feather/top_events_{year}_1.0_degrees_0613.feather') # load top events df with lats/lons/mags/times for filtering
    df_hail = pd.read_csv(f'csv/{year}_hail_events.csv') # load hail events with n_hail for day for filtering
    try:
        df_hail = df_hail[df_hail['day'] == int(date)]
        hail_events = int(df_hail['hail_events'])
    except:
        sys.exit()
    if hail_events < 2:
        sys.exit()
    df = pd.read_feather(f'{out_path}/csv/storms.feather')
    tds = df['timedate'].tolist()
    td_set = set()
    for td in tds:
        td_set.add(td[:12])
    df_extract = pd.DataFrame(columns=['timedate', 'lat', 'lon',])
    df_archive = pd.DataFrame(columns=['timedate', 'lat', 'lon','class'])
    for timedate in td_set:
        t_0 = f'{timedate}000'
        accumulate_MESH(date, out_path, t_0)
        groupings = check_clusters(date, out_path, t_0)
        if groupings != -1: # check for exception, likely in file
            for grouping in groupings:
                lat, lon = convert_to_latlon(grouping[0], grouping[1])
                if grouping[2] == 3:
                    row = {'timedate': timedate, 'lat': lat, 'lon': lon}
                    df_extract = df_extract.append(row, ignore_index=True)
                else:
                    row = {'timedate': timedate, 'lat': lat, 'lon': lon, 'class': grouping[2]}
                    df_archive = df_archive.append(row, ignore_index=True)
    df_extract.to_feather(f'{out_path}/feather/extract.feather')
    df_archive.to_feather(f'{out_path}/feather/archive.feather')
    out_path = f'/condo/swatwork/mcmontalbano/MYRORSS/data/dev/{year}/{date}'
    extract_storms(date, out_path, df_extract)
    return None
