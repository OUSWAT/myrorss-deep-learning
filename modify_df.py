#
# Author: Michael Montalbano
# Purpose: Alter datasets to form 'new' datasets (subsets with different statistics)
# title: alter.py
import settings as s
import numpy as np
import pandas as pd
import util, random
import sys, stats, glob, datetime
import matplotlib.pyplot as plt
from collections import Counter
from math import sin, cos, sqrt, atan2, radians
from sys import exit
'''
Sample method for deleting elements of a dataset:
bin_indices, ratios, shave_ratios = my_hist(outs_shave, outs_2008)
print(ratios,shave_ratios) # pick group with very different proportion


outs = delete_some(outs_2008,bindices, index=1,percent=10)
# Repeat until satisfactory 
'''
DATA_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'

def relate_information_about_year(year='2011'):
    # given a year and bounding boxes df, switch off cases outside the bounding box
  #  df['Storm_Number'] = np.arange(0,len(
    
    storms_csv_paths = glob.glob(f'{DATA_HOME}/{year}/**/csv/storms.csv') # returns the full path to the directory
    for storm_path in storms_csv_path:
       df = pd.read_csv(f'{storm_path}')
       print(df['is_Storm'].value_counts())

def merge_df(year):
    dfl = glob.glob(f'csv/*loc*{year}*') # Location dataframes names
    dfd = glob.glob(f'csv/*det*{year}*') # Details dataframe names

    dfl = pd.read_csv(dfl[0])
    dfd = pd.read_csv(dfd[0])

    df_details = dfd[dfd['EVENT_TYPE']=='Hail']
    df_merged = pd.merge(df_details, dfl, on='EVENT_ID')
    # make day column of YYYYMMDD
    yearmonth_arr = np.asarray(df_merged['BEGIN_YEARMONTH'])
    day_arr = np.asarray(df_merged['BEGIN_DAY'])
    yearmonthday_arr = np.asarray([f'{str(y)}{str(d).zfill(2)}' for y,d in zip(yearmonth_arr,day_arr)])
    df_merged['date'] = yearmonthday_arr # column added
    
    df_hail = pd.read_csv(f'csv/{year}_hail_events.csv') # 
    df_hail = df_hail[df_hail['hail_events'] > 5] # only keep events with more than 5 events
    df_locations = pd.DataFrame(columns=['date','lats','lons','begin_times', 'end_times', 'magnitudes']) 
    #df_bb = pd.DataFrame(columns=['day','bounds'])
    for idx, row in df_hail.iterrows():
        day = str(int(row['day'])) # int to get number then convert to str for comparison
        df_day = df_merged[df_merged['date'] == day] # get only the rows for that da
        lats = np.asarray(df_day['LATITUDE'])
        lons = np.asarray(df_day['LONGITUDE'])
        begin_times = np.asarray(df_day['BEGIN_TIME'])
        end_times = np.asarray(df_day['END_TIME'])
        magnitudes = np.asarray(df_day['MAGNITUDE'])
        row = {'date':day, 'lats': lats, 'lons':lons, 'begin_times':begin_times, 'end_times':end_times, 'magnitudes':magnitudes}
        df_locations.loc[len(df_locations.index)] = row
    return df_locations

def switch_off_storms(storms_df, top_events_df, date):
    # given a storm df, switch off it outside some set of regulations
    # Can be used during extraction after comp ref is localmaxed
    dfb = pd.read_csv('csv/bounding_boxes.csv')
    month = int(date[4:6])
    row = dfb[dfb['Month'] == month]
    latNW = float(row['latNW'])
    lonNW = float(row['lonNW'])
    latSE = float(row['latSE'])
    lonSE = float(row['lonSE'])
    df_top = top_events_df[top_events_df['date'] == date]
    dfr = pd.read_csv('csv/nexrad-stations.csv')
    radarInfo = [list([lat,lon]) for lat, lon in zip(dfr['LAT'], dfr['LON'])]
    for idx, storm in storms_df.iterrows(): # iterate through rows of storms df
        lat = storm['Latitude']
        lon = storm['Longitude']
        hour = storm['timedate'].split('-')[1][:-2]
        try:
            events_lons = df_top['lons']
        except:
            print(f'There is an error with {date}')
            print(f'This is the df {df_top}')
        events_lats = df_top['lats']
        events_begin_times = df_top['begin_times']
        events_end_times = df_top['end_times']
        events_magnitudes = df_top['magnitudes']
        # check if storm is within 300 km of any of the top lons/lats
        for i in range(len(events_lons)):
            try:
                event_lat = events_lats.iloc[0][i]
                event_lon = events_lons.iloc[0][i]
                event_begin_time = str(events_begin_times.iloc[0][i]).zfill(4)
            except IndexError or KeyError:
                print(f'IndexError with event on {date}')
                continue
            distance_to_event = calculateDist(lat, lon, event_lat, event_lon)
            if distance_to_event < 300:
                event_begin_time_dt = datetime.datetime.strptime(event_begin_time, '%H%M')
                hour_dt = datetime.datetime.strptime(hour, '%H%M')    
                if hour_dt - event_begin_time_dt < datetime.timedelta(hours=3):         
                    storms_df.loc[idx, 'is_Storm'] = True
                else:
                    storms_df.loc[idx, 'is_Storm'] = False
            else:
                storms_df.loc[idx, 'is_Storm'] = False
        if lonNW <= lon <= lonSE and latSE <= lat <= latNW:
            storms_df.loc[idx, 'is_Storm'] = True
        else:
            storms_df.loc[idx, 'is_Storm'] = False
            continue
        nearest_dist = 1000 # dummy var
        for radar_loc in radarInfo: # radar loc is a list(lat,lon)
            latR, lonR = radar_loc
            dist = calculateDist(lat, lon, latR, lonR)
            if dist < nearest_dist:
                nearest_dist = dist
            if nearest_dist > 100:
                storms_df.loc[idx, 'is_Storm'] = False
            else:
                storms_df.loc[idx, 'is_Storm'] = True
    try:
        storms_df.reset_index().to_feather(f'/condo/swatwork/mcmontalbano/MYRORSS/data/{date[:4]}/{date}/csv/storms.feather')
    except:
        storms_df.to_feather(f'{s.data_path}/{date[:4]}/{date}/csv/storms.feather')
    return storms_df

def keep_top(df_year_events, num=10):
    df_top = pd.DataFrame(columns=['date','lats','lons','begin_times', 'end_times', 'magnitudes'])
    for idx, row in df_year_events.iterrows():
        # get the indices of the top 10 magnitudes
        try:
            indices = np.arange(len(row['magnitudes']))[row['magnitudes'].argsort()[-num:]][::-1]
        except IndexError:
            indices = np.arange(len(row['magnitudes']))[row['magnitudes'].argsort()[::-1]]
        # get the lats and lons of the top 10 magnitudes
        lats = row['lats'][indices]
        lons = row['lons'][indices]
        # get the begin and end times of the top 10 magnitudes
        begin_times = row['begin_times'][indices]
        end_times = row['end_times'][indices]
        row = {'date':row['date'], 'lats': lats, 'lons':lons, 'begin_times':begin_times, 'end_times':end_times, 'magnitudes':row['magnitudes'][indices]}
        df_top.loc[len(df_top.index)] = row
    return df_top

def calculateDist(lat1, lon1, lat2, lon2):
    R = 6373.0
    rlat1 = radians(lat1)
    rlon1 = radians(lon1)
    rlat2 = radians(lat2)
    rlon2 = radians(lon2)
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = sin(dlat / 2)**2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def find_bounding_box(points):
    lats, lons = zip(*points)
    return [(max(lats), min(lons)), (min(lats), max(lons))]

def cluster(year):
    # feather
    dfls = glob.glob(f'csv/*loc*{year}*') # Location dataframes names
    dfds = glob.glob(f'csv/*det*{year}*') # Details dataframe names

    dfl = pd.read_csv(dfls[0]) # grab 2011
    dfd = pd.read_csv(dfds[0]) # grab 2011

    df_details = dfd[dfd['EVENT_TYPE']=='Hail']
    df_merged = pd.merge(df_details, dfl, on='EVENT_ID')
    # make day column of YYYYMMDD
    yearmonth_arr = np.asarray(df_merged['BEGIN_YEARMONTH'])
    day_arr = np.asarray(df_merged['BEGIN_DAY'])
    yearmonthday_arr = np.asarray([f'{str(y)}{d}' for y,d in zip(yearmonth_arr,day_arr)])
    df_merged['date'] = yearmonthday_arr # column added

    df_hail = pd.read_csv(f'csv/{year}_hail_events.csv')
    df_hail = df_hail[df_hail['hail_events'] > 5]
    df_locations = pd.DataFrame(columns=['date','lats','lons','begin_times', 'end_times'])
    df_bb = pd.DataFrame(columns=['date','bounds'])
    for idx, row in df_hail.iterrows():
        date = str(int(row['day'])) # int to get number then convert to str for comparison
        df_day = df_merged[df_merged['date'] == date] # comparison
        lats = np.asarray(df_day['LATITUDE'])
        lons = np.asarray(df_day['LONGITUDE'])
        begin_times = np.asarray(df_day['BEGIN_TIME'])
        end_times = np.asarray(df_day['END_TIME'])
        row = {'day':day, 'lats': lats, 'lons':lons, 'begin_times':begin_times, 'end_times':end_times}
        df_locations.loc[len(df_locations.index)] = row

        # get clusters
        max_dist = 1 # max distance to belong in same cluster
        df_day = df_locations[df_locations['day'] == day]
        clusters = [[(lats[0], lons[0])]]
        for lat,lon in zip(lats,lons):
           for cluster in clusters:
               for point in cluster:
                   if abs(point[0] - lat) < max_dist and abs(point[1] - lon) < max_dist:
                       cluster.append((lat,lon))
                   else:
                       clusters.append([(lat,lon)])
        bounding_boxes = []
        for cluster in clusters:
            points = find_bounding_box(cluster)
            bounding_boxes.append(points)
        row = {'day':day, 'bounds': bounding_boxes}
        df_bb.loc[len(df_bb.index)] = row
    df_bb.to_csv(f'csv/{year}_bounding_boxes.csv')

def cluster_2(df_day,day='20110409'):
    max_dist = 1 # max distance to belong in same cluster
    df_day = df_locations[df_locations['day'] == day]
    clusters = [[(lats[0], lons[0])]]
    for lat,lon in zip(lats,lons):
       for cluster in clusters:
           for point in cluster:
               if abs(point[0] - lat) < max_dist and abs(point[1] - lon) < max_dist:
                   cluster.append((lat,lon))
               else:
                   clusters.append([(lat,lon)])
    bounding_boxes = []
    for cluster in clusters:
        points = find_bounding_box(cluster)
        bounding_boxes.append(points)
    row = {'day':day, 'bounds': bounding_boxes}
    df_bb.loc[len(df_bb.index)] = row
    df_bb.to_csv(f'csv/{year}_bounding_boxes.csv')

 
def main():
    for year in np.arange(1999,2011,1): 
        creating = True 
        try:
            if creating==True:
                df_year_events = merge_df(year)
                df_year_events.reset_index().to_feather(f'csv/events_{year}.feather')
                df_year_events = pd.read_feather(f'csv/events_{year}.feather')
                df_year_top_events = keep_top(df_year_events)
                df_year_top_events.to_feather(f'csv/top_events_{year}.feather')
            storms = glob.glob(f'/condo/swatwork/mcmontalbano/MYRORSS/data/{year}/**/csv/storms.csv')
            for storm in storms:
                date = storm.split('/')[-3]
                df_year_top_events = pd.read_feather(f'csv/top_events_{year}.feather')
                try:
                    storms_df = pd.read_csv(storm)
                except:
                    print(storm)
                    continue
                storms_after = switch_off_storms(storms_df, df_year_top_events, date)
                storms_after.reset_index().to_feather(f'/condo/swatwork/mcmontalbano/MYRORSS/data/{date[:4]}/{date}/csv/storms.feather')
        except:
            continue       
                  

if __name__ == "__main__":
    main()
