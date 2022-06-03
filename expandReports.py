mport sys, time, argparse
import pandas as pd
import numpy as np
from calendar import timegm
from os import path
from sys import argv
from math import sin, cos, sqrt, atan2, radians
from tqdm import tqdm
# nexrad csv : https://github.com/Eli-S-Bridge/NEXRAD/blob/master/nexrad_site_list_with_utm.csv
#############################################################################
#
# Reads in StormEvent files from NCEI from a specified folder and year range and writes
# a comma separated .csv file to specified output directory of tornado, severe hail and severe thunderstorm wind.
# You can set a time threshold to not include severe reports within a certain time of a tornado.
#
# Example command: python expandReports.py /Users/theasandmael/2011_Severe_Storm_Reports_Dual-pol_Only.csv /Users/theasandmael/Documents/nexrad-stations.csv /Users/theasandmael
#
# Author:  Thea Sandmael - 20190514
# Modified by: Michael Montalbano - 20220420
# 
#############################################################################


def main(inFile, radarFilePath, outputDir, outputFile = ''):
   if outputFile == '':
      outputFile = inFile[inFile.rfind('/')+1:-4] + '_Expanded'

   print('This is the output file name: ' + outputFile)

   if 'Dual-pol' in inFile: dualpol = 1
   else: dualpol = 0

   dfo = pd.DataFrame()
   
   # Read in CSV and store as a dataframe
   try:
      dfi = pd.read_csv(inFile,encoding='ISO-8859-1')
   except OSError as err:
      print('***ERROR READING RADAR CSV FILE***: {}\n{}\nEXITING......'.format(inFile,err))
      exit()
'''   
   # Read in CSV and store as a dataframe
   try:
      dfr = pd.read_csv(radarFilePath,encoding='ISO-8859-1')
   except OSError as err:
      print("***ERROR READING RADAR CSV FILE***: {}\n{}\nEXITING......".format(radarFilePath,err))
      exit()
'''
#   print("Reading radar info:")
       
#   radarInfo = [list([icao,lat,lon,date,region]) for icao,lat,lon,date,region in zip(tqdm(dfr['ICAO']), dfr['LAT'], dfr['LON'], dfr['DUALPOL_DATE'], dfr['REGION'])]

 #  print('Size of data before expansion:')
 #  print(len(dfi))

   timeDiff = [abs(timegm(time.strptime(btime, '%Y%m%d-%H%M')) - timegm(time.strptime(etime, '%Y%m%d-%H%M'))) for btime, etime in zip(tqdm(dfi["begin_timestamp"]), dfi["end_timestamp"])]

   print('Size of data after time filter:')
   dfi.reindex(np.where(np.array(timeDiff) < 7200)[0])

   print(len(dfi))
   
   times = [np.arange(timegm(time.strptime(btime, '%Y%m%d-%H%M')), timegm(time.strptime(etime, '%Y%m%d-%H%M')) + 60, 60) for btime, etime in zip(tqdm(dfi["begin_timestamp"]), dfi["end_timestamp"])]

   #print(times[0])
   #print("Calculating lats:")
   #lats  = [np.linspace(dfi["begin_lat"][i], dfi["end_lat"][i], len(times[i])) for i in tqdm(range(len(times)))]
   #
   #print("Calculating lats:")
   #lats  = [np.linspace(blat, elat, len(timeran)) for blat, elat, timeran in zip(tqdm(dfi["begin_lat"]), dfi["end_lat"], times)]
   #
   #
   #print("Calculating lons:")
   #lons  = [np.linspace(blon, elon, len(timeran)) for blon, elon, timeran in zip(tqdm(dfi["begin_lon"]), dfi["end_lon"], times)]
   #
   #print("Calculating lminr:")
   #lminr = [np.min([np.arange(0, len(timeran)), np.arange(0, len(timeran))[::-1]], axis=0) for timeran in tqdm(times)]

   ran = []
   rad = []
   reg = []
   lats = []
   lons = []
   lminr = []
   
   for i in tqdm(range(len(times))):
      lat = np.linspace(dfi["begin_lat"][i], dfi["end_lat"][i], len(times[i]))
      lon = np.linspace(dfi["begin_lon"][i], dfi["end_lon"][i], len(times[i]))
      lats.extend(lat)
      lons.extend(lon)
      lminr.extend(np.min([np.arange(0, len(times[i])), np.arange(0, len(times[i]))[::-1]], axis=0))
      
      if dfi["begin_radar"][i] == dfi["end_radar"][i]:
         ran.extend(np.linspace(dfi["begin_range"][i], dfi["end_range"][i], len(times[i])))
         rad.extend([dfi["begin_radar"][i]]*len(times[i]))
         reg.extend([dfi["begin_region"][i]]*len(times[i]))
      else:
         info1 = radarInfo[np.where(np.array(radarInfo) == dfi["begin_radar"][i])[0][0]]
         info2 = radarInfo[np.where(np.array(radarInfo) == dfi["end_radar"][i])[0][0]]
         dist1 = [calculateDist(lat1, lon1, info1[1], info1[2]) for lat1, lon1 in zip(lat, lon)]
         dist2 = [calculateDist(lat2, lon2, info2[1], info2[2]) for lat2, lon2 in zip(lat, lon)]
         
         minin = np.argmin([np.array(dist1),np.array(dist2)], axis=0) 
         
         if dfi["begin_region"][i] == dfi["end_region"][i]:
            reg.extend([dfi["begin_region"][i]]*len(times[i]))
         else:
            reg.extend([[dfi["begin_region"][i], dfi["end_region"][i]][mini] for mini in minin])
                
         ran.extend(np.min([np.array(dist1),np.array(dist2)], axis=0))
         rad.extend([[dfi["begin_radar"][i], dfi["end_radar"][i]][mini] for mini in minin])
      
   timesflat = [item for sublist in times for item in sublist]

   timeStamp = [time.strftime('%Y%m%d-%H%M', time.gmtime(reptime)) for reptime in timesflat]

   dfo = pd.DataFrame()
   
   dfo['timestamp']           = timeStamp
   dfo['year']                = [stamp[0:4]  for stamp in timeStamp]
   dfo['month']               = [stamp[4:6]  for stamp in timeStamp]
   dfo['day']                 = [stamp[6:8]  for stamp in timeStamp]
   dfo['time']                = [stamp[9:13] for stamp in timeStamp]
   dfo['lat']                 = lats#[item for sublist in lats  for item in sublist]
   dfo['lon']                 = lons#[item for sublist in lons  for item in sublist]
   dfo['tornado']             = [dfi["tornado"][i] for i in range(len(dfi["tornado"])) for _ in range(len(times[i]))]
   dfo['hail']                = [dfi["hail"][i]    for i in range(len(dfi["hail"]))    for _ in range(len(times[i]))]
   dfo['wind']                = [dfi["wind"][i]    for i in range(len(dfi["wind"]))    for _ in range(len(times[i]))]
   dfo['severe_type']         = [dfi["severe_type"][i] for i in range(len(dfi["severe_type"])) for _ in range(len(times[i]))]
   dfo['type']                = [dfi["type"][i]    for i in range(len(dfi["type"]))    for _ in range(len(times[i]))]
   dfo['magnitude']           = [dfi["magnitude"][i] for i in range(len(dfi["magnitude"])) for _ in range(len(times[i]))]
   dfo['tornado_length']      = [dfi["tornado_length"][i] for i in range(len(dfi["tornado_length"])) for _ in range(len(times[i]))]
   dfo['tornado_width']       = [dfi["tornado_width"][i] for i in range(len(dfi["tornado_width"])) for _ in range(len(times[i]))]
   dfo['source']              = [dfi["source"][i] for i in range(len(dfi["source"])) for _ in range(len(times[i]))]
   dfo['minutes_from_report'] = lminr#[item for sublist in lminr for item in sublist]
   dfo["season"]              = ['MAM' if month in range(3,6) else 'JJA' if month in range(6,9) else 'SON' if month in range(6,9) else 'DJF' for month in tqdm(dfo['month'])]
   dfo['region']              = reg
   dfo['radar']               = rad
   dfo['range']               = ran

   print("Final size of data:")
   print(len(dfo))            
   
   dfo = dfo.sort_values('timestamp')
   
   # Write out dataframe to csv
   dfo.to_csv(path_or_buf = str('{}/{}.csv'.format(outputDir,outputFile)), index=False)

   return 0

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

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Reads in filtered storm reports and expands them in time.\
   \n Usage: {} <path_to_storm_reports_csv> <path_to_radar_csv> <output_directory>')
   parser.add_argument('-o', metavar='outputFile', type=str,  nargs='?', default='',    help='Output file name (default: [inFile]_Expanded.csv)')    
   parser.add_argument('inFile',        type=str, nargs=1,   help='Storm report csv')
   parser.add_argument('radarFilePath', type=str, nargs=1,   help='List of NEXRAD radars csv')
   parser.add_argument('outputDir',     type=str, nargs=1,   help='Output directory where the output file will be saved')
   args = parser.parse_args(argv[1:])
   
   # Check if required input paths are valid
   if not path.exists(args.inFile[0]):
      print("***inFile DOES NOT EXISTS*** Exiting.....")
      exit()

   if not path.exists(args.outputDir[0]):
      print("***OUTPUT DIRECTORY DOES NOT EXISTS*** Exiting.....")
      exit()
      
   main(args.inFile[0], args.radarFilePath[0], args.outputDir[0], outputFile = args.o)
