# General use function for running loops
# Here it is configured to generate storm_dfs and filter those storms for a given year

import basic as b
import glob
import settings as s
import modify_df as mod
import pandas as pd
import sys

def main():
    # get cases in year
    year = sys.argv[1] 
    date_paths = glob.glob(f'{s.data_path}/{year}/20*')
    # first generate the feathers
    for date_path in date_paths:
        date_str = str(date_path.split('/')[-1])
        storm_path = glob.glob(f'{date_path}/csv/storms.feather')

        top_events_df = pd.read_feather(f'csv/top_events_{year}.feather')

        if storm_path == []:
            storm_before = b.get_storm_info(date_path)
        else:
            storms_before = pd.read_feather(f'{storm_path}')
        storms_after = mod.switch_off_storms(storms_before, top_events_df, date_str)

if __name__ == '__main__':
    main()

