import pandas as pd
import numpy as np
import datetime, calendar
import glob



def main():
    dfs = glob.glob('csv/StormEvents_details-ftp*')
    for df_unit in dfs:
        pieces = df_unit.split('_')
        year = pieces[3][1:] # grab year  from file form
        df_hail = pd.DataFrame(columns=['day','hail_events','storms'])
        try:
            df = pd.read_csv(df_unit)
        except:
            print('exception')
            continue
        year = int(str(df['BEGIN_YEARMONTH'][0])[:4])
        months = np.arange(1,13,1)
        for m in months:
            print(f'{year} and {m}')
            num_days = calendar.monthrange(year, m)[1]
            days = [datetime.date(year, m, day) for day in range(1, num_days+1)]
            for day in days:
                year_str = str(day.year)
                yearmonth = int(str(day.year) + "%02d" % (day.month,))
                n_day = day.day
                yearmonthday = str(yearmonth) + "%02d" % n_day
                df = df[df['BEGIN_YEARMONTH']==yearmonth]         
                df = df[df['BEGIN_DAY'] == day]
                try:
                    print('trying to get dummies for df')
                    y = pd.get_dummies(df.EVENT_TYPE) 
                    num_hail_events = y['Hail'].sum()
                    print(f'there are {num_hail_events}')
                except KeyError:
                    num_hail_events = 0
                    print('there was a KeyError')
                # also check the number of valid storms for each day
                try:
                    storms_df = pd.read_csv(f'/condo/swatwork/mcmontalbano/MYRORSS/data/{year_str}/{yearmonthday}/csv/storms.csv')
                    num_storms = storms_df['is_Storm'].value_counts()[0] # grab the number of True storms ([1] is the number of false
                except FileNotFoundError:
                    print('file not found')
                    num_storms = np.nan                    
                df_hail.loc[len(df_hail.index)] = {'day': yearmonthday, 'hail_events': num_hail_events, 'storms': num_storms}
        df_hail.to_csv(f'csv/{year_str}_hail_events.csv')
            
if __name__ == '__main__':
    main()
