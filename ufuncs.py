#%%
import pandas as pd
from constants import STORAGE_DIR
from glob import glob
import numpy as np

# load metadata df
mdf = pd.read_csv('./metadata.csv')

VARIABLE_LOOKUP = {'%RH':'RH',
                   'C':'T'}

#%%

def read_ibutton_metadata(csv_path):
    # read metadata at top of csv
    df = pd.read_csv(csv_path, nrows=18, sep=r'@|@@',
                    engine='python', header=None)
    df = df[0].str.split(": ",expand = True)
    col = df[0].str.split("  ",expand = True)[1]
    df.loc[df[1].isnull(), 1] = col.loc[df[1].isnull()]
    df.columns = ['key', 'value']
    sr = df.set_index('key').squeeze()
    return sr

def find_ibutton_site(mdf, meta, year):
    serial = meta.loc['1-Wire/iButton Registration Number'][-8:-2]
    site = mdf.loc[mdf[f'ibutton_serial_{year}'] == serial, 'site'].item()
    return site

def read_ibutton_csv(csv_path):

    df = pd.read_csv(csv_path, sep=r'@|,', engine='python',
                    header=18, parse_dates=False)

    if not df.index[0] == 0:
        df = df.reset_index()
        time_strs = df['index'] + df['Date/Time']
        date_cols = ['index', 'Date/Time']
    else:
        time_strs =  df['Date/Time']
        date_cols = ['Date/Time']
    
    meta = read_ibutton_metadata(csv_path)
    

    variable = VARIABLE_LOOKUP[df['Unit'].unique().item()]
    df = df.drop('Unit', axis=1)

    
    df.index = pd.to_datetime(time_strs, format='mixed')
    df.index.name = 'datetime'
    df = df.drop(date_cols, axis=1).squeeze()

    year = df.index.year.unique().item()
    site = find_ibutton_site(mdf, meta, year)
    df.name = (site, variable)
    return df

def is_ibutton_csv(csv_path):
    try:
        first_line = pd.read_csv(csv_path, nrows=1, header=None)
    except:
        return False
    if len(first_line.columns) == 1:
        if (first_line.loc[0,0][:6] == '1-Wire'):
            return True
        else:
            return False 

#%%
csv_paths = glob('./monitoring_data/20*/*/*.csv')
csv_paths.extend(glob('./monitoring_data/20*/*.csv'))
for csv_path in csv_paths:
    if is_ibutton_csv(csv_path):
        df = read_ibutton_csv(csv_path)
        print(csv_path)

