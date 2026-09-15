#%%
import pandas as pd
from constants import STORAGE_DIR, TMS_DEPLOTMENT_DATE
from glob import glob
import numpy as np
import re

# load metadata df
mdf = pd.read_csv('./metadata.csv')

VARIABLE_LOOKUP = {'%RH':'RH',
                   'C':'T'}

class TMSLookup:
    def __init__(self, mdf):
        self.lookup = mdf[['site', 'tms_serial_2023', 'triangle_point_anti-clockwise']].dropna()
        self.lookup = self.lookup.set_index(['site', 'triangle_point_anti-clockwise']).squeeze()
        self.lookup = {v: k for k, v in self.lookup.to_dict().items()}

    def get_site(self, tms_serial):
        return self.lookup[tms_serial]

tms_lookup = TMSLookup(mdf)

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

def is_tms_csv(csv_path):
    fname = csv_path.split('/')[-1]
    pattern = r'^data_\d{8}_\d{4}_\d{2}_\d{2}_\d\.csv$'
    if re.fullmatch(pattern, fname):
        return True
    else:
        return False

#%%
csv_paths = glob('./monitoring_data/20*/*/*.csv')
csv_paths.extend(glob('./monitoring_data/20*/*.csv'))

def load_ibutton_df(csv_paths):
    srs = {}
    for csv_path in csv_paths:
        if is_ibutton_csv(csv_path):
            sr = read_ibutton_csv(csv_path)
            print(csv_path)
            if sr.name not in srs.keys():
                srs[sr.name] = sr
            else:
                srs[sr.name] = pd.concat([srs[sr.name], sr])

    sr = pd.concat(srs)
    sr_clean = sr[~sr.index.duplicated(keep='first')]
    df = sr_clean.unstack(level=[0, 1])

    return df

#%%




def read_tms_csv(csv_path):
    headers = ['datetime', 'time_zone', 'T1', 'T2', 'T3', 'soil moisture count',
            'shake','error_flag', 'soil moisture']
    df = pd.read_csv(csv_path, sep=';', header=None, index_col=0,
                    decimal=',')
    if df.empty:
        return df
    
    df.columns = headers

    for str_len, addition in zip((10, 13, 16), [' 00:00:00', '00:00', ':00']):
        df['datetime'] = df['datetime'].mask(df['datetime'].str.len() == str_len, 
                                            df['datetime'] + addition)

    datetime = df['datetime'][0]
    if '/' in datetime:
        date_sep = '/'
    elif '.' in datetime:
        date_sep = '.'
    
    if len(datetime.split(date_sep)[0]) == 4: #yearfirst
        format_str = f'%Y{date_sep}%m{date_sep}%d %H:%M:%S'
    elif len(datetime.split(date_sep)[0]) == 2: #dayfirst
        format_str = f'%d{date_sep}%m{date_sep}%Y %H:%M:%S'
    else:
        raise ValueError(f"Unexpected datetime format: {df['datetime'][0]}")

    df['datetime'] = pd.to_datetime(df['datetime'], format=format_str)


    df = df.set_index('datetime')

    df = df.loc[slice(TMS_DEPLOTMENT_DATE, None), :]

    fname = csv_path.split('/')[-1]
    tms_serial = fname.split('_')[1]
    site_name = tms_lookup.get_site(float(tms_serial))

    # append outer multiindex column level
    df.columns = pd.MultiIndex.from_tuples([(site_name[0], site_name[1], col) for col in df.columns], 
                                           names=['site', 'device', 'variable'])
    return df

def load_tms_df(csv_paths):
    dfs = []
    for csv_path in csv_paths:
        if is_tms_csv(csv_path):
            df = read_tms_csv(csv_path)
            print(csv_path)
            if not df.empty:
                dfs.append(df.stack(level=[0,1,2]))

    df_stacked = pd.concat(dfs)
    df_clean = df_stacked[~df_stacked.index.duplicated(keep='first')]
    df = df_clean.unstack([1,2,3]).astype(float)
    return df

