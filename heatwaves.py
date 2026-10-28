#%%
from ufuncs import load_data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from constants import TEMPERATURE_COLUMNS
import numpy as np
import matplotlib.transforms as transforms


def zscore(s, window, thresh=3, return_all=False):
    roll = s.rolling(window=window, min_periods=1, center=True)
    avg = roll.mean()
    std = roll.std(ddof=0)
    z = s.sub(avg).div(std)   
    m = z.between(-thresh, thresh)
    
    if return_all:
        return z, avg, std, m
    return s.where(m, avg)

def qc_temp(df, col_list=[]):
    variable_index_level = df.columns.names.index('variable')

    for column in df.columns:
        variable = column[variable_index_level]
        if variable in col_list:
            sr = df[column].copy()
   
            sr_qc = zscore(sr, window='7D', thresh=3)
            df[column] = sr_qc
    return df



idf, tdf = load_data(reset_caches=False)
idf = qc_temp(idf, col_list=TEMPERATURE_COLUMNS)
tdf = qc_temp(tdf, col_list=TEMPERATURE_COLUMNS+['soil moisture count'])

#%%

def mc_calc_vwc(
    raw_moisture, 
    temp_values, 
    a, b, c, 
    ref_t=24.0, 
    acor_t=1.91, 
    wcor_t=0.64, 
    cal_cor_factor=0.0, 
    cal_cor_slope=0.0, 
    frozen2na=True
):
    """
    Python implementation of myClim's mc_calc_vwc for TOMST TMS loggers.
    Supports single values or NumPy arrays for vectorised processing.
    """
    
    # Ensure inputs are numpy arrays for vectorisation
    raw_moisture = np.asarray(raw_moisture, dtype=float)
    temp_values = np.asarray(temp_values, dtype=float)
    
    # Step 1: Initial VWC estimation
    vwc = a * (raw_moisture ** 2) + b * raw_moisture + c
    
    # Step 2: Calculate temperature correction dynamics
    dcor_t = wcor_t - acor_t
    
    # Apply temperature correction shift if temperature is available
    # myClim logic: raw_values + (ref_t - temp) * (acor_t + dcor_t * vwc)
    tcor = np.where(
        np.isnan(temp_values),
        raw_moisture,
        raw_moisture + (ref_t - temp_values) * (acor_t + dcor_t * vwc)
    )
    
    # Step 3 & 4: Recalculate VWC incorporating logger specific air/water calibrations
    corrected_signal = tcor + cal_cor_factor + cal_cor_slope * vwc
    vwc_cor = a * (corrected_signal ** 2) + b * corrected_signal + c
    
    # Clamp boundaries between 0.0 (0%) and 1.0 (100% volumetric moisture)
    result = np.clip(vwc_cor, 0.0, 1.0)
    
    # Handle frozen soil rule (TMS measurements become erratic below freezing)
    if frozen2na:
        result = np.where(temp_values < 0.0, np.nan, result)
        
    return result


for site in tdf.columns.get_level_values('site').unique():
    for device in [1, 2, 3]:

        raw_counts = tdf.xs((site, device, 'soil moisture count'), level=['site', 'device', 'variable'], axis=1)
        soil_temps = tdf.xs((site, device, 'T1'), level=['site', 'device', 'variable'], axis=1)

        vwc_results = mc_calc_vwc(
            raw_moisture=raw_counts,
            temp_values=soil_temps,
            a=-1.22e-7, 
            b=7.55e-4,
            c=-0.123,
            ref_t=24.0,
            acor_t=1.91,
            wcor_t=0.64
        )

        tdf.loc[:, (site, device, 'soil moisture')] = vwc_results * 100


#%%
HEATWAVES = [
    slice(pd.Timestamp('2026-05-22'), pd.Timestamp('2026-05-31')),
    slice(pd.Timestamp('2026-06-19'), pd.Timestamp('2026-06-28')),
    slice(pd.Timestamp('2026-07-04'), pd.Timestamp('2026-07-26')),
    # slice(pd.Timestamp('2026-07-28'), pd.Timestamp('2026-07-30')),
    # slice(pd.Timestamp('2026-08-03'), pd.Timestamp('2026-08-05')),
    slice(pd.Timestamp('2026-08-09'), pd.Timestamp('2026-08-15')),
]

heatwave_lengths = [len(tdf.loc[hw]) for hw in HEATWAVES]

fig  = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, len(HEATWAVES), figure=fig, 
                             width_ratios=heatwave_lengths, wspace=0.2)

for i, hw in enumerate(HEATWAVES):
    ax = fig.add_subplot(gs[i])

#%% 
def plot_timeseries_with_heatwaves(df, variable, heatwaves, aggregation='mean'):
    temp = df.xs(variable, level='variable', axis=1).T.groupby(level='site').mean().T
    temp = temp.loc['2026-05-01':'2026-09-01']

    if aggregation == 'mean':
        temp = temp.resample('D').mean().mean(1)
    elif aggregation == 'max':
        temp = temp.resample('D').max().mean(1)

    fig, ax = plt.subplots(figsize=(12, 6))
    temp.plot(ax=ax)
    for hw in heatwaves:
        ax.axvspan(hw.start, hw.stop, color='red', alpha=.2)

    ax.set_xlabel('')
    return fig, ax

plot_timeseries_with_heatwaves(tdf, 'T3', HEATWAVES, aggregation='max')
plot_timeseries_with_heatwaves(tdf, 'T1', HEATWAVES, aggregation='max')
plot_timeseries_with_heatwaves(tdf, 'soil moisture', HEATWAVES, aggregation='mean')
#%%

temp = tdf.xs('T3', level='variable', axis=1).T.groupby(level='site').mean().T
temp = temp.loc['2026-05-01':'2026-09-01']

sm = tdf.xs('soil moisture', level='variable', axis=1).T.groupby(level='site').mean().T
sm = sm.loc['2026-05-01':'2026-09-01']

#%%
fig, ax = plt.subplots(figsize=(11, 5.5))
ax2 = ax.twinx()
temp.resample('D').max().mean(axis=1).plot(ax=ax, color='red', linestyle='',
                                      marker='.')
temp.resample('D').max().mean(axis=1).rolling(window=7, center=True).mean().plot(ax=ax, color='red', 
                                                               linestyle='-', alpha=.5,
                                                               lw=4)
ax.set_ylabel('Maximum Temperature (°C)', color='red')

sm.resample('D').mean().mean(axis=1).plot(ax=ax2, color='blue', linestyle='', marker='.')
sm.resample('D').mean().mean(axis=1).rolling(window=7, center=True).mean().plot(ax=ax2, color='blue', 
                                                               linestyle='-', alpha=.5,
                                                               lw=4)
ax2.set_ylabel('Soil moisture (%)', color='blue')

transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
for hw in HEATWAVES:
        ax.axvspan(hw.start, hw.stop, color='darkorange', alpha=.2)
        ax.text(hw.start + (hw.stop - hw.start)/2, 0, 'heatwave', color='darkorange', fontsize=12,
                ha='center', va='bottom', alpha=1, transform=transform, fontweight='bold')

ax.set_xlabel('')

#%%

control_temp = tdf.xs('regen1', level='site', axis=1).xs('T3', level='variable', axis=1).mean(axis=1)
site_temps = tdf.xs('T3', level='variable', axis=1).T.groupby(level='site').mean().T.drop('regen1', axis=1)

cooling = -site_temps.sub(control_temp, axis=0)

cooling = cooling.resample('D').mean().loc['2026-05-01':'2026-09-01']

metadata = pd.read_csv('metadata.csv', index_col=0)

low = ['regen2', 'regen5', 'regen7']
high = ['regen3', 'regen4', 'regen6']

df = pd.DataFrame({
    'younger regen': cooling[low].mean(axis=1),
    'older regen': cooling[high].mean(axis=1),
    'great wood': cooling['great_wood'],
    'plantation': cooling['norway']
})

site_colors = {'younger regen': 'limegreen', 'older regen': 'forestgreen', 
               'great wood': 'brown', 'plantation': 'navy'}

fig, ax = plt.subplots(figsize=(11, 5.5))

df.plot(ax=ax, color=[site_colors.get(x, '#333333') for x in df.columns], 
        marker='.', linestyle='', legend=None)
df.rolling(window=7, center=True).mean().plot(ax=ax, color=[site_colors.get(x, '#333333') for x in df.columns],
        linestyle='-', alpha=.7, lw=4)
ax.set_xlabel('')
ax.set_ylabel('Cooling (°C)')
# 0 line
ax.axhline(0, color='black', linestyle='--', alpha=.5)

transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
for hw in HEATWAVES:
        ax.axvspan(hw.start, hw.stop, color='darkorange', alpha=.2)
        ax.text(hw.start + (hw.stop - hw.start)/2, 0, 'heatwave', color='darkorange', fontsize=12,
                ha='center', va='bottom', alpha=1, transform=transform, fontweight='bold')