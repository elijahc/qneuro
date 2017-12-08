import pandas as pd
import numpy as np
from peakutils.peak import indexes as find_indexes
from peakutils.plot import plot as peak_plot

from .nwb_wrappers import Alm1DataSet

def tag_filter(tag_set, tag, cond='in'):
    if cond is 'in':
        idxs = [i for i,t in enumerate(tag_set) if tag in t]
    else:
        idxs = [i for i,t in enumerate(tag_set) if tag not in t]
    return idxs

def _bad_trls_idxs(data_set,
        filter_stim=True, filter_lick_early=True, filter_non_lick=True):

    trial_inx = data_set.trial_idxs.tolist()
    bad_trls = []
    trial_tags = data_set.get_trial_tags()

    if filter_stim:
        bad_trls.extend(tag_filter(trial_tags, b'StimTrials'))

    if filter_lick_early:
        bad_trls.extend( tag_filter(trial_tags, b'LickEarly') )

    if filter_non_lick:
        lick_t = data_set._fetch('/acquisition/timeseries/lick_trace/timestamps')
        licks = data_set._fetch('/acquisition/timeseries/lick_trace/data')
        non_lick_trls = []

        # Only search through good trials
        trial_inx = list(set(trial_inx)-set(bad_trls))

        start_t, stop_t = data_set.get_start_stop_times()
        for t_id in trial_inx:

            trl_start_time = start_t[t_id]
            trl_stop_time = stop_t[t_id]

            licktrl_idxs = np.where((lick_t>trl_start_time)&(lick_t<trl_stop_time))[0]

            if len(licktrl_idxs) < 2:
                non_lick_trls.append(t_id)

        bad_trls.extend(non_lick_trls)

    return bad_trls

def get_good_trials(data_set, return_idxs=False, filter_na=True,
        filter_stim=True, filter_lick_early=True, filter_non_lick=True):

    tt = data_set.get_trial_table()
    trials = pd.DataFrame.from_records(tt)
    trial_inx = trials.index.values.tolist()
    bad_trls = _bad_trls_idxs(data_set,filter_stim=filter_stim, filter_lick_early=filter_lick_early,filter_non_lick=filter_non_lick)

    # Exclude these 'bad' trials (don't have lick data)
    trial_inx = list( set(trial_inx) - set(bad_trls) )

    if filter_na:
        out = trials.loc[trial_inx].dropna()
    else:
        out = trials.loc[trial_inx]

    if return_idxs:
        return out.index.values
    else:
        return out

def get_delay_spikes(data_set, df, start_offset=0, stop_offset=0.5):
    spks = data_set.get_spike_times()
    pole_outs = df.pole_out.values

    # Preallocate output array
    delay_spikes = np.empty((len(pole_outs),len(spks)))

    for pnum,p in enumerate(pole_outs):
        delay_window = [p+start_offset, p+stop_offset]

        for snum,spkt in enumerate(spks):
            s_start = np.argmax(spkt>delay_window[0])
            s_end = np.argmax(spkt[s_start:]>delay_window[1])+s_start
            delay_spikes[pnum,snum] = len(spkt[s_start:s_end])
    
    return delay_spikes

def lick_window(dat,trl_idx):
    lick_t = dat.lick_timestamps
    licks = dat.lick_data
    trl_start = dat.trial_start_timestamps[trl_idx]
    trl_stop = dat.trial_stop_timestamps[trl_idx]
    find_nearest = lambda arr,val: (np.abs(arr-val)).argmin()
    ts = find_nearest(lick_t,trl_start)+15
    te = find_nearest(lick_t,trl_stop)-15
    # ts = np.argmax(lick_t>trl_start)+15
    # te = np.argmax(lick_t[ts:]>trl_stop)+ts
    return slice(ts,te)


def get_licks(dat,trl_idx):
    lick_win = lick_window(dat,trl_idx)

    licks_t = dat.lick_timestamps
    licks = dat.lick_data
    licks_norm = licks[lick_win]-licks[lick_win.start+15:lick_win.start+10000].mean()
    ick = find_indexes(licks_norm,.8,min_dist=400)
    l_idxs = ick+lick_win.start
    return l_idxs

def get_movement_spikes(data_set, df, start_offset=-0.05, stop_offset=0.45):
    idxs = df.index.values
    trials = df
    first_licks = []
    lick_t = data_set.lick_timestamps
    licks = data_set.lick_data

    for idx in idxs:
        l_idxs = get_licks(data_set,idx)

        if len(l_idxs)==0:
            # first_licks.append(np.nan)
            pass
        else:
            first_licks.append(lick_t[l_idxs[0]])


    spks = data_set.get_spike_times()
    mvmt_spikes = np.zeros((len(first_licks),len(data_set.unit_ids)))
    for i,l_t in enumerate(first_licks):
        mvmt_window = [l_t+start_offset,l_t+stop_offset]

        for j,spkt in enumerate(spks):
            isx = (np.abs(spkt-mvmt_window[0])).argmin()
            iex = (np.abs(spkt-mvmt_window[1])).argmin()
            spk_rate = len(spkt[isx:iex])
            mvmt_spikes[i,j] = spk_rate

    return mvmt_spikes

def gen_raster(df,tid,num_units,t_s=0):
    grouped = df[df['trial']==tid][['time','unit']].groupby('unit')

    df = grouped.aggregate(lambda x: [t-t_s for t in x.tolist()])
    raster = [[]]*25
    for u in df.index.values:
        raster[u-1]=df.loc[u].values[0]
    return raster
