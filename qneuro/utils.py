import pandas as pd
import numpy as np
import peakutils
from .nwb_wrappers import Alm1DataSet

def get_good_trials(data_set, filter_lick_early=True, filter_na=True, filter_non_lick=True):
    tt = data_set.get_trial_table()
    trials = pd.DataFrame.from_records(tt)
    trial_inx = []

    if filter_lick_early:
        trial_tags = data_set.get_trial_tags()        
        non_early_trials = [i for i,t in enumerate(trial_tags) if b'LickEarly' not in t]
        trial_inx = non_early_trials

    if filter_na:
        non_na = trials.loc[trial_inx].dropna()
        trial_inx = np.array(non_na.index.values)

    if filter_non_lick:
        bad_trls = [] 
        lick_t = data_set._fetch('/acquisition/timeseries/lick_trace/timestamps')
        licks = data_set._fetch('/acquisition/timeseries/lick_trace/data')
        bad_trls = []

        for t_id in trial_inx:
            trl = trials.loc[t_id]
            # print(trl)
            licktrl_idxs = np.where((lick_t>trl.start_time)&(lick_t<trl.stop_time))[0]
            
            if len(licktrl_idxs) < 2:
                bad_trls.append(t_id)

        # Exclude these 'bad' trials (don't have lick data)
        trial_inx = set(trial_inx)-set(bad_trls)
        trial_inx = list(trial_inx)

    return trials.loc[trial_inx]

def get_delay_spikes(data_set, df, start_offset=0, stop_offset=0.5):
    spks = data_set.get_spike_times()    
    pole_outs = df.pole_out.values
    
    # Preallocate output array
    delay_spikes = np.zeros((len(pole_outs),len(spks)))

    for pnum,p in enumerate(pole_outs):
        delay_window = [p+start_offset, p+stop_offset]

        for snum in range(0,len(spks)):
            spk_rate = len(spks[snum][(spks[snum]>=delay_window[0])&(spks[snum]<=delay_window[1])])
            delay_spikes[pnum,snum] = spk_rate
    
    return delay_spikes

def get_movement_spikes(data_set, df, start_offset=-0.05, stop_offset=0.45):
    idxs = df.index.values
    trials = df
    first_licks = []
    lick_t = data_set._fetch('/acquisition/timeseries/lick_trace/timestamps')
    licks = data_set._fetch('/acquisition/timeseries/lick_trace/data')

    for idx in idxs:
        trl = trials.loc[idx]
        licktrl_idxs = np.where((lick_t>trl.start_time)&(lick_t<trl.stop_time))[0]
        lick = np.array([lick_t[licktrl_idxs].tolist(),licks[licktrl_idxs].tolist()])
        
        # Sometimes the lick data ends with some negative numbers, which messes up peakutils. Take those values out.
        relick = lick[:,(lick[1,:]>0)]

        ick = peakutils.peak.indexes((relick[1,:]),.3,min_dist=200)
        first_lick=relick[0,ick[0]]
        first_licks.append(first_lick)

    spks = data_set.get_spike_times()
    mvmt_spikes = np.zeros((len(first_licks),len(spks)))
    for lnum in range(0,len(first_licks)):
        mvmt_window = [first_licks[lnum]+start_offset,first_licks[lnum]+stop_offset]
    
        for snum in range(0,len(spks)):
            spk_rate = len(spks[snum][(spks[snum]>=mvmt_window[0])&(spks[snum]<=mvmt_window[1])])
            mvmt_spikes[lnum,snum] = spk_rate

    return mvmt_spikes