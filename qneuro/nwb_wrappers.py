import h5py
import numpy as np
import pandas as pd
import re

class Alm1DataSet(object):
    """ A very simple interface for exracting electrophysiology data
    from an NWB file.
    """
    def __init__(self, file_name,trial_win=8):
        """ Initialize the NwbDataSet instance with a file name
        Parameters
        ----------
        file_name: string
           NWB file name
        """
        self.file_name = file_name
        self.trial_win = 8
        self.bitmasks = {
                'all_trials':[0,1,2,3,4,5,6,7],
                'R': [0,2,4],
                'L': [1,3,5],
                'hit': [0,1],
                'err': [2,3],
                'nolick': [4,5],
                'lick_early': [6],
                'stim_trials':[7],}

        with h5py.File(self.file_name, 'r') as f:
            self.session_start_time = f['/session_start_time'].value
            self.unit_ids = f['/processing/extracellular_units/UnitTimes/unit_list'].value.tolist()
            self.cell_types = f['/processing/extracellular_units/UnitTimes/cell_types'].value.tolist()
            self.epoch_ids = list(f['/epochs'].keys())

            start = []
            stop = []
            for e in self.epoch_ids:
                start.append(f['/epochs/%s/start_time'%e].value)
                stop.append(f['/epochs/%s/stop_time'%e].value)
            self.trial_start_timestamps = np.array(start)
            self.trial_stop_timestamps = np.array(stop)

            self.type_mat = f['/analysis/trial_type_mat']
            self.audiocue_timepoints = f['/stimulus/presentation/auditory_cue/timestamps'].value
            self.pole_in_timestamps = f['/stimulus/presentation/pole_in/timestamps'].value
            self.pole_out_timestamps = f['/stimulus/presentation/pole_out/timestamps'].value
            self.lick_data = f['/acquisition/timeseries/lick_trace/data'].value
            self.lick_timestamps = f['/acquisition/timeseries/lick_trace/timestamps'].value
            dips = np.argwhere(self.lick_data<0).flatten()
            self.lick_stop_times = np.zeros_like(self.trial_stop_timestamps)
            cut = np.argmax(self.trial_start_timestamps>self.lick_timestamps[dips][0])-1
            patch = len(self.trial_stop_timestamps)-len(dips)
            self.lick_stop_times[:cut] = np.array([np.nan]*patch)
            self.lick_stop_times[cut:] = np.squeeze(self.lick_timestamps[dips])

            filt = lambda crit: np.nonzero(self.type_mat[self.bitmasks[crit]].sum(axis=0))[0]
            self.trial_idxs = filt('all_trials')
            self.L_trials = filt('L')
            self.R_trials = filt('R')
            self.hit_trials = filt('hit')
            self.err_trials = filt('err')
            self.nolick_trials = filt('nolick')
            self.lick_early_trials = filt('lick_early')
            self.stim_trials = filt('stim_trials')


    def _fetch(self,key):
        out = None
        with h5py.File(self.file_name, 'r') as f:
            if key in f:
                out = f[key].value
        return out

    def get_start_stop_times(self):
        return (self.trial_start_timestamps,self.trial_stop_timestamps)

    def trial_outcomes(self):
        outcomes = np.zeros(len(self.epoch_ids),dtype=np.uint8)
        outcomes[self.hit_trials]=0
        outcomes[self.err_trials]=1
        outcomes[self.nolick_trials]=2
        str_o = ['Hit','Err',float('nan')]
        return [str_o[o] for o in outcomes]

    def get_trial_tags(self):
        return [self._fetch('/epochs/%s/tags'%e) for e in self.epoch_ids] 

    def get_spike_trial_ids(self):
        spk_trial_ids = []
        with h5py.File(self.file_name, 'r') as f:
            prefix='/processing/extracellular_units/UnitTimes/'
            unit_keys = [ prefix+u.decode()+'/trial_ids' for u in self.unit_ids ]
            spk_trial_ids = [f[uk].value for uk in unit_keys]
        return spk_trial_ids

    def print_keys(self,start=None):
        if start is not None:
            print(start)
            prefix='|- '
        else:
            start='/'
            prefix=''

        with h5py.File(self.file_name,'r') as f:  
            if isinstance(f[start],h5py._hl.dataset.Dataset):
                if f[start].size == 1:
                    print(start+'='+str(f[start].value))
                else:
                    print(type(f[start].value))
                    print(f[start].value)
            else:
                for k in f[start]:
                    print(prefix + str(k))

    def get_trial_table(self):
        trial_tbl = []
        dirs = np.array(['R']*len(self.epoch_ids))
        dirs[self.L_trials]='L'
        outcomes = self.trial_outcomes()

        num_trials = len(self.epoch_ids)

        p = 4
        list_round = lambda arr,places: [round(e,places) for e in arr]

        records = {'start_time': list_round(self.trial_start_timestamps, p),
                 'auditory_cue': list_round(self.audiocue_timepoints, p),
                 'pole_in': list_round(self.pole_in_timestamps, p),
                 'pole_out': list_round(self.pole_out_timestamps, p),
                 'lick_stop_time': list_round(self.lick_stop_times, p),
                 'stop_time': list_round(self.trial_stop_timestamps, p),
                 'outcome': outcomes,
                 'direction': dirs,
                }
        # trial_tbl.append(entry)

        return records

    def get_trial_timestamps(self,trl_idxs=None,cols=['start_time','stop_time','lick_stop_time']):
        if trl_idxs is None:
            trl_idxs = np.arange(len(self.epoch_ids))

        recs = {k:np.array(v)[trl_idxs] for k,v in self.get_trial_table().items()}
        tts = [{k:recs[k][i] for k in cols} for i in np.arange(len(trl_idxs))]
        return tts


    def get_spike_times(self,unit_ids=None,trl_idxs=None):
        if unit_ids is None:
            unit_ids = self.unit_ids

        if trl_idxs is None:
            trl_idxs = np.arange(len(self.epoch_ids))

        spk_tms = []
        with h5py.File(self.file_name, 'r') as f:
            prefix='/processing/extracellular_units/UnitTimes/'

            unit_keys = [ prefix+u.decode()+'/times' for u in unit_ids ]
            for uk in unit_keys:
                spks = [round(s,4) for s in f[uk].value]
                spk_tms.append(spks)

        return spk_tms

    def get_trial_time_ranges(self,trl_idxs=None,end_mark='lick_stop_time'):
        if trl_idxs is None:
            trl_idxs = np.arange(len(self.epoch_ids))

        tts = self.get_trial_timestamps(trl_idxs=trl_idxs,cols=['start_time',end_mark])
        ttr = [np.arange(d['start_time']*1000,d[end_mark]*1000,1)/1000 for d in tts]
        return ttr


    def get_spiketrains(self,unit_ids=None,trl_idxs=None):
        if unit_ids is None:
            unit_ids = self.unit_ids

        if trl_idxs is None:
            trl_idxs = np.arange(len(self.epoch_ids))

        spks = self.get_spike_times()
        spk_trl_ids = self.get_spike_trial_ids()
        names = [n.decode() for n in self.unit_ids]
        uids = [int(re.match(r'unit_(\d+)',n).groups()[0]) for n in names]

        spk_tbl = []
        for uid,spkt,spktids in zip(uids,spks,spk_trl_ids):
            spk_tbl.extend([{'time':s,'unit':uid,'trial':tid-1} for s,tid in zip(spkt,spktids) if tid-1 in trl_idxs])
        return spk_tbl

