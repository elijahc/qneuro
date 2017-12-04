import h5py
import numpy as np
import pandas as pd

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
                'R': [0,2,4],
                'L': [1,3,5],
                'hit': [0,1],
                'err': [2,3],
                'nolick': [4,5],
                'lick_early': [6],
                'stim_trials':[7]}

        with h5py.File(self.file_name, 'r') as f:
            self.session_start_time = f['/session_start_time'].value
            self.unit_ids = f['/processing/extracellular_units/UnitTimes/unit_list'].value.tolist()
            self.cell_types = f['/processing/extracellular_units/UnitTimes/cell_types'].value.tolist()
            self.epoch_ids = list(f['/epochs'].keys())
            self.type_mat = f['/analysis/trial_type_mat']
            self.audiocue_timepoints = f['/stimulus/presentation/auditory_cue/timestamps'].value
            self.pole_in_timestamps = f['/stimulus/presentation/pole_in/timestamps'].value
            self.pole_out_timestamps = f['/stimulus/presentation/pole_out/timestamps'].value

            filt = lambda crit: np.nonzero(self.type_mat[self.bitmasks[crit]].sum(axis=0))
            self.L_trials = filt('L')
            self.R_trials = filt('R')
            self.hit_trials = filt('hit')
            self.err_trials = filt('err')
            self.nolick_trials = filt('nolick')
            self.lick_early_trials = filt('lick_early')

    def _fetch(self,key):
        out = None
        with h5py.File(self.file_name, 'r') as f:
            if key in f:
                out = f[key].value
        return out

    def trial_outcomes(self):
        outcomes = np.zeros(len(self.epoch_ids),dtype=np.uint8)
        outcomes[self.hit_trials]=0
        outcomes[self.err_trials]=1
        outcomes[self.nolick_trials]=2
        return outcomes

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
        str_o = ['Hit','Err',float('nan')]
        dirs = np.array(['R']*len(self.epoch_ids))
        dirs[self.L_trials]='L'
        outcomes = [str_o[o] for o in self.trial_outcomes()]

        num_trials = len(outcomes)
        with h5py.File(self.file_name, 'r') as f:
            for i,e,o,d in zip(np.arange(num_trials),self.epoch_ids,outcomes,dirs):
                entry = {'start_time': f['/epochs/%s/start_time'%e].value.round(4),
                         'auditory_cue': self.audiocue_timepoints[i].round(4),
                         'pole_in': self.pole_in_timestamps[i].round(4),
                         'pole_out': self.pole_out_timestamps[i].round(4),
                         'stop_time': f['/epochs/%s/stop_time'%e].value.round(4),
                         'outcome': o,
                         'direction': d
                        }
                trial_tbl.append(entry)

        return trial_tbl

    def get_spike_times(self,unit_ids=None):
        if unit_ids is None:
            unit_ids = self.unit_ids
        spk_tms = []
        with h5py.File(self.file_name, 'r') as f:
            prefix='/processing/extracellular_units/UnitTimes/'

            unit_keys = [ prefix+u.decode()+'/times' for u in unit_ids ]
            for uk in unit_keys:
                spks = f[uk].value.round(4)
                spk_tms.append(spks)
        return spk_tms