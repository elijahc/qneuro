import pandas as pd
import numpy as np
import os
from . import utils as qutils

def load_unit_metadata():
    pass

def load_animal_metadata():
    src_path = qutils.__file__.split('utils.py')[0]
    metadata_path = os.path.join(src_path,'animal_metadata.pkl')
    return pd.read_pickle(metadata_path)

def load_prep_vs_exec():
    src_path = qutils.__file__.split('utils.py')[0]
    # exp_path = os.path.join(src_path,'

