import pandas as pd
import numpy as np
import os
from . import utils as qutils

def load_metadata_df():
    src_path = qutils.__file__.split('utils.py')[0]
    metadata_path = os.path.join(src_path,'animal_metadata.pkl')
    return pd.read_pickle(metadata_path)
