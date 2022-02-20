import os
import pymc3 as pm
import numpy as np

from typing import Tuple

def calculate_BIC(model:pm.Model, n)->Tuple[np.float64, np.float64]:
    MAP = pm.find_MAP(model = model)
    logP = -model.logp(MAP)
    k = 0.5 * len(list(MAP.keys()))
    return k*np.log(n) + 2*logP, logP

def create_dirs(data_path:str):
    if not os.path.exists('output_info'):
        os.mkdir('output_info')
        
    dirs_to_build = ['summaries', 'tracePlots', 'ppcPlots', 'selection_metrics', 'prior_plot']
    
    try:
        data_path = data_path.replace('.csv', '')
        data_path = os.path.join('output_info', data_path)
        os.mkdir(data_path)
    except FileExistsError:
        pass
    
    try:
        for dir_ in dirs_to_build:
            os.mkdir(os.path.join(data_path, dir_))
    except FileExistsError:
        pass