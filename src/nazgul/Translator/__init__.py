import os
import warnings
import importlib

import nazgul.configurations as conf

sims = {}
subsims = {}
path_trnsl = conf.nazgul_path/"Translator"

for file in path_trnsl.iterdir():
    file_name = str(file.name)
    if file.is_dir() and not file_name.startswith('.') and not file_name.startswith('__'):
        # those should be only directories representing availalbe simulations
        simsuite = file_name
        simsuite_module = importlib.import_module(f'.{simsuite}',"nazgul.Translator")
        sim = simsuite_module.sim
        # Not all simsuites have subsims
        subsim = getattr(simsuite_module,"subsim",{})
        sims[simsuite] = sim
        subsims[simsuite] = subsim
        
        
# Define the standard simulation (for convenience)
std_simsuite  = conf.std_simsuite
std_sim       = sims[std_simsuite][0]
try:
    std_subsim = subsims[std_simsuite][std_sim][0]
except KeyError:
    std_subsim = None
try:
    # use the following simulation only as test case
    test_sim      = sims[std_simsuite][1]
except IndexError:
    test_sim = None
    pass
    #warnings.warn("Test simulation not implemented")

# used for tutorial -linked ONLY snap 20 of test_sim
tutorial_sim  = sims["EAGLE"][2]

# Default values - kept here fore backwards compatability but now chosen in configurations
min_z        = conf.min_z
max_z        = conf.max_z
min_mass     = conf.min_mass

def get_simsuite_code(simsuite):
    simsuite_module = importlib.import_module(f'.{simsuite}',"nazgul.Translator")
    simsuite_code = simsuite_module.simsuite_short_name
    return simsuite_code

def get_simsuite_from_code(simsuite_code):
    # opposite of get_simsuite_code
    simsuites = list(sims.keys())
    for ss in simsuites:
        if simsuite_code==get_simsuite_code(ss):
            return ss
    raise RuntimeError(f"Could not find sim. suite corresponding to code {simsuite_code}")
