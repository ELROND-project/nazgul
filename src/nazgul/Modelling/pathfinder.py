from python_tools.tools import mkdir
from nazgul.Translator import get_simsuite_code,get_simsuite_from_code,std_sim,std_simsuite,std_subsim

from nazgul.pathfinder import get_sim_dir,results_dir

from pathlib import Path
model_res_base      = results_dir/"models/"

def get_model_res_dir_from_gallens(gallens, res_dir):
    gal = gallens.Gal
    name_suffix = gallens.name.replace("Sub","LS")
    model_res_dir = _model_res_dir(gallens.Gal, 
                                   gallens.name.replace("Sub", "LS"), 
                                   res_dir)
    return model_res_dir

def _get_sim_dir_name(simsuite_code,sim,subsim=None):
    sim_dir = f"{simsuite_code}_{sim}"
    if subsim:
        sim_dir +=f"_{str(subsim)}"
    return sim_dir
    
def _get_sim_dir(gal):
    try:
        simsuite_code =  str(gal.simsuite_code)
    except AttributeError:
        # MONKEY PATCH
        simsuite = gal.simsuite
        simsuite_code = get_simsuite_code(simsuite)
    sim    = str(gal.sim)
    subsim = getattr(gal,"subsim",False)
    sim_dir = _get_sim_dir_name(simsuite_code=simsuite_code,
                            sim=sim,subsim=subsim)
    return sim_dir


def get_res_dir(res_dir_base,simsuite,sim,subsim=None,run_type=0):
    res_dir = Path(res_dir_base)
    simsuite_code = get_simsuite_code(simsuite)
    sim_dir = _get_sim_dir_name(simsuite_code=simsuite_code,
                            sim=sim,subsim=subsim)
    res_dir = res_dir/sim_dir
    if run_type!=0:
        res_dir = res_dir/"test"
    mkdir(res_dir)
    return res_dir
    
def _model_res_dir(gal, name_suffix, res_dir):
    sim_dir = _get_sim_dir(gal)
    res_dir = Path(res_dir)
    if str(sim_dir) not in str(res_dir):
        res_dir = res_dir / sim_dir
    model_res_dir = res_dir / f"snap_{gal.snap}_{name_suffix}"
    # create the dir
    mkdir(model_res_dir)
    return model_res_dir

def get_model_res_dir(lens_or_gal, res_dir):
    if hasattr(lens_or_gal, "gallens"):      # it's a Lens
        return _model_res_dir(lens_or_gal.gallens.Gal, lens_or_gal.name, res_dir)
    elif hasattr(lens_or_gal, "Gal"):        # it's a GalLens
        gallens = lens_or_gal
        return _model_res_dir(gallens.Gal, gallens.name.replace("Sub", "LS"), res_dir)
    else:
        raise TypeError(f"Expected Lens or GalLens, got {type(lens_or_gal)}")

    
def get_link_lens_path(lens):
    return lens.model_res_dir/"link_gallens.pkl"

#from importlib import import_module
#def get_res_dir_base(model):
#    return import_module(f".model_{model}","nazgul.Modelling").res_dir_base

    