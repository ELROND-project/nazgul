"""
Simulate an image with SIS and try to model it  (without LOS)
"""
import sys
import numpy as np
from python_tools.tools import mkdir
from python_tools.get_res import load_whatever

from nazgul.Modelling.pathfinder import model_res_base
from nazgul.mount_doom.lens_system import LensSystem
# LOS simulated as dependent on gal name (not z)
from nazgul.mount_doom.fuel_tank import std_lensfuel as lensfuel

mask_center = True
gauss_tE_prior = True
overwrite_results = True
check_if_workin_on_it = True

lens_model_list   = ['SIS']
source_model_list = ["SERSIC_ELLIPSE"]
res_dir_base      = model_res_base/"testSIS_noLOS/"
mkdir(res_dir_base)

description_model = "Try to model a single SIS, and model the lens with SIS, and an elliptical sersic source (without LOS)"


from nazgul.Modelling.model_testSIS import _get_kwargs_params,create_SIS_gallens
    
def get_kwargs_params(lens):
    return _get_kwargs_params(lens,los_included=False)
    

kw_model = {"lens_model_list":lens_model_list,
            "source_model_list":source_model_list,
            "get_kwargs_params":get_kwargs_params}


import argparse

if __name__=="__main__":
    from nazgul.Modelling.lib_models import setup_lens,get_res_dir,model_lens,n_it_std,n_burn_std,n_run_std,n_part_std

    parser = argparse.ArgumentParser(prog=sys.argv[0],description=description_model)
    parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Run type, changes where the results are stored. Should change the lenght of the modelling as well, but not implemented yet""")

    #parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Type of run:
    #    0 = standard, PSO_it = {n_it_std} PSO_prt = {n_part_std} MCMCb = {n_burn_std} MCMCr = {n_run_std}  
    #    1 = test run  PSO_it = 3      PSO_prt = 3      MCMCb = 1     MCMCr = 2 
    #    2 = test run (longer)  PSO_it = 100      PSO_prt = 20      MCMCb = 100     MCMCr = 200 
    #    (PSO_it: PSO iterations, PSO_prt: PSO particles, MCMCr: MCMC run steps, MCMCb: MCMC burn in steps)\n""")
    
    args     = parser.parse_args()
    run_type = args.run_type
    simsuite = "ANL_TEST"
    sim      = "SIS"
    subsim   = None
    res_dir = get_res_dir(res_dir_base,simsuite,sim,
                          subsim=subsim,run_type=run_type)

    lens_sys = load_whatever("/cosma/home/do019/dc-quei1/nazgul/src/nazgul/RingBearer/ANL_TEST/SIS/snap_0/TEST_GAL_SIS_tE1.5_nS1.00e+06/Sub/Sub_Lens_TEST_GAL_SIS_tE1.5_nS1.00e+06_Prj0_Qvz7NA.pkl")
    lens_sys.unpack()
    lens2model = LensSystem.from_GalLens(lens_sys,lensFuel=lensfuel)
    # the SIS has no radial CC, so the sampling of the source pos fails
    # we can trick it by defining it a priori
    # can't be 0,0 exactly as it would trigger an error :
    kwargs_source_pos = {"center_x":1e-9,"center_y":1e-9}
    from nazgul.mount_doom.cracks_of_doom import kwargs_source_default
    lens2model.kwargs_source = kwargs_source_default|kwargs_source_pos
    lens2model.setup(reload=False)
    lens = setup_lens(lens=lens2model,
                      res_dir=res_dir,
                      check_if_workin_on_it=check_if_workin_on_it,
                      overwrite=overwrite_results)           
    model_lens(lens,kw_model,
               n_it   = n_it_std,
               n_part = n_part_std,
               n_burn = n_burn_std,
               n_run  = n_run_std,
               mask_center=mask_center,              
               gauss_tE_prior=gauss_tE_prior)
