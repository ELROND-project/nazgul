"""
Simulate an image with SIS and try to model it - leave gamma OD and omega LOS free

-> assume optimistic HST observations
"""
import sys
import argparse
import numpy as np

from python_tools.tools import mkdir

from nazgul.MountDoom.lens_system import LensSystem
from nazgul.Modelling.pathfinder import model_res_base

# LOS simulated as dependent on gal name (not z)
from nazgul.MountDoom.fuel_tank import lensfuel_los_gal_optimistic_band

from nazgul.MountDoom.generate_gal_lens import GalLens
from nazgul.MountDoom.cracks_of_doom import kwargs_source_default

reload_SIS = True
mask_center = True
gauss_tE_prior = True
overwrite_results = True
check_if_workin_on_it = False

res_dir_base      = model_res_base/"testSIS_freeOG_optimisticObs/"
mkdir(res_dir_base)

description_model = "Try to model a single SIS w LOS, and model the lens with EPL +LOS minimal (with free gamma OD and omega LOS), and an elliptical sersic source - now with fixed lens pos - and Optimistic Observation (def. lnstr. HST band)"


from nazgul.Modelling.model_testSIS import n_smpl,tE_in,z_lens
from nazgul.Modelling.model_testSIS_freegOD import create_SIS_gallens,kw_model


if __name__=="__main__":
    from nazgul.Modelling.lib_models import setup_lens,get_res_dir,model_lens,n_it_std,n_burn_std,n_run_std,n_part_std

    parser = argparse.ArgumentParser(prog=sys.argv[0],description=description_model)
    parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Type of run:
        0 = standard, PSO_it = {n_it_std} PSO_prt = {n_part_std} MCMCb = {n_burn_std} MCMCr = {n_run_std}  
        1 = test run  PSO_it = 3      PSO_prt = 3      MCMCb = 1     MCMCr = 2 
        2 = test run (longer)  PSO_it = 100      PSO_prt = 20      MCMCb = 100     MCMCr = 200 
        (PSO_it: PSO iterations, PSO_prt: PSO particles, MCMCr: MCMC run steps, MCMCb: MCMC burn in steps)\n""")
    
    args     = parser.parse_args()
    run_type = args.run_type
    simsuite = "ANL_TEST"
    sim      = "SIS"
    subsim   = None
    
    if run_type==0:
        n_it   = n_it_std
        n_part = n_part_std
        n_burn = n_burn_std
        n_run  = n_run_std
        n_smpl_SIS = n_smpl
    elif run_type == 1:
        n_it   = 3
        n_part = 3
        n_burn = 1
        n_run  = 2
        n_smpl_SIS = int(1e3)
        check_if_workin_on_it= False
    elif run_type==2:
        n_it   = 100
        n_part = 20
        n_burn = 100
        n_run  = 200
        check_if_workin_on_it = False
    else:
        raise ValueError(f"run_type must be 0,1 or 2, not {run_type}")
    
    res_dir = get_res_dir(res_dir_base,simsuite,sim,
                          subsim=subsim,run_type=run_type)
    
    lens_sys = create_SIS_gallens(n_smpl=n_smpl_SIS)
    lens_sys.unpack()
    lens2model = LensSystem.from_GalLens(lens_sys,lensFuel=lensfuel_los_gal_optimistic_band)
    # the SIS has no radial CC, so the sampling of the source pos fails
    # we can trick it by defining it a priori
    # can't be 0,0 exactly as it would trigger an error :
    kwargs_source_pos = {"center_x":1e-9,"center_y":1e-9}
    lens2model.kwargs_source = kwargs_source_default|kwargs_source_pos
    lens2model.setup(reload=False)
    
    lens = setup_lens(lens=lens2model,
                      res_dir=res_dir,
                      check_if_workin_on_it=check_if_workin_on_it,
                      overwrite=overwrite_results)           
    
    model_lens(lens,kw_model,
               n_it   = n_it,
               n_part = n_part,
               n_burn = n_burn,
               n_run  = n_run,
               mask_center=mask_center,              
               gauss_tE_prior=gauss_tE_prior)
