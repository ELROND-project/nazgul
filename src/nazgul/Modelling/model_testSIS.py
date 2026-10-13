"""
Simulate an image with SIS and try to model it 
"""

import sys
import argparse
import numpy as np
from python_tools.tools import mkdir
from python_tools.get_res import load_whatever

from nazgul.Modelling.pathfinder import model_res_base
from nazgul.mount_doom.lens_system import LensSystem
# LOS simulated as dependent on gal name (not z)
from nazgul.mount_doom.fuel_tank import lensfuel_los_gal as lensfuel

from nazgul.Translator.translator import PartGal
from nazgul.mount_doom.generate_gal_lens import GalLens
from nazgul.particle_lenses import default_kwlens_part_AS
from nazgul.mount_doom.cracks_of_doom import kwargs_source_default

mask_center = True
gauss_tE_prior = True
overwrite_results = True
check_if_workin_on_it = True

lens_model_list   = ['SIS','LOS_MINIMAL']
source_model_list = ["SERSIC_ELLIPSE"]
res_dir_base      = model_res_base/"testSIS/"
mkdir(res_dir_base)

description_model = "Try to model a single SIS w LOS, and model the lens with EPL +LOS minimal, and an elliptical sersic source"

def _get_kwargs_params(lens,los_included=True):
    # Params:
    # initial guess of non-linear parameters, we chose different starting parameters than the truth #
    tE = lens.gallens.thetaE.value

    # LOS added separately    
    kwargs_lens_init = [{'theta_E':     tE *np.random.normal(1,.1,1)[0], 
                         'center_x':    0.0, 
                         'center_y':    0.0}]
    kwargs_source_init = [{'R_sersic': 0.03, 
                           'n_sersic': 1.0,
                           'e1':       0.0, 
                           'e2':       0.0, 
                           'center_x': 0.0, 
                           'center_y': 0.0}]
    
    # initial spread in parameter estimation #
    kwargs_lens_sigma = [{'theta_E':     0.3, 
                          'center_x':    0.1, 
                          'center_y':    0.1}]
    kwargs_source_sigma = [{'R_sersic': 0.1, 
                            'n_sersic': 0.5, 
                            'e1':       0.2, 
                            'e2':       0.2, 
                            'center_x': 0.1, 
                            'center_y': 0.1}]
    
    # hard bound lower limit in parameter space #
    kwargs_lower_lens = [{'theta_E':     0.0, 
                          'center_x':   -10., 
                          'center_y':   -10.}]
    kwargs_lower_source = [{'R_sersic': 0.001, 
                            'e1':       -0.5, 
                            'e2':       -0.5,
                            'n_sersic': .5, 
                            'center_x': -10, 
                            'center_y': -10}]
    # hard bound upper limit in parameter space #
    kwargs_upper_lens = [{'theta_E':     3*tE, 
                          'center_x':    10.0,
                          'center_y':    10.0}]
    kwargs_upper_source = [{'R_sersic': 10, 
                            'e1':       0.5, 
                            'e2':       0.5,
                            'n_sersic': 5., 
                            'center_x': 10, 
                            'center_y': 10}]

    # add LOS params
    # First we fix to 0 kappa and omega (not gamma_od for now -> gamma_od should be fixed
    # omega_LOS should not be fixed! the LOS shears in combination induce a small rotation
    # allowing for freedom in omega_LOS accounts for this and prevents bias in the shears
    # -> fix omage_LOS for now
    gamma_prior = 0.5
    #omega_prior = 0.5
    gamma_sigma = 0.1
    #omega_sigma = 0.1

    # this is for the minimal model
    kwargs_fixed_los = {'kappa_od': 0.0, 
                        'kappa_los': 0.0, 
                        'omega_od': 0.0,
                        'omega_los':0.0, 
                        'gamma1_od':0.0,  
                        'gamma2_od':0.0
                       }
    kwargs_fixed_lens = [{}]
    if los_included:
        kwargs_lens_init.append({#'gamma1_od':0, 'gamma2_od': 0,'omega_los': 0,
                                 'gamma1_los':0, 
                                 'gamma2_los':0
                                 })
    
        kwargs_lens_sigma.append({#'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,'omega_los': omega_sigma
                                  'gamma1_los': gamma_sigma, 
                                  'gamma2_los': gamma_sigma})
    
        kwargs_lower_lens.append({#'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,'omega_los': -omega_prior,
                                  'gamma1_los': -gamma_prior, 
                                  'gamma2_los': -gamma_prior})
    
        kwargs_upper_lens.append({#'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,'omega_los': omega_prior,
                                  'gamma1_los': gamma_prior, 
                                  'gamma2_los': gamma_prior})
        kwargs_fixed_lens.append(kwargs_fixed_los)
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}
    return kwargs_params



def create_SIS_gallens(tE_true=1.5,z=0.2,n_smpl=5e7,
                       kwlens_part = default_kwlens_part_AS,
                       reload=False):
    SISGal = PartGal({"n_smpl":n_smpl,
                      "theta_E":tE_true,"z_lens":z},
                     sim="SIS",simsuite="ANL_TEST") 
    
    SISGal.run(reload=reload) 
    SISGal.store_gal()
    
    lensgal = GalLens(SISGal,0,kwlens_part=kwlens_part)
    lensgal.run(read_prev=reload)
    lensgal.store()
    return lensgal
    
def get_kwargs_params(lens):
    return _get_kwargs_params(lens,los_included=True)
    
kw_model = {"lens_model_list":lens_model_list,
          "source_model_list":source_model_list,
         "get_kwargs_params":get_kwargs_params}


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
    elif run_type == 1:
        n_it   = 3
        n_part = 3
        n_burn = 1
        n_run  = 2
    elif run_type==2:
        n_it   = 100
        n_part = 20
        n_burn = 100
        n_run  = 200
    else:
        raise ValueError(f"run_type must be 0,1 or 2, not {run_type}")
    
    res_dir = get_res_dir(res_dir_base,simsuite,sim,
                          subsim=subsim,run_type=run_type)
    
    lens_sys = create_SIS_gallens()
    lens_sys.unpack()
    lens2model = LensSystem.from_GalLens(lens_sys,lensFuel=lensfuel)
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
