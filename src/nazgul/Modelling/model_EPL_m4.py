# Default model: only consider EPL + multipole m4 + Shear
# But LOS is not simulated to be 0,0
import sys
import numpy as np
from python_tools.tools import mkdir
from nazgul.Modelling.pathfinder import model_res_base

# LOS simulated as dependent on gal name (not z)
from nazgul.mount_doom.fuel_tank import lensfuel_los_gal as lensfuel

lens_model_list   = ['EPL_BOXYDISKY_ELL','LOS_MINIMAL']
source_model_list = ["SERSIC_ELLIPSE"]
res_dir_base      = model_res_base/"EPL_m4/"
mkdir(res_dir_base)

description_model = "Simulate variable LOS, and model the lens with EPL + multipole m4 + LOS minimal, and an elliptical sersic source"

def get_kwargs_params(lens):
    # Params:
    # initial guess of non-linear parameters, we chose different starting parameters than the truth #
    tE = lens.gallens.thetaE.value

    # LOS added separately    
    kwargs_lens_init = [{'theta_E':     tE *np.random.normal(1,.1,1)[0], 
                         'gamma':       2.0,
                         'e1':          0.0, 
                         'e2':          0.0, 
                         'center_x':    0.0, 
                         'center_y':    0.0,
                         'a4_a':        0.0}]
    kwargs_source_init = [{'R_sersic': 0.03, 
                           'n_sersic': 1.0,
                           'e1':       0.0, 
                           'e2':       0.0, 
                           'center_x': 0.0, 
                           'center_y': 0.0}]
    # initial spread in parameter estimation #
    kwargs_lens_sigma = [{'theta_E':     0.3, 
                          'gamma':       0.2,
                          'e1':          0.2, 
                          'e2':          0.2, 
                          'center_x':    0.1, 
                          'center_y':    0.1,
                          'a4_a':        0.05}]
    kwargs_source_sigma = [{'R_sersic': 0.1, 
                            'n_sersic': 0.5, 
                            'e1':       0.2, 
                            'e2':       0.2, 
                            'center_x': 0.1, 
                            'center_y': 0.1}]
    
    # hard bound lower limit in parameter space #
    kwargs_lower_lens = [{'theta_E':     0.0, 
                          'gamma':       1.0,
                          'e1':         -0.5, 
                          'e2':         -0.5, 
                          'center_x':   -10., 
                          'center_y':   -10.,
                          'a4_a':       -0.1}]
    kwargs_lower_source = [{'R_sersic': 0.001, 
                            'e1':       -0.5, 
                            'e2':       -0.5,
                            'n_sersic': .5, 
                            'center_x': -10, 
                            'center_y': -10}]
    # hard bound upper limit in parameter space #
    kwargs_upper_lens = [{'theta_E':     3*tE, 
                          'gamma':       3.0,
                          'e1':          0.5, 
                          'e2':          0.5, 
                          'center_x':    10.0,
                          'center_y':    10.0,
                          'a4_a':       +0.1}]
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
    kwargs_fixed_los = {'kappa_od':  0.0, 
                        'kappa_los': 0.0, 
                        'omega_od':  0.0,
                        'omega_los': 0.0, 
                        'gamma1_od': 0.0,  
                        'gamma2_od': 0.0
                       }

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
    
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, kwargs_fixed_los], kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}
    return kwargs_params

kw_model = {"lens_model_list":lens_model_list,
          "source_model_list":source_model_list,
         "get_kwargs_params":get_kwargs_params}

if __name__=="__main__":
    from nazgul.Modelling.lib_models import run_model
    run_model(res_dir_base,
              lensfuel=lensfuel,
              kw_model=kw_model,
              gauss_tE_prior=True,
              prog_name=sys.argv[0],
              description=description_model)