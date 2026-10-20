# Model all lenses where LOS IS simulated AND w. LOS in the model (adapted from model_SNS_DCE)
# using GEP
# here the center is not masked, to try to fit the central image to constrain theta_core
import sys
import numpy as np
from python_tools.tools import mkdir
from nazgul.Modelling.pathfinder import model_res_base

# LOS simulated as dependent on gal name (not z)
from nazgul.mount_doom.fuel_tank import lensfuel_los_gal as lensfuel

res_dir_base      = model_res_base/"GEP_ellexp/"
mkdir(res_dir_base)
description_model = "Simulate variable LOS, and model the lens with generalised elliptical potential (GEP with exponential elliptical profile) + LOS minimal, and an elliptical sersic source and not masking the center"
lens_model_list   = ['GEP','LOS_MINIMAL']


GEP_models = {'parameterization': 'rectangular',
            'ellipticity_model': 'exponential',
            'pointing_angle_model': None,
            'position_model': 'linear',
            'radial_potential': 'cored_power_law'}
profile_kwargs_list = [GEP_models,None]

source_model_list = ["SERSIC_ELLIPSE"]
"""
# the following would be nice but it's too risky (e.g. OD component in LOS)
# and doesn't account for initial values, sigmas and fixed params
from lenstronomy.LensModel.lens_model import LensModel
lens_model = LensModel(lens_model_list=lens_model_list,
                       profile_kwargs_list=profile_kwargs_list)
lens_model.lens_model.func_list[1].lower_limit_default
"""
def get_kwargs_params(lens):
    # Params:
    # initial guess of non-linear parameters, we chose different starting parameters than the truth #
    tE = lens.gallens.thetaE.value
    dx_centre = 1.5*tE # the freedom for the central position (before too large, +-10'')
    # LOS added separately    
    kwargs_lens_init = [{'theta_E':     tE *np.random.normal(1,.1,1)[0], 
                         'theta_core':  tE/10,
                         "sharpness":   2.0,
                         "gamma":       2.0,
                         "v_x":         0.0,
                         "v_y":         0.0,
                         'center_x':    0.0, 
                         'center_y':    0.0,
                         'e1_0':        0.0, 
                         'e2_0':        0.0, 
                         'e1_inf':      0.0,
                         'Dzeta_e1':    1.0,
                         'e2_inf':      0.0,
                         'Dzeta_e2':    1.0}]
    kwargs_source_init = [{'R_sersic': 0.03, 
                           'n_sersic': 1.0,
                           'e1':       0.0, 
                           'e2':       0.0, 
                           'center_x': 0.0, 
                           'center_y': 0.0}]
    
    # initial spread in parameter estimation #
    kwargs_lens_sigma = [{'theta_E':     0.3, 
                          'theta_core':  0.03,
                          "sharpness":   0.3,
                          "gamma":       0.1,
                          "v_x":         0.1,
                          "v_y":         0.1,
                          'center_x':    0.5, 
                          'center_y':    0.5,
                          'e1_0':        0.2, 
                          'e2_0':        0.2, 
                          'e1_inf':      0.2,
                          'Dzeta_e1':    0.3,
                          'e2_inf':      0.2,
                          'Dzeta_e2':    0.3}]

    kwargs_source_sigma = [{'R_sersic': 0.1, 
                            'n_sersic': 0.5, 
                            'e1':       0.2, 
                            'e2':       0.2, 
                            'center_x': 0.1, 
                            'center_y': 0.1}]
    
    # hard bound lower limit in parameter space #
    kwargs_lower_lens = [{'theta_E':     0.0, 
                          'theta_core':  0.0,
                          "sharpness":   1.0,
                          "gamma":       1.0,
                          "v_x":         -0.5,
                          "v_y":         -0.5,
                          'center_x':    -dx_centre, 
                          'center_y':    -dx_centre,
                          'e1_0':        -0.5, 
                          'e2_0':        -0.5, 
                          'e1_inf':      -0.5,
                          'Dzeta_e1':    0.0,
                          'e2_inf':      -0.5,
                          'Dzeta_e2':    0.0}]
    kwargs_lower_source = [{'R_sersic': 0.001, 
                            'e1':       -0.5, 
                            'e2':       -0.5,
                            'n_sersic': .5, 
                            'center_x': -dx_centre, 
                            'center_y': -dx_centre}]
    # hard bound upper limit in parameter space #
    kwargs_upper_lens = [{'theta_E':     3*tE,
                          'theta_core':  3*tE,
                          "sharpness":   3.0,
                          "gamma":       3.0,
                          "v_x":         0.5,
                          "v_y":         0.5,
                          'center_x':    +dx_centre, 
                          'center_y':    +dx_centre,
                          'e1_0':        0.5, 
                          'e2_0':        0.5, 
                         'e1_inf':       0.5,
                         'Dzeta_e1':     3*tE,
                         'e2_inf':       0.5,
                         'Dzeta_e2':     3*tE}]
    kwargs_upper_source = [{'R_sersic': 10, 
                            'e1':       0.5, 
                            'e2':       0.5,
                            'n_sersic': 5., 
                            'center_x': +dx_centre, 
                            'center_y': +dx_centre}]

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
    kwargs_fixed_los = {'kappa_od': 0.0, 'kappa_los': 0.0, 'omega_od': 0.0,
                        'omega_los':0.0, 'gamma1_od':0.0,  'gamma2_od':0.0
                       }

    kwargs_lens_init.append({#'gamma1_od':0, 'gamma2_od': 0,'omega_los': 0,
                             'gamma1_los':0, 'gamma2_los':0
                             })

    kwargs_lens_sigma.append({#'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,'omega_los': omega_sigma
                              'gamma1_los': gamma_sigma, 'gamma2_los': gamma_sigma})

    kwargs_lower_lens.append({#'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,'omega_los': -omega_prior,
                              'gamma1_los': -gamma_prior, 'gamma2_los': -gamma_prior})

    kwargs_upper_lens.append({#'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,'omega_los': omega_prior,
                              'gamma1_los': gamma_prior, 'gamma2_los': gamma_prior})
    
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, kwargs_fixed_los], kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}
    return kwargs_params
    
kw_model = {"lens_model_list":     lens_model_list,
            "source_model_list":   source_model_list,
            "get_kwargs_params":   get_kwargs_params,
            "profile_kwargs_list": profile_kwargs_list}

if __name__=="__main__":
    from nazgul.Modelling.lib_models import run_model 
    
    run_model(res_dir_base,
              lensfuel=lensfuel,
              kw_model=kw_model,
              gauss_tE_prior=True,
              prog_name=sys.argv[0],
              mask_center=False,
              #lenses2model= lenses2model,
              description=description_model)
