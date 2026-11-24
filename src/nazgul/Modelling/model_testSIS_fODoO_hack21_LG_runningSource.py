"""
Simulate an image with SIS and try to model it - leave gamma OD and omega LOS free
-> SUBSTANCIALLY DIFFERENT MODEL: we now run several models with the source moving in the background and passing trough the center of the SIS. We then estimate the tension and relate that to the position of the osurce

-> assume optimistic HST observations 

-> hack2: use native lenstronomy to produce the image and model it, but with optimistic Obs as setup for the observations 
-> 2.1: actually get every ref to nazgul out and only put explicit values
-> move the source -> now move a lot
-> larger gamma LOS values
"""
import sys
import dill
import argparse
import numpy as np
from copy import deepcopy
from corner import corner
import matplotlib.pyplot as plt

from python_tools.tools import mkdir
from python_tools.get_res import load_whatever

from nazgul.Modelling.pathfinder import model_res_base

from nazgul.lib_stat import tension_multidim

# LOS simulated as dependent on gal name (not z)
tE_in = 1.5 # arcsec
z_lens = 0.2
n_smpl = int(1e7)


# this is the only one we should be fine loading -> long to load and not used
#from nazgul.Modelling.lib_models  import plot_model_plot
mask_center = True
gauss_tE_prior = True
overwrite_results = True
check_if_workin_on_it = False

res_dir_base      = model_res_base/"testSIS_fODoO_hack21_LG_runningSource/"
mkdir(res_dir_base)

description_model = "Try to model a single SIS w LOS, and model the lens with EPL +LOS minimal (with free gamma OD and omega LOS), and an elliptical sersic source - now with fixed lens pos - and Optimistic Observation (def. lnstr. HST band) - hacked II: use lenstronomy to produce the image, not nazgul - now without any ref. to nazgul. Now w. Larger Gamma LOS values - Now with a 'running source' in the background, to estimate the tension w. gamma_LOS"
lens_model_list   = ['SIS','LOS_MINIMAL']
source_model_list = ["SERSIC_ELLIPSE"]

from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from astropy.cosmology import Planck13 as cosmo

z_source = 1.0556582737985163
kw_add_lenses = [{'kappa_od': 0, 'kappa_os': 0, 'kappa_ds': 0,
              'omega_od': 0, 'omega_os': 0, 'omega_ds': 0,
              'gamma1_od':  0.01, 'gamma2_od': 0.02,
              'gamma1_os':  0.0, 'gamma2_os': 0.01,
              'gamma1_ds':  0.04, 'gamma2_ds': -0.01}]

kw_gamma_los_true = {"gamma1_los": kw_add_lenses[0]["gamma1_od"] +kw_add_lenses[0]["gamma1_os"] - kw_add_lenses[0]["gamma1_ds"]    ,
                     "gamma2_los": kw_add_lenses[0]["gamma2_od"] +kw_add_lenses[0]["gamma2_os"] - kw_add_lenses[0]["gamma2_ds"]}
                     
kw_lens_true = [{"theta_E":tE_in,
                 "center_x":0,
                 "center_y":0},
                  kw_add_lenses[0]]
kwargs_lensmodel = {"z_lens":z_lens,
                    "z_source":z_source,
                    "cosmo":cosmo}

lensModel = LensModel(["SIS","LOS"],
                      **kwargs_lensmodel)
lightModel_source = LightModel(light_model_list=["SERSIC_ELLIPSE"])

from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
kwargs_psf = {"psf_type":"GAUSSIAN",
                "truncation":5,
                "fwhm":0.08}

pixel_num_obs = 109
psf = PSF(**kwargs_psf)

kwargs_model_sim = {"source_light_model_list":source_model_list,
                    "cosmo":cosmo,
                    "z_source":z_source}

band = HST()
kwargs_single_band = band.kwargs_single_band()
        
SimObs = SimAPI(num_pix = pixel_num_obs, # N of pixels in "observed" image
                kwargs_single_band = kwargs_single_band, # telescope specific keyword arguments (eg HST, see above)
                kwargs_model = kwargs_model_sim,# kwargs source model (in principle kw lens as well)
                )
data_class = SimObs.data_class
kwargs_data = SimObs.kwargs_data
kwargs_numerics = {'supersampling_factor': 1, 
                  'supersampling_convolution': False}

imageModel = ImageModel(data_class=data_class, 
                        psf_class=psf, 
                        lens_model_class=lensModel,
                        source_model_class=lightModel_source,
                        lens_light_model_class=None,
                        point_source_class=None,
                        kwargs_numerics=kwargs_numerics)

kwargs_constraints = {"solver_type":"NONE"}

kwargs_model = {'lens_model_list': lens_model_list,
                'source_light_model_list': source_model_list}


pos_x_source_ratio_tE = np.linspace(0,0.3,30)
pos_x_source = pos_x_source_ratio_tE * tE_in
def get_kwargs_data_joint_pos_y(pos_source):
    kwargs_source_pos = {"center_x":pos_source,
                      "center_y":1e-9} 
    
    # Compute kwargs_source luminosity
    kwargs_source_init = {'abs_magnitude': -21,
     'R_sersic': 0.5,
     'n_sersic': 1.5,
     'e1': 0.0,
     'e2': 0.0}| kwargs_source_pos
    
    kwargs_source = deepcopy(kwargs_source_init)
    if "abs_magnitude" in kwargs_source.keys():
        # convert absolute magnitude in apparent
        Mag = kwargs_source["abs_magnitude"]
        dL  = cosmo.luminosity_distance(z_source)
        m = Mag + 5*np.log10(dL.to("pc").value/10)
        del kwargs_source["abs_magnitude"]
        kwargs_source["magnitude"] = m
        kwargs_source_list       = [kwargs_source]
        # the following only depends on -kwargs_source_params, -magnitude_0_point -source_model_list
        _, kwargs_source_list, _ = SimObs.magnitude2amplitude(kwargs_source_mag = kwargs_source_list)
    
    # Create obs. images
    image_lnstr       = imageModel.image(kwargs_lens=kw_lens_true, 
                                   kwargs_source=kwargs_source_list)
    image_lnstrnoisy  = image_lnstr + SimObs.noise_for_model(model=image_lnstr)
    error_image       = SimObs.estimate_noise(image_lnstrnoisy) 
    
    # update kwargs_data
    kwargs_data["image_data"] = image_lnstrnoisy
    kwargs_data["noise_map"]  = error_image
    
    kw_numerics  = {'point_source_supersampling_factor':1}
    
    image_band      = [kwargs_data, kwargs_psf, kw_numerics]
    multi_band_list = [image_band]
    
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    return kwargs_data_joint

# the mask is tricky to reproduce, we simply load it form prev. model testSIS_freeOG_optimisticObs

from nazgul.configurations import nazgul_path
#kw_input_for_testSIS.dll -> results/models/testSIS_freeOG_optimisticObs/ANLT_SIS/snap_0_LS_Lens_TEST_GAL_SIS_tE1.5_nS1.00e+07_Prj0/kw_input.dll
kw_in = load_whatever(nazgul_path/"tmp/kw_input_for_testSIS.dll")
kwargs_likelihood = kw_in["kwargs_likelihood"]

kw_prms_lens_init = [{'theta_E': 1.7,
    'center_x': 0.0,
    'center_y': 0.0},
   {'gamma1_los': 0,
    'gamma2_los': 0,
    'gamma1_od': 0.0,
    'gamma2_od': 0.0,
    'omega_los': 0.0}]
kw_prms_lens_sigma = [{'theta_E': 0.3, 'center_x': 0.1, 'center_y': 0.1},
   {'gamma1_los': 0.1,
    'gamma2_los': 0.1,
    'gamma1_od': 0.1,
    'gamma2_od': 0.1,
    'omega_los': 0.0005}]
kw_prms_lens_fix = [{}, {'kappa_od': 0.0, 'kappa_los': 0.0, 'omega_od': 0.0}]
kw_prms_lens_min = [{'theta_E': 0.0, 'center_x': -10.0, 'center_y': -10.0},
   {'gamma1_los': -0.5,
    'gamma2_los': -0.5,
    'gamma1_od': -0.5,
    'gamma2_od': -0.5,
    'omega_los': -0.002}]

kw_prms_lens_max = [{'theta_E': np.float64(4.379109991947686),
    'center_x': 10.0,
    'center_y': 10.0},
   {'gamma1_los': 0.5,
    'gamma2_los': 0.5,
    'gamma1_od': 0.5,
    'gamma2_od': 0.5,
    'omega_los': 0.002}]

kw_params_lens = kw_prms_lens_init,kw_prms_lens_sigma,kw_prms_lens_fix,kw_prms_lens_min,kw_prms_lens_max

kw_prms_source_init =  [{'R_sersic': 0.03,
                        'n_sersic': 1.0,
                        'e1': 0.0,
                        'e2': 0.0,
                        'center_x': 0.0,
                        'center_y': 0.0}]
kw_prms_source_sigma=   [{'R_sersic': 0.1,
                        'n_sersic': 0.5,
                        'e1': 0.2,
                        'e2': 0.2,
                        'center_x': 0.1,
                        'center_y': 0.1}]
kw_prms_source_fix =  [{}]
kw_prms_source_min =   [{'R_sersic': 0.001,
                        'e1': -0.5,
                        'e2': -0.5,
                        'n_sersic': 0.5,
                        'center_x': -10,
                        'center_y': -10}]
kw_prms_source_max= [{'R_sersic': 10,
                    'e1': 0.5,
                    'e2': 0.5,
                    'n_sersic': 5.0,
                    'center_x': 10,
                    'center_y': 10}]

kw_params_source = kw_prms_source_init,kw_prms_source_sigma,kw_prms_source_fix,kw_prms_source_min,kw_prms_source_max
kwargs_params = {'lens_model': kw_params_lens,
                'source_model': kw_params_source}

fitting_kwargs_list = kw_in["fitting_kwargs_list"] 

if __name__=="__main__":
    from nazgul.Modelling.lib_models import n_it_std,n_burn_std,n_run_std,n_part_std

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
        raise ValueError("not really working")
    else:
        raise ValueError(f"run_type must be 0,1 or 2, not {run_type}")
    
    model_res_dir =str(res_dir_base)+"/ANLT_SIS/"
    if run_type>0:
        model_res_dir+="/test/"
    mkdir(model_res_dir)
    # where to store the individual runs:
    model_res_dir_runs = f"{model_res_dir}/runs/"
    mkdir(model_res_dir_runs)
    
    from lenstronomy.Workflow.fitting_sequence import FittingSequence
    kw_input = {"kwargs_model":        kwargs_model, 
                "kwargs_constraints":  kwargs_constraints, 
                "kwargs_likelihood":   kwargs_likelihood, 
                "kwargs_params":       kwargs_params,
                "fitting_kwargs_list": fitting_kwargs_list,
                'pos_x_source': pos_x_source
               }
    nm_input = f"{model_res_dir}/kw_input_gen.dll"
    with open(nm_input,"wb") as f:
        dill.dump(kw_input,f)
    result_tension = []
    for i,pos_x_source_i in enumerate(pos_x_source):
        print(i)
        kwargs_data_joint = get_kwargs_data_joint_pos_y(pos_x_source_i)
        
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()

        print("kwargs_result",kwargs_result)
        nm_res = f"{model_res_dir_runs}/kw_res_{i}.dll"
        with open(nm_res,"wb") as f:
            dill.dump(kwargs_result,f)
        # we need to extract the updated multi_band_list object since the coordinate shifts were updated in the kwargs_data portions of it
        multi_band_list_out = fitting_seq.multi_band_list
        nm_mblo = f"{model_res_dir_runs}/multi_band_list_out_{i}.dll"
        
        with open(nm_mblo,"wb") as f:
            dill.dump(multi_band_list_out,f)
        
        #plot_model_plot(multi_band_list_out,kwargs_model,kwargs_result,kwargs_likelihood,res_dir=model_res_dir)
        # don't store pso results
        emcee = chain_list[-1]
        sampler_type, mc_sample, param_mcmc, mc_logL   = emcee
        emcee_path = f'{model_res_dir_runs}/emcee_chain_{i}.dll'
        with open(emcee_path,"wb") as f:
            dill.dump(emcee,f)


        indexes_gLOS = param_mcmc.index("gamma1_los_lens1"),param_mcmc.index("gamma2_los_lens1")
        distr  =  np.asarray(mc_sample)[:,indexes_gLOS].T
        gamma_los_tension = tension_multidim(distr,true_value_list)
        print("Tension :",gamma_los_tension)
        result_tension.append([i,pos_x_source_i,gamma_los_tension])

    nm_results_tension = f'{model_res_dir}/result_tension.dll'
    with open(nm_results_tension,"wb") as f:
        dill.dump(result_tension,f)
    print(f"Saved {nm_results_tension}")
        
"""   
corner(mc_sample,labels=param_mcmc,show_titles=True,plot_datapoints=False,hist_kwargs= {"density":True})
nm = f'{model_res_dir}/mcmc_post_{i}.pdf'
plt.savefig(nm)
print(f"Saving {nm}")
plt.close()
"""
