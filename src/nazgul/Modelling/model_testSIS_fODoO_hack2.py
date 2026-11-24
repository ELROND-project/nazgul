"""
Simulate an image with SIS and try to model it - leave gamma OD and omega LOS free

-> assume optimistic HST observations 

-> hack2: use native lenstronomy to produce the image and model it, but with optimistic Obs as setup for the observations 
"""
import sys
import argparse
import numpy as np
from corner import corner
import matplotlib.pyplot as plt

from python_tools.tools import mkdir
from python_tools.get_res import load_whatever

from nazgul.Modelling.pathfinder import model_res_base

# LOS simulated as dependent on gal name (not z)
from nazgul.MountDoom.fuel_tank import lensfuel_los_gal_optimistic_band
from nazgul.Modelling.model_testSIS_freegOD_optimisticObs import n_smpl


from nazgul.Modelling.lib_models  import save_data,plot_model_plot
mask_center = True
gauss_tE_prior = True
overwrite_results = True
check_if_workin_on_it = False

res_dir_base      = model_res_base/"testSIS_fODoO_hack2/"
mkdir(res_dir_base)

description_model = "Try to model a single SIS w LOS, and model the lens with EPL +LOS minimal (with free gamma OD and omega LOS), and an elliptical sersic source - now with fixed lens pos - and Optimistic Observation (def. lnstr. HST band) - hacked II: use lenstronomy to produce the image, not nazgul"
from nazgul.Modelling.model_testSIS import tE_in,z_lens #1.5 ",0.2,
from nazgul.Modelling.model_testSIS import lens_model_list,source_model_list

from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

# Gallens used only to reproduce same parameters:
from nazgul.Modelling.model_testSIS import create_SIS_gallens

gl  = create_SIS_gallens(reload=True)
cosmo = gl.cosmo
z_source = gl.z_source
lensfuel_los_gal_optimistic_band.setup(gl)
kw_add_lenses = lensfuel_los_gal_optimistic_band.get_kw_add_lenses(gl)

kw_lens_true = [{"theta_E":tE_in,
                 "center_x":0,
                 "center_y":0},
                  kw_add_lenses["kwargs_lens"][0]]
kwargs_lensmodel = {"z_lens":z_lens,
                    "z_source":z_source,
                    "cosmo":cosmo}

lensModel = LensModel(["SIS","LOS"],
                      **kwargs_lensmodel)
lightModel_source = LightModel(light_model_list=["SERSIC_ELLIPSE"])

SimObs = lensfuel_los_gal_optimistic_band.SimObs
kwargs_psf = SimObs.kwargs_psf
psf = PSF(**kwargs_psf)

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

# get kwargs source w. correct amplitude given the obs. simobs
from nazgul.MountDoom.cracks_of_doom import get_kwargs_sourceSim
class _fakelens():
    def __init__(self):
        self.cosmo = gl.cosmo
        self.gallens = gl

_fkl = _fakelens()
kw_source_init = lensfuel_los_gal_optimistic_band.kwargs_source_init
kwargs_source = get_kwargs_sourceSim(SimObs,kwargs_source=kw_source_init,lens=_fkl)
kwargs_source_list = [kwargs_source]

image_lnstr       = imageModel.image(kwargs_lens=kw_lens_true, 
                               kwargs_source=kwargs_source_list)
image_lnstrnoisy  = image_lnstr + SimObs.noise_for_model(model=image_lnstr)
error_image       = SimObs.estimate_noise(image_lnstrnoisy) 

kwargs_data["image_data"] = image_lnstrnoisy
kwargs_data["noise_map"]  = error_image

kw_numerics  = {'point_source_supersampling_factor':1}

image_band      = [kwargs_data, kwargs_psf, kw_numerics]
multi_band_list = [image_band]

kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
kwargs_constraints = {"solver_type":"NONE"}

kwargs_model = {'lens_model_list': lens_model_list,
                'source_light_model_list': source_model_list}

from nazgul.configurations import nazgul_path
#kw_input_for_testSIS.dll -> results/models/testSIS_freeOG_optimisticObs/ANLT_SIS/snap_0_LS_Lens_TEST_GAL_SIS_tE1.5_nS1.00e+07_Prj0/kw_input.dll
kw_in = load_whatever(nazgul_path/"tmp/kw_input_for_testSIS.dll")

kwargs_likelihood = kw_in["kwargs_likelihood"]
# WARNING: using kw_params as well - it should be fine
kwargs_params = kw_in["kwargs_params"]
fitting_kwargs_list = kw_in["fitting_kwargs_list"] 

if __name__=="__main__":
    from nazgul.Modelling.lib_models import get_res_dir,n_it_std,n_burn_std,n_run_std,n_part_std

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
    
    model_res_dir = get_res_dir(res_dir_base,simsuite,sim,
                          subsim=subsim,run_type=run_type)

    from lenstronomy.Workflow.fitting_sequence import FittingSequence
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

    kw_input = {"kwargs_data_joint":   kwargs_data_joint,
                "kwargs_model":        kwargs_model, 
                "kwargs_constraints":  kwargs_constraints, 
                "kwargs_likelihood":   kwargs_likelihood, 
                "kwargs_params":       kwargs_params,
                "fitting_kwargs_list": fitting_kwargs_list,
               }
    nm_input = f"{model_res_dir}/kw_input.dll"
    save_data(kw_input,nm_input,"input")
    
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    print("kwargs_result",kwargs_result)
    nm_res = f"{model_res_dir}/kw_res.dll"
    save_data(kwargs_result,nm_res,"result output")
    
    # we need to extract the updated multi_band_list object since the coordinate shifts were updated in the kwargs_data portions of it
    multi_band_list_out = fitting_seq.multi_band_list
    nm_mblo = f"{model_res_dir}/multi_band_list_out.dll"
    save_data(multi_band_list_out,nm_mblo,"output multiband list")
    
    plot_model_plot(multi_band_list_out,kwargs_model,kwargs_result,kwargs_likelihood,res_dir=model_res_dir)
    # don't store pso results
    emcee = chain_list[-1]
    sampler_type, mc_sample, param_mcmc, mc_logL   = emcee
    emcee_path = f'{model_res_dir}/emcee_chain.dll'
    save_data(emcee,emcee_path,"emcee chain")

    corner(mc_sample,labels=param_mcmc,show_titles=True,plot_datapoints=False,hist_kwargs= {"density":True})
    nm = f'{model_res_dir}/mcmc_post.pdf'
    plt.savefig(nm)
    print(f"Saving {nm}")
    plt.close()

