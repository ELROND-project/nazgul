# Model all lenses where LOS is not simulated but w. LOS in the model
# we add multipoles and see if we manage to reduce the LOS tension (still sim to be 0)
# copy from SNS_multipole_gausstE
# now:
#  - with elliptical source (fitted, not simulated)
#  - with m1,m3,m4 multipoles
#  - with gaussian tE prior
#  - gamma (power law) min value brought from 1.5 to 1
import os,gc
import argparse
import warnings
import numpy as np
import sys,json,dill
from corner import corner
import matplotlib.pyplot as plt

from lenstronomy.Plots import chain_plot

from python_tools.get_res import load_whatever

from nazgul.plot_PL import plot_kappamap
from nazgul.lens_part_LOS import get_kw_los
from nazgul.mount_doom.lens_system import LensSystem

from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.Modelling.lib_models import setup_lens,setup_sim_obs,get_kwargs_likelihood,get_lenses2model,add_gaussian_tE_prior
from nazgul.Modelling.lib_models import save_data,plot_model_plot
from nazgul.Modelling.lib_models import model_res_base,n_it_std,n_part_std,n_burn_std,n_run_std,get_res_dir # default values

# WOI cross-machine lock
from python_tools.tools import mkdir
from python_tools.tools_WOI import set_workin_on_it

lens_model_list   = ['EPL_MULTIPOLE_M1M3M4_ELL','LOS_MINIMAL']
source_model_list = ["SERSIC_ELLIPSE"]
res_dir_base      = model_res_base/"SNS_m134_gausstE"
mkdir(res_dir_base)

def get_kwargs_params(lens):
    # Params:
    # initial guess of non-linear parameters, we chose different starting parameters than the truth #
    tE = lens.gallens.thetaE.value

    # LOS added separately    
    kwargs_lens_init = [{'theta_E': tE + np.random.normal(0,.1,1)[0]*tE, 
                    'e1': 0, 'e2': 0, 
                    'gamma': 2., 
                    'center_x': 0., 'center_y': 0,
                    "a1_a":0,
                    "delta_phi_m1":0,
                    "a3_a":0,
                    "delta_phi_m3":0,
                    "a4_a":0,
                    "delta_phi_m4":0}]
    kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 1., 
                           'center_x': 0, 'center_y': 0,
                          'e1':0,'e2':0}]
    
    # initial spread in parameter estimation #
    kwargs_lens_sigma = [{'theta_E': 0.3, 
                          'e1': 0.2, 'e2': 0.2, 
                          'gamma': .2, 
                          'center_x': 0.1, 'center_y': 0.1,
                          "a1_a":0.05,
                          "delta_phi_m1":np.pi/4,
                          "a3_a":0.05,
                          "delta_phi_m3":np.pi/10,
                          "a4_a":0.05,
                          "delta_phi_m4":np.pi/15}]
    kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1,
                            'e1':0.2,'e2':0.2}]
    
    # hard bound lower limit in parameter space #
    # hard bounds of multipoles are taken from the model script itself
    kwargs_lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 
                          'gamma': 1., 
                          'center_x': -10., 'center_y': -10,
                          "a1_a":-0.2,
                          "delta_phi_m1":-np.pi,
                          "a3_a":-0.2,
                          "delta_phi_m3":-np.pi/6,
                          "a4_a":-0.2,
                          "delta_phi_m4":-np.pi/8
                         }]
    kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10,
                            'e1':-.5,
                            'e2':-.5}]
    # hard bound upper limit in parameter space #
    kwargs_upper_lens = [{'theta_E': 3*tE, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10,
                        "a1_a": 0.2,
                        "delta_phi_m1": np.pi,
                        "a3_a": 0.2,
                        "delta_phi_m3": np.pi / 6,
                        "a4_a": 0.2,
                        "delta_phi_m4": np.pi / 8,}]
    kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10,
                            'e1':0.5,'e2':0.5}]

    # add LOS params
    # First we fix to 0 kappa and omega (not gamma_od for now)
    # omega_LOS should not be fixed! the LOS shears in combination induce a small rotation
    # allowing for freedom in omega_LOS accounts for this and prevents bias in the shears
    
    gamma_prior = 0.5
    omega_prior = 0.5
    gamma_sigma = 0.1
    omega_sigma = 0.1

    # this is for the minimal model
    kwargs_fixed_los = {'kappa_od': 0.0, 'kappa_los': 0.0, 'omega_od': 0.0}

    kwargs_lens_init.append({'gamma1_od':0, 'gamma2_od': 0,
                             'gamma1_los':0, 'gamma2_los':0,
                             'omega_los': 0 })

    kwargs_lens_sigma.append({'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,
                              'gamma1_los': gamma_sigma, 'gamma2_los': gamma_sigma,
                              'omega_los': omega_sigma})

    kwargs_lower_lens.append({'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,
                              'gamma1_los': -gamma_prior, 'gamma2_los': -gamma_prior,
                              'omega_los': -omega_prior})

    kwargs_upper_lens.append({'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,
                              'gamma1_los': gamma_prior, 'gamma2_los': gamma_prior,
                              'omega_los': omega_prior})

    """
    # this is for the full LOS
    kwargs_fixed_los = {'kappa_od': 0, 'kappa_os': 0,'kappa_ds':0,
                        'omega_od': 0.0, 'omega_os': 0.0, 'omega_ds': 0.0}
    kwargs_lens_init.append({'gamma1_od': 0, 'gamma2_od': 0,
                             'gamma1_os': 0, 'gamma2_os': 0,
                             'gamma1_ds': 0, 'gamma2_ds': 0})
    kwargs_lens_sigma.append({'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,
                              'gamma1_os': gamma_sigma, 'gamma2_os': gamma_sigma,
                              'gamma1_ds': gamma_sigma, 'gamma2_ds': gamma_sigma})
    kwargs_lower_lens.append({'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,
                              'gamma1_os': -gamma_prior, 'gamma2_os': -gamma_prior,
                              'gamma1_ds': -gamma_prior, 'gamma2_ds': -gamma_prior})

    kwargs_upper_lens.append({'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,
                              'gamma1_os': gamma_prior, 'gamma2_os': gamma_prior,
                              'gamma1_ds': gamma_prior, 'gamma2_ds': gamma_prior})
    """
    
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, kwargs_fixed_los], kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}
    return kwargs_params


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Simulate (without shear) and model the lens (with shear) - Study the Internal Shear")
    parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Type of run: 
        0 = standard, PSO_it = {n_it_std} PSO_prt = {n_part_std} MCMCb = {n_burn_std} MCMCr = {n_run_std}  
        1 = test run  PSO_it = 3      PSO_prt = 3      MCMCb = 1     MCMCr = 2 
        2 = test run (longer)  PSO_it = 100      PSO_prt = 20      MCMCb = 100     MCMCr = 200 
       (PSO_it: PSO iterations, PSO_prt: PSO particles, MCMCr: MCMC run steps, MCMCb: MCMC burn in steps)\n""")
    parser.add_argument('-nl','--n_lenses',type=int,dest="n_lenses",default=5,help=f"Number of lenses to model")
    parser.add_argument('-mtE','--min_thetaE',type=float,dest="min_thetaE",default=None,help=f"Min theta_E for the gal to be considered a lens")
    parser.add_argument('-snap','--snap',nargs="+",type=str,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-ovw','--overwrite',action="store_true",dest="overwrite",default=False,help=f"Overwrite previous results (def False, unless test - than always true)")

    args       = parser.parse_args()
    run_type   = args.run_type
    n_lenses   = args.n_lenses
    min_thetaE = args.min_thetaE 
    snaps      = args.snaps #[25,26,27]
    sim        = args.sim
    subsim     = args.subsim
    simsuite   = args.simsuite
    overwrite  = args.overwrite

    check_if_workin_on_it = True
    if run_type==0:
        n_iterations = int(n_it_std) #number of iteration of the PSO run
        n_particles  = int(n_part_std) #number of particles in PSO run
        n_burn = int(n_burn_std) #MCMC burn in steps
        n_run  = int(n_run_std) #MCMC total steps 
    elif run_type ==1:
        print("Test Run")
        n_iterations = int(3) #number of iteration of the PSO run
        n_particles  = int(3) #number of particles in PSO run
        n_run  = int(2) #MCMC total steps 
        n_burn = int(1) #MCMC burn in steps
        if n_lenses>3:
            print("Resetting n_lenses to 3 because test")
            n_lenses = 3
        overwrite = True
    elif run_type ==2:
        print("Test Run - longer")
        check_if_workin_on_it = False
        n_iterations = int(100) #number of iteration of the PSO run
        n_particles  = int(20) #number of particles in PSO run
        n_run  = int(200) #MCMC total steps 
        n_burn = int(100) #MCMC burn in steps
        if n_lenses>1:
            print("Resetting n_lenses to 1 because test")
            n_lenses = 1
        overwrite = True
    else:
        raise RuntimeError("Give a valid run_type or implement it your own")

    # picked by hand "bad" lenses ->
    lenses2skip = []
    
    kw_get_all_gallens = {"sim":sim,
                          "subsim":subsim,
                           "simsuite":simsuite,
                            "snaps":snaps}
    res_dir = get_res_dir(res_dir_base,simsuite,sim,
                          subsim=subsim,run_type=run_type)
    mkdir(res_dir)

    print("\nGetting catalogue of lenses 2 model\n###################\n")
    if run_type==0:
        gal_lenses  = get_lenses2model(res_dir=res_dir,
                                       reload=True,
                                       kw_get_all_gallens=kw_get_all_gallens,
                                       n_lenses=np.nan, # has to load all of them anyway
                                       min_thetaE=min_thetaE,
                                       skip_lenses=lenses2skip)
    else:
        warnings.warn("VERY patchy way to deal with the issue, but fast")
        try:
            cat = load_whatever(res_dir/"cat_lens2model.dll")
        except FileNotFoundError:
            raise RuntimeError("To save time, the catalogue should be linked from the original model, and not recomputed for each test")
        _gal_lenses = [ load_whatever(c) for c in cat["lens_cat"][:n_lenses]]
        gal_lenses = []
        for g in _gal_lenses:
            g.unpack()
            #from nazgul.stat_lenses import monkey_patch_naming
            #monkey_patch_naming(g)
            g.run()
            gal_lenses.append(g)
    print("\nCatalogue of lenses 2 model obtained\n###################\n")
    
    for i,gal_lens in enumerate(gal_lenses):            
        print("\nLoading lens "+gal_lens.name)
        lens = LensSystem.from_GalLens(gal_lens)
        lens = setup_lens(lens,res_dir=res_dir,
                          overwrite=overwrite,
                          check_if_workin_on_it=check_if_workin_on_it)
        if lens is None: #means that someone is workin on it
            continue
        print("\nModelling lens "+gal_lens.name+"\n###############################")   
        plot_kappamap(lens.gallens.kappa_map, 
                      extent_kpc=lens.gallens.kw_extents["extent_kpc"],
                      savename=f"{lens.model_res_dir}/kappa_gal.png")
        multi_band_list = setup_sim_obs(lens)
        image_obs = multi_band_list[0][0]["image_data"]
        
        # models
        kwargs_model = {'lens_model_list': lens_model_list,
                        'source_light_model_list': source_model_list}
        
        
        kwargs_likelihood = get_kwargs_likelihood(lens,image_obs=image_obs)
        kwargs_likelihood = add_gaussian_tE_prior(lens,kwargs_likelihood)
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    
        # Params:
        
        kwargs_params = get_kwargs_params(lens)
    
        kwargs_constraints = {#'joint_source_with_point_source': [[0, 0]], 
        #    'joint_source_with_point_source': list [[i_point_source, k_source], [...], ...],
        #     joint position parameter between lens model and source light model
                               #   'num_point_source_list': [4],
                                  'solver_type':'NONE'# 'PROFILE_SHEAR',  # 'PROFILE', \
                              #'PROFILE_SHEAR', 'ELLIPSE', 'CENTER', 'NONE'
                                  }
    
        
        # actual fit:
        from lenstronomy.Workflow.fitting_sequence import FittingSequence
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': n_particles, 'n_iterations':n_iterations}]
                          ,
                           ['MCMC', {'n_burn': n_burn, 'n_run': n_run, 'walkerRatio': 5, 'sigma_scale': .1}]
            ]
        kw_input = {"kwargs_data_joint":   kwargs_data_joint,
                    "kwargs_model":        kwargs_model, 
                    "kwargs_constraints":  kwargs_constraints, 
                    "kwargs_likelihood":   kwargs_likelihood, 
                    "kwargs_params":       kwargs_params,
                    "fitting_kwargs_list": fitting_kwargs_list,
                    #"kw_add_lenses":       kw_add_lenses
                   }
        nm_input = f"{lens.model_res_dir}/kw_input.dll"
        save_data(kw_input,nm_input,"input")
        
        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()
        print("kwargs_result",kwargs_result)
        nm_res = f"{lens.model_res_dir}/kw_res.dll"
        save_data(kwargs_result,nm_res,"result output")
        
        
        # we need to extract the updated multi_band_list object since the coordinate shifts were updated in the kwargs_data portions of it
        multi_band_list_out = fitting_seq.multi_band_list
        nm_mblo = f"{lens.model_res_dir}/multi_band_list_out.dll"
        save_data(multi_band_list_out,nm_mblo,"output multiband list")
        
        plot_model_plot(multi_band_list_out,kwargs_model,kwargs_result,kwargs_likelihood,res_dir=lens.model_res_dir)
        # don't store pso results
        emcee = chain_list[-1]
        sampler_type, mc_sample, param_mcmc, mc_logL   = emcee
        emcee_path = f'{lens.model_res_dir}/emcee_chain.dll'
        save_data(emcee,emcee_path,"emcee chain")

        corner(mc_sample,labels=param_mcmc,show_titles=True,plot_datapoints=False,hist_kwargs= {"density":True})
        nm = f'{lens.model_res_dir}/mcmc_post.pdf'
        plt.savefig(nm)
        print(f"Saving {nm}")
        plt.close()
            
        # test
        for i in range(len(chain_list)):
            chain_plot.plot_chain_list(chain_list, i)
        nm = f'{lens.model_res_dir}/chain_plot.pdf'
        plt.savefig(nm)
        print(f"Saving {nm}")
        # reset flag to false
        set_workin_on_it(lens.model_res_dir,wrk = False)

        # Cleanup to save memory
        plt.close("all")
        del lens
        del chain_list
        del kw_input
        del multi_band_list
        gc.collect()

        # only do n lenses:
        i+=1
        if i==n_lenses:
            break