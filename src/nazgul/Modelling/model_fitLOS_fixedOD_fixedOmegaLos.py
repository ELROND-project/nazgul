# Copy from modelling_severals, now we fit for LOS shear instead of external shear
import os
import argparse
import numpy as np
import sys,json,dill
from pathlib import Path
from corner import corner
from copy import copy,deepcopy
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST

from python_tools.read_fits import load_fits
from python_tools.tools import mkdir,to_dimless
from python_tools.get_res import load_whatever

from nazgul.plot_PL import plot_all
from nazgul.masking import mask_SEAGLE,mask_max_dens,mask_bright_center,resize_mask
from nazgul.mount_doom.cracks_of_doom import LoadLens,get_extents
from nazgul.mount_doom.lens_system import LensSystem
from nazgul.plot_PL import plot_kappamap
from nazgul.stat_lenses import get_all_gallens

from nazgul.lens_part_LOS import get_kw_los
from nazgul.pathfinder import get_sim_dir,tmp_dir
from nazgul.Modelling.lib_models import model_res_base,n_it_std,n_part_std,n_burn_std,n_run_std,get_res_dir # default values

lens_model_list   = ['EPL',"LOS_MINIMAL"]
source_model_list = ["SERSIC"]
res_dir_base      = tmp_dir/"models/fitLOS_fixedOD_fixedOmegaLos/"

def _get_model_res_dir(lens,res_dir=res_dir_base):
    res_dir = Path(f"{res_dir}/snap_{lens.gallens.Gal.snap}_{lens.name}")
    return res_dir
def get_link_lens_path(lens,res_dir=res_dir_base):
    if not hasattr(lens,"model_res_dir"): 
        lens.model_res_dir = _get_model_res_dir(lens,res_dir=res_dir)
    return lens.model_res_dir/"link_gallens.pkl"


def setup_lens(lens,res_dir=res_dir_base):
    lens.model_res_dir = _get_model_res_dir(lens,res_dir=res_dir)
    lens.image_sim = lens.get_lensed_image(unconvolved=False)
    mkdir(lens.model_res_dir)
    
    # For conveniency, but likely not the best idea:
    print("TODO: This is valid ONLY when we are only modelling a galaxy")
    lens.kw_extents = get_extents(arcXkpc=lens.gallens.arcXkpc,
                                  _radec=lens.gallens._radec)
    lens.kappa_map = lens.gallens.kappa_map
    lens.Gal = lens.gallens.Gal
    lens.z_lens = lens.gallens.z_lens
    lens.z_source = lens.gallens.z_source
    lens.deltaPix = lens.gallens.deltaPix
    lens.pixel_num = lens.gallens.pixel_num
    
    #lens.kwargs_lens = lens.gallens.kwargs_lens
    plot_all(lens,skip_caustic=True)

    # create link to lens
    src = lens.gallens.pkl_path
    dst = get_link_lens_path(lens,res_dir=res_dir)
    if not os.path.islink(dst):
        os.symlink(src,dst)
    return lens


def setup_sim_obs(lens,band_str="HST_F160W",pssf=3):
    if band_str=="HST_F160W":
        band     = HST(band='WFC3_F160W', psf_type="PIXEL")
        # TOOD: bug (?) in lenstronomy, psf_type is not passed to band
        band.obs["psf_type"] = "PIXEL"
        psf_path = Path(f"./ObsData/HST/WFC3/F160W/PSFSTD_WFC3IR_F160W.fits")
    else:
        raise RuntimeError("Pragma no cover: to implement other bands and PSFs")
    
    # the following contain 9 PSF depending on their position in the CCD
    # we will consider the central position -> i think it should be the second to last (not too important here )
    psf = load_fits(psf_path)[-2]
    # we can supersample it
    if pssf!=1:
        psf_ss  = zoom(psf,pssf,order=3)
        kwargs_psf = {"kernel_point_source":psf_ss,
                      "point_source_supersampling_factor":pssf}
    else:
        kwargs_psf = {"kernel_point_source":psf}
    
    multi_band_list = lens.sim_multi_band_list(band=band,
                                               kwargs_psf=kwargs_psf)
    return multi_band_list


def get_lens_mask(lens,image_obs,plot_mask=True):
    # masking inner and outer of thetaE -> nope, follow SEAGLE approach
    #image = kwargs_data["image_data"]
    mask_HD = mask_SEAGLE(lens)*mask_bright_center(lens,rad=lens.gallens.thetaE*.5) #mask_max_dens(lens)
    mask_LD = resize_mask(mask_HD,image_obs)
    if plot_mask:
        plt.close()
        plt.close("all")
        kw_extents = lens.kw_extents
        extent_arcsec = kw_extents["extent_arcsec"]
        kw_plot = {"cmap":"hot","extent":extent_arcsec,"origin":"lower"}
        fig,axes = plt.subplots(2,2, figsize=(10, 10))
        ax =axes[0][0]
        im0 = ax.imshow(np.log10(lens.image_sim),**kw_plot)
        ax.set_title("Image")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   
        
        ax =axes[0][1]
        ax.set_title("Masked Image")
        mask_nan = copy(mask_HD)
        mask_nan[np.where(mask_HD==0)] = np.nan
        im0 = ax.imshow(np.log10(mask_nan*lens.image_sim),**kw_plot)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   

        ax =axes[1][0]
        im0 = ax.imshow(np.log10(image_obs),**kw_plot)
        ax.set_title("Image (Realistic)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   
        
        ax =axes[1][1]
        ax.set_title("Masked Image (Realistic)")
        mask_nan = copy(mask_LD)
        mask_nan[np.where(mask_LD==0)] = np.nan
        im0 = ax.imshow(np.log10(mask_nan*image_obs),**kw_plot)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   
        
        nm = f"{lens.model_res_dir}/{lens.name}_masked_im.png"
        print(f"Saving {nm}")
        plt.savefig(nm)
    return mask_LD

def get_kwargs_likelihood(lens,image_obs,plot_mask=True):
    mask = get_lens_mask(lens,image_obs,plot_mask=plot_mask)
    
    kwargs_likelihood = {'check_bounds': True, # punish out-of-bound soulutions
                     #'force_no_add_image': False,
                     'source_marg': False, # marginalization addition on the imaging likelihood based on the covariance of the inferred linear coefficients 
                     #'image_position_uncertainty': 0.004,
                     #'check_matched_source_position': True,
                     #'source_position_tolerance': 0.001,
                     'source_position_sigma': 0.01,
                     #'prior_lens': prior_lens
                      "image_likelihood_mask_list": [mask]  }
    return kwargs_likelihood

def get_kwargs_params(lens):
    # Params:
    # initial guess of non-linear parameters, we chose different starting parameters than the truth #
    tE = lens.gallens.thetaE.value

    # LOS added separately    
    kwargs_lens_init = [{'theta_E': tE + np.random.normal(0,.1,1)[0]*tE, 
                    'e1': 0, 'e2': 0, 
                    'gamma': 2., 
                    'center_x': 0., 'center_y': 0}]
    kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 1., 'center_x': 0, 'center_y': 0}]
    
    # initial spread in parameter estimation #
    kwargs_lens_sigma = [{'theta_E': 0.3, 
                          'e1': 0.2, 'e2': 0.2, 'gamma': .2, 
                          'center_x': 0.1, 'center_y': 0.1}]
    kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1}]
    
    # hard bound lower limit in parameter space #
    kwargs_lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10}]
    kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10}]
    # hard bound upper limit in parameter space #
    kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10}]
    kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10}]

    # add LOS params
    # First we fix to 0 kappa and omega (not gamma_od for now)
    # omega_LOS should not be fixed! the LOS shears in combination induce a small rotation
    # allowing for freedom in omega_LOS accounts for this and prevents bias in the shears

    gamma_prior = 0.5
    gamma_sigma = 0.1

    # this is for the minimal model
    kwargs_fixed_los = {'kappa_od': 0.0, 'kappa_los': 0.0, 'omega_od': 0.0,'gamma1_od':0,'gamma2_od': 0,'omega_los': 0 }

    kwargs_lens_init.append({'gamma1_los':0, 
                             'gamma2_los':0})

    kwargs_lens_sigma.append({'gamma1_los': gamma_sigma, 
                              'gamma2_los': gamma_sigma})

    kwargs_lower_lens.append({'gamma1_los': -gamma_prior, 
                              'gamma2_los': -gamma_prior})

    kwargs_upper_lens.append({'gamma1_los': gamma_prior, 
                              'gamma2_los': gamma_prior})
    
    lens_params   = [kwargs_lens_init, kwargs_lens_sigma, 
                     [{}, kwargs_fixed_los], kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}
    return kwargs_params

def get_lenses2model(kw_get_all_gallens={"snaps":[27]},n_lenses=5,skip_lenses=[]):
    precomputed_lenses = get_all_gallens(**kw_get_all_gallens)
    
    theta = []
    for lens  in precomputed_lenses:
        theta.append(lens.thetaE.value)
    theta = np.array(theta)
    lenses_sort = np.array(precomputed_lenses)[theta.argsort()][::-1]
    lenses_selected = lenses_sort[:n_lenses]
    if len(skip_lenses)!=0:
        lenses_accepted = []
        for l in lenses_selected:
            if l.name not in skip_lenses:
                lenses_accepted.append(l)
            else:
                print(f"Ignoring lens {l} because in the list of lenses to skip")
        lenses_selected = lenses_accepted
    return lenses_selected

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Simulate and model the lens")
    parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Type of run: 
        0 = standard, PSO_it = {n_it_std} PSO_prt = {n_part_std} MCMCb = {n_burn_std} MCMCr = {n_run_std}  
        1 = test run  PSO_it = 3      PSO_prt = 3      MCMCb = 1     MCMCr = 2 
       (PSO_it: PSO iterations, PSO_prt: PSO particles, MCMCr: MCMC run steps, MCMCb: MCMC burn in steps)\n""")
    
    args         = parser.parse_args()
    run_type     = args.run_type
    
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
    else:
        raise RuntimeError("Give a valid run_type or implement it your own")
    
    lenses2skip = [] # ["Sub_Gn22SGn0_Npix200_PartAS_Prj2"]
    #["Sub_Gn3SGn0_Npix200_PartAS_Prj1","Sub_Gn22SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj2"]
    gal_lenses  = get_lenses2model(skip_lenses=lenses2skip)
    for gal_lens in gal_lenses: 
        print("Loading lens "+gal_lens.name+"\n")
        """print("Adding LOS effects")
        kw_los = get_kw_los()
        kw_add_lenses = {"lens_model_list":["LOS"],
                        "kwargs_lens":[kw_los]}
        lens = LensSystem.from_GalLens(gal_lens,kwargs_add_lenses=kw_add_lenses)
        """
        lens = LensSystem.from_GalLens(gal_lens)
        """
        if lens.gallens.thetaE.value<min_thetaE:
            raise RuntimeError(f"Ensure that the thetaE of the input lens is larger than min_thetaE: {lens.gallens.thetaE.value}<{min_thetaE}")
            
        if lens.gallens.Gal.M < min_mass:
            raise RuntimeError(f"Ensure that the M of the input lens is larger than min_mass:{lens.gallens.Gal.M} < {min_mass}")
        """
        lens = setup_lens(lens)
        plot_kappamap(lens.gallens.kappa_map, 
                      extent_kpc=lens.gallens.kw_extents["extent_kpc"],
                      savename=f"{lens.model_res_dir}/kappa_gal.png")
        multi_band_list = setup_sim_obs(lens)
        image_obs = multi_band_list[0][0]["image_data"]
        
        # models
        kwargs_model = {'lens_model_list': lens_model_list,
                        'source_light_model_list': source_model_list}
        
        
        kwargs_likelihood = get_kwargs_likelihood(lens,image_obs=image_obs)
    
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
                    "fitting_kwargs_list": fitting_kwargs_list
                   }
        nm_input = f"{lens.model_res_dir}/kw_input.dll"
        with open(nm_input,"wb") as f:
            dill.dump(kw_input,f)
        print(f"Saving input in kw: {nm_input}")
        
        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()
        print("kwargs_result",kwargs_result)
        nm_res = f"{lens.model_res_dir}/kw_res.json"
        with open(nm_res,"w") as f:
            json.dump(kwargs_result,f)
        print(f"Saving result output in kw:{nm_res}")
    
        print(kwargs_result)
    
        
        # we need to extract the updated multi_band_list object since the coordinate shifts were updated in the kwargs_data portions of it
        multi_band_list_out = fitting_seq.multi_band_list
        nm_mblo = f"{lens.model_res_dir}/multi_band_list_out.json"
        print(f"Saving output multiband list as {nm_mblo}")
        with open(nm_mblo,"wb") as f:
            dill.dump(multi_band_list_out,f)
        
        modelPlot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result, 
                              arrow_size=0.02, cmap_string="gist_heat",
                              image_likelihood_mask_list=kwargs_likelihood["image_likelihood_mask_list"])
    
        band_index_plot = 0
        f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
        
        modelPlot.data_plot(ax=axes[0,0], band_index=band_index_plot)#,v_min=,v_max=band_i.vmax)
        modelPlot.model_plot(ax=axes[0,1], band_index=band_index_plot)#,v_min=band_i.vmin,v_max=band_i.vmax)
        modelPlot.normalized_residual_plot(ax=axes[0,2], band_index=band_index_plot)#,v_min=band_i.res_vmin,v_max=band_i.res_vmax)
        modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100, band_index=band_index_plot)#,v_min=band_i.vmin,v_max=band_i.vmax)
        modelPlot.convergence_plot(ax=axes[1, 1], band_index=band_index_plot,v_max=1)
        modelPlot.magnification_plot(ax=axes[1, 2], band_index=band_index_plot)
        f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        nm = f'{lens.model_res_dir}/mass_model.pdf'
        plt.savefig(nm)
        print(f"Saving {nm}")
        
        f, axes = plt.subplots(1,2, figsize=(8, 4), sharex=False, sharey=False)
    
        modelPlot.decomposition_plot(ax=axes[0], band_index=band_index_plot, text='Source light', source_add=True, unconvolved=True)#,v_min=band_i.vmin,v_max=band_i.vmax)
        modelPlot.decomposition_plot(ax=axes[1], band_index=band_index_plot, text='Source light convolved', source_add=True)#,v_min=band_i.vmin,v_max=band_i.vmax)
    
        f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        #plt.show()
        nm = f'{lens.model_res_dir}/light_model.pdf'
        plt.savefig(nm)
        print(f"Saving {nm}")
        
        kwargs_result_wo_tracer = deepcopy(kwargs_result)
        keys = copy(list(kwargs_result_wo_tracer.keys()))
        for k in keys:
            if "tracer" in str(k):
                del kwargs_result_wo_tracer[k]
        ### 
        logL,_prms   = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result_wo_tracer)
        
        n_data = modelPlot._imageModel.num_data_evaluate
        reduced_chi2  = -logL * 2 / n_data
        print(f'{np.round(reduced_chi2,2)} reduced X^2 of all evaluated imaging data combined\n')
        print("################################\n")
    
        #Normalised plot
        f, axes = plt.subplots(figsize=(10,7))
        modelPlot.normalized_residual_plot(ax=axes,v_min=-3, v_max=3,text=r"Norm. Resid $\chi^2_{red.}$="+str(np.round(reduced_chi2,2)))
        nm = f'{lens.model_res_dir}/normalised_residuals.png'
        plt.savefig(nm)
        print(f"Saving {nm}")
        plt.close()
    
        #Caustics
        f, axes = plt.subplots(figsize=(10,7))
        modelPlot.source_plot(ax=axes, deltaPix_source=0.01, numPix=1000, with_caustics=True)
        f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        nm = f'{lens.model_res_dir}/caustics.png'
        plt.savefig(nm)
        print(f"Saving {nm}")
        plt.close()
        """
        f, axes = plt.subplots(figsize=(10,7))
        modelPlot.decomposition_plot(ax=axes, text='Point source position', source_add=False, \
                        lens_light_add=False, point_source_add=True, v_min=-1, v_max=1)
        nm = f'{lens.model_res_dir}/point_source_position.png'
        plt.savefig(nm)
        print(f"Saving {nm}")
        plt.close()
        """
        
        if run_type!=1:
            sampler_type, mc_sample, param_mcmc, mc_logL  = chain_list[-1]
            chnl_path = f'{lens.model_res_dir}/chain_list.dll'
            with open(chnl_path,"wb") as f:
                dill.dump(chain_list,f)
            print(f"Saving {chnl_path}")
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
        plt.close()
