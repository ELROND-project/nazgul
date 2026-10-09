# Copy from model_allLOS.py
# adapted to have all the required components for the specific models

import os
import warnings
import argparse
import numpy as np
import sys,dill
from pathlib import Path
from corner import corner
from copy import copy,deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.Util import util
from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot

from python_tools.read_fits import load_fits
from python_tools.get_res import load_whatever
from python_tools.tools import mkdir,to_dimless,dict_equal

from nazgul.plot_PL import plot_all
from nazgul.masking import mask_SEAGLE,mask_max_dens,mask_bright_center,resize_mask
#from nazgul.mount_doom.cracks_of_doom import LoadLens #,get_extents
from nazgul.mount_doom.lens_system import LensSystem
from nazgul.plot_PL import plot_kappamap
from nazgul.stat_lenses import get_all_gallens


from nazgul.lens_part_LOS import get_kw_los

from nazgul.Modelling.pathfinder import model_res_base,get_link_lens_path, get_model_res_dir_from_gallens

# WOI cross-machine lock
from python_tools.tools_WOI import workin_on_it, set_workin_on_it, is_someone_workin_on_it

lens_model_list_def   = ['EPL']
source_model_list_def = ["SERSIC"]
#PSO
n_it_std   = 1000
n_part_std = 300
#MCMC
n_burn_std = 700
n_run_std  = 7000

from nazgul.mount_doom.fuel_tank import std_lensfuel
def setup_lens(lens,
               res_dir,
               _plot=True,
               overwrite=False,
               check_if_workin_on_it=True,
               workin_on_it=True,
               verbose=True):
    model_res_dir = get_model_res_dir_from_gallens(lens.gallens,res_dir=res_dir)
    # verify that no-one is working on it
    if check_if_workin_on_it:
        if is_someone_workin_on_it(model_res_dir):
            warnings.warn(f"This lens, {lens.name} is being worked on, skipping - if not, delete the WOI.dll file:\n{model_res_dir}/WOI.dll") 
            return None        
    set_workin_on_it(model_res_dir,wrk = workin_on_it)
    # verify that there aren't previous results
    if not overwrite:
        if Path(f"{model_res_dir}/kw_res.dll").is_file():
            warnings.warn(f"This lens, {lens.name} has already results, and I should not overwrite them. If that's not what you want, set overwrite=True") 
            return None
            
    lens.setup(verbose=verbose)
    
    lens.model_res_dir = model_res_dir
    # Note: the following is only used for plotting
    # and mask definition (to find the bright center of the image)
    lens.image_true = lens.true_image()
    
    if verbose:
        print(f"Saving modelling results in {lens.model_res_dir}") 
    # For conveniency, but likely not the best idea: -> it's not bad if we don't change the map
    #lens.kw_extents = get_extents(arcXkpc=lens.gallens.arcXkpc,
    #                              _radec=lens.gallens._radec)
    #lens.kappa_map = lens.gallens.kappa_map
    #lens.Gal       = lens.gallens.Gal
    #lens.z_lens    = lens.gallens.z_lens
    #lens.z_source  = lens.gallens.z_source
    
    # for the masking:
    lens.deltaPix  = lens.gallens.deltaPix
    lens.pixel_num = lens.gallens.pixel_num
    
    #lens.kwargs_lens = lens.gallens.kwargs_lens
    if _plot:
        gal_plot = deepcopy(lens.gallens)
        gal_plot.image_sim = lens.image_true
        plot_all(gal_plot,skip_caustic=True)
        
    # create link to lens
    src = lens.gallens.pkl_path
    dst = get_link_lens_path(lens)
    if not os.path.islink(dst):
        try:
            os.symlink(src,dst)
        except FileExistsError as e:
            print(f"This error \n{e}\n should not happen, but it's not too important")
    """
    # not useful at the moment - possibly in the future
    # store the kwargs_lenssystem
    kwargs_lenssystem = lens._kwargs_lenssystem
    pth_kw_lnsstm     = str(res_dir)+"/kw_lenssystem.dll"
    with open(pth_kw_lnsstm,"wb") as f:
        dill.dump(kwargs_lenssystem,f)
    if verbose:
        print(f"Stored {pth_kw_lnsstm}")
    """
    return lens


def get_kw_lens_mask(lens,image_obs,mask_center=True):
    # masking inner and outer of thetaE -> nope, follow SEAGLE approach
    #image = kwargs_data["image_data"]
    mask_SE = mask_SEAGLE(lens,image=image_obs) 
    if mask_center:
        mask_HD = mask_bright_center(lens)
    else:
        mask_HD = mask_bright_center(lens,rad_pix=0)
    #,rad=lens.gallens.thetaE*.5) #mask_max_dens(lens)
    mask_LD = resize_mask(mask_HD,image_obs)*mask_SE
    mask_SE_HD = resize_mask(mask_SE,mask_HD)
    mask_comb_HD = mask_HD*mask_SE_HD
    kw_mask = {"mask_SE": mask_SE}
    kw_mask["mask_HD"] = mask_HD
    kw_mask["mask_LD"] = mask_LD
    kw_mask["mask_SE_HD"] = mask_SE_HD
    kw_mask["mask_comb_HD"] = mask_comb_HD
    return kw_mask
    
    
def get_lens_mask(lens,image_obs,mask_center=True,plot_mask=True):
    kw_mask = get_kw_lens_mask(lens,image_obs,mask_center=mask_center)
    mask_SE	=	kw_mask["mask_SE"]
    mask_HD	=	kw_mask["mask_HD"]
    mask_LD	=	kw_mask["mask_LD"]
    mask_SE_HD	=	kw_mask["mask_SE_HD"]
    mask_comb_HD	=	kw_mask["mask_comb_HD"]

    if plot_mask:
        plt.close()
        plt.close("all")
        kw_extents = lens.gallens.kw_extents
        image_true  = lens.image_true
        extent_arcsec = kw_extents["extent_arcsec"]
        kw_plot = {"cmap":"hot","extent":extent_arcsec,"origin":"lower"}
        fig,axes = plt.subplots(2,2, figsize=(10, 10))
        ax =axes[0][0]
        im0 = ax.imshow(np.log10(image_true),**kw_plot)
        ax.set_title("Image")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   
        
        ax =axes[0][1]
        ax.set_title("Masked Image")
        mask_nan = copy(mask_comb_HD)
        mask_nan[np.where(mask_nan==0)] = np.nan
        im0 = ax.imshow(np.log10(mask_nan*image_true),**kw_plot)
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

def add_gaussian_tE_prior(lens,kwargs_likelihood,sig_tE=0.5):
    print("Adding gaussian prior for the theta_E")
    tE = lens.gallens.thetaE.value
    print(f"centered on 'true_value' tE={np.round(tE,3)} and sigma={np.round(sig_tE,3)}")
    kwargs_likelihood["prior_lens"] = [[0,"theta_E",tE,sig_tE]]
    return kwargs_likelihood
        

def get_kwargs_likelihood(lens,image_obs,mask_center=True,plot_mask=True):
    mask = get_lens_mask(lens,image_obs,
                         mask_center=mask_center,
                         plot_mask=plot_mask)
    
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
    
def get_kwargs_params_def(lens):
    raise RuntimeError("This function should be taken as a reference and re-implemented in the specific model")

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
    kwargs_upper_lens = [{'theta_E': 3*tE, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10}]
    kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10}]
 
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, kwargs_fixed_los], kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}
    return kwargs_params

def _get_lenses2model(kw_get_all_gallens={"snaps":[27]},n_lenses=None,min_thetaE=None,skip_lenses=[],verbose=True):
    precomputed_lenses =  np.array(get_all_gallens(verbose=verbose,**kw_get_all_gallens))
    if min_thetaE is None and n_lenses is None:
        n_lenses = 5
        print(f"No boundary given - assuming n_lenses={n_lenses} - If you really want all the lenses, give n_lenses=np.nan")
    
    theta = []
    for lens  in precomputed_lenses:
        theta.append(lens.thetaE.value)
    theta = np.array(theta)
    if min_thetaE is None:
        if np.isnan(n_lenses):
            print("Assuming no boundary")
            lenses_selected = precomputed_lenses
        else:
            lenses_sort     = precomputed_lenses[theta.argsort()][::-1]
            lenses_selected = lenses_sort[:n_lenses]
    else:
        lenses_selected = precomputed_lenses[np.where(theta>min_thetaE)]
        if n_lenses:
            if not np.isnan(n_lenses):
                lenses_selected = lenses_selected[:n_lenses]
                
    if len(skip_lenses)!=0:
        lenses_accepted = []
        for l in lenses_selected:
            if l.name not in skip_lenses:
                lenses_accepted.append(l)
            else:
                print(f"Ignoring lens {l} because in the list of lenses to skip")
        lenses_selected = lenses_accepted

    if len(lenses_selected)==0:
        raise RuntimeError("The boundary given are too strict - no galaxy satisfy it")
    return lenses_selected

def get_lenses2model(res_dir,reload=True,verbose=True,**kw_get_lenses2model):
    cat_l2m = Path(res_dir)/"cat_lens2model.dll"
    update_cat  = True
    recompute   = False
    if cat_l2m.is_file():
        if reload:
            print(f"Loading previously computed catalogue of lenses to models {cat_l2m}") 
            kw_cat_lens = load_whatever(cat_l2m)
            if not dict_equal(kw_cat_lens["kw_require"],kw_get_lenses2model):
                print(f"Catalogue {cat_l2m} exists, but doen't have the same requirements. Ignored and updated")
                recompute  = True
            else:
                update_cat = False
                recompute  = False
                cat_lens = kw_cat_lens["lens_cat"]
                lenses_2unpack = [load_whatever(l) for l in cat_lens]
                lenses = [l.unpack() for l in lenses_2unpack]
        else:
            print(f"Catalogue {cat_l2m} exists, but ignored - recomputing it and updating it")
            recompute = True
    else:
        print(f"Catalogue {cat_l2m} doesn't exists - creating it now")
        recompute = True

    if recompute:
        update_cat = True
        lenses = _get_lenses2model(verbose=verbose, **kw_get_lenses2model)
        
    if update_cat:
        lenses_cat = [l.pkl_path for l in lenses]
        kw_cat_lens = {"lens_cat":lenses_cat,
                       "kw_require":kw_get_lenses2model}
        with open(cat_l2m,"wb") as f:
            dill.dump(kw_cat_lens,f)
        print(f"Saving catalogue of lenses to models {cat_l2m}") 
    return lenses
    
def save_data(data,nm_data,str_data_type=""):
    with open(nm_data,"wb") as f:
        dill.dump(data,f)
    print(f"Saving {str_data_type}: {nm_data}")

def load_kwargs_result(res_dir):
    nm_res = f"{res_dir}/kw_res.dll"   
    kwargs_result = load_whatever(nm_res)
    return kwargs_result
    
def load_kwargs_input(res_dir):
    nm_res = f"{res_dir}/kw_input.dll"   
    kwargs_result = load_whatever(nm_res)
    return kwargs_result
    
def get_model_plot(res_dir,
                   multi_band_list_out = None,
                   kw_input= None,
                   kwargs_result=None):
    if multi_band_list_out is None:
        multi_band_list_out = load_mblo(res_dir)
    if kw_input  is None:
        kw_input = load_kwargs_input(res_dir)
    if kwargs_result is None:
        kwargs_result = load_kwargs_result(res_dir)
    kwargs_model      = kw_input["kwargs_model"]
    kwargs_likelihood = kw_input["kwargs_likelihood"]
    modelPlot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result,
                          image_likelihood_mask_list=kwargs_likelihood["image_likelihood_mask_list"])
    return modelPlot

def load_mblo(res_dir):
    nm_mblo = f"{res_dir}/multi_band_list_out.dll"
    multi_band_list_out = load_whatever(nm_mblo)
    return multi_band_list_out

def plot_modelplot_massmodel(modelPlot,band_index_plot=0):
    im0 = plt.imshow(np.log10(modelPlot._band_plot_list[band_index_plot]._data))
    vmin,vmax = im0.get_clim()
    plt.close()
    
    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
    kw_modelplot = {"cmap":"gist_heat",
                    "vmin":vmin,
                    "vmax":vmax,
                    "band_index":band_index_plot}
    kw_modelplot_res = deepcopy(kw_modelplot) 
    kw_modelplot_res["cmap"] = "bwr"
    kw_modelplot_res["vmin"] = -3
    kw_modelplot_res["vmax"] = +3
    modelPlot.data_plot(ax=axes[0,0], **kw_modelplot)
    modelPlot.model_plot(ax=axes[0,1],**kw_modelplot)
    modelPlot.normalized_residual_plot(ax=axes[0,2],**kw_modelplot_res)
    modelPlot.source_plot(ax=axes[1, 0], delta_pix_source=0.01, num_pix=100, **kw_modelplot)
    modelPlot.convergence_plot(ax=axes[1, 1], band_index=band_index_plot,vmax=1)
    modelPlot.magnification_plot(ax=axes[1, 2], band_index=band_index_plot,cmap="PuOr")
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    return f

def plot_model_plot(multi_band_list_out,kwargs_model,kwargs_result,kwargs_likelihood,res_dir,plot_point_sources=False):
    modelPlot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result, 
                          image_likelihood_mask_list=kwargs_likelihood["image_likelihood_mask_list"])
    
    band_index_plot = 0
    f_massmodel =  plot_modelplot_massmodel(modelPlot=modelPlot,band_index_plot=band_index_plot)
    nm = f'{res_dir}/mass_model.pdf'
    f_massmodel.savefig(nm)
    plt.close(f_massmodel)
    print(f"Saving {nm}")
    
    f, axes = plt.subplots(1,2, figsize=(8, 4), sharex=False, sharey=False)
    
    modelPlot.decomposition_plot(ax=axes[0], band_index=band_index_plot, kwargs_title={"text":'Source light'}, source_add=True, unconvolved=True,cmap="gist_heat")#,vmin=band_i.vmin,vmax=band_i.vmax)
    modelPlot.decomposition_plot(ax=axes[1], band_index=band_index_plot, kwargs_title={"text":'Source light convolved'}, source_add=True,cmap="gist_heat")#,vmin=band_i.vmin,vmax=band_i.vmax)
    
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    nm = f'{res_dir}/light_model.pdf'
    plt.savefig(nm)
    plt.close(f)
    print(f"Saving {nm}")
    
    reduced_chi2 = get_red_chi2(modelPlot=modelPlot,verbose=True)    
    #Normalised plot
    f, axes = plt.subplots(figsize=(10,7))
    modelPlot.normalized_residual_plot(ax=axes,vmin=-3, vmax=3,kwargs_title={"text":r"Norm. Resid $\chi^2_{red.}$="+str(np.round(reduced_chi2,2))})
    nm = f'{res_dir}/normalised_residuals.png'
    plt.savefig(nm)
    print(f"Saving {nm}")
    plt.close(f)
    
    #Caustics
    f, axes = plt.subplots(figsize=(10,7))
    modelPlot.source_plot(ax=axes, delta_pix_source=0.01, num_pix=1000,cmap="gist_heat")
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    nm = f'{res_dir}/caustics.png'
    plt.savefig(nm)
    print(f"Saving {nm}")
    plt.close(f)
    if plot_point_sources:
        f, axes = plt.subplots(figsize=(10,7))
        modelPlot.decomposition_plot(ax=axes, kwargs_title={"text":'Point source position'}, source_add=False, \
                        lens_light_add=False, point_source_add=True, vmin=-1, vmax=1,cmap="gist_heat")
        nm = f'{res_dir}/point_source_position.png'
        plt.savefig(nm)
        print(f"Saving {nm}")
        plt.close(f)

def get_kwres_wo_tracer(kwargs_result):
    keys = copy(list(kwargs_result.keys()))
    if any([True for k in keys if "tracer" in k]):
        kwargs_result_wo_tracer = deepcopy(kwargs_result)
        for k in keys:
            if "tracer" in str(k):
                del kwargs_result_wo_tracer[k]
        return kwargs_result_wo_tracer
    else:
        return kwargs_result
    
def get_red_chi2(modelPlot,verbose=True):
    """
    kwargs_result_wo_tracer = get_kwres_wo_tracer(kwargs_result)
    ### 
    logL,_prms   = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result_wo_tracer)
    
    n_data = modelPlot._imageModel.num_data_evaluate
    reduced_chi2  = -logL * 2 / n_data
    """
    reduced_chi2 = modelPlot._band_plot_list[0].reduced_x2
    if verbose:
        print(f'{np.round(reduced_chi2,2)} reduced X^2 of all evaluated imaging data combined\n')
        print("################################\n")
    return reduced_chi2

"""
# Wrong idea - kwargs_lenssystem will in principle be different for each lens
class LensSystemFactory:
    def __init__(self, res_dir,**kwargs_lenssystem):
        path_kw_lnss = str(res_dir)+"/kwargs_lenssystem.dll"
        with open(path_kw_lnss,"wb") as f:
            dill.dump(kwargs_lenssystem,f)
        print(f"Stored {path_kw_lnss}")
        self._kwargs = kwargs_lenssystem

    def from_GalLens(self, GalLens, **override):
        return LensSystem.from_GalLens(GalLens, **{**self._kwargs, **override})
"""
import gc
from nazgul.Modelling.pathfinder import get_res_dir 
from nazgul.Translator import std_sim,std_simsuite,std_subsim

def model_lens(lens,kw_model,
               mask_center=True,
               n_it = n_it_std,
               n_part= n_part_std,
               n_burn = n_burn_std,
               n_run = n_run_std,
              gauss_tE_prior=True):
    lens_model_list   = kw_model["lens_model_list"]
    source_model_list = kw_model["source_model_list"]
    get_kwargs_params = kw_model["get_kwargs_params"]

    print("\nModelling lens "+lens.name+"\n###############################")   
    plot_kappamap(lens.gallens.kappa_map, 
                  extent_kpc=lens.gallens.kw_extents["extent_kpc"],
                  savename=f"{lens.model_res_dir}/kappa_gal.png")
    multi_band_list = lens.sim_multi_band_list()
    image_obs = multi_band_list[0][0]["image_data"]
    
    # models
    kwargs_model = {'lens_model_list': lens_model_list,
                    'source_light_model_list': source_model_list}
    
    
    kwargs_likelihood = get_kwargs_likelihood(lens,image_obs=image_obs,mask_center=mask_center)
    if gauss_tE_prior:
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
    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': n_part, 'n_iterations':n_it}]
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
    return None 
    
def run_model(res_dir_base,
                lensfuel,
                kw_model,
                gauss_tE_prior=True,
                prog_name=sys.argv[0],
              mask_center=True,
              description="Simulate and model the lens",
              n_it = n_it_std,
              n_part= n_part_std,
              n_burn = n_burn_std,
              n_run = n_run_std,
             lenses2skip = [],
             lenses2model = [],
              **kwargs
             ):
    parser = argparse.ArgumentParser(prog=prog_name,description=description)
    parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Type of run:
        0 = standard, PSO_it = {n_it} PSO_prt = {n_part} MCMCb = {n_burn} MCMCr = {n_run}  
        1 = test run  PSO_it = 3      PSO_prt = 3      MCMCb = 1     MCMCr = 2 
        2 = test run (longer)  PSO_it = 100      PSO_prt = 20      MCMCb = 100     MCMCr = 200 
        (PSO_it: PSO iterations, PSO_prt: PSO particles, MCMCr: MCMC run steps, MCMCb: MCMC burn in steps)\n""")
    parser.add_argument('-nl','--n_lenses',type=int,dest="n_lenses",default=np.nan,help=f"Number of lenses to model")
    parser.add_argument('-mtE','--min_thetaE',type=float,dest="min_thetaE",default=.3,help=f"Min theta_E for the gal to be considered a lens")
    parser.add_argument('-ovr','--overwrite_results',dest="overwrite_results",
                        default=False,action="store_true",help=f"Overwrite previous results")
    if not "sim" in kwargs:
        parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    if not "snap" in kwargs:
        parser.add_argument('-snap','--snap',nargs="+",type=str,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    if not "simsuite" in kwargs:
        parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    if not "subsim" in kwargs:
        parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    args       = parser.parse_args()
    run_type   = args.run_type
    n_lenses   = args.n_lenses
    min_thetaE = args.min_thetaE 
    if not "snap" in kwargs:
        snaps      = args.snaps #[25,26,27]
    if not "sim" in kwargs:
        sim        = args.sim
    if not "subsim" in kwargs:
        subsim     = args.subsim
    if not "simsuite" in kwargs:
        simsuite   = args.simsuite
    
    check_if_workin_on_it = True
    overwrite_results = args.overwrite_results
    if run_type==0:
        n_it    = int(n_it) #number of iteration of the PSO run
        n_part  = int(n_part) #number of particles in PSO run
        n_burn  = int(n_burn) #MCMC burn in steps
        n_run   = int(n_run) #MCMC total steps 
    elif run_type ==1:
        print("Test Run")
        overwrite_results = True
        check_if_workin_on_it = False
        n_it   = int(3) #number of iteration of the PSO run
        n_part = int(3) #number of particles in PSO run
        n_run  = int(2) #MCMC total steps 
        n_burn = int(1) #MCMC burn in steps
        if n_lenses>3 or np.isnan(n_lenses):
            print("Resetting n_lenses to 3 because test")
            n_lenses = 3
    elif run_type ==2:
        print("Test Run - longer")
        overwrite_results = True
        check_if_workin_on_it = False
        n_it   = int(100) #number of iteration of the PSO run
        n_part = int(20) #number of particles in PSO run
        n_run  = int(200) #MCMC total steps 
        n_burn = int(100) #MCMC burn in steps
        if n_lenses>2 or np.isnan(n_lenses):
            print("Resetting n_lenses to 1 because long test")
            n_lenses = 2
    else:
        raise RuntimeError("Give a valid run_type or implement it your own")
    
    kw_get_all_gallens = {"sim":sim,
                          "subsim":subsim,
                          "simsuite":simsuite,
                          "snaps":snaps}
    res_dir = get_res_dir(res_dir_base,simsuite,sim,
                          subsim=subsim,run_type=run_type)

    if lenses2model == []:
        print("\nGetting catalogue of lenses 2 model\n###################\n")
        gal_lenses  = get_lenses2model(res_dir=res_dir,
                                       reload=True,
                                       kw_get_all_gallens=kw_get_all_gallens,
                                       n_lenses=np.nan, # has to load all of them anyway
                                       min_thetaE=min_thetaE,
                                       skip_lenses=lenses2skip)
        print("\nCatalogue of lenses 2 model obtained\n###################\n")
        
        for i,gal_lens in enumerate(gal_lenses):  
            print("\nLoading lens "+gal_lens.name)

            lens = LensSystem.from_GalLens(gal_lens,lensFuel=lensfuel)
            lens = setup_lens(lens,
                              res_dir=res_dir,
                              check_if_workin_on_it=check_if_workin_on_it,
                              overwrite=overwrite_results)
            
            if lens is None: #means that someone is workin on it
                continue
            model_lens(lens,kw_model,
                       n_it   = n_it,
                       n_part = n_part,
                       n_burn = n_burn,
                       n_run  = n_run,
                       mask_center=mask_center,
                       gauss_tE_prior=gauss_tE_prior)
            # only do n lenses:
            if i==n_lenses:
                break
    else:
        for i,lens2setup in enumerate(lenses2model):
            lens = setup_lens(lens=lens2setup,
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
            # only do n lenses:
            if i==n_lenses:
                break
