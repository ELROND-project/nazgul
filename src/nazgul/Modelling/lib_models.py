# Copy from model_allLOS.py
# adapted to have all the required components for the specific models

import os
import warnings
import argparse
import importlib
import numpy as np
import sys,dill
from pathlib import Path
from corner import corner
from copy import copy,deepcopy
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.Util import util
from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot

from python_tools.read_fits import load_fits
from python_tools.get_res import load_whatever
from python_tools.tools import mkdir,to_dimless,dict_equal

from nazgul.plot_PL import plot_all
from nazgul.Translator import get_simsuite_code,get_simsuite_from_code,std_sim,std_simsuite,std_subsim
from nazgul.masking import mask_SEAGLE,mask_max_dens,mask_bright_center,resize_mask
from nazgul.mount_doom.cracks_of_doom import LoadLens,get_extents
from nazgul.mount_doom.lens_system import LensSystem
from nazgul.plot_PL import plot_kappamap
from nazgul.stat_lenses import get_all_gallens


from nazgul.lens_part_LOS import get_kw_los
from nazgul.pathfinder import get_sim_dir,results_dir,path_nazgul

# WOI cross-machine lock
from python_tools.tools_WOI import workin_on_it, set_workin_on_it, is_someone_workin_on_it

lens_model_list_def   = ['EPL']
source_model_list_def = ["SERSIC"]
model_res_base      = results_dir/"models/"
#PSO
n_it_std   = 1000
n_part_std = 300
#MCMC
n_burn_std = 700
n_run_std  = 7000

def get_res_dir(res_dir_base,simsuite,sim,subsim=None,run_type=0):
    res_dir = Path(res_dir_base)
    simsuite_code = get_simsuite_code(simsuite)
    sim_dir = _get_sim_dir_name(simsuite_code=simsuite_code,
                            sim=sim,subsim=subsim)
    res_dir = res_dir/sim_dir
    if run_type!=0:
        res_dir = res_dir/"test"
    mkdir(res_dir)
    return res_dir
    
def _get_sim_dir_name(simsuite_code,sim,subsim=None):
    sim_dir = f"{simsuite_code}_{sim}"
    if subsim:
        sim_dir +=f"_{str(subsim)}"
    return sim_dir
    
def _get_sim_dir(gal,simsuite_code=None):
    if simsuite_code is None:
        try:
            simsuite_code =  str(gal.simsuite_code)
        except AttributeError:
            # MONKEY PATCH
            simsuite = gal.simsuite
            simsuite_code = get_simsuite_code(simsuite)
    sim    = str(gal.sim)
    subsim = getattr(gal,"subsim",False)
    sim_dir = _get_sim_dir_name(simsuite_code=simsuite_code,
                            sim=sim,subsim=subsim)
    return sim_dir

def get_model_res_dir(lens,res_dir):
    gal = lens.gallens.Gal
    sim_dir = _get_sim_dir(gal)
    if str(sim_dir) not in str(res_dir):
        res_dir =  Path(res_dir)/sim_dir
    res_dir = Path(f"{str(res_dir)}/snap_{gal.snap}_{lens.name}/")
    return res_dir
    
def get_link_lens_path(lens,res_dir):
    if not hasattr(lens,"model_res_dir"): 
        lens.model_res_dir = get_model_res_dir(lens,res_dir=res_dir)
    return lens.model_res_dir/"link_gallens.pkl"
    
def setup_lens(lens,res_dir,kwargs_source=None,
               _plot=True,
               overwrite=False,
               check_if_workin_on_it=True,
               workin_on_it=True,
               verbose=True):
    lens.model_res_dir = get_model_res_dir(lens,res_dir=res_dir)
    mkdir(lens.model_res_dir)
    # verify that no-one is working on it
    if check_if_workin_on_it:
        if is_someone_workin_on_it(lens.model_res_dir):
            warnings.warn(f"This lens, {lens.name} is being worked on, skipping- if not, delete the {workin_on_it} file:\n{lens.model_res_dir}/{workin_on_it}") 
            return None        
    set_workin_on_it(lens.model_res_dir,wrk = workin_on_it)
    # verify that there aren't previous results
    if not overwrite:
        if Path(f"{lens.model_res_dir}/kw_res.dll").is_file():
            warnings.warn(f"This lens, {lens.name} has already results, and I should not overwrite them. If that's not what you want, set overwrite=True") 
            return None
            
    lens.setup()
    # Note: the following is only used for plotting
    # and mask definition (to find the bright center of the image)
    Sim = lens.get_Sim() 
    lens.image_sim = lens.get_lensed_image(Sim=Sim,kwargs_source=kwargs_source, unconvolved=False)
    
    if verbose:
        print(f"Saving modelling results in {lens.model_res_dir}") 
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
    if _plot:
        plot_all(lens,skip_caustic=True)
        
    # create link to lens
    src = lens.gallens.pkl_path
    dst = get_link_lens_path(lens,res_dir=res_dir)
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


######################################
# kwargs_of realistic HST observations used to simulate the "observed" images 
kwargs_band_HST_camera = {
    'read_noise': 2,                      # Readout noise
    'pixel_scale':0.065,                  # 0.065 F160W after drizzling (could also do 0.08 to be more conservative
    'ccd_gain': 2.35,                     # averaged over the 4 amplifier (does not matter)
}
# inspired by F160W taken from idgc07c[nlpq]q_flt.fits 
sky_count      = 0.11 # after drizzling, clip outliers and take median  (e-/sec)
exp_time_1exp  = 550 # ~average over 4 exposures
num_exposures  = 4   #  
# taken from https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
# the following ZP computation is also correct, returns 25.937 and the error is 0.008 so it's consistent
# PHOTFLAM is the inverse sensitivity at the infinite aperture, taken from
#PHOTFLAM_f160w = 1.9429e-20 
#PHOTPLAM_f160w = 15369.18
#ZP_AB_f160w = -2.5*np.log10(PHOTFLAM_f160w) - 21.1 - 5*np.log10(PHOTPLAM_f160w) + 18.6921
ZP_AB_f160w    = 25.941 

sky_brightness = -np.log10(sky_count) * 2.5 + ZP_AB_f160w
kwargs_band_HST_obs = {
    'sky_brightness':sky_brightness,      # ~21.5 mag
    'exposure_time':exp_time_1exp,        # average time for 1 exposure
    'magnitude_zero_point':ZP_AB_f160w,   # ~25.9 mag
    'num_exposures': num_exposures,       # stnd n* of exposures combined in drizzing
    'psf_type':'PIXEL'                    # kernel to be provided later on
}
class band_HST():
    """
    Inspired by class HST in lenstronomy.SimulationAPI.ObservationConfig.py 
    """
    def __init__(self,
                 kwargs_camera = kwargs_band_HST_camera,
                 kwargs_obs    = kwargs_band_HST_obs):
        self.camera = kwargs_camera
        self.obs = kwargs_obs
        # obtained from https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/psf
        self.psf_path =  Path(f"{path_nazgul}/ObsData/HST/WFC3/F160W/PSFSTD_WFC3IR_F160W.fits")
    def kwargs_single_band(self):
        """
        :return: merged kwargs from camera and obs dicts
        """
        kwargs = util.merge_dicts(self.camera, self.obs)
        return kwargs
    def get_kwargs_psf(self,pssf_effective=5):
        if np.abs(int(pssf_effective)-pssf_effective)>1e-7:
            raise RuntimeError("We should have an integer pssf_effective") 
        pssf_effective = int(pssf_effective)

        psf_path = self.psf_path
        
        delta_pix_native = 0.128          # arcsec/pix, native F160W
        pssf_orig        = 4              # STScI PSF supersampling vs native
        delta_pix_psf    = delta_pix_native / pssf_orig   # = 0.032 arcsec/pix

        delta_pix_band   = self.camera["pixel_scale"]     # = 0.08 arcsec/pix (lenstronomy target)
        pssf_band        = delta_pix_native / delta_pix_band # = 1.6
        # pssf is the ratio of image pixel scale to PSF pixel scale,
        # as lenstronomy expects. The PSF must be zoomed to achieve this.
        # Current PSF pixel scale: delta_pix_psf = 0.032 "/pix
        # Target PSF pixel scale for given pssf: delta_pix_band / pssf
        zoom_factor = pssf_effective*pssf_band/pssf_orig        
    
        psf = load_fits(psf_path)[-2]
        psf = _positivise_psf(psf)
        
        if zoom_factor<1:
            warnings.warn("PSSF should be set s.t. zoom_factor>1")
        
        if not np.isclose(zoom_factor, 1):
            psf = zoom(psf, zoom_factor, order=3)
            psf = _positivise_psf(psf)
            
        kwargs_psf = {
            "psf_type":"PIXEL",
            "kernel_point_source_normalisation":True,
            "kernel_point_source": psf,
            "point_source_supersampling_factor": pssf_effective
        }
        return kwargs_psf
        
def setup_sim_obs(lens, band_str="HST_F160W", pssf_effective=5):
    if band_str == "HST_F160W":
        band = band_HST() 
    else:
        raise RuntimeError("Pragma no cover: to implement other bands and PSFs")

    kwargs_psf = band.get_kwargs_psf(pssf_effective=pssf_effective)

    return lens.sim_multi_band_list(band=band, kwargs_psf=kwargs_psf)
    
def _positivise_psf(psf):
    if np.any(psf<0):
        warnings.warn("Some negative pixels in the PSF")
        i_psf0,j_psf0 = np.where(psf<0)
        if i_psf0.shape[0]*100/psf.ravel().shape[0]>30:
            raise ValueError("PSF has more than 30% negative pixels, something is not right")
        warnings.warn("Setting minimum value for negative PSF pixels")
        psf[psf<0] = np.min(psf[psf>0])/100
    # renormalise it aftwards
    psf /= psf.sum()
    return psf


def get_kw_lens_mask(lens,image_obs):
    # masking inner and outer of thetaE -> nope, follow SEAGLE approach
    #image = kwargs_data["image_data"]
    mask_SE = mask_SEAGLE(lens,image=image_obs) 
    mask_HD = mask_bright_center(lens)
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
    
    
def get_lens_mask(lens,image_obs,plot_mask=True):
    kw_mask = get_kw_lens_mask(lens,image_obs)
    mask_SE	=	kw_mask["mask_SE"]
    mask_HD	=	kw_mask["mask_HD"]
    mask_LD	=	kw_mask["mask_LD"]
    mask_SE_HD	=	kw_mask["mask_SE_HD"]
    mask_comb_HD	=	kw_mask["mask_comb_HD"]

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
        mask_nan = copy(mask_comb_HD)
        mask_nan[np.where(mask_nan==0)] = np.nan
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

def add_gaussian_tE_prior(lens,kwargs_likelihood,sig_tE=0.5):
    print("Adding gaussian prior for the theta_E")
    tE = lens.gallens.thetaE.value
    print(f"centered on 'true_value' tE={np.round(tE,3)} and sigma={np.round(sig_tE,3)}")
    kwargs_likelihood["prior_lens"] = [[0,"theta_E",tE,sig_tE]]
    return kwargs_likelihood
        

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
                          arrow_size=0.02, cmap_string="gist_heat",
                          image_likelihood_mask_list=kwargs_likelihood["image_likelihood_mask_list"])
    return modelPlot

def load_mblo(res_dir):
    nm_mblo = f"{res_dir}/multi_band_list_out.dll"
    multi_band_list_out = load_whatever(nm_mblo)
    return multi_band_list_out
    
def plot_model_plot(multi_band_list_out,kwargs_model,kwargs_result,kwargs_likelihood,res_dir,plot_point_sources=False):
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
    nm = f'{res_dir}/mass_model.pdf'
    plt.savefig(nm)
    plt.close(f)
    print(f"Saving {nm}")
    
    f, axes = plt.subplots(1,2, figsize=(8, 4), sharex=False, sharey=False)
    
    modelPlot.decomposition_plot(ax=axes[0], band_index=band_index_plot, text='Source light', source_add=True, unconvolved=True)#,v_min=band_i.vmin,v_max=band_i.vmax)
    modelPlot.decomposition_plot(ax=axes[1], band_index=band_index_plot, text='Source light convolved', source_add=True)#,v_min=band_i.vmin,v_max=band_i.vmax)
    
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    nm = f'{res_dir}/light_model.pdf'
    plt.savefig(nm)
    plt.close(f)
    print(f"Saving {nm}")
    
    reduced_chi2 = get_red_chi2(modelPlot=modelPlot,verbose=True)    
    #Normalised plot
    f, axes = plt.subplots(figsize=(10,7))
    modelPlot.normalized_residual_plot(ax=axes,v_min=-3, v_max=3,text=r"Norm. Resid $\chi^2_{red.}$="+str(np.round(reduced_chi2,2)))
    nm = f'{res_dir}/normalised_residuals.png'
    plt.savefig(nm)
    print(f"Saving {nm}")
    plt.close(f)
    
    #Caustics
    f, axes = plt.subplots(figsize=(10,7))
    modelPlot.source_plot(ax=axes, deltaPix_source=0.01, numPix=1000, with_caustics=True)
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    nm = f'{res_dir}/caustics.png'
    plt.savefig(nm)
    print(f"Saving {nm}")
    plt.close(f)
    if plot_point_sources:
        f, axes = plt.subplots(figsize=(10,7))
        modelPlot.decomposition_plot(ax=axes, text='Point source position', source_add=False, \
                        lens_light_add=False, point_source_add=True, v_min=-1, v_max=1)
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
if __name__=="__main__":
    raise RuntimeError("Do not run this - kept only as a reference")
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Simulate and model the lens")
    parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Type of run: 
        0 = standard, PSO_it = {n_it_std} PSO_prt = {n_part_std} MCMCb = {n_burn_std} MCMCr = {n_run_std}  
        1 = test run  PSO_it = 3      PSO_prt = 3      MCMCb = 1     MCMCr = 2 
       (PSO_it: PSO iterations, PSO_prt: PSO particles, MCMCr: MCMC run steps, MCMCb: MCMC burn in steps)\n""")
    parser.add_argument('-nl','--n_lenses',type=int,dest="n_lenses",default=5,help=f"Number of lenses to model")
    parser.add_argument('-mtE','--min_thetaE',type=float,dest="min_thetaE",default=None,help=f"Min theta_E for the gal to be considered a lens")
    parser.add_argument('-snap','--snap',nargs="+",type=int,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")

    args       = parser.parse_args()
    run_type   = args.run_type
    n_lenses   = args.n_lenses
    min_thetaE = args.min_thetaE 
    snaps      = args.snaps #[25,26,27]
    sim        = args.sim
    subsim     = args.subsim
    simsuite   = args.simsuite
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

    lenses2skip = ["Sub_Gn22SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj1","Sub_Gn3SGn0_Npix200_PartAS_Prj2"]
    #["Sub_Gn3SGn0_Npix200_PartAS_Prj1","Sub_Gn22SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj2"]
    kw_get_all_gallens = {"sim":sim,
                          "subsim":subsim,
                           "simsuite":simsuite,
                            "snaps":snaps}
    res_dir = get_res_dir(res_dir_base,simsuite,sim,
                          subsim=subsim,run_type=run_type)
    gal_lenses  = get_lenses2model(res_dir=res_dir,
                                   reload=True,
                                   kw_get_all_gallens=kw_get_all_gallens,
                                   n_lenses=n_lenses,
                                   min_thetaE=min_thetaE,
                                   skip_lenses=lenses2skip)
    for gal_lens in gal_lenses: 
        print("Loading lens "+gal_lens.name+"\n")
        print("Adding LOS effects")
        kw_los = get_kw_los()
        kw_add_lenses = {"lens_model_list":["LOS"],
                        "kwargs_lens":[kw_los]}
        lens = LensSystem.from_GalLens(gal_lens,kwargs_add_lenses=kw_add_lenses)

        lens = setup_lens(lens,res_dir=res_dir,check_if_workin_on_it=True) #change it with res_dir_base of the given model
        if lens is None:
            continue
        plot_kappamap(lens.gallens.kappa_map, 
                      extent_kpc=lens.gallens.kw_extents["extent_kpc"],
                      savename=f"{res_dir}/kappa_gal.png")
        multi_band_list = setup_sim_obs(lens)
        image_obs = multi_band_list[0][0]["image_data"]
        
        # models
        kwargs_model = {'lens_model_list': lens_model_list,
                        'source_light_model_list': source_model_list}
        
        
        kwargs_likelihood = get_kwargs_likelihood(lens,image_obs=image_obs)
    
        kwargs_data_joint = {'multi_bakw[nd_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    
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
                    "kw_add_lenses":       kw_add_lenses
                   }
        nm_input = f"{res_dir}/kw_input.dll"
        save_data(kw_input,nm_input,"input")
        
        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()
        print("kwargs_result",kwargs_result)
        nm_res = f"{res_dir}/kw_res.dll"
        save_data(kwargs_result,nm_res,"result output")
        
        
        # we need to extract the updated multi_band_list object since the coordinate shifts were updated in the kwargs_data portions of it
        multi_band_list_out = fitting_seq.multi_band_list
        nm_mblo = f"{res_dir}/multi_band_list_out.dll"
        save_data(multi_band_list_out,nm_mblo,"output multiband list")


        plot_model_plot(multi_band_list_out,kwargs_model,kwargs_result,kwargs_likelihood,res_dir=res_dir)
        
        if run_type==0:
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
        plt.close()
        set_workin_on_it(lens.model_res_dir,wrk = False)
    