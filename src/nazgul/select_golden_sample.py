import os,sys
import dill
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from copy import copy, deepcopy
from scipy.optimize import least_squares
from chainconsumer import Chain,ChainConsumer

from python_tools.get_res import load_whatever
from python_tools.tools import silencer,to_uid_base64

from nazgul.Translator import std_kw_sim
from nazgul.mount_doom.lens_system import LensSystem
from nazgul.combined_modelling_results import get_res_dir,get_emcee,get_modelling_prms,get_partial_chain,_convert_shear_params,name_models #,get_lensfuel

from nazgul.Modelling.lib_models import get_red_chi2,get_model_plot

from nazgul.Modelling.pathfinder import get_model_res_dir_from_gallens #,get_model_res_dir

def load_cat_gallens_model(model,kw_sim=std_kw_sim):
    model_res_dir = get_res_dir(model,kw_sim=kw_sim)
    nm_cat  = model_res_dir/"cat_lens2model.dll"
    cat_gal = load_whatever(nm_cat)
    return cat_gal
    

def load_gallens_model(model,kw_sim=std_kw_sim):
    cat_gal = load_cat_gallens_model(model,kw_sim=kw_sim)
    gal_lenses = [load_whatever(ctg) for ctg in cat_gal["lens_cat"]]    
    return gal_lenses
        
# Verify that the isofit is a good 2D fit
def _verify_type(type="psi"):
    allowed_type =  ["psi","dens","kappa","pot"]
    if type not in allowed_type:
        raise ValueError(f"type must be one of {allowed_type}") 
    # monkey patch correction for wrong naming
    if type == "kappa":
        type = "dens"
    if type == "pot":
        type = "psi"
    return type
    
def get_isofile_name(gallens,type="psi"):
    type = _verify_type(type=type)
    nm_iso_file = str(gallens.pkl_path.parent/f"kw_res_iso{type}_prj{gallens.proj_index}.dll")
    return nm_iso_file
    
def get_isofit(gallens,type="psi"):
    type = _verify_type(type=type)
    # try to load prev. res.
    nm_isofile = get_isofile_name(gallens,type=type)
    isofile = glob(nm_isofile)
    if len(isofile)==1:    
        isofit = load_whatever(isofile[0])["isofit"]
    elif len(isofile)==0:
        # if failed, recompute it
        isofit = recompute_isofit(lens,type=type)
    return isofit
    
def recompute_isofit(gallens,type="psi"):
    raise RuntimeError("to implement")
    type = _verify_type(type=type)
    nm_isopsi_file  = get_isofile_name(gallens,type=type)
    ###
    return isofit_psi

def get_chi2_isofit(isofit):
    # note: we do not have sigma, so this is not really a chi2
    mod = copy(isofit["model"])
    mod[mod==0.0] = np.nan
    chi2 = np.nansum((isofit["map"]- mod)**2) 
    return chi2
    
## Verify that the D2 fit is a good fit 

def get_E_sqrt_inv(isolist):
    # E^(-1/2) = 1/sqrt(E) = A Q A'
    # A  = (cos(phi) -sin(phi)) (sin(phi)  cos(phi)) 
    # Q  = (q^-1/2    0       ) ( 0        q^1/2)
    # A'  = (cos(phi)  sin(phi)) (-sin(phi) cos(phi)) 
    phi    = isolist.pa # already in rad right?
    q_sqrt = np.sqrt(1-isolist.eps)  # sqrt(axis ratio)
    E_sqrt_inv_list = []
    for i in range(len(q_sqrt)):
        Q_mtrx = np.array([[1/q_sqrt[i],0],
                           [0,q_sqrt[i]]])
        A_mtrx = np.array([[np.cos(phi[i]), -np.sin(phi[i])],
                           [np.sin(phi[i]),np.cos(phi[i])]])
        A_prime_mtrx = np.array([[np.cos(phi[i]), np.sin(phi[i])],
                                 [-np.sin(phi[i]),np.cos(phi[i])]])
        E_sqrt_inv_list.append(A_mtrx.dot(Q_mtrx.dot(A_prime_mtrx)))
    return np.asarray(E_sqrt_inv_list)

def get_xi(isolist):
    """Get circularised radius from isolist"""
    q = 1-isolist.eps
    #b = isolist.sma*q
    xi = isolist.sma*np.sqrt(q) # = np.sqrt(a*b)
    return xi
    
def get_D_theo(isolist,N2_D,phi_D):
    E_sqrt_inv = get_E_sqrt_inv(isolist)
    phi_D_arr  = np.array([np.cos(phi_D),np.sin(phi_D)])
    return np.sqrt(N2_D) * E_sqrt_inv.dot(phi_D_arr)

def get_x_vec(isolist,N2_D,phi_D):
    D  = get_D_theo(isolist,N2_D,phi_D)
    return _get_x_vec(isolist,D)
    
def _get_x_vec(isolist,D):
    xi = get_xi(isolist)
    return xi[:,np.newaxis]*D

"""def f_fit_D(N2_D_phi_D,isolist):
    N2_D,phi_D = N2_D_phi_D
    xy         = get_x_vec(isolist,N2_D,phi_D)
    xy_iso     = np.array([isolist.x0 - isolist.x0[0],
                           isolist.y0 - isolist.y0[0]]).T
    diff       = xy-xy_iso
    return diff.flatten()
"""

def f_fit_xy(N2_D_phi_D,isolist):
    N2_D,phi_D = N2_D_phi_D
    D  = get_D_theo(isolist,N2_D,phi_D)
    xi = get_xi(isolist)
    xy_fit = xi[:,np.newaxis]*D 
    x_fit,y_fit = xy_fit.T
    xy_iso      = np.column_stack([isolist.x0 - isolist.x0[0],
                           isolist.y0 - isolist.y0[0]])
    diff = xy_fit - xy_iso 
    return diff.ravel()

def get_D_fit_from_isolist(isolist):
    # a simple least square fit
    results = []
    diffs2  = []
    def _f_fit_xy(N_phi):
        return f_fit_xy(N_phi,isolist)
    for phi0 in np.linspace(0,2*np.pi-1e-5,6):
        res = least_squares(_f_fit_xy,[1,phi0], # initial guess at starting point
                     bounds=([1e-8, 0], [np.inf, 2*np.pi])
                    )
        diffs2.append(np.sum(res.fun**2))
        results.append(res.x)
    result = results[np.argmin(diffs2)]
    N2_D,phi_D = result
    D = get_D_theo(isolist,N2_D,phi_D)
    return D
    
def get_xy_fit_from_isolist(isolist):
    D   = get_D_fit_from_isolist(isolist)
    xy  = _get_x_vec(isolist,D)
    x,y = xy.T
    return x,y
"""
def get_D_fit_from_isolist(isolist):
    # a simple least square fit
    result = leastsq(f_fit_D,[0.5,1.0], # initial guess at starting point
                     args = (isolist) # alternatively you can do this with closure variables in f if you like
                    )
    N2,phi_D = result[0]
    D = get_D_theo(isolist,N2,phi_D)
    return D
"""
def get_RE_pix(gallens):
    #RE_pix = lens.thetaE.value*lens.pixel_num/(2*lens.radius.value)
    RE_pix = gallens.thetaE.value/gallens.deltaPix.value
    return RE_pix
    
def _get_cropped_isolist_RE(isolist,RE_pix,frac_tE=0.1):
    isocrop = deepcopy(isolist)

    sma_RE = isolist.sma/RE_pix
    i_crop = np.argmin(np.abs(sma_RE-frac_tE))

    isocrop.__dict__["_list"] = isocrop.__dict__["_list"][i_crop:]
    return isocrop

def get_isolist_cropped(isolist,gallens,frac_tE=0.1):
    RE_pix = get_RE_pix(gallens)
    isolist_cropped = _get_cropped_isolist_RE(isolist=isolist,RE_pix=RE_pix,frac_tE=frac_tE)
    return isolist_cropped
    
def get_chi2_Dfit(isolist,gallens,sigma_r = 2,frac_tE=0.1):
    # frac_tE: define the isofotes to discard i.e. sma(isophote_cropped) > frac*tE 
    # sigma_r: as an indication of the expected radial error in position of the center
    
    isolist_cropped = get_isolist_cropped(isolist,gallens,frac_tE=frac_tE)

    x,y = get_xy_fit_from_isolist(isolist_cropped)
    x_iso = np.array([isolist_cropped.x0 - isolist_cropped.x0[0]])
    y_iso = np.array([isolist_cropped.y0 - isolist_cropped.y0[0]])
    Dr    = np.hypot(x-x_iso,y-y_iso)
    """
    xi = get_xi(isolist_cropped)
    # plot comparison
    plt.scatter(xi,x)
    plt.scatter(xi,x_iso)
    plt.show()
    plt.scatter(xi,y)
    plt.scatter(xi,y_iso)
    plt.show()
    plt.scatter(xi,Dr)
    plt.show()
    """
    chi2 = np.sum(Dr**2)/sigma_r**2
    return chi2
    # the reduced chi2 is too small?
    #red_chi2 = chi2/(len(x)-2)
    #return red_chi2

# Verifying that the axis ratio is well behaved
def get_chi2_qfit(isolist,gallens,sigma_q = 2,frac_tE=0.1):
    warnings.warn("Not implemented, for now passing with a warning")
    return 0

# Verify chi2 of lens model
@silencer
def get_chi2_lens_model(gallens,model,kw_sim=std_kw_sim,verbose=False):
    # verify that said lens was modelled
    res_dir       = get_res_dir(model,kw_sim)
    model_res_dir = get_model_res_dir_from_gallens(gallens,res_dir)
    #lensFuel          = get_lensfuel(model=model)
    #lens              = LensSystem.from_GalLens(gallens,lensFuel=lensFuel)
    #lens.setup()
    #model_res_dir     = get_model_res_dir(lens,res_dir=res_dir)
    try:
        modelPlot     = get_model_plot(res_dir=model_res_dir)
    except FileNotFoundError:
        return np.nan
    reduced_chi2  = get_red_chi2(modelPlot,verbose=False)
    return reduced_chi2


# tmp function, to improve

std_important_params= ["theta_E","e1_lens0","e2_lens0","gamma_lens0", # main lensing params
                       "gamma_ext","psi_ext", # any shear component (if present)
                       "gamma_los"]# any los components
def get_golden_sample(gal_lenses,
                      model,
                      min_thetaE=0.5,
                      max_thetaE=1.5,
                      min_chi2_psi_2D = 1.5,
                      min_chi2_psi_drift = 2.5,
                      min_chi2_psi_q = 1.5,
                      min_chi2_lens = 1.5,
                      frac_tE = 0.1,
                      important_params=std_important_params,
                      kw_sim=std_kw_sim,
                      verbose=True
                     ):
    # min_thetaE,max_thetaE: self-explanatory
    # min_chi2_psi_2D: min chi2 for 2D iso-potential fit
    #  min_chi2_psi_drift: min chi2 drift x0,y0 fit of the isopotential fit
    # min_chi2_psi_q: min chi2 q fit of the isopotential fit
    # min_chi2_lens: min chi2 of the lens model
    # frac_tE: fraction of theta_E to ignore for isopot. parameters

    # select range of theta_E
    tE = np.asarray([gl.thetaE.value for gl in gal_lenses])
    gal_lenses_crop = np.asarray(gal_lenses)[(tE>min_thetaE) & (tE<max_thetaE)]

    if verbose:
        n_tot_lens = len(gal_lenses)
        n_tot_crop = len(gal_lenses_crop)
        rt_crop    = np.round(n_tot_crop*100/n_tot_lens,2)
        n_discd_tE = n_tot_lens-n_tot_crop
        
        print(f"N* of selected lenses via theta_E crop: {n_tot_crop}/{n_tot_lens} = {rt_crop}%")
        
    """
    # To plot the hist of tE and the selected ones
    _,bins,_ = plt.hist(tE,bins=40,label=r"N$_{\rm{lenses}}$="+str(len(tE)))
    
    tE_crop = tE[(tE>min_tE) & (tE<max_tE)]
    plt.hist(tE_crop,bins=bins,alpha=.5,label=str(min_tE)+r"<$\theta_E$<"+str(max_tE) +": "+str(len(tE_crop)))
    plt.xlabel(r'$\theta_E$ ["]')
    plt.title("Lenses modelled")
    plt.legend()
    plt.show()
    """
    
    golden_lens_sample = []
    n_discd_Dfit = 0
    n_discd_qfit = 0
    n_discd_2D   = 0
    n_discd_lens = 0
    n_discd_nml  = 0
    n_discd_ncst = 0
    gallens_disc_Dfit = []
    
    for gallens in gal_lenses_crop:
        gallens.unpack(verbose=False)
        isofit_psi  = get_isofit(gallens,type="psi")
        chi2_psi_2D = get_chi2_isofit(isofit_psi)
        if chi2_psi_2D>min_chi2_psi_2D:
            n_discd_2D+=1
            continue
        """
        # for now, we ignore kappa map bc too small
        isofit_kappa = get_isofit(lens,type="kappa")
        chi2_kappa   = get_chi2_isofit(isofit_kappa)
        if chi2_kappa>2:
            continue
        """
        chi2_Dfit = get_chi2_Dfit(isofit_psi["isolist"],
                                  gallens=gallens,
                                  frac_tE=frac_tE,
                                  sigma_r=2)
        if chi2_Dfit>min_chi2_psi_drift:
            n_discd_Dfit+=1
            gallens_disc_Dfit.append(gallens)
            continue
    
        chi2_qfit = get_chi2_qfit(isofit_psi["isolist"],
                                  gallens=gallens,
                                  frac_tE=frac_tE,
                                  sigma_q=.1)
        if chi2_qfit>min_chi2_psi_q:
            n_discd_qfit+=1
            continue
    
        chi2_lens = get_chi2_lens_model(gallens=gallens,
                                        model=model,
                                       kw_sim=kw_sim,
                                       verbose=False)
        if np.isnan(chi2_lens):
            print(f"{gallens.name} not yet lens-modelled")
            n_discd_nml+=1
            continue
            
        if chi2_lens>min_chi2_lens:
            n_discd_lens+=1
            continue
            
        if important_params is not None:
            unconstrained_imp_prms = get_unconstrained_important_params(gallens,
                                                                   model=model,
                                                                   important_params=important_params,
                                                                   kw_sim=kw_sim)
            if len(unconstrained_imp_prms)>0:
                print(f"{gallens.name} discarded because it has important unconstrained params: \n{unconstrained_imp_prms}")
                n_discd_ncst +=1
                continue
                
        if verbose:
            print(f"Selected {gallens.name}")
        golden_lens_sample.append(gallens)
    if verbose:
        n_golden   = len(golden_lens_sample)
        rt_golden  = np.round(n_golden*100/n_tot_lens,2) 
        tot_disc   = n_discd_tE+n_discd_2D+n_discd_Dfit+n_discd_lens+n_discd_nml+n_discd_ncst
        print(f"N* of selected golden lenses: {n_golden}/{n_tot_lens} = {rt_golden}%")
        print("Discarded due to:")
        print("small thetaE, bad fit 2D isopot. map , bad drift fit, bad lens model, no lens mod., unconstr.= total discarded")
        print(n_discd_tE, n_discd_2D,n_discd_Dfit,n_discd_lens,n_discd_nml,n_discd_ncst,"=",
             tot_disc)
        print(np.round(n_discd_tE*100/tot_disc,1),"% ",
              np.round(n_discd_2D*100/tot_disc,1),"% ",
              np.round(n_discd_Dfit*100/tot_disc,1),"% ",
              np.round(n_discd_lens*100/tot_disc,1),"% ",
              np.round(n_discd_nml*100/tot_disc,1),"% ",
              np.round(n_discd_ncst*100/tot_disc,1),"% ",
             )
        
        assert tot_disc==(n_tot_lens-n_golden)
    return golden_lens_sample
    
def get_unconstrained_params(chain, important_params=None):
    """Return the subset of important_params that chainconsumer can't bound for this chain."""
    chainconsumer = ChainConsumer()
    chainconsumer.add_chain(chain)
    summary = chainconsumer.analysis.get_summary(chains=[chain])[chain.name]  # dict: param -> ChainSummary/tuple
    bad = []
    if important_params is None:
        # consider all of them
        important_params = list(summary.keys())
    for p in important_params:
        bound = summary.get(p)
        if bound is None:
            bad.append(p)
            continue
        low = bound.lower
        high = bound.upper
        if low is None or high is None:
            bad.append(p)
    return bad

"""
from nazgul.combined_modelling_results import get_model_module
    model_module = get_model_module(model_name)
    # awful way to trick my own code :/
    class faketE:
        def __init__(self):
            self.value=3
    class fakegallens:
        def __init__(self):
            self.thetaE = faketE()
    class fakelens:
        def __init__(self):
            self.gallens = fakegallens()
    kwargs_params = model_module.get_kwargs_params(fake_lens())
    kwargs_params["lens_model"]
    """

std_important_params= ["theta_E","e1_lens0","e2_lens0","gamma_lens0", # main lensing params
                       "gamma_ext","psi_ext", # any shear component (if present)
                       "gamma1_los","gamma2_los","gamma_los"]# any los components
def get_present_important_params(model,
                         important_params=std_important_params,
                         kw_sim=std_kw_sim):
    params = get_modelling_prms(model,kw_sim=kw_sim)
    present_important_params = []
    for ip in important_params:
        for prm in params:
            if ip.lower() in prm.lower():
                present_important_params.append(prm)
    #present_important_params = _convert_shear_params(model=model,
    #                                                 params=present_important_params)
    return present_important_params

def get_unconstrained_important_params(gallens,
                                        model,
                                       important_params=std_important_params,
                                       kw_sim=std_kw_sim):
    
    present_imp_prms = get_present_important_params(model,important_params,kw_sim=kw_sim)
    partial_chain    = get_partial_chain(lens=gallens,
                                      model=model,
                                      wanted_param_list=present_imp_prms,
                                      do_prettify_prm_nm=False)
    chain_imp_prms = Chain(samples=partial_chain, name=f"{str(gallens.name)}_{str(model)}", 
              shade=True, color='#2c7fb8', smooth=1, bins=30,
              shade_gradient = 0.4, linewidth=3.0)
    not_constrained_prms = get_unconstrained_params(chain_imp_prms)
    return not_constrained_prms

# Default values
min_thetaE = 0.5
max_thetaE = 1.5
min_chi2_psi_2D=1.5
min_chi2_psi_drift = 2.5
min_chi2_psi_q = 1.5
min_chi2_lens = 1.5
frac_tE = 0.1

def get_GS_cat(model,kw_sim=std_kw_sim):
    res_dir  = get_res_dir(model,kw_sim=kw_sim)
    gs_path  = glob(f"{res_dir}/kw_golden_sample_*.dll")
    if len(gs_path)==1:
        return load_whatever(gs_path[0])
    elif len(gs_path)==0:
        raise RuntimeError(f"No golden path catalogues available for model {model}. First run {sys.argv[0]}")
    elif len(gs_path)>1:
        raise RuntimeError(f"Pragma no cover: multiple available catalogues:\n{gs_path}")
        
def get_GS_names(kw_GS):
    """
    Convert kwargs_GS into list of names compatible with Profile Lens
    """
    golden_sample_paths = kw_GS["list_golden_sample"]
    golden_sample_names = []
    for p in golden_sample_paths:
        nm_split_sub = str(p).split("Sub_Lens_")[1]
        nm,ind = nm_split_sub.split("_Prj") 
        gsn = nm+"_Prj"+ind[0]
        golden_sample_names.append(gsn)
    return golden_sample_names
    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Create golden sample of lenses based on parameters and modelling results ")
    parser.add_argument('-m','--model',type=str,
                        dest="model",
                        help=f"Name of type of model - accepted: {name_models}",required=True)
    parser.add_argument('-rc','--recompute',default=False,action="store_true",
                        dest="recompute",
                        help=f"If golden sample already present, recompute - default False")
    parser.add_argument('-mintE','--min_thetaE', dest="min_thetaE",
                        default=min_thetaE, type=float,
                        help=f"Min theta_E in arcsec - default {min_thetaE}")
    parser.add_argument('-maxtE','--max_thetaE', dest="max_thetaE",
                        default=max_thetaE, type=float,
                        help=f"Max theta_E in arcsec - default {max_thetaE}")
    parser.add_argument('-min_chi2_psi_2D','--min_chi2_psi_2D', dest="min_chi2_psi_2D",
                        default=min_chi2_psi_2D, type=float,
                        help=f"Min chi^2 for psi isofitting routine: 2D - default {min_chi2_psi_2D}")
    parser.add_argument('-min_chi2_psi_drift','--min_chi2_psi_drift', dest="min_chi2_psi_drift",
                        default=min_chi2_psi_drift, type=float,
                        help=f"Min chi^2 for psi isofitting routine: drift - default {min_chi2_psi_drift}")
    parser.add_argument('-min_chi2_psi_q','--min_chi2_psi_q', dest="min_chi2_psi_q",
                        default=min_chi2_psi_q, type=float,
                        help=f"Min chi^2 for psi isofitting routine: axis ratio q - default {min_chi2_psi_q}")
    parser.add_argument('-min_chi2_lens','--min_chi2_lens', dest="min_chi2_lens",
                    default=min_chi2_lens, type=float,
                    help=f"Min chi^2 for lens modelling - default {min_chi2_lens}")
    parser.add_argument('--frac_tE', dest="frac_tE",
                default=frac_tE, type=float,
                help=f"Define the isofotes to discard i.e. sma(isophote_cropped) > frac*tE - default {frac_tE}")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_kw_sim["sim"],help=f"Simulation name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_kw_sim["subsim"],help=f"Sub-Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_kw_sim["simsuite"],help=f"Simulation suite name")

    args     = parser.parse_args()
    model    = args.model
    kw_sim   = {"sim":args.sim,
               "subsim":args.subsim,
               "simsuite":args.simsuite}
    res_dir  = get_res_dir(model,kw_sim=kw_sim)
    base     = to_uid_base64(args.__dict__)
    nm_smpl  = f"{res_dir}/kw_golden_sample_{base}.dll"
    if not args.recompute and os.path.exists(nm_smpl):
        print(f"Golden sample already computed for this model and parameters: {nm_smpl}\nIf you want to recompute it, run this script with -recompute") 
    
    frac_tE = args.frac_tE
    min_thetaE = args.min_thetaE
    max_thetaE = args.max_thetaE
    min_chi2_lens = args.min_chi2_lens
    min_chi2_psi_q = args.min_chi2_psi_q
    min_chi2_psi_2D = args.min_chi2_psi_2D
    min_chi2_psi_drift = args.min_chi2_psi_drift
    
    gal_lenses = load_gallens_model(model=model)
    golden_sample = get_golden_sample(gal_lenses,
                                      model=model,
                                      min_thetaE=min_thetaE,
                                      max_thetaE=max_thetaE,
                                      min_chi2_psi_2D = min_chi2_psi_2D,
                                      min_chi2_psi_drift = min_chi2_psi_drift,
                                      min_chi2_psi_q = min_chi2_psi_q,
                                      min_chi2_lens = min_chi2_lens,
                                      frac_tE = frac_tE,
                                      important_params=std_important_params,
                                      verbose=True)
    list_nms_golden_sample = [g.pkl_path for g in golden_sample]
    kw_golden_sample  = {"list_golden_sample":list_nms_golden_sample,
                         "kw_args":args.__dict__}
    with open(nm_smpl,"wb") as f:
        dill.dump(kw_golden_sample,f)
    print(f"Saved {nm_smpl}")

