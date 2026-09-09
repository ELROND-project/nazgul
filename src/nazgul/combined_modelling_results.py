# Try to generalise it
import gc

import os,sys
import argparse
import warnings
import tracemalloc
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from textwrap import wrap
from copy import copy,deepcopy
from importlib import import_module

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from chainconsumer import Chain #, ChainConsumer
from chainconsumer.plotting import plot_contour,plot_dist #,plot_truths

from lenstronomy.Util.param_util import shear_polar2cartesian,ellipticity2phi_q

from python_tools.tools import short_SciNot
from python_tools.get_res import load_whatever
from python_tools.tools_WOI import is_someone_workin_on_it

from nazgul.lens_part_LOS import get_kw_los
from nazgul.mount_doom.cracks_of_doom import LoadLens
from nazgul.Translator import std_sim,std_simsuite,std_subsim,get_simsuite_from_code
# debugging memory leak tools 
from python_tools.memory_leak_tools import log_memory,log_top_allocs
from nazgul.Modelling.lib_models import get_red_chi2,get_model_plot,model_res_base

matplotlib.use('Agg') 

# thank you Nat!
green        = ['#a6dba0','#5aae61','#1b7837']
purple       = ['#c2a5cf', '#9970ab', '#762a83']
analogous    = ['#a0c3db', '#dbb7a0']
warm         = ['#fdcc8a', '#fc8d59', '#d7301f']
cool         = ['#41b6c4', '#2c7fb8', '#253494']

colors_Nat = {"green":green,
              "purple":purple,
              "analogous":analogous,
              "warm":warm,
              "cool":cool}

#plt.style.use('sanglier')
plt.rcParams.update({'font.size': 10})
# scale of figure
scale_fig = 7

def shear_stdev(gamma, gamma1, gamma2, covmat):
    '''
    computes standard deviation on magnitude of gamma_LOS from covariance matrix
    '''
    first_term = ((gamma1.array[1]/gamma)**2.)*covmat[0,0]
    second_term = ((gamma2.array[1]/gamma)**2.)*covmat[1,1]
    third_term = ((2*gamma1.array[1]*gamma2.array[1])/gamma)*covmat[0,1]
    variance = first_term + second_term + third_term
    st_dev = np.sqrt(variance)
    return st_dev

def shear_magnitude(gamma1, gamma2):
    '''
    computes magnitude of gamma_LOS
    '''
    shear = np.sqrt(gamma1.array[1]**2. + gamma2.array[1]**2.)
    return shear

def lims_getter(g1, g2):
    '''
    gets x and y limits for the contour plot based on the size of the posterior, and ensures a square aspect ratio
    '''
    nsig = 5 # number of sigmas from the mean sets the plot lims
    g1_nsig = nsig*(g1.array[1] - g1.array[0]) # assuming Gaussian posterior i.e. upper and lower lims are the same
    g2_nsig = nsig*(g2.array[1] - g2.array[0])

    x_lower = g1.array[1] - g1_nsig
    x_upper = g1.array[1] + g1_nsig

    y_lower = g2.array[1] - g2_nsig
    y_upper = g2.array[1] + g2_nsig

    xdiff = x_upper - x_lower
    ydiff = y_upper - y_lower

    if max(xdiff, ydiff) == xdiff:
        y_lower_new = g2.array[1] - g1_nsig
        y_upper_new = g2.array[1] + g1_nsig
        return x_lower, x_upper, y_lower_new, y_upper_new
    else:
        x_lower_new = g1.array[1] - g2_nsig
        x_upper_new = g1.array[1] + g2_nsig
        return x_lower_new, x_upper_new, y_lower, y_upper
        
def _glob_exactly_1(cnd_name,_str_info=""):
    candidate = glob(cnd_name)
    if len(candidate)!=1:
        if _str_info!="":
            print(_str_info)#"lens dir =",_lens_dir)
        if len(candidate)==0:
            raise RuntimeError(f"No compatible data found matching name {cnd_name}")
        else:
            raise RuntimeError("Too many candidate files found")
    else:
        return candidate[0]

def get_all_lens_model_paths(res_dir,snaps=[],check_if_workin_on_it=True):
    if snaps ==[]:
        pth_modlenses_res = glob(f"{res_dir}/snap*/kw_res.*")
    else:
        pth_modlenses_res = []
        for s in snaps:
            for pth in glob(f"{res_dir}/snap_{s}_*/kw_res.*"):
                pth_modlenses_res.append(pth)
    pth_modlenses     = [Path(pth).parent for pth in pth_modlenses_res]
    if pth_modlenses==[]:
        raise RuntimeError(f"No modelled lenses found in {res_dir}")
    if check_if_workin_on_it:
        flag = np.invert([is_someone_workin_on_it(i) for i in pth_modlenses])
        if np.any(flag):
            pth_modlenses = np.array(pth_modlenses)[flag]
            pth_modlenses = pth_modlenses.tolist()
    return pth_modlenses
    


def _convert_shear2LOS(mc_sample,param_mcmc):
    """
    Only for convenience - rewrite gamma_ext, psi_ext as gamma_LOS_1,gamma_LOS_2
    """
    # needed to convert gamma_shear, psi_shear into gamma_shear1,gamma_shear2

    mc_sampleT =  np.array(mc_sample).T
    param_mcmc = list(param_mcmc)
    i_g1 = param_mcmc.index("gamma_ext_lens1")
    i_g2 = param_mcmc.index("psi_ext_lens1")
    gext,psiext = mc_sampleT[i_g1],mc_sampleT[i_g2]
    g1,g2 = shear_polar2cartesian(psiext,gext)
    mc_sampleT[i_g1] = g1
    mc_sampleT[i_g2] = g2

    param_mcmc[i_g1] = 'gamma1_los_lens1'
    param_mcmc[i_g2] = 'gamma2_los_lens1'
    
    return mc_sampleT.T, np.array(param_mcmc)

def _convert_polarshear2LOS(mc_sample,param_mcmc):
    """
    Only for convenience - rewrite gamma1, gamma2 as gamma_LOS_1,gamma_LOS_2
    """
    param_mcmc = list(param_mcmc)
    i_g1 = param_mcmc.index("gamma1_lens1")
    i_g2 = param_mcmc.index("gamma2_lens1")
    param_mcmc[i_g1] = 'gamma1_los_lens1'
    param_mcmc[i_g2] = 'gamma2_los_lens1'
    
    return mc_sample, np.array(param_mcmc)

def get_model_title(model):
    # TODO: implement better
    if model=="fitLOS":    
        ttl = "LOS not simulated, but modelled"
    elif model=="allLOS":    
        ttl = "LOS simulated and modelled"
    elif model=="fitLOS_fixedOD":
        ttl = r"LOS not simulated, but modelled (with fixed $\vec{\gamma_{od}}=0$)"
    elif model=="fitLOS_fixedOD_fixedOmgaLos":
        ttl = r"LOS not simulated, but modelled (with fixed $\vec{\gamma_{od}}=0$, $\omega_{\rm{LOS}}=0$)"
    elif model=="noLOS":
        ttl = r"LOS not simulated and external shear modelled"
    elif model=="noLOS_g12":
        ttl = r"LOS not simulated and external shear modelled (polar shear)"
    elif model=="simNoShear" or "SNS" in model:
        ttl = r"LOS not simulated, but modelled - nothing fixed!"
    elif model=="simNoShear_gausstE":
        ttl = r"LOS not simulated, but modelled - nothing fixed! (gaussian theta_E prior)"
    else:
        raise RuntimeError("Define title of model")
    if "multipole" in model:
        ttl +=" with multipoles"
    
    return ttl

def plot_los_outVsin(lenses,_rnd=3):
    model = "allLOS"
    fig,axes = plt.subplots(2,1,figsize=(12,16))
    #gamma_los_name =  r"$\gamma_{\rm{LOS}}$"
    str_glos   = r"$\gamma^{\rm{LOS}}$"
    str_glos_1 = r"$\gamma^{\rm{LOS}}_{1}$"
    str_glos_2 = r"$\gamma^{\rm{LOS}}_{2}$"
    
    str_glos_out_1 = r"$\gamma^{\rm{LOS}}_{1,\;\rm{out}}$"
    str_glos_out_2 = r"$\gamma^{\rm{LOS}}_{2,\;\rm{out}}$"
    str_glos_in_1 = r"$\gamma^{\rm{LOS}}_{1,\;\rm{in}}$"
    str_glos_in_2 = r"$\gamma^{\rm{LOS}}_{2,\;\rm{in}}$"

    str_glos_out = r"$\gamma^{\rm{LOS}}_{\rm{out}}$"
    str_glos_in = r"$\gamma^{\rm{LOS}}_{\rm{in}}$"

    g_los1_out_med,g_los2_out_med = [],[]
    g_los1_out_std,g_los2_out_std = [],[]
    g_los1_in,g_los2_in = [],[]

    g_los_out_med = []
    g_los_out_std = []
    g_los_in      = []
    
    phi_los_med  = []
    phi_los_std  = []
    phi_lens_med = []
    phi_lens_std = []
    q_lens_med   = []

    for i,lens in enumerate(lenses):
        lens_name = lens.name.replace("Sub_","")
        nm_input = f"{lens.model_res_dir}/kw_input.dll"            
        kw_input = load_whatever(nm_input)

        chnl_path = f'{lens.model_res_dir}/emcee_chain.dll'
        emcee_chain = load_whatever(chnl_path)
        sampler_type, mc_sample, param_mcmc, mc_logL  = emcee_chain
        mc_sample  = np.array(mc_sample)
        param_mcmc = np.array(param_mcmc)
        i_gamma_los1 = np.where(param_mcmc=="gamma1_los_lens1")
        gamma_los1 = mc_sample.T[i_gamma_los1][0]
        i_gamma_los2 = np.where(param_mcmc=="gamma2_los_lens1")
        gamma_los2 = mc_sample.T[i_gamma_los2][0]
        
        phi_los,q_los = ellipticity2phi_q(gamma_los1,gamma_los2)
        i_e1_lens = np.where(param_mcmc=="e1_lens0")
        e1_lens   = mc_sample.T[i_e1_lens][0]
        i_e2_lens = np.where(param_mcmc=="e2_lens0")
        e2_lens   = mc_sample.T[i_e2_lens][0]
        phi_lens,q_lens = ellipticity2phi_q(e1_lens,e2_lens)
        
        phi_los_med.append(np.median(phi_los))
        phi_los_std.append(np.std(phi_los))
        
        phi_lens_med.append(np.median(phi_lens))
        phi_lens_std.append(np.std(phi_lens))
        q_lens_med.append(np.median(q_lens))
        
        gamma_los  = np.sqrt(gamma_los1**2 + gamma_los2**2)
        gamma_los_meas = np.median(gamma_los)
        gamma_los_std  = np.std(gamma_los)
        
        kw_add_lns = kw_input['kw_add_lenses']
        kw_los     = kw_add_lns["kwargs_lens"][kw_add_lns["lens_model_list"].index("LOS")]
        gamma_los1_true = kw_los["gamma1_od"] + kw_los["gamma1_os"] - kw_los["gamma1_ds"]
        gamma_los2_true = kw_los["gamma2_od"] + kw_los["gamma2_os"] - kw_los["gamma2_ds"]
        gamma_los_true =  np.sqrt(gamma_los1_true**2 + gamma_los2_true**2)
        
        g_los_out_med.append(gamma_los_meas)
        g_los_out_std.append(gamma_los_std)
        g_los_in.append(gamma_los_true)
        
        g_los1_out_med.append(np.median(gamma_los1))
        g_los1_out_std.append(np.std(gamma_los1))
        g_los2_out_med.append(np.median(gamma_los2))
        g_los2_out_std.append(np.std(gamma_los2))
        g_los1_in.append(gamma_los1_true)
        g_los2_in.append(gamma_los2_true)

    if np.std(g_los1_in)>1e-10:
        axes = plot_los_outVsin_var(axes,
                             g_los1_in,g_los2_in,
                             g_los1_out_med,g_los1_out_std,
                             g_los2_out_med,g_los2_out_std,
                             str_glos_1,str_glos_2,
                             str_glos_in_1,str_glos_in_2,
                             str_glos_out_1,str_glos_out_2)
    else:
        print("NOTE: input glos is constant")
        axes = plot_los_outVsin_const(axes,
                             g_los1_in,g_los2_in,
                             g_los1_out_med,g_los1_out_std,
                             g_los2_out_med,g_los2_out_std,
                             str_glos_1,str_glos_2,
                             str_glos_in_1,str_glos_in_2,
                             str_glos_out_1,str_glos_out_2)

    
    #ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="grey")
    plt.suptitle(r"$\gamma_{\rm{in}}^{\rm{LOS}}$ vs $\gamma_{\rm{out}}^{\rm{LOS}}$")
    plt.tight_layout()   

    fig2,axes2 = plt.subplots(1,figsize=(9,6))
    if np.std(g_los_in)>1e-10:
        axes2.scatter(g_los_in,g_los_out_med)
        axes2.errorbar(g_los_in,g_los_out_med,yerr=g_los_out_std)
    else:
        print("NOTE: input glos is constant")
        ax = axes2
        x = np.arange(len(g_los_in))

        ax.set_ylabel(str_glos)
        ax.get_xaxis().set_ticks([])
        ax.set_xlabel("Lenses")
        ax.axhline(np.mean(g_los_in),ls="--",color="grey",label=str_glos_in)
        ax.scatter(x,g_los_out_med,marker="o",c="r",label=str_glos_out)
        ax.errorbar(x,g_los_out_med,yerr=g_los_out_std,c="r",fmt="ko",ecolor="k",elinewidth=.8)
        ax.legend()
    fig2.suptitle(r"$\gamma_{\rm{in}}^{\rm{LOS}}$ vs $\gamma_{\rm{out}}^{\rm{LOS}}$")
    
    
    # study of Etherington et al. '24
    phi_lens = np.array(phi_lens_med)*180/np.pi
    phi_los  = np.array(phi_los_med)*180/np.pi
    phi_lens_std = np.array(phi_lens_std)*180/np.pi
    phi_los_std = np.array(phi_los_std)*180/np.pi
    fig3,axis3 = plt.subplots()
    axis3.errorbar(phi_lens,phi_los,
                 xerr=phi_lens_std,yerr=phi_los_std,
                 ecolor="k",fmt="None",elinewidth=.8,markersize=4)
    axis3.scatter(phi_lens,phi_los,color="r",marker="o")
    axis3.set_xlabel(r"$\phi_{\rm{Lens}}$ [$^{\rm{o}}$]")
    axis3.set_ylabel(r"$\phi_{\rm{LOS}}$ [$^{\rm{o}}$]")
    fig3.suptitle("\n".join(wrap(r"Pointing angles ($\phi$) of mass model (lens) and LOS shear",50)))
    nm3 = "tmp/phi_los_vs_lens.png"
    fig3.tight_layout()
    fig3.savefig(nm3)
    print(f"Saved {nm3}")
    plt.close(fig3)
    
    ## comp fig 3 Etherington et al. '24
    Dphi = np.abs(phi_lens-phi_los)
    Dphi_err = np.sqrt(phi_lens_std**2 + phi_los_std**2)
    fig4,axis4 = plt.subplots()
    axis4.errorbar(g_los_out_med,Dphi,
                   xerr=g_los_out_std,yerr=Dphi_err,
                 ecolor="k",fmt="ko",elinewidth=.8,markersize=4)
    im0 = axis4.scatter(g_los_out_med,Dphi,c=np.array(q_lens_med),cmap="viridis")
    divider = make_axes_locatable(axis4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig4.colorbar(im0, cax=cax, orientation='vertical',label=r"q$_{\rm{lens}}$")
    fig4.suptitle("\n".join(wrap(r"$\Delta$ P.A. ($\phi$) of mass model (lens) and LOS shear vs LOS shear strenght ($\gamma$) and axis ratio (q)",50)))
    axis4.set_ylabel(r"|$\phi_{\rm{lens}}$ - $\phi_{\rm{LOS}}$| [$^{\rm{o}}$]")
    axis4.set_xlabel(r"$\gamma_{\rm{LOS}}$")
    nm4 = "tmp/Dphi_vs_gamma.png"
    fig4.tight_layout()
    fig4.savefig(nm4)
    print(f"Saved {nm4} - compare w. Etherington et al. '24, fig 3")
    plt.close(fig4)

    return fig, fig2


def plot_los_outVsin_var(axes,g_los1_in,g_los2_in,
                             g_los1_out_med,g_los1_out_std,
                             g_los2_out_med,g_los2_out_std,
                             str_glos_1,str_glos_2,
                             str_glos_in_1,str_glos_in_2,
                             str_glos_out_1,str_glos_out_2):
    # sort them in growing order:
    i_g1 = copy(sorted(range(len(g_los1_in)), key=lambda k: g_los1_in[k]))
    i_g2 = copy(sorted(range(len(g_los2_in)), key=lambda k: g_los2_in[k]))

    g_los1_in      = np.array(g_los1_in)[i_g1]
    g_los1_out_med = np.array(g_los1_out_med)[i_g1]
    g_los1_out_std = np.array(g_los1_out_std)[i_g1]

    g_los2_in      = np.array(g_los2_in)[i_g2]
    g_los2_out_med = np.array(g_los2_out_med)[i_g2]
    g_los2_out_std = np.array(g_los2_out_std)[i_g2]
    
    ax = axes[0]
    ax.set_title(str_glos_1)
    ax.set_xlabel(str_glos_in_1)
    ax.set_ylabel(str_glos_out_1)
    ax.scatter(g_los1_in,g_los1_out_med,marker="o",c="k")
    ax.errorbar(g_los1_in,g_los1_out_med,yerr=g_los1_out_std,c="k",fmt="ko",ecolor="k",elinewidth=.8)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="grey")
    
    ax = axes[1]
    ax.set_title(str_glos_2)
    ax.set_xlabel(str_glos_in_2)
    ax.set_ylabel(str_glos_out_2)
    ax.scatter(g_los2_in,g_los2_out_med,marker="o",c="k")
    ax.errorbar(g_los2_in,g_los2_out_med,yerr=g_los2_out_std,c="k",fmt="ko",ecolor="k",elinewidth=.8)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="grey")
    return axes
    
def plot_los_outVsin_const(axes,g_los1_in,g_los2_in,
                             g_los1_out_med,g_los1_out_std,
                             g_los2_out_med,g_los2_out_std,
                             str_glos_1,str_glos_2,
                             str_glos_in_1,str_glos_in_2,
                             str_glos_out_1,str_glos_out_2): 
    
    x = np.arange(len(g_los1_in))

    ax = axes[0]
    
    #ax.set_title(str_glos_1)
    #ax.set_xlabel()
    ax.set_ylabel(str_glos_1)
    ax.get_xaxis().set_ticks([])
    ax.axhline(np.mean(g_los1_in),ls="--",color="grey",label=str_glos_in_1)
    ax.scatter(x,g_los1_out_med,marker="o",c="k",label=str_glos_out_1)
    ax.errorbar(x,g_los1_out_med,yerr=g_los1_out_std,c="k",fmt="ko",ecolor="k",elinewidth=.8)
    #ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="grey")
    ax.legend()
    ax = axes[1]
    #ax.set_title(str_glos_2)
    ax.set_ylabel(str_glos_2)
    ax.get_xaxis().set_ticks([])
    ax.set_xlabel("Lenses")
    ax.axhline(np.mean(g_los2_in),ls="--",color="grey",label=str_glos_in_2)
    ax.scatter(x,g_los2_out_med,marker="o",c="k",label=str_glos_out_2)
    ax.errorbar(x,g_los2_out_med,yerr=g_los2_out_std,c="k",fmt="ko",ecolor="k",elinewidth=.8)
    ax.legend()
    return axes

def get_full_chain(lens,model):
    chnl_path = f'{lens.model_res_dir}/emcee_chain.dll'
    emcee_chain = load_whatever(chnl_path)
    sampler_type, mc_sample, param_mcmc, mc_logL  = emcee_chain
    param_mcmc = np.array(param_mcmc)
    if model == "noLOS":
        mc_sample,param_mcmc = _convert_shear2LOS(mc_sample,param_mcmc)
    elif model=="noLOS_g12":
        mc_sample,param_mcmc = _convert_polarshear2LOS(mc_sample,param_mcmc)
    full_chain = pd.DataFrame( np.array(mc_sample) , columns=param_mcmc)
    del mc_sample,emcee_chain
    return full_chain

def load_kw_data(model,lens):
    full_chain = get_full_chain(lens,model)
    
    res_dir = lens.model_res_dir
    modelPlot = get_model_plot(res_dir=res_dir)
    
    kw_data = dict(full_chain=full_chain,
                   modelPlot=modelPlot)
    return kw_data

def plot_result_line(model,lens,axes,i_row,nrows,columns_ttl,kw_data=None,_rnd=3,overlay_ellipticity=False,no_thetaE=False):
    fig       = axes.flatten()[0].get_figure()
    lens_name = lens.name.replace("Sub_","")
    if kw_data is None:    
        kw_data = load_kw_data(model,lens)
    modelPlot  = kw_data["modelPlot"]
    full_chain = kw_data["full_chain"]
    
    model_band = modelPlot._band_plot_list[0]
    kw_modelplot = {"vmin":model_band._v_min_default,
                    "vmax":model_band._v_max_default,
                    "extent":model_band._image_extent,
                    "origin":"lower",
                    "cmap":model_band._cmap}
    # Sim Image
    ax = axes[i_row][0]
    ax.set_ylabel(lens_name)
    if i_row ==0:
        ax.set_title(columns_ttl[0])
    ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])
    
    im0 = ax.imshow(np.log10(model_band._data),**kw_modelplot)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"flux$_{data}$")

    # Model
    ax = axes[i_row][1]
    if i_row ==0:
        ax.set_title(columns_ttl[1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    im0 = ax.imshow(np.log10(model_band._model),**kw_modelplot)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"flux$_{model}$")


    # Residual
    ax = axes[i_row][2]
    if i_row ==0:
        ax.set_title(columns_ttl[2])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    reduced_chi2  = get_red_chi2(modelPlot,verbose=False)
    
    kw_modelplot_resid = {**kw_modelplot, "vmin": -3, "vmax": 3, "cmap": "bwr"}
    im0 = ax.imshow(model_band._norm_residuals,**kw_modelplot_resid)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"(f$_{model}$-f$_{data}$)/$\sigma$")
    
    x_txt = (model_band._image_extent[1]-model_band._image_extent[0])*3.5/5
    y_txt = (model_band._image_extent[3]-model_band._image_extent[2])*4/5
    ax.text(x_txt,y_txt,s=r"$\chi^2_{red.}$="+str(np.round(reduced_chi2,2)),color="k",backgroundcolor="w")
        
    # Posterior Theta_E
    if not no_thetaE:
        ax = axes[i_row][3]
        if i_row ==0:
            ax.set_title(columns_ttl[3])
        if i_row==nrows-1:
            ax.set_xlabel(r"$\theta_E$")
        thetaE = full_chain["theta_E_lens0"].to_numpy() 
        plot_dist(ax, Chain(samples=full_chain, name=lens_name, shade=True, color='#2c7fb8', smooth=20, bins=10,
                                       shade_gradient = 0.4, linewidth=3.0), px="theta_E_lens0",)
        lbl_tE_meas = r"$\theta_{\rm{E}}=$"+str(np.round(np.median(thetaE),_rnd))+"+-"+str(np.round(np.std(thetaE),_rnd))+'"'
        ax.axvline(np.median(thetaE),ls="-.",color="k",label=lbl_tE_meas)
        lbl_tE_true = r"$\theta_{\rm{E, True}}=$"+short_SciNot(lens.thetaE.value)+'"'
        ax.axvline(lens.thetaE.value,ls="-",color="r",label=lbl_tE_true)
        ax.legend(loc="upper right")
    
    # Posterior gamma_ext
    index = 4
    if no_thetaE:
        index = 3
    ax = axes[i_row][index]
    chain4plotcont = Chain(samples=full_chain, name=lens_name, shade=True, 
                           color=warm[1], shade_gradient = 0.8, linewidth=3.0)
    if i_row ==0:
        ax.set_title(columns_ttl[index])
    plot_contour(ax, chain4plotcont, px="gamma1_los_lens1", py="gamma2_los_lens1")
    if overlay_ellipticity:
        plot_contour(ax, chain4plotcont, px="e1_lens0", py="e2_lens0")
        
    if overlay_ellipticity:
        column_g12e12 = ['gamma1_los_lens1', 'gamma2_los_lens1',"e1_lens0","e2_lens0"]
    else:
        column_g12e12 = ['gamma1_los_lens1', 'gamma2_los_lens1']
    g12e12 = full_chain.loc[:,column_g12e12]
    med_g12e12 = g12e12.median()
    sig_g12e12 = g12e12.std()
    for i in range(4):
        if i<2:
            col= warm[0]
        else:
            if not overlay_ellipticity:
                continue
            col = cool[1]
        _nm = column_g12e12[i].replace("_lens0","").replace("_lens1","")
        _nm = _nm.replace("gamma",r"$\gamma$")
        if "1" in _nm:
            _nm = _nm.replace("1","")
            _nm = _nm.replace("_los",r"$_{\rm{LOS}, 1}$")
        elif "2" in _nm:
            _nm = _nm.replace("2","")
            _nm = _nm.replace("_los",r"$_{\rm{LOS}, 2}$")
        
        if model=="noLOS":
           _nm = _nm.replace("LOS",r"Shear")
         
        
        lbl = _nm+f"={np.round(med_g12e12[i],_rnd)}+-{np.round(sig_g12e12[i],_rnd)}"
        col="k"
        if i%2==0:
            ax.axvline(med_g12e12[i],ls="-.",c=col,label=lbl)
        else:
            ax.axhline(med_g12e12[i],ls="-.",c=col,label=lbl)

    clmns = [r"$\gamma_{\rm{LOS},1}$",r"$\gamma_{\rm{LOS},2}$"]
    if "noLOS" in model:
        clmns = [r"$\gamma_{\rm{shear},1}$",r"$\gamma_{\rm{shear},2}$"]
        
    if i_row==nrows-1:
        if overlay_ellipticity:
            ax.set_xlabel(clmns[0]+r"/e$_1$")
        else:
            ax.set_xlabel(clmns[0])
    if nrows==1:
        ax.set_xlabel(clmns[0])
    if overlay_ellipticity:
        ax.set_ylabel(clmns[1]+r"/e$_2$")
    else:
        ax.set_ylabel(clmns[1])

    if model=="allLOS":
        kw_add_lns = kw_input['kw_add_lenses']
        kw_los     = kw_add_lns["kwargs_lens"][kw_add_lns["lens_model_list"].index("LOS")]
        gamma_los1_true = kw_los["gamma1_od"] + kw_los["gamma1_os"] - kw_los["gamma1_ds"]
        gamma_los2_true = kw_los["gamma2_od"] + kw_los["gamma2_os"] - kw_los["gamma2_ds"]
    elif "fitLOS" in model or "noLOS" in model or "simNoShear" in model or "SNS" in model:
        #TODO : better implementation of this
        gamma_los1_true,gamma_los2_true = 0,0
    else:
        raise RuntimeError(f"{model} not defined for gamma_los_truth")
    ax.axvline(gamma_los1_true,ls="-",label="Truth "+clmns[0]+f"= {np.round(gamma_los1_true,_rnd)}",c="r")
    ax.axhline(gamma_los2_true,ls="-",label="Truth "+clmns[1]+f"= {np.round(gamma_los2_true,_rnd)}",c="r")
        
    #ax.legend(loc="lower left",framealpha=1)
    leg = ax.legend(framealpha=1)
    leg.set_zorder(1000)
    # free memory:
    del modelPlot
    del full_chain
    del g12e12
    if not no_thetaE:
        del thetaE
    gc.collect()
    return axes
    
warnings.filterwarnings("ignore")

name_models = ["noLOS","noLOS_g12","fitLOS","allLOS","fitLOS_fixedOD","fitLOS_fixedOD_fixedOmegaLos","simNoShear","SNS_multipole","SNS_m134_gausstE"]

kw_names_models = {}
for nm in name_models:
    if nm=="noLOS":
        kw_names_models[nm] = "model_ext_shear"
    elif nm=="noLOS_g12":
        kw_names_models[nm] = "model_ext_shear_g12"
    else:
        kw_names_models[nm] = f"model_{nm}"
kw_models_names =  {v: k for k, v in kw_names_models.items()}


def get_res_dir(model,simsuite=std_simsuite,sim=std_sim,subsim=std_subsim):
    model_name = kw_names_models[model]
    try:
        model_module = import_module(f'.{model_name}',"nazgul.Modelling")
        get_res_dir = model_module.get_res_dir
        res_dir_base = model_module.res_dir_base
    except:
        if model in name_models:
            print(f"{model} to implement") 
        raise RuntimeError(f"model {model} not known")
    res_dir = get_res_dir(res_dir_base,simsuite,sim,subsim=subsim,run_type=0)
    return res_dir 


def get_model_from_res_dir(res_dir,model_res=model_res_base):
    model_subpath = str(Path(model_res).name)+"/" #"models/" by def.
    _sub_res_dir = str(res_dir).split(model_subpath)[1]
    _model_name  = _sub_res_dir.split("/")[0]
    model_name   = kw_models_names["model_"+_model_name] 
    #verify
    _sub_res_dir_split = Path(_sub_res_dir.split("/")[1]).parts[0]
    simsuite_code =  _sub_res_dir_split.split("_")[0]
    simsuite = get_simsuite_from_code(simsuite_code)
    sim_subsim = "_".join(_sub_res_dir_split.split("_")[1:])
    res_dir_recovered = get_res_dir(model_name,simsuite=simsuite,sim=sim_subsim,
                subsim=None)
    res_dir = str(res_dir)
    res_dir_recovered = str(res_dir_recovered)
    if len(res_dir)>len(res_dir_recovered):
        assert res_dir_recovered in res_dir
    else:
        assert res_dir in res_dir_recovered
    return model_name
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot Combined results for all the lens model of given run")
    parser.add_argument('-m','--model',type=str,
                        dest="model",
                        help=f"Name of type of model - accepted: {name_models}")
    parser.add_argument('-ove','--overlay_ellipticity', dest="overlay_ellipticity", 
                        default=False, action="store_true",
                        help=f"If true, overlay ellipticity posterior")
    parser.add_argument('-snap','--snap',dest="snaps",default=[],nargs="+",help="(Optional) Define a specific snap list")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-lpp','--lines_per_page', dest="lines_per_page",
                        default=5, type=int,
                        help="Number of lens rows per page in combined PDF (default=5)")

    args     = parser.parse_args()
    model    = args.model
    snaps    = args.snaps
    sim      = args.sim
    subsim   = args.subsim
    simsuite = args.simsuite
    overlay_ellipticity = args.overlay_ellipticity
    lines_per_page      = args.lines_per_page

    _rnd = 3
    res_dir = get_res_dir(model,simsuite=simsuite,sim=sim,subsim=subsim)
    nm_combined = f"{res_dir}/combined_result.pdf"

    lens_resdir_paths = get_all_lens_model_paths(res_dir,snaps=snaps)  # paths only, no loading
    n_lenses          = len(lens_resdir_paths)

    columns_ttl = ["Sim Image", "Model", "Norm. Resid.", r"P($\theta_E$|S.I.)"]
    if overlay_ellipticity:
        columns_ttl.append(r"P($\gamma_{\rm{LOS},1}$,$\gamma_{\rm{LOS},2}$|S.I.) + P(e$_1$,e$_2$|S.I.)")
    else:
        columns_ttl.append(r"P($\gamma_{\rm{LOS},1}$,$\gamma_{\rm{LOS},2}$|S.I.)")
    if "noLOS" in model:
        columns_ttl[-1] = columns_ttl[-1].replace("LOS", "Shear")

    ncols = len(columns_ttl)

    # accumulators for plot_los_outVsin, only needed if model=="allLOS"
    lenses_for_los = []
    tracemalloc.start()
    with PdfPages(nm_combined) as pdf:

        # iterate over pages
        for page_start in range(0, n_lenses, lines_per_page):
            log_memory("beggining page loop")
            page_slice  = lens_resdir_paths[page_start : page_start + lines_per_page]
            nrows_page  = len(page_slice)
            fig, axes = plt.subplots(nrows_page, ncols,
                                     figsize=(scale_fig * ncols, scale_fig * nrows_page),
                                     squeeze=False)
            log_memory("after subplots instantiation")
            for i_row, model_res_dir in enumerate(page_slice):
                log_memory("beggining row loop")
                # ── load lens ──────────────────────────────────────────────
                lens = LoadLens(model_res_dir/"link_gallens.pkl")
                lens.unpack()
                lens.model_res_dir = model_res_dir
                log_memory("loaded lens")
                kw_data = load_kw_data(model,lens)
                log_memory("loaded kw_data")
                # ── single-lens PDF ────────────────────────────────────────
                nm_single = f"{model_res_dir}/single_result.pdf"
                fig_s, axes_s = plt.subplots(1, ncols,
                                             figsize=(scale_fig * ncols, scale_fig),
                                             squeeze=False)
                log_memory("before plot_result_line 1")
                plot_result_line(model, lens, axes_s, 0, 1,
                                 columns_ttl, _rnd=_rnd,
                                 kw_data =kw_data,
                                 overlay_ellipticity=overlay_ellipticity)
                log_memory("after plot_result_line 1")
                fig_s.suptitle(get_model_title(model))
                fig_s.tight_layout()
                fig_s.savefig(nm_single)
                plt.close(fig_s)
                plt.close("all")
                del fig_s,axes_s
                gc.collect()
                print(f"Saved {nm_single}")

                # ── row in combined page ───────────────────────────────────
                log_memory("before plot_result_line 2")
                plot_result_line(model, lens, axes, i_row, nrows_page,
                                 columns_ttl, _rnd=_rnd,kw_data=kw_data,
                                 overlay_ellipticity=overlay_ellipticity)
                
                log_memory("after plot_result_line 2")
                # ── keep lightweight ref for LOS plot if needed ────────────
                if model == "allLOS":
                    lenses_for_los.append(lens)
                else:
                    del lens
                del kw_data
                gc.collect()
                log_memory("End of row loop - deleted kw_data")
                
            fig.suptitle(get_model_title(model))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            del fig
            print(f"Saved page {page_start // lines_per_page + 1} to {nm_combined}")
            gc.collect()
            log_memory("End of page loop")
            log_top_allocs()
    print(f"Combined PDF complete: {nm_combined}")

    # ── LOS comparison plots (allLOS only) ────────────────────────────────
    if model == "allLOS":
        fig, fig2 = plot_los_outVsin(lenses_for_los, _rnd=_rnd)

        nm = f"{res_dir}/comp_glos_12_in_vs_out.pdf"
        fig.savefig(nm)
        plt.close(fig)
        del fig
        print(f"Saved {nm}")

        nm = f"{res_dir}/comp_glos_in_vs_out.pdf"
        fig2.savefig(nm)
        plt.close(fig2)
        del fig2
        print(f"Saved {nm}")

        del lenses_for_los
        gc.collect()
        
"""
def get_all_lens_models(res_dir):
    pth_modlenses = get_all_lens_model_paths(res_dir)
    lenses = []
    for pth_lens in pth_modlenses:
        try:
            # if res exists AND is loaded correctly
            load_whatever(pth_lens/"kw_res.*")
            # then we consider the lens
            model_res_dir = pth_lens
            lens_link     = model_res_dir/"link_gallens.pkl"
            if not os.path.exists(lens_link):
                warnings.warn("MONKEY-PATCH- update name of the lens on the fly")
                _lens_dir = Path(os.readlink(lens_link)).parent
                _nm_lns = str(model_res_dir.name)
                prj_index = int(_nm_lns.split("Prj")[1][0])
                nmlns = _nm_lns.split("_")[2]
                candidate_lens_name = "Sub_Lens_"+nmlns+"_Prj"+str(prj_index)+"*"
                cnd_name =  str(_lens_dir)+"/"+candidate_lens_name
                lens_link = _glob_exactly_1(cnd_name,_str_info="lens dir ="+str(_lens_dir)) 
            lens = LoadLens(lens_link)
            lens.unpack() 
            lens.model_res_dir = model_res_dir
            lenses.append(lens)
        except Exception as e:
            print(f"Failed to load {pth_lens} due to {e} - skipping.")
    return lenses


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot Combined results for all the lens model of given run")
    parser.add_argument('-m','--model',type=str,
                        dest="model",
                        help=f"Name of type of model - accepted: {name_models}")
    
    parser.add_argument('-ove','--overlay_ellipticity', dest="overlay_ellipticity", 
                        default=False, action="store_true",
                        help=f"If true, overlay ellipticity posterior")
    args     = parser.parse_args()
    model    = args.model
    overlay_ellipticity = args.overlay_ellipticity

    # to which cifra significativa to round
    _rnd = 3
    res_dir = get_res_dir(model)
    nm_combined = f"{res_dir}/combined_result.pdf"    
    lenses_modelled = get_all_lens_models(res_dir)
    columns_ttl = ["Sim Image","Model","Norm. Resid.",r"P($\theta_E$|S.I.)"]
    if overlay_ellipticity:
        columns_ttl.append(r"P($\gamma_{\rm{LOS},1}$,$\gamma_{\rm{LOS},2}$|S.I.) + P(e$_1$,e$_2$|S.I.)")
    else:
        columns_ttl.append(r"P($\gamma_{\rm{LOS},1}$,$\gamma_{\rm{LOS},2}$|S.I.)")

    if "noLOS" in model:
        columns_ttl[-1] = columns_ttl[-1].replace("LOS","Shear")
    nrows  = len(lenses_modelled)
    ncols  = len(columns_ttl) # n* of wanted columns
    scl = 8
    fig, axes = plt.subplots(nrows, ncols, figsize=(scl*ncols,scl*nrows))
    for i_row,lens in enumerate(lenses_modelled):
        plot_result_line(model,lens,axes,i_row,nrows,columns_ttl,\
                         _rnd=_rnd,overlay_ellipticity=overlay_ellipticity)
    plt.suptitle(get_model_title(model))
    plt.tight_layout()
    plt.savefig(nm_combined)
    print(f"Saved {nm_combined}")
    plt.close()
    
    for i_row,lens in enumerate(lenses_modelled):
        nm_single = f"{lens.model_res_dir}/single_result.pdf"    

        fig, axes = plt.subplots(1, ncols, figsize=(scl*ncols,scl*1))
        plot_result_line(model,lens,np.array([axes]),0,1,columns_ttl,\
                         _rnd=_rnd,overlay_ellipticity=overlay_ellipticity)
        plt.suptitle(get_model_title(model))
        plt.tight_layout()
        plt.savefig(nm_single)
        print(f"Saved {nm_single}")
        plt.close(fig) 
    if model=="allLOS":
        fig,fig2 = plot_los_outVsin(lenses_modelled,_rnd=_rnd)
        _nm = "comp_glos_12_in_vs_out.pdf"
        nm = f"{res_dir}/{_nm}"
        fig.savefig(nm)
        plt.close(fig)
        print(f"Saved {nm}")
        _nm = "comp_glos_in_vs_out.pdf"
        nm = f"{res_dir}/{_nm}"
        fig2.savefig(nm)
        print(f"Saved {nm}")
        plt.close(fig2)
##########
##########
c = ChainConsumer()

if model!="noLOS":
    gamma1_full = c.analysis.get_parameter_summary(chain=Chain(samples=full_chain, name='lenstronomy_mcmc_emcee'),column=r"gamma1_los_lens1")
    gamma2_full = c.analysis.get_parameter_summary(chain=Chain(samples=full_chain, name='lenstronomy_mcmc_emcee'),column=r'gamma2_los_lens1')

    nc_full_g1 =  np.isfinite(gamma1_full.array[0]) # checks if the lower bound is finite (CC returns NaN for both upper and lower bounds when param is unconstrained)
    nc_full_g2 =  np.isfinite(gamma2_full.array[0])
    covmat_full = Chain(samples=full_chain.loc[:, ['gamma1_los_lens1', 'gamma2_los_lens1']], name='lenstronomy_mcmc_emcee').get_covariance().matrix
    shear_mag_full = shear_magnitude(gamma1_full, gamma2_full)
    shear_stdev_full = shear_stdev(shear_mag_full, gamma1_full, gamma2_full, covmat_full)        
 
gamma_epl_full = c.analysis.get_parameter_summary(chain=Chain(samples=full_chain, name='lenstronomy_mcmc_emcee'),column=r'gamma_lens0')
"""

