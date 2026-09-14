import sys
import argparse
import numpy as np
import pandas as pd
from copy import copy,deepcopy
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer, Chain
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Util.param_util import shear_cartesian2polar,ellipticity2phi_q



from python_tools.get_res import load_whatever
from nazgul.combined_modelling_results import get_all_lens_models,get_res_dir

def _convert_sample2qphi(mc_sample,param_mcmc):
    print("Very specific function - do not use outside of here")
    
    mc_sampleT =  np.array(mc_sample).T
    param_mcmc = list(param_mcmc)
    i_g1 = param_mcmc.index("gamma1_los_lens1")
    i_g2 = param_mcmc.index("gamma2_los_lens1")
    g1_los,g2_los = mc_sampleT[i_g1],mc_sampleT[i_g2]
    g_los,psi_los = shear_cartesian2polar(g1_los,g2_los)
    mc_sampleT[i_g1] = g_los
    mc_sampleT[i_g2] = psi_los
    param_mcmc[i_g1] = "$\gamma_{LOS}$"
    param_mcmc[i_g2] = r"$\phi_{LOS}$"

    mc_sampleT =  np.array(mc_sample).T
    param_mcmc = list(param_mcmc)
    i_g1 = param_mcmc.index("gamma1_od_lens1")
    i_g2 = param_mcmc.index("gamma2_od_lens1")
    g1_od,g2_od = mc_sampleT[i_g1],mc_sampleT[i_g2]
    g_od,psi_od = shear_cartesian2polar(g1_od,g2_od)
    mc_sampleT[i_g1] = g_od
    mc_sampleT[i_g2] = psi_od
    param_mcmc[i_g1] = r"$\gamma_{OD}$"
    param_mcmc[i_g2] = r"$\phi_{OD}$"

    i_e1 = param_mcmc.index("e1_lens0")
    i_e2 = param_mcmc.index("e2_lens0")
    e1,e2 = mc_sampleT[i_e1],mc_sampleT[i_e2]
    phi_eps,q_eps = ellipticity2phi_q(e1,e2)
    mc_sampleT[i_e1] = q_eps
    mc_sampleT[i_e2] = phi_eps
    param_mcmc[i_e1] =  r"$\rm{q}_{\rm{eps}}$"
    param_mcmc[i_e2] = r"$\phi_{\rm{eps}}$"

    return mc_sampleT.T,np.array(param_mcmc)


def embellish_params(params):
    params_list = copy(list(params))
    prm2embellish = ["gamma1_los_lens1","gamma2_los_lens1",
                          "gamma1_od_lens1","gamma2_od_lens1",
                          "e1_lens0","e2_lens0"]
    embellishedprms = [r"$\gamma_{1,\rm{LOS}}$",r"$\gamma_{2,\rm{LOS}}$",
                       r"$\gamma_{1,\rm{OD}}$", r"$\gamma_{2,\rm{OD}}$",
                       r"$\rm{e}_{1}$",r"$\rm{e}_{2}$"]
    for i in range(len(prm2embellish)):
        if prm2embellish[i] in params:
            params_list[i] = embellishedprms[i]
    return params_list

def plot_corner_lens(lens):
    nm_mblo = f"{lens.model_res_dir}/multi_band_list_out.json"
    multi_band_list_out = load_whatever(nm_mblo)
    nm_input = f"{lens.model_res_dir}/kw_input.dll"            
    kw_input = load_whatever(nm_input)
    kwargs_data_joint= kw_input["kwargs_data_joint"] 
    kwargs_model = kw_input["kwargs_model"]
    kwargs_constraints  = kw_input["kwargs_constraints"] 
    kwargs_likelihood  = kw_input["kwargs_likelihood"]
    kwargs_params  = kw_input["kwargs_params"]
    fitting_kwargs_list  = kw_input["fitting_kwargs_list"]

    nm_res = f"{lens.model_res_dir}/kw_res.json"
    kwargs_result = load_whatever(nm_res)
    modelPlot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result, 
                          arrow_size=0.02, cmap_string="gist_heat",
                          image_likelihood_mask_list=kwargs_likelihood["image_likelihood_mask_list"])

    model_band = modelPlot._band_plot_list[0]
    kw_modelplot = {"vmin":model_band._v_min_default,
                    "vmax":model_band._v_max_default,
                    "extent":model_band._image_extent,
                    "origin":"lower",
                    "cmap":model_band._cmap}
    
    chnl_path = f'{lens.model_res_dir}/chain_list.dll'
    chain_list = load_whatever(chnl_path)
    sampler_type, mc_sample, param_mcmc, mc_logL  = chain_list[-1]
    param_mcmc = np.array(param_mcmc)
    mc_sample = np.array(mc_sample)

    params_of_interest = ["gamma1_los_lens1","gamma2_los_lens1",
                          "gamma1_od_lens1","gamma2_od_lens1",
                          "e1_lens0","e2_lens0"]
    beauty_params_of_interest = embellish_params(params_of_interest)
    beauty_param_mcmc = embellish_params(param_mcmc)
    c = ChainConsumer()
    c.add_chain(Chain(
        samples=pd.DataFrame(mc_sample, columns=beauty_param_mcmc),
        name="g12",)
               )
    fig = c.plotter.plot(columns=beauty_params_of_interest)
    nm = f"{lens.model_res_dir}/corner_plot_LosVsEll.png"
    
    
    ax =fig.axes[4]
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    kwargs_result_wo_tracer = deepcopy(kwargs_result)
    keys = copy(list(kwargs_result_wo_tracer.keys()))
    for k in keys:
        if "tracer" in str(k):
            del kwargs_result_wo_tracer[k]

    logL,_prms   = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result_wo_tracer)

    n_data = modelPlot._imageModel.num_data_evaluate
    reduced_chi2  = -logL * 2 / n_data
    kw_modelplot_resid = deepcopy(kw_modelplot)
    kw_modelplot_resid["vmin"] = -3
    kw_modelplot_resid["vmax"] = 3
    kw_modelplot_resid["cmap"] = "bwr"
    im0 = ax.imshow(model_band._norm_residuals,**kw_modelplot_resid)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"(f$_{model}$-f$_{data}$)/$\sigma$")
    
    x_txt = (model_band._image_extent[1]-model_band._image_extent[0])*3.5/5
    y_txt = (model_band._image_extent[3]-model_band._image_extent[2])*4/5
    ax.text(x_txt,y_txt,s=r"$\chi^2_{red.}$="+str(np.round(reduced_chi2,2)),color="k",backgroundcolor="w")

    fig.savefig(nm)
    print(f"Saved {nm}")
    fig.close()
    ###
    # Add a plot where gamma12 and e12 are converted into angles and power
    

    params_of_interest = [r"$\gamma_{LOS}$",r"$\phi_{LOS}$",
                          r"$\gamma_{OD}$",r"$\phi_{OD}$",
                          r"$\rm{q}_{\rm{eps}}$",r"$\phi_{\rm{eps}}$"]

    mc_sample,param_mcmc = _convert_sample2qphi(mc_sample,param_mcmc)
    c = ChainConsumer()
    c.add_chain(Chain(
        samples=pd.DataFrame(mc_sample, columns=param_mcmc),
        name="gamma_phi",)
               )
    fig = c.plotter.plot(columns=params_of_interest)
    nm = f"{lens.model_res_dir}/corner_plot_LosVsEll_qphi.png"
    
    ax =fig.axes[4]
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im0 = ax.imshow(model_band._norm_residuals,**kw_modelplot_resid)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"(f$_{model}$-f$_{data}$)/$\sigma$")
    
    x_txt = (model_band._image_extent[1]-model_band._image_extent[0])*3.5/5
    y_txt = (model_band._image_extent[3]-model_band._image_extent[2])*4/5
    ax.text(x_txt,y_txt,s=r"$\chi^2_{red.}$="+str(np.round(reduced_chi2,2)),color="k",backgroundcolor="w")

    fig.savefig(nm)
    print(f"Saved {nm}")



name_models = ["noLOS","fitLOS","allLOS","fitLOS_fixedOD"]
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot Corner plot for lens")
    parser.add_argument('-m','--model',type=str,
                        dest="model",
                        help=f"Name of type of model - accepted: {name_models}")    
    args     = parser.parse_args()
    model    = args.model
    # to which cifra significativa to round
    _rnd = 3
    if model not in name_models:
        raise RuntimeError(f"model {model} not known - accepted: {name_models}")
    # get_res_dir maps the model name to its Modelling/ module and appends the
    # {SimCode}_{Sim}[_{subsim}] level, which is where the snap_* dirs live
    res_dir = get_res_dir(model)

    lenses_modelled = get_all_lens_models(res_dir)
    for lens in lenses_modelled:
        plot_corner_lens(lens)