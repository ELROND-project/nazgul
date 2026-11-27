import sys,os
import argparse
import numpy as np
from glob import glob
from copy import deepcopy
import matplotlib.pyplot as plt
from importlib import import_module
from chainconsumer import Chain ,ChainConsumer,Truth

from python_tools.tools import mkdir
from python_tools.get_res import load_whatever

from nazgul.configurations import nazgul_path
from nazgul.profiler_lens import _standardise_truths
from nazgul.combined_modelling_results import show_contour_post_2,get_full_chain


def _get_kw_lens_true_testSIS(model_module):
    try:
        kw_lens_true = model_module.kw_lens_true
    except AttributeError as err:
        if "hack" in model_name:
            err_str = str(err) +"\n this is weird, because we expect to have kw_lens_true implemented in clear text in the 'hack' models (non-nazgul)" 
            raise RuntimeError(err_str)
        else:
            # this should have a lensfuel -> it does, but more natural to simply get the params
            #lf = model_module.lensfuel
            kw_lens_los = model_module.kw_los_gal["kwargs_lens"][0]
            tE_in  = model_module.tE_in
            kw_lens_SIS = {"theta_E":tE_in,"center_x":0,"center_y":0}
            
            kw_lens_true = [kw_lens_SIS,
                            kw_lens_los]
    return kw_lens_true

def _get_kw_source_true_testSIS(model_module):
    try:
        kw_source_true = model_module.kwargs_source_init
    except AttributeError as err:
        if "hack" in model_name:
            err_str = str(err) +"\n this is weird, because we expect to have kwargs_source_init implemented in clear text in the 'hack' models (non-nazgul)" 
            raise RuntimeError(err_str)
        else:
            # this should have a lensfuel 
            lf = model_module.lensfuel
            kw_source_true = lf.kwargs_source_init
    return kw_source_true

    
def get_kw_LensAndSource_true(model_name):
    model_module = import_module(".model_"+model_name,"nazgul.Modelling")
    kw_lens_true = _get_kw_lens_true_testSIS(model_module)
    kw_source_true = _get_kw_source_true_testSIS(model_module)
    return kw_lens_true,kw_source_true
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot testSIS results posterior")
    parser.add_argument('-mn','--model_name',type=str,dest="model_name",help= f"Model name, e.g. testSIS_fODoO_hack2_1_mvS_LargerGamma")

    args       = parser.parse_args()
    model_name = args.model_name

    path_res = nazgul_path/str("results/models/"+model_name+"/ANLT_SIS/")
    if not "hack" in model_name:
        # ugly but easy:
        _path_res =  glob(str(path_res)+"/*")
        # we assume it's a singular directory
        assert len(_path_res)== 1
        path_res = _path_res[0]
            
    res_path = f"{str(nazgul_path)}/tmp/res_{model_name}/"
    mkdir(res_path)
    kw_lens_true,kw_source_true = get_kw_LensAndSource_true(model_name)
    
    # ugly way to get the "kw_truths"
    kwargs_lens_los = deepcopy(kw_lens_true[1])
    if 'gamma1_os' in kwargs_lens_los.keys():
        kwargs_lens_los["gamma1_los"]  = kwargs_lens_los["gamma1_os"] + kwargs_lens_los["gamma1_od"] - kwargs_lens_los["gamma1_ds"]
        kwargs_lens_los["gamma2_los"]  = kwargs_lens_los["gamma2_os"] + kwargs_lens_los["gamma2_od"] - kwargs_lens_los["gamma2_ds"]
    elif 'gamma1' in  kwargs_lens_los.keys():
        print("the simulated lens doesn't have LOS, but shear")
        kwargs_lens_los["gamma1_los"]  = kwargs_lens_los["gamma1"]
        kwargs_lens_los["gamma2_los"]  = kwargs_lens_los["gamma2"]
    else:
        print("DEBUG",kwargs_lens_los.keys())
        raise 
    kw_truths_by_hand = {"theta_E_lens0":kw_lens_true[0]["theta_E"],
                         "center_x_lens0":kw_lens_true[0]["center_x"],
                         "center_y_lens0":kw_lens_true[0]["center_y"],
                         "kwargs_source":kw_source_true,
                         "kwargs_add_lenses":{"lens_model_list":["LOS"],
                                              "kwargs_lens":[kwargs_lens_los ]}
                        }
                    
                    
    # plotting in a ~ to profiler
    kw_sim_testsis = {'sim': 'SIS', 'simsuite': 'ANL_TEST', 'subsim': None}

    chnl_path  =  f"{str(path_res)}/emcee_chain.dll"
    emcee_chain = load_whatever(chnl_path)
    sampler_type, mc_sample, param_mcmc, mc_logL  = emcee_chain
    kw_truths = _standardise_truths(kw_truths_by_hand)


    # plot 2D gamma12_los vs true value
    full_chain = get_full_chain(lens="",model="",emcee_chain=emcee_chain,kw_sim=kw_sim_testsis, do_prettify_prm_nm=True)
    chain = Chain(samples=full_chain, name="chain", 
              shade=True, color='#2c7fb8', smooth=1, bins=30,
              shade_gradient = 0.4, linewidth=3.0)
    gamma_los_prms = ["gamma1_los_lens1","gamma2_los_lens1"]

    show_contour_post_2(chain,*gamma_los_prms,kw_truths=kw_truths)
    nm1 = res_path+"/gamma12LOS.pdf"
    plt.savefig(nm1)
    print(f"Saving {nm1}")


    # plot full chain vs true values
    c = ChainConsumer()
    c.add_chain(chain)
    limited_truth = {}
    for kw in chain.data_columns:
        if kw in kw_truths.keys():
            limited_truth[kw] = float(kw_truths[kw])
    c.add_truth(Truth(location=limited_truth,line_style="--",color="k"))
    # Verify Truths are within range of plotting
    extents = {}
    for kw in limited_truth.keys():
        _truth = limited_truth[kw]
        _arr = chain.data_samples[kw].array
        min_arr = _arr.min()
        max_arr = _arr.max()
        Darr = max_arr-min_arr
        padd = Darr/5
        if _truth<(min_arr+padd):
            extents[kw] = (_truth-padd,max_arr)
        elif _truth>(max_arr-padd):
            extents[kw] = (min_arr,_truth+padd)  
        else:
            extents[kw] = (min_arr,max_arr)
    c.plotter.config.extents=extents
    
    fig = c.plotter.plot()
    nm2 = f"{res_path}/full_post.pdf"
    fig.savefig(nm2)
    print(f"Saving {nm2}")
    
    print("DONE\n\n")