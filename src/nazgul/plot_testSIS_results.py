import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
from python_tools.get_res import load_whatever

from nazgul.configurations import nazgul_path
model_name= "testSIS_fODoO_hack2_1_mvS_LargerGamma"
path_res = nazgul_path/str("results/models/"+model_name+"/ANLT_SIS/")



from python_tools.tools import mkdir

res_path = f"{str(nazgul_path)}/tmp/res_{model_name}/"
mkdir(res_path)


from importlib import import_module

model_module = import_module(".model_"+model_name,"nazgul.Modelling")
kw_lens_true = model_module.kw_lens_true

from nazgul.combined_modelling_results import show_contour_post_2,get_full_chain,show_contour_post
from nazgul.profiler_lens import _standardise_truths
kwargs_lens_los = deepcopy(kw_lens_true[1])
kwargs_lens_los["gamma1_los"]  = kwargs_lens_los["gamma1_os"] + kwargs_lens_los["gamma1_od"] - kwargs_lens_los["gamma1_ds"]
kwargs_lens_los["gamma2_los"]  = kwargs_lens_los["gamma2_os"] + kwargs_lens_los["gamma2_od"] - kwargs_lens_los["gamma2_ds"]
kw_truths_by_hand = {"theta_E_lens0":kw_lens_true[0]["theta_E"],
                     "center_x_lens0":kw_lens_true[0]["center_x"],
                     "center_y_lens0":kw_lens_true[0]["center_y"],
                     "kwargs_source":model_module.kwargs_source_init,
                     "kwargs_add_lenses":{"lens_model_list":["LOS"],
                                          "kwargs_lens":[kwargs_lens_los ]}
                    }
                    
                    
# has to reinvet profiler bc for some reason it fails to reload prev. res.
from chainconsumer import Chain ,ChainConsumer,Truth
#from nazgul.profiler_lens import _standardise_truths
#from nazgul.combined_modelling_results import get_full_chain,get_emcee,show_contour_post_2

kw_sim_testsis = {'sim': 'SIS', 'simsuite': 'ANL_TEST', 'subsim': None}

chnl_path  =  f"{str(path_res)}/emcee_chain.dll"
emcee_chain = load_whatever(chnl_path)
sampler_type, mc_sample, param_mcmc, mc_logL  = emcee_chain
kw_truths = _standardise_truths(kw_truths_by_hand)


full_chain = get_full_chain(lens="",model="",emcee_chain=emcee_chain,kw_sim=kw_sim_testsis, do_prettify_prm_nm=True)
chain = Chain(samples=full_chain, name="chain", 
              shade=True, color='#2c7fb8', smooth=1, bins=30,
              shade_gradient = 0.4, linewidth=3.0)
gamma_los_prms = ["gamma1_los_lens1","gamma2_los_lens1"]

show_contour_post_2(chain,*gamma_los_prms,kw_truths=kw_truths)
plt.savefig(res_path+"/gamma12LOS.pdf")


from chainconsumer import Chain ,ChainConsumer, Truth
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
    if _truth<min_arr:
        extents[kw] = (_truth-padd,max_arr)
    if _truth>max_arr:
        extents[kw] = (min_arr,_truth+padd)  
c.plotter.config.extents=extents

fig = c.plotter.plot()
fig.savefig(f"{res_path}/full_post.pdf")

print("DONE\n\n")