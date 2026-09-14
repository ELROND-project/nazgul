# very simple question:
# does multipole reduce tension with gamma_los?
import gc
import glob
import warnings
import numpy as np
import argparse,sys,os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from python_tools.get_res import load_whatever
from python_tools.tools_WOI import is_someone_workin_on_it

from nazgul.mount_doom.lens_system import LensSystem
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.combined_modelling_results import get_full_chain,get_res_dir,name_models

from nazgul.Modelling.lib_models import model_res_base,save_data,get_model_res_dir,get_red_chi2
from nazgul.Modelling.lib_models import load_kwargs_result,get_model_plot

def get_g1g2_from_lens(lens,full_chain):
    g1  = full_chain.gamma1_los_lens1.mean()
    sg1 = full_chain.gamma1_los_lens1.std()
    g2  = full_chain.gamma2_los_lens1.mean()
    sg2 = full_chain.gamma2_los_lens1.std()

    glos_hist = np.hypot(full_chain.gamma1_los_lens1,full_chain.gamma2_los_lens1)
    glos  = glos_hist.mean()
    sglos = glos_hist.std() 
    kw_g1g2 = {"g1":g1,"g2":g2,
               "sg1":sg1,"sg2":sg2,
               "glos":glos,"sglos":sglos}
    return kw_g1g2


def get_lenses_names(path_lenses,lenses2ignore=[]):
    lenses_names = []
    for path in glob.glob(path_lenses):
        if any(ln2i in str(path) for ln2i in lenses2ignore):
            continue
        lenses_names.append(Path(path).name)
    return np.array(lenses_names)

def _get_g1g2(path_lenses,lenses2ignore=[]):
    g1g2T,glosT = [],[]
    lens_paths = []
    chi2 = []
    for path in glob.glob(path_lenses):
        for ln2i in lenses2ignore:
            if ln2i in str(path):
                continue
        else:
            model_res_dir = Path(path)
            if is_someone_workin_on_it(model_res_dir):
                # Still under work - results not updated
                continue
            # the following way to get the model name is very weak
            model   = model_res_dir.parent.parent.name
            gallens = load_whatever(model_res_dir/"link_gallens.pkl")
            gallens.unpack()
            lens    = LensSystem.from_GalLens(gallens)
            lens.unpack()
            lens.model_res_dir = model_res_dir #get_model_res_dir(lens,res_dir = res_dir)
            try:
                full_chain = get_full_chain(lens=lens,model=model)
            except FileNotFoundError:
                warnings.warn(f"Lens {lens} results not found - skipping")
                continue
            kw_g1g2 = get_g1g2_from_lens(lens,full_chain)
            g1g2T.append([kw_g1g2["g1"],kw_g1g2["g2"],kw_g1g2["sg1"],kw_g1g2["sg2"]])
            glosT.append([kw_g1g2["glos"],kw_g1g2["sglos"]])
            del full_chain
            kwargs_result  = load_kwargs_result(model_res_dir)
            model_plot     = get_model_plot(model_res_dir,kwargs_result=kwargs_result)
            chi2.append(get_red_chi2(model_plot,verbose=False))
            del model_plot

            lens_paths.append(model_res_dir)
            gc.collect()
    g1g2 = np.transpose(g1g2T)
    glos = np.transpose(glosT)
    kw_glos_tot = {"lens_path":lens_paths,"g1g2":g1g2,"glos":glos,"chi2":np.array(chi2)}
    return kw_glos_tot
    
def get_g1g2(path_lenses,nm_g1g2_data,lenses2ignore=[],reload=True):
    try:
        assert reload
        kw_glos_tot = load_whatever(nm_g1g2_data)
        print(f"Loaded previous result {nm_g1g2_data}")
    except:
        print("Computing g1g2 ex novo")
        kw_glos_tot = _get_g1g2(path_lenses,lenses2ignore=lenses2ignore)
        save_data(kw_glos_tot,nm_g1g2_data,"g1 g1 sg1 sg2 glos sglos")
    return kw_glos_tot
    
def tau(val_i,val_j,sig_i,sig_j): 
    # from dirty_TDC-WST
    tau_val = np.abs(val_i-val_j)/np.sqrt(sig_i**2+sig_j**2) #~ Z val
    return tau_val
    
def compute_tension(out,truth):
    if np.shape(out)[0]==2:
        out_val,out_sig = out
    else:
        out_val = out
        out_sig = None
        raise RuntimeError("Not implemented")
    if np.shape(truth)==():
        truth = truth*np.ones_like(out_val)
    tension = tau(out_val,truth,out_sig,np.zeros_like(truth))
    return tension
    

def str_mod(model_name):
    #hard coded for plot
    if model_name=="simNoShear_gausstE":
        str_m = "EPL + LOS Shear"
    elif model_name=="SNS_multipole":
        str_m = "EPL + LOS Shear + m4"
    elif model_name=="SNS_m134_gausstE":
        str_m = "EPL + LOS Shear + m134"
    else:
        warnings.warn("str_mod not defined, defaulting to actual name")
        str_m = model_name
    return str_m
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="LOS shear results for lens model")

    parser.add_argument('-m1','--model1',type=str,
                        dest="model1",
                        help=f"Name of model 1 - accepted: {name_models}")
    parser.add_argument('-m2','--model2',type=str,
                        dest="model2",
                        help=f"Name of model 2 - accepted: {name_models}")

    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-nr','--no_reload',dest="no_reload",
                        default=False,action="store_true",help=f"Do not reload prev. results")
    parser.add_argument('-mc','--min_chi2',dest="min_chi2",
                        default=None,type=float,help=f"Minimum chi^2 threshold")
    args       = parser.parse_args()
    model1    = args.model1
    model2    = args.model2
    #snaps    = args.snaps # by def. take all snaps
    sim      = args.sim
    subsim   = args.subsim
    simsuite = args.simsuite

    reload       = not args.no_reload
    min_chi2     = args.min_chi2
    min_chi2_str = ""
    if min_chi2 is not None:
        min_chi2_str = f"_minX2_{int(min_chi2)}"
    
    lenses2ignore= [""]
    res_dir1 = get_res_dir(model1,simsuite=simsuite,sim=sim,subsim=subsim)
    res_dir2 = get_res_dir(model2,simsuite=simsuite,sim=sim,subsim=subsim)
    path_lenses1 = str(res_dir1/"snap_*")
    nm_g1g2_data1 = res_dir1/"g1g2.dll"
    kw_glos_tot1  = get_g1g2(path_lenses1,nm_g1g2_data1,lenses2ignore=lenses2ignore,reload=reload)
    glos1,s_glos1 = kw_glos_tot1["glos"]
    chi2_1        = kw_glos_tot1["chi2"]
    lenses_names1 = load_whatever(nm_g1g2_data1)['lens_path']
    lenses_names1 = np.array([str(l.name) for l in lenses_names1])

    if min_chi2 is not None:
        glos1 = glos1[chi2_1<min_chi2]
        s_glos1 = s_glos1[chi2_1<min_chi2]
        lenses_names1 = lenses_names1[chi2_1<min_chi2]

    #####
    # hist of tension
    #
    true_glos = 0
    warnings.warn("Bad coding - truth for now set by hand to 0")
    glos_tension1 = compute_tension(out=[glos1,s_glos1],truth=true_glos)

    path_lenses2 = str(res_dir2/"snap_*")
    #lenses_names2 = get_lenses_names(path_lenses2)
    nm_g1g2_data2 = res_dir2/"g1g2.dll"
    kw_glos_tot2  = get_g1g2(path_lenses2,nm_g1g2_data2,lenses2ignore=lenses2ignore,reload=reload)
    glos2,s_glos2 = kw_glos_tot2["glos"]
    chi2_2        = kw_glos_tot2["chi2"]
    lenses_names2 = load_whatever(nm_g1g2_data2)['lens_path']
    lenses_names2 = np.array([str(l.name) for l in lenses_names2])
    if min_chi2 is not None:
        glos2 = glos2[chi2_2<min_chi2]
        s_glos2 = s_glos2[chi2_2<min_chi2]
        lenses_names2 = lenses_names2[chi2_2<min_chi2]
    #####
    # hist of tension
    #
    true_glos = 0
    warnings.warn("Bad coding - truth for now set by hand to 0")
    glos_tension2 = compute_tension(out=[glos2,s_glos2],truth=true_glos)
    gt1_list,gt2_list  = [],[] 
    for i1,l1 in enumerate(lenses_names1):
        if l1 in lenses_names2:
            i2 = list(lenses_names2).index(l1)
            gt1 = glos_tension1[i1]
            gt1_list.append(gt1)
            gt2 = glos_tension2[i2]
            gt2_list.append(gt2)

    gt1_m = np.mean(gt1_list)
    gt2_m = np.mean(gt2_list)
    model1_str = str_mod(model1)
    model2_str = str_mod(model2)
    for i1,(gt1,gt2) in enumerate(zip(gt1_list,gt2_list)):
            if i1==0:
                tau_str = r" $\left\langle \tau \right\rangle$="
                label1 = model1_str+tau_str+str(np.round(gt1_m,1))
                label2 = model2_str+tau_str+str(np.round(gt2_m,1))
            else:
                label1 = None
                label2 = None
            lnm = lenses_names2[i1].split("LS_Lens_")[1]
            plt.scatter(lnm,gt1,c="r",label=label1)
            plt.axhline(gt1_m,c="r",ls="--")
            plt.scatter(lnm,gt2,c="b",label=label2)
            plt.axhline(gt2_m,c="b",ls="--")
    plt.ylabel(r"$\tau$")
    plt.title(r"Tension with expected $\gamma_{\rm{LOS}}$")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    nm = f"results/models/multipoles_tension_m1{model1}_m2{model2}{min_chi2_str}.png"
    plt.savefig(nm)
    print(f"Saving {nm}")