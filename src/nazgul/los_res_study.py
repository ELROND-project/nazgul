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
    

        
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="LOS shear results for lens model")

    parser.add_argument('-m','--model',type=str,
                        dest="model",
                        help=f"Name of type of model - accepted: {name_models}")

    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-nr','--no_reload',dest="no_reload",
                        default=False,action="store_true",help=f"Do not reload prev. results")
    parser.add_argument('-mc','--min_chi2',dest="min_chi2",
                        default=None,type=float,help=f"Minimum chi^2 threshold")
    args       = parser.parse_args()
    model    = args.model
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
    res_dir = get_res_dir(model,simsuite=simsuite,sim=sim,subsim=subsim)
    path_lenses = str(res_dir/"snap_*")
    nm_g1g2_data = res_dir/"g1g2.dll"
    kw_glos_tot  = get_g1g2(path_lenses,nm_g1g2_data,lenses2ignore=lenses2ignore,reload=reload)
    chi2             = kw_glos_tot["chi2"]
    g1,g2,sg1,sg2    = kw_glos_tot["g1g2"]
    glos,s_glos = kw_glos_tot["glos"]

    if min_chi2 is not None:
        g1  = g1[chi2<min_chi2]
        g2  = g2[chi2<min_chi2]
        sg1 = sg1[chi2<min_chi2]
        sg2 = sg2[chi2<min_chi2]
        glos = glos[chi2<min_chi2]
        s_glos = s_glos[chi2<min_chi2]
        chi2 = chi2[chi2<min_chi2]
    fig_g1g2,ax_g1g2 = plt.subplots(1)
    ax_g1g2.errorbar(g1,g2,xerr= sg1,yerr=sg2,fmt='',marker='',mew=0,ls="",
                           ecolor="k",elinewidth=.8,label=r"N$_{\rm{models}}$="+str(len(g1)))
    im0 = ax_g1g2.scatter(g1,g2,c=chi2,cmap="viridis")
    divider = make_axes_locatable(ax_g1g2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_g1g2.colorbar(im0, cax=cax, orientation='vertical',label=r"$\chi^2_{\rm{red.}}$")

    g1_wa = np.average(g1,weights=1/np.array(sg1))
    g2_wa = np.average(g2,weights=1/np.array(sg2))
    ax_g1g2.axvline(g1_wa,ls="-.",c="g",label=r"<$\gamma_1$> "+str(np.round(g1_wa,2)))
    ax_g1g2.axhline(g2_wa,ls="-.",c="g",label=r"<$\gamma_2$> "+str(np.round(g2_wa,2)))
    ax_g1g2.axvline(0,ls="-",c="grey",alpha=.3)
    ax_g1g2.axhline(0,ls="-",c="grey",alpha=.3)
    ax_g1g2.set_xlabel(r"$\gamma_{\rm{LOS, 1}}$")
    ax_g1g2.set_ylabel(r"$\gamma_{\rm{LOS, 2}}$")

    ax_g1g2.set_title(r"Scatter of $\gamma_{\rm{LOS}}$ components for "+str(model) )  
    nm_g1g2_fig = res_dir/f"g1g2_scatter{min_chi2_str}.png"
    ax_g1g2.legend()
    fig_g1g2.savefig(nm_g1g2_fig)
    print(f"Saving {nm_g1g2_fig}")
    plt.close(fig_g1g2)

    
    plt.hist(glos,label=r"N$_{\rm{models}}$="+str(len(g1)),bins=24)
    plt.xlabel(r"$\gamma_{\rm{LOS}}$")
    plt.title(r"Histogram of $\gamma_{\rm{LOS}}$ for "+str(model))  
    plt.legend()
    nm_glos_fig = res_dir/f"glos_hist{min_chi2_str}.png"
    plt.savefig(nm_glos_fig)
    print(f"Saving {nm_glos_fig}")
    plt.close()

    plt.errorbar(chi2,glos,yerr=s_glos,fmt="ko")
    plt.title(r"$\gamma_{\rm{LOS}}$ for "+str(model)+r" wrt $\chi^2_{\rm{red.}}$")  
    plt.xlabel(r"$\chi^2{\rm{red.}}$")
    plt.ylabel(r"$\gamma_{\rm{LOS}}$")
    
    nm_glosChi2_fig = res_dir/f"glosVschi2{min_chi2_str}.png"
    plt.savefig(nm_glosChi2_fig)
    print(f"Saving {nm_glosChi2_fig}")
    plt.close()
    
    #####
    # hist of tension
    #
    true_glos = 0
    warnings.warn("Bad coding - truth for now set by hand to 0")
    glos_tension = compute_tension(out=[glos,s_glos],truth=true_glos)
    plt.hist(glos_tension,label=r"N$_{\rm{models}}$="+str(len(g1)),bins=24)
    plt.xlabel(r"$\tau(\gamma_{\rm{LOS}})$")
    plt.title(r"Histogram of Tension of $\gamma_{\rm{LOS}}$ for "+str(model))  
    plt.legend()
    nm_glos_tens_fig = res_dir/f"glos_tension_hist{min_chi2_str}.png"
    plt.savefig(nm_glos_tens_fig)
    print(f"Saving {nm_glos_tens_fig}")
    plt.close()
