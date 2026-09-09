# Study statistically core and power law index from the 1D mass distribution
# to add: compare with lens model results
import sys
import dill
import warnings
import argparse
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module
from scipy.optimize import curve_fit
from chainconsumer import Chain, ChainConsumer

from python_tools.get_res import load_whatever
from nazgul.lib_plot import warm_colors as warm
from nazgul.mount_doom.lens_system import LensSystem
from nazgul.plot_AMRxpart import get_kw_1D_density,get_savedir_plots
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.combined_modelling_results import get_full_chain,get_res_dir,get_model_from_res_dir
# we select lenses the same way we select them for modelling:
from nazgul.Modelling.lib_models import get_lenses2model,get_model_res_dir
# thus we need to define which model to consider
# for now simNoShear:
std_lens_model = "simNoShear"
#from nazgul.Modelling.model_simNoShear import res_dir_base
#warnings.warn(f"Using lens cat. obtained from {res_dir_base}")
# note: this is for convenience, but we could implement it exnovo, indep. of modelling (at least for the study of the mass profile - if we compare to the lens model it's different)

def rho_cored(r,rho0,r_core,gamma):
    rho = rho0/(r**2+  r_core**2)**(gamma)
    return rho

def rho_cored_smooth(r,rho0,r_core,gamma,alpha):
    # following Pierre's function (eq. to mine if alpha==2 and scaling appropriately gamma and rho0)
    rho = rho0 * (1 + (r/r_core)**alpha)**((3-gamma)/alpha)
    return rho
    
def log_rho_cored(log_r,log_rho0,log_r_core,gamma):
    r      = 10**log_r
    r_core = 10**log_r_core
    rho0   = 10**log_rho0
    rho    = rho_cored(r,rho0,r_core,gamma)
    return np.log10(rho)

# to consider if not using the fit log_rho_cored

std_kw_get_all_gallens = {"sim":std_sim,
                          "subsim":std_subsim,
                          "simsuite":std_simsuite,
                          "snaps":[]}

r_lbl = r"log$_{10}$r [kpc]"
sig_lbl = r"log$_{10}\Sigma$ [10$^9$ M$_{\odot}$/kpc]"

def get_core_gamma_1gal(gal_lens,res_dir,single_plot=True,smooth=False):
    print("\nLoading gal lens "+gal_lens.name)
    gal_lens.unpack()
    kw_1d = get_kw_1D_density(gal_lens.Gal,gal_lens.proj_index)
            
    r = kw_1d["r_all"].value #kpc
    Sigma = kw_1d["Sigma_encl_all"].value
    Sigma_scaled = Sigma/1e9 # 1e9 SolMass/kpc
    if not smooth:
        fit_prm,fit_cov = curve_fit(rho_cored,r,Sigma_scaled)
    else:
        print("Fitting for smoothed power law")
        fit_prm,fit_cov = curve_fit(rho_cored_smooth,r,Sigma_scaled)
    rho0    = fit_prm[0]
    core    = fit_prm[1]
    gamma   = fit_prm[2]
    print("Results from 1D density Cored Power Law fit")
    print("Core :",np.round(core,2))
    print("Gamma :",np.round(gamma,2))
    log_r,log_Sig = np.log10(r),np.log10(Sigma_scaled)
    savedir = get_savedir_plots(gal_lens.Gal)
    if single_plot:
        nm_fig = f"{savedir}/1D_Density_fit.png"
        # if plot doesn't exits (to implement)  
        fig,ax = plt.subplots()
        ax.set_title("Density profile fit with cored power law")
        ax.plot(log_r,log_Sig,c="b",label="data")
        ax.plot(log_r,np.log10(rho_cored(r,*fit_prm)),c="g",
                 ls="--",label="fit cored PL")
        ax.set_xlabel(r_lbl)
        ax.set_ylabel(sig_lbl)
        ax.axvline(np.log10(core),ls="-.",c="g",label="core")
        ax.legend()
        fig.savefig(nm_fig)
        print(f"Saving {nm_fig}")
        plt.close(fig)

    # get gamma from lens modelling to compare it
    # the following is just used for convenience
    lens = LensSystem.from_GalLens(gal_lens)
    gal_lens.model_res_dir = get_model_res_dir(lens,res_dir=res_dir)
    try:
        lens_model = get_model_from_res_dir(res_dir)
        full_chain = get_full_chain(gal_lens,model=lens_model)
        gamma_lns = full_chain["gamma_lens0"].to_numpy() 
        gamma_med,gamma_std = np.median(gamma_lns),np.std(gamma_lns)
        gamma_lens = [gamma_med,gamma_std]
        del full_chain,gamma_lns
    except FileNotFoundError as e:
        warnings.warn(f"{e} \nWe ignore it and set it to NaN")
        gamma_lens = [np.nan,np.nan]
    del gal_lens,lens
    # for plot overlap of 1D density profiles (normalised by core and max dens) 
    log_r_scaled   = log_r/np.log10(core)
    log_Sig_scaled = log_Sig/np.log10(rho0)
    return gamma,core,gamma_lens,log_r_scaled,log_Sig_scaled

def get_core_gamma(reload=True,
                    res_dir=None,  
                    min_thetaE=None,
                    single_plot=True,
                    lenses2skip=[],
                    kw_get_all_gallens=std_kw_get_all_gallens):
    nm_kw_core_gamma = f"{res_dir}/kw_res_core_gamma.dll"
    if reload:
        try:
            kw_core_gamma = load_whatever(nm_kw_core_gamma)
            print(f"Loaded {nm_kw_core_gamma}") 
            kw_get_all_gallens_old = kw_core_gamma.get("kw_get_all_gallens",None)
            if kw_get_all_gallens_old is None:
                warnings.warn("MONKEY PATCH- update kw_get_all_gallens with kw_core_gamma")
                kw_core_gamma["kw_get_all_gallens"] = kw_get_all_gallens
                with open(nm_kw_core_gamma,"wb") as f:
                    dill.dump(kw_core_gamma,f)
            if kw_get_all_gallens == kw_get_all_gallens_old:
                return kw_core_gamma
            else:
                print(f"File {nm_kw_core_gamma} does not match the lenses - recomputing")
                raise FileNotFoundError
        except FileNotFoundError:
            print("Tried and failed to load previous result. Recomputing")

    gal_lenses  = get_lenses2model(res_dir=res_dir,
                                   reload=True,
                                   min_thetaE=min_thetaE,
                                   skip_lenses=lenses2skip,
                                   n_lenses=np.nan, #select all of the them
                                   kw_get_all_gallens=kw_get_all_gallens)
    gammas,cores = [],[]
    gamma_lenses   = []
    fig_overlap,ax_overlap = plt.subplots()
    nm_fig_overap = f"{res_dir}/1D_overlap.png"
    ax_overlap.set_xlabel(r_lbl)
    ax_overlap.set_ylabel(sig_lbl)
    ax_overlap.set_title("Overlap of scaled 1D density profiles")

    lenses_names = []
    for i,gal_lens in enumerate(gal_lenses): 
        
        gamma,core,gamma_lens,log_r_scaled,log_Sig_scaled = get_core_gamma_1gal(gal_lens,res_dir=res_dir,
                                                                  single_plot=single_plot)

        _lbl = None
        if i==0:
            _lbl = "N = "+str(len(gal_lenses))
        ax_overlap.plot(log_r_scaled,log_Sig_scaled,
                        alpha=.5,c="grey",label=_lbl)

        # store res
        cores.append(core)
        gammas.append(gamma)
        gamma_lenses.append(gamma_lens)
        lenses_names.append(gal_lens.name)
        
    fig_overlap.savefig(nm_fig_overap)
    print(f"Saving {nm_fig_overap}")
    plt.close(fig_overlap)

    gammas = np.array(gammas)
    cores  = np.array(cores)
    gamma_lenses = np.array(gamma_lenses).T # [0] =<gamma>, [1] = std(gamma)
    kw_core_gamma = {"gammas":gammas,
                     "cores":cores,
                     "gamma_lenses":gamma_lenses,
                     "lenses_names":lenses_names,
                     "kw_get_all_gallens":kw_get_all_gallens}

    with open(nm_kw_core_gamma,"wb") as f:
        dill.dump(kw_core_gamma,f)
    print(f"Saved {nm_kw_core_gamma}") 
    return kw_core_gamma

def plot_core_gamma(kw_core_gamma,res_dir):
    cores = kw_core_gamma["cores"]
    gammas = kw_core_gamma["gammas"]
    gamma_lenses = kw_core_gamma["gamma_lenses"]
    
    # plot summ. stat of core and gamma
    c = ChainConsumer()
    prms_nms = ["Core",r"$\gamma$"]
    c.add_chain(Chain(
        samples=pd.DataFrame(np.array([cores,gammas]).T,columns=prms_nms),
        name="",
        shade=True,
        color=warm[0],
        shade_gradient = 0.8, linewidth=3.0) )
    fig = c.plotter.plot(columns=prms_nms)
    nm_fig = f"{res_dir}/distr_coreVsGamma_1Ddens.png"
    plt.savefig(nm_fig)
    print(f"Saving {nm_fig}")
    plt.close(fig)
    
    # compare gamma vs core
    ind_sort = np.argsort(gammas)
    plt.scatter(gammas[ind_sort],cores[ind_sort],c="k")
    plt.title(r"1D dens. cored PL fit: $\gamma$ vs r$_{\rm{core}}$")
    plt.ylabel(r"log$_{10}$r$_{\rm{core}}$ [kpc]")
    plt.xlabel(r"$\gamma$")
    nm_fig = f"{res_dir}/dens_1dfit_gammaVs_core.png"
    plt.savefig(nm_fig)
    print(f"Saving {nm_fig}")
    plt.close()
    
    # compare gammas
    plt.errorbar(gammas[ind_sort],gamma_lenses[0][ind_sort],
                 yerr=gamma_lenses[1][ind_sort])
    plt.scatter(gammas[ind_sort],gamma_lenses[0][ind_sort],c="k")
    plt.title(r"$\gamma_{\rm{Density}}$ vs $\gamma_{\rm{PEMD}}$")
    plt.xlabel(r"$\gamma_{\rm{Density}}$")
    plt.ylabel(r"$\gamma_{\rm{Lens Model}}$")
    nm_fig = f"{res_dir}/gamma_dens_Vs_gamma_lens.png"
    plt.savefig(nm_fig)
    print(f"Saving {nm_fig}")
    plt.close()
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description=" Study statistically core and power law index from the 1D mass distribution")
    parser.add_argument('-mtE','--min_thetaE',type=float,dest="min_thetaE",default=None,help=f"Min theta_E for the gal to be considered a lens")
    parser.add_argument('-snap','--snap',nargs="+",type=str,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-lm','--lens_model',type=str,dest="lens_model",default=std_lens_model,help=f"Lens model to compare the power law to (and get the lens list from)")
    parser.add_argument('-nsp','--no_single_plot',dest="single_plot",
                        default=True,action="store_false",help=f"Do not plot each 1D density fit")
    parser.add_argument('-nr','--no_reload',dest="reload",
                        default=True,action="store_false",help=f"Do not try to reload prev. res.")

    args        = parser.parse_args()
    min_thetaE  = args.min_thetaE 
    snaps       = args.snaps #[25,26,27]
    sim         = args.sim
    subsim      = args.subsim
    simsuite    = args.simsuite
    lens_model  = args.lens_model
    single_plot = args.single_plot
    reload      = args.reload
    
    # picked by hand "bad" lenses ->
    lenses2skip = ["LS_Lens_Gn75SGn0_Prj1","LS_Lens_Gn4SGn0_Prj2","LS_Lens_Gn4SGn0_Prj0","LS_Lens_Gn14SGn0_Prj1",
                   "LS_Lens_Gn71SGn0_Prj2","LS_Lens_Gn7SGn1_Prj2","LS_Lens_Gn15SGn1_Prj0","LS_Lens_Gn15SGn1_Prj0",
                   "LS_Lens_Gn6SGn0_Prj2","LS_Lens_Gn1SGn2_Prj1","LS_Lens_Gn42SGn0_Prj1","LS_Lens_Gn18SGn0_Prj2",
                   "LS_Lens_Gn18SGn0_Prj0","LS_Lens_Gn18SGn0_Prj0","LS_Lens_Gn18SGn0_Prj2","LS_Lens_Gn22SGn1_Prj2",
                   "LS_Lens_Gn22SGn1_Prj1","LS_Lens_Gn66SGn0_Prj1","LS_Lens_Gn45SGn0_Prj0","LS_Lens_Gn33SGn0_Prj2"]

    res_dir            = get_res_dir(lens_model)
    kw_get_all_gallens = {"sim":sim,
                          "subsim":subsim,
                          "simsuite":simsuite,
                          "snaps":snaps}
    kw_core_gamma = get_core_gamma(reload=reload,
                                   res_dir = res_dir,  
                                   min_thetaE=min_thetaE,
                                   kw_get_all_gallens = kw_get_all_gallens,
                                   single_plot=single_plot,
                                   lenses2skip=lenses2skip)
    plot_core_gamma(kw_core_gamma,res_dir)