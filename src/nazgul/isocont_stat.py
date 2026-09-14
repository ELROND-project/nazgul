"""
Study the statistic of isocontours of available lenses
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from python_tools.tools import to_dimless

from nazgul.stat_lenses import get_all_gallens_paths
from nazgul.mount_doom.cracks_of_doom import LoadLens

from nazgul.fit_iso_ell import fit_isodens,fit_isopot,_err_map_type #,plot_isodens,plot_isopot
from nazgul.Translator import std_sim,std_simsuite,std_subsim

from nazgul.pathfinder import std_data_dir,tmp_dir

def plot_first_column(axis,logr,th_nrm,gamma_der,eps,Dpa,x0,y0,bxdk,*args,**kwargs):
    ax = axis[0][0]
    ax.plot(logr,gamma_der,alpha=.3,color="grey",ls="-")
    if i==0:
        ax.set_xlabel(r'log10(Semimajor axis [kpc])')
        ax.set_ylabel(r"$\gamma$ []")
        ax.set_title(r"Fit $\gamma$(r)")
    ax = axis[1][0]
    ax.plot(th_nrm,eps,alpha=.3,color="grey",ls="-")
    if i==0:
        ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
        ax.set_ylabel(r"$\epsilon/\epsilon(\theta_E)$ []")
        ax.set_title(r"Ellipticity $\epsilon$ normalised for $\epsilon(\theta_E$)")
    ax = axis[2][0]
    ax.plot(th_nrm,Dpa,alpha=.3,color="grey",ls="-")
    if i==0:
        ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
        ax.set_ylabel(r"P.A.-P.A.($\theta_E$) [$^o$]")
        ax.set_title(r"Pointing Angle P.A. rescaled by P.A.($\theta_E$)")
    
    ax = axis[3][0]
    ax.plot(th_nrm,x0,alpha=.3,color="grey",ls="-")
    if i==0:
        ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
        ax.set_ylabel(r'(X0-Xcnt)/$\theta_E$ []')
        ax.set_title(r"Drift$_x$ = (Center$_x$-Cnt[0]$_x$)/$\theta_E$)")
    ax = axis[4][0]
    ax.plot(th_nrm,y0,alpha=.3,color="grey",ls="-")
    if i==0:
        ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
        ax.set_ylabel(r'(Y0-Ycnt)/$\theta_E$ []')
        ax.set_title(r"Drift$_y$ = (Center$_y$-Cnt[0]$_y$)/$\theta_E$)")
    
    ax = axis[5][0]
    ax.plot(th_nrm,bxdk,alpha=.3,color="grey",ls="-")
    if i==0:
        ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
        ax.set_ylabel(r"b4/b4($\theta_E$) []")
        ax.set_title(r"Boxy-diskiness b4 normalised for b4($\theta_E$)")
    return axis
    

def get_theta_norm(logr,gal_lens):
    r       = 10**np.array(logr) #kpc
    theta   = r*to_dimless(gal_lens.arcXkpc) #arcsec
    th_nrm  = theta/to_dimless(gal_lens.thetaE) #dimless
    return th_nrm

def get_gamma_der(isolist,logr):    
    y = np.log10(isolist.intens[1:])
    return -np.gradient(y,logr)

def debug_cat_failed_isofit_lens(lens_path,type_map="kappa",message=None):
    with open(f"tmp/debug_cat_failed_isofit_lens_{type_map}.txt", "a") as myfile:
        str_to_write = str(lens_path)
        if message:
            str_to_write += ": "+message
        str_to_write += "\n"
        myfile.write(str_to_write)

def study_fit(gal_lens,type_map="kappa",reload=True):
    _err_map_type(type_map)

    try:
        if type_map=="kappa":
            kw_res  = fit_isodens(gal_lens,reload=reload)
        elif type_map=="psi":
            kw_res  = fit_isopot(gal_lens,reload=reload)

    except Exception as e:
        print(f"Lens {gal_lens_path} failed:\n{e}")
        print("skipping")
        debug_cat_failed_isofit_lens(gal_lens_path,type_map=type_map,message=str(e))
        return None
    logr    = kw_res["loglogfit"]["fitx"]
    th_nrm  = get_theta_norm(logr,gal_lens)
    i_thetaE = np.argmin(np.abs(th_nrm-1))
    if type_map=="kappa":
        isocont = "isodens"
    else:
        isocont = "isopot"
    isolist       = kw_res[isocont]["isolist"]
    gamma_der     = get_gamma_der(isolist,logr)
    gamma_fit_fix = -kw_res["loglogfit"]["popt_log"][1]

    eps    = isolist.eps[1:]
    pa     = isolist.pa[1:]/np.pi*180. # deg
    # correct for pointing angle around ~0
    if np.any(np.abs(np.diff(pa))>135):
        pa[np.where(pa>135)] -=180
    pa_i = pa[i_thetaE]
    Dpa  = pa-pa_i

    geom  = kw_res[isocont]["geom"]
    dx    = (isolist.x0[1:]-geom.x0) #pix
    dxarc = dx*to_dimless(gal_lens.deltaPix) #arcsec
    x0    = dxarc/to_dimless(gal_lens.thetaE) # dimless
    
    dy    = (isolist.y0[1:]-geom.y0) #pix
    y0    = to_dimless(dy*gal_lens.deltaPix/gal_lens.thetaE) #dimless

    bxdk   = isolist.b4[1:] # boxy-diskyness
    bxdk   = bxdk
    
    kw_res = {"logr":logr,
              "gamma_fit_fix":gamma_fit_fix,
              "gamma_der":gamma_der,
              "eps":eps,
              "Dpa":Dpa,
              "bxdk":bxdk,
              "x0":x0,
              "y0":y0}
    return kw_res
    
def plot_second_column(kw_res,fig,axis):
    gamma_distr = kw_res["gamma_distr"]
    ellipt_distr = kw_res["ellipt"]
    DPA_distr = kw_res["DPA"]
    drift_x   = kw_res["drift_x"]
    drift_y    = kw_res["drift_y"]
    bxdk_distr = kw_res["bxdk"]
    # define limits to ignore outliers

    n_bins = 30
    ax = axis[0][1]
    ax.hist(gamma_distr,bins=n_bins)
    med_gamma = np.nanmedian(gamma_distr)
    std_gamma = np.nanstd(gamma_distr)
    ax.axvline(med_gamma,ls="--",c="r",label=r"median($\gamma$)="+str(np.round(med_gamma,2))
               +"+-"+str(np.round(std_gamma,2)))
    ax.set_xlabel(r"$\gamma$(r)")
    ax.set_title(r"Distr. fit $\gamma$")
    ax.legend()

    ax = axis[1][1]
    ax.hist(ellipt_distr,bins=n_bins)
    med_ell = np.nanmedian(ellipt_distr)
    std_ell = np.nanstd(ellipt_distr)
    ax.axvline(med_ell,ls="--",c="r",label=r"median($\epsilon$)="+str(np.round(med_ell,2))  +"+-"+str(np.round(std_ell,2)))
    ax.set_xlabel(r"$\epsilon$")
    ax.set_title(r"Distr. ellipticity $\epsilon$")
    ax.legend()

    ax = axis[2][1]
    ax.hist(DPA_distr,bins=n_bins)
    med_dpa = np.nanmedian(DPA_distr)
    std_dpa = np.nanstd(DPA_distr)
    ax.axvline(med_dpa,ls="--",c="r",label=r"median(D P.A.)="+str(np.round(med_dpa,2))+"+-"+str(np.round(std_dpa,2)))
    ax.set_xlabel(r"P.A. - P.A.($\theta_E$)")
    ax.set_title(r"Distr. Pointing Angle - PA$(\theta_E)$")
    ax.legend()

    ax = axis[3][1]
    ax.hist(drift_x,bins=n_bins)
    med_dx = np.nanmedian(drift_x)
    std_dx = np.nanstd(drift_x)

    ax.axvline(med_dx,ls="--",c="r",label=r"median(Drift x)="+str(np.round(med_dx,2))+"+-"+str(np.round(std_dx,2)))
    ax.set_xlabel(r"Drift x")
    ax.set_title(r"Distr. Drift x")
    ax.legend()

    ax = axis[4][1]
    ax.hist(drift_y,bins=n_bins)
    med_dy = np.nanmedian(drift_y)
    std_dy = np.nanstd(drift_y)

    ax.axvline(med_dy,ls="--",c="r",label=r"median(Drift y)="+str(np.round(med_dy,2))+"+-"+str(np.round(std_dy,2)))
    ax.set_xlabel(r"Drift y")
    ax.set_title(r"Distr. Drift y")
    ax.legend()

    ax = axis[5][1]

    ax.hist(bxdk_distr,bins=n_bins)
    med_fb4 = np.nanmedian(bxdk_distr)
    std_fb4 = np.nanstd(bxdk_distr)
    ax.axvline(med_fb4,ls="--",c="r",label=r"median(b4)="+str(np.round(med_fb4,2))+"+-"+str(np.round(std_fb4,2)))
    ax.set_xlabel(r"b4$")
    ax.set_title(r"Distr. Boxy-Diskyness parameter b4")

    ax.legend()
    fig.tight_layout()
    return fig,axis

def study_isoconts(kw_res_i,map):
    kw_cont = kw_res_i["isocont"]
    """
    # map should be rescaled accordingly to the isocont fit 
    fig.imshow(map) 
    plot_conts(fig,conts)
    """
    raise RuntimeError("Pragma: no cover (to implement)")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Study statistic of isocontours (kappa and psi) of lenses")
    parser.add_argument('-snap','--snap',nargs="+",dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-nr','--no_reload',dest="reload",
                        default=True,action="store_false",help=f"Do not try to reload prev. res.")
    
    args      = parser.parse_args()
    snaps     = args.snaps
    sim       = args.sim
    subsim    = args.subsim
    simsuite  = args.simsuite
    reload    = args.reload


    res_dir         = tmp_dir #tmp res dir, to improve
    gal_lenses_path = get_all_gallens_paths(snaps=snaps,
                                            simsuite=simsuite,
                                            sim=sim,
                                            subsim=subsim,
                                            data_dir=std_data_dir)
    
    list_res = ["ellipt","gamma_distr","DPA","bxdk","drift_x","drift_y"]
    kw_res_kappa,kw_res_psi = {},{}
    for l in list_res:
        kw_res_kappa[l] = []
        kw_res_psi[l] = []
    fig_p,axis_p = plt.subplots(6,2,figsize=(10,23))
    fig_k,axis_k = plt.subplots(6,2,figsize=(10,23))
    for i,gal_lens_path in enumerate(gal_lenses_path):   
        gal_lens = LoadLens(gal_lens_path)
        if gal_lens is False:
            continue
        print("\nIsofitting of Lens "+gal_lens.name+"\n")
        print("\n\nFittting kappa\n#############\n#########")
        kw_res_kappa_i = study_fit(gal_lens,type_map="kappa",reload=reload)
        print("\n\nFittting psi\n#############\n#########")
        kw_res_psi_i   = study_fit(gal_lens,type_map="psi",reload=reload)

        if kw_res_kappa_i is not None:
            ellipt_k   = np.nanmedian(kw_res_kappa_i["eps"])
            DPA_k      = np.nanmedian(kw_res_kappa_i["Dpa"])
            bxdk_k     = np.nanmedian(kw_res_kappa_i["bxdk"])
            drift_x_k  = np.nanmedian(kw_res_kappa_i["x0"])
            drift_y_k  = np.nanmedian(kw_res_kappa_i["y0"])
            gamma_distr_k = kw_res_kappa_i["gamma_fit_fix"]
            gamma_der_k   = kw_res_kappa_i["gamma_der"]
            kw_res_kappa["ellipt"].append(ellipt_k)
            kw_res_kappa["gamma_distr"].append(gamma_distr_k)
            kw_res_kappa["DPA"].append(DPA_k)
            kw_res_kappa["bxdk"].append(bxdk_k)
            kw_res_kappa["drift_x"].append(drift_x_k)
            kw_res_kappa["drift_y"].append(drift_y_k)
            logr_k      = kw_res_kappa_i["logr"]
            th_nrm_k    = get_theta_norm(logr_k,gal_lens)
            kw_res_kappa_i["th_nrm"] = th_nrm_k
            axis_k      = plot_first_column(axis_k,**kw_res_kappa_i)
                            
        if kw_res_psi_i is not None:
            ellipt_p   = np.nanmedian(kw_res_psi_i["eps"])
            DPA_p      = np.nanmedian(kw_res_psi_i["Dpa"])
            bxdk_p     = np.nanmedian(kw_res_psi_i["bxdk"])
            drift_x_p  = np.nanmedian(kw_res_psi_i["x0"])
            drift_y_p  = np.nanmedian(kw_res_psi_i["y0"])
            gamma_distr_p = kw_res_psi_i["gamma_fit_fix"]
            gamma_der_p   = kw_res_psi_i["gamma_der"]
            
            kw_res_psi["ellipt"].append(ellipt_p)
            kw_res_psi["gamma_distr"].append(gamma_distr_p)
            kw_res_psi["DPA"].append(DPA_p)
            kw_res_psi["bxdk"].append(bxdk_p)
            kw_res_psi["drift_x"].append(drift_x_p)
            kw_res_psi["drift_y"].append(drift_y_p)
                                    
            logr_p      = kw_res_psi_i["logr"]
            th_nrm_p    = get_theta_norm(logr_p,gal_lens)
            kw_res_psi_i["th_nrm"] = th_nrm_p
            
            axis_p      = plot_first_column(axis_p,**kw_res_psi_i)
    
    fig_k,axis_k = plot_second_column(kw_res_kappa,fig_k,axis_k)
    fig_p,axis_p = plot_second_column(kw_res_psi,fig_p,axis_p)
    n_lenses_k   = str(len(kw_res_kappa["gamma_distr"]))
    n_lenses_p   = str(len(kw_res_psi["gamma_distr"]))
    nm_k = f"{res_dir}/distr_isoparams_kappa.png"
    print(f"Saving {nm_k}")
    fig_k.suptitle(r"Iso-$\kappa$ fit results ["+n_lenses_k+" lenses]")
    fig_k.savefig(nm_k)
    nm_p = f"{res_dir}/distr_isoparams_psi.png"
    print(f"Saving {nm_p}")
    fig_p.suptitle(r"Isopotential fit results ["+n_lenses_p+" lenses]")
    fig_p.savefig(nm_p)
