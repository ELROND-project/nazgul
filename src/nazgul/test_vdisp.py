# Addition: Testing Cored profile as possible reason for theta_E issue
# see notion for 29th may 2026
# mainly, if cored profile instead of SIS, we have 
# theta_E/theta_E(sigma_v) = sqrt(1 - 2 \theta_c/theta_E(sigma_v)) = sqrt( 1- 1/kappa_max )

# So we just need to compute kappa_max to correct theta_E/theta_E(sigma_v) 

import sys
import dill
import argparse
import warnings
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const
from mpl_toolkits.axes_grid1 import make_axes_locatable

from python_tools.get_res import load_whatever

from nazgul.stat_lenses import get_all_gallens
from nazgul.configurations import min_z,max_z,min_mass


from nazgul.Translator.translator import get_vdisp
from nazgul.Translator import std_sim,std_simsuite,std_subsim

from nazgul.Translator.EAGLE.particle_galaxy import compute_principal_axes
from nazgul.project_gal import project_Gal

def get_tE_SIS(lensgal,**kwargs_query):
    lensgal.unpack()
    cosmo = lensgal.cosmo
    Ds    = cosmo.angular_diameter_distance(lensgal.z_source)
    Dls   = cosmo.angular_diameter_distance_z1z2(lensgal.z_lens,lensgal.z_source)
    simsuite = lensgal.Gal.simsuite
    if simsuite=="EAGLE":
        # we do not pass it, but we verify it's the same
        assert kwargs_query["sim"] == lensgal.Gal.sim
    vdisp_stars = get_vdisp(simsuite=simsuite,
                            simpartgal=lensgal.Gal,
                            **kwargs_query)
    
    theta_E_sis = _get_tE_SIS(vdisp_stars,Dls=Dls,Ds=Ds)
    return theta_E_sis

def _get_tE_SIS(v_disp,Dls,Ds):
    vdis_c_ratio = v_disp/const.c
    theta_E_sis_radns = 4*np.pi*(vdis_c_ratio.value)*Dls/Ds
    theta_E_sis = theta_E_sis_radns.value*u.radian.to("arcsec")*u.arcsec
    return theta_E_sis    

def get_kw_tE(lenses,
              out_dll="tmp/del_theta_sis.dll",
              kwargs_query={},
             correct_core=True,
             compute_q=True):
    theta_E_list = []
    theta_sis_list = []
    zs = []
    
    if correct_core:
        rcc_list = []
    if compute_q:
        q_list = []
    for lensgal in lenses:
        lensgal.unpack()
        theta_E_sis = get_tE_SIS(lensgal,**kwargs_query)
        theta_sis_list.append(theta_E_sis.value)
        theta_E_list.append(lensgal.thetaE.value)
        zs.append(lensgal.z_lens)
        print("tE=",np.round(lensgal.thetaE,2))
        print("tE_sis=",np.round(theta_E_sis,2))
        print("Ratio of theta_E computed vs theta_E_sis from vel. disp:",np.round(lensgal.thetaE/theta_E_sis,2))
        if correct_core:
            # MD is actually obtained by averaging 5% of the most massive cells - its center is very stable and fairly accurate, but the MD value ends up being taken as weighted average over > theta_E^2 area -> not an accurate rep. of the maximum kappa
            # take instead the max kappa
            """
            try:
                kw_prj = load_whatever(lensgal.Gal.projection_path)
            except FileNotFoundError:
                # recomputing 
                kw_prj  = project_Gal(lensgal.Gal,z_source_max=lensgal.z_source_max,sample_z_source =lensgal.sample_z_source,min_thetaE=lens.min_thetaE)
            kmax = kw_prj["MD_value"]/lensgal.SigCrit
            """
            # this might still be wrong by being shot-noise dominated
            kmax = np.max(lensgal.kappa_map)
            
            rcc  = np.sqrt(1-(1/kmax))
            print("Ratio of theta_E computed vs theta_E_sis from vel. disp  (core corr.):",np.round(lensgal.thetaE/(theta_E_sis*rcc),2))
            rcc_list.append(rcc)
        if compute_q:
            """
            # the following are the 3D principal axes - not what we want here
            if not hasattr(lensgal,"principal_axes"):
                print("MONKEY PATCH - Adding principal_axes to galaxies that did not have it yet (and updating them)")
                lensgal.Gal.compute_principal_axes()
                lensgal.Gal.store_gal()
            axes = lensgal.Gal.principal_axes
            """
            pth_proj    = lensgal.Gal.projection_path
            kw_proj_res = load_whatever(pth_proj)
            try:
                axes    =  kw_proj_res["principal_axes_2D"]
            except KeyError:
                print("MONKEY PATCH\n###############\nComputing and updating 2D principal_axes to projection results that did not have it yet\n")
                from project_gal import Gal2kwMXYZ,project_kw_parts,get_principal_axis_2D
                kw_parts      = Gal2kwMXYZ(lensgal.Gal)
                kw_parts_proj = project_kw_parts(kw_parts=kw_parts,
                                                 proj_index=lensgal.Gal.proj_index)
                axes = get_principal_axis_2D(kw_parts_proj)
                del kw_parts,kw_parts_proj
                kw_proj_res["principal_axes_2D"] = axes
                with open(lensgal.Gal.projection_path,"wb") as f:
                    dill.dump(kw_proj_res,f)
                print(f"Updating {lensgal.Gal.projection_path}")

            q    = axes["b"]/axes["a"]
            q_list.append(q)

        
            
    theta_sis_list = np.array(theta_sis_list)
    theta_E_list = np.array(theta_E_list)
    kw_tE = {"theta_E_sis":theta_sis_list,"theta_E":theta_E_list,"z_lens":zs}
    if correct_core:
        kw_tE["Ratio_Core_Corr"] = np.array(rcc_list)
    if compute_q:
        kw_tE["q"] = np.array(q_list)
    with open(out_dll,"wb") as f:
        dill.dump(kw_tE,f)
    print(f"Saving {out_dll}")
    return kw_tE
    
def comp_tE_vs_tEsis(reload=True,
                     out_dll = "tmp/theta_sis.dll",
                     name_plot = "tmp/tE_sis_vs_comp.png",
                     title = r"Compare $\theta_E$",
                    kw_get_gallens={},
                    kwargs_query={},
                    correct_core=True,
                    compute_q=True):
    
    if reload:
        try:
            kw_tE = load_whatever(out_dll)
            print(f"Loaded prev. res. {out_dll}")
        except:
            print(f"Failed to load {out_dll}")
            reload = False
    if reload == False:
        lenses  = get_all_gallens(verbose=False,**kw_get_gallens)

        kw_tE = get_kw_tE(lenses=lenses,
                          out_dll=out_dll,
                          kwargs_query=kwargs_query,
                          correct_core=correct_core,
                          compute_q=compute_q)
    theta_sis_list = kw_tE["theta_E_sis"]
    theta_E_list   = kw_tE["theta_E"]
    zs             = kw_tE["z_lens"]
    if correct_core:
        R_cc = kw_tE["Ratio_Core_Corr"]
    if compute_q:
        q = kw_tE["q"]
    fig,axis = plt.subplots(1,2, figsize=(8, 4))
    plt.suptitle(title)
    ax = axis[0]
    sis_range = theta_sis_list.min(),theta_sis_list.max()
    sis_11 = np.linspace(*sis_range,10)
    tE_11 = np.linspace(*sis_range,10)
    ax.plot(sis_11,tE_11,ls="--",c="k",label="1:1")
    im0 = ax.scatter(theta_sis_list,theta_E_list,marker=".",c=zs,cmap="viridis")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if correct_core:
        ax.scatter(theta_sis_list*R_cc,theta_E_list,marker="x",c=zs,
                   cmap="viridis",label=r"$\theta_c$ correction")
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"z$_{lens}$")
    ax.axis('equal')
    ax.set_ylabel(r"$\theta_E$")
    ax.set_xlabel(r"$\theta_E(\sigma_{\rm{v}})$")
    ax.legend()
    
    ax = axis[1]
    x = np.arange(len(theta_sis_list))
    im0 = ax.scatter(x,theta_E_list/theta_sis_list,marker=".",c=zs,cmap="viridis")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if correct_core:
        ax.scatter(x,theta_E_list/(theta_sis_list*R_cc),marker="x",c=zs,
                   cmap="viridis",label=r"$\theta_c$ correction")

    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"z$_{lens}$")
    ax.plot(x,np.ones_like(x),ls="--",c="k",label="1")
    
    ax.set_xlabel(r"Gal")
    ax.set_ylabel(r"$\theta_E$/$\theta_E(\sigma_{\rm{v}})$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(name_plot)
    print(f"Saving {name_plot}")
    plt.close()
    if correct_core:
        tE_ratio = theta_E_list/(theta_sis_list)
        tE_ratio_corr = tE_ratio/R_cc
        nbins =10
        print("N=",len(tE_ratio))
        plt.hist(tE_ratio,bins=nbins,alpha=.5,color="b",label=r"Original N="+str(len(tE_ratio)))
        plt.axvline(np.nanmedian(tE_ratio),c="b",ls="--",label=r"$\theta_E/\theta_{\rm{E, SIS}}(\sigma_{\rm{v}})$="+str(np.round(np.nanmedian(tE_ratio),1)))
        plt.hist(tE_ratio_corr,bins=nbins,color="r",alpha=.5,label=r"$\theta_c$ correction")
        plt.axvline(np.nanmedian(tE_ratio_corr),c="r",ls="--",label=r"$\theta_E/(\theta_{\rm{E, SIS}}(\sigma_{\rm{v}}*\rm{R}_{\rm{cc}})$="+str(np.round(np.nanmedian(tE_ratio_corr),1)))
        plt.title(r"$\theta_E$ ratio from mass map and from $\sigma_v$ with/wo cored profile correction")
        plt.yticks([])
        plt.legend()
        plt.xlabel(r"$\theta_E/\theta_{E,SIS}$")
        plt.tight_layout()
        name_plot2 = name_plot.replace(".png","_2.png")
        name_plot2 = name_plot2.replace(".pdf","_2.pdf")
        plt.savefig(name_plot2)
        print(f"Saving {name_plot2}")
        plt.close()
        
        K_c = 1/(1 -R_cc**2)
        plt.scatter(K_c,theta_E_list/theta_sis_list,c="g")
        x_kc = np.linspace(1,9,100)
        y_kc = np.sqrt(1-1/x_kc)
        plt.plot(x_kc,y_kc,ls="--",c="k",label=r"f($\kappa_{\rm{max}}$) = $\sqrt{1-\frac{1}{\kappa_{\rm{max}}}}$")
        plt.xlabel(r"$\kappa_{\rm{max}}$")
        plt.ylabel(r"$\theta_E$/$\theta_E(\sigma_{\rm{v}})$")
        plt.title(r"$\theta_E$/$\theta_E(\sigma_{\rm{v}})$ wrt $\kappa_{\rm{max}}$ and fit for $\kappa_{\rm{c}}=\kappa_{\rm{max}}$")
        plt.legend()
        name_plot3 = name_plot2.replace("_2.","_kc_vs_tEratio.")
        plt.savefig(name_plot3)
        print(f"Saving {name_plot3}")
        plt.close()
        
        print("Ugly but fast: output lenses for which the correction is not sufficient and plot their kappa map")
        if not reload:
            lenses  = get_all_gallens(verbose=False,**kw_get_gallens)
        #np.where(tE_ratio_corr<thresh_tE_ratio_corr,
        if compute_q:
            fig,ax = plt.subplots(1) 
            x = np.arange(len(theta_sis_list))
            im0 = ax.scatter(x,theta_E_list/theta_sis_list,marker=".",c=q,cmap="viridis")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if correct_core:
                ax.scatter(x,theta_E_list/(theta_sis_list*R_cc),marker="x",c=q,
                           cmap="viridis",label=r"$\theta_c$ correction")
        
                fig.colorbar(im0, cax=cax, orientation='vertical',label=r"q$_{lens}$")
            ax.plot(x,np.ones_like(x),ls="--",c="k",label="1")
            
            ax.set_xlabel(r"Gal")
            ax.set_ylabel(r"$\theta_E$/$\theta_E(\sigma_{\rm{v}})$")
            ax.legend()
            plt.tight_layout()
            name_plot4 = name_plot2.replace("_2.","_q.")
            plt.savefig(name_plot4)
            print(f"Saving {name_plot4}")
            plt.close()

from nazgul.stat_lenses import get_catdir_stat
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Compare computed theta E with expected one obtained from SIS (vel.disp.)")
    parser.add_argument('-snap','--snap',nargs="+",type=int,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-icc','--ignore_correct_core',dest="ignore_correct_core",
                        default=False,action="store_true",help=f"Ignore core correction")
    parser.add_argument('-nr','--no_reload',dest="no_reload",
                        default=False,action="store_true",help=f"Do not reload prev. results")
    parser.add_argument('-ncq','--no_compute_q',dest="no_compute_q",
                        default=False,action="store_true",help=f"Do not compute axis ratio (q)")
    args      = parser.parse_args()
    snaps     = args.snaps #[25,26,27]
    sim       = args.sim
    subsim    = args.subsim
    simsuite  = args.simsuite
    reload    = not args.no_reload
    compute_q = not args.no_compute_q

    cat_dir = get_catdir_stat(snaps=snaps,sim=sim,simsuite=simsuite,
                              subsim=subsim)
    correct_core = not args.ignore_correct_core
    snaps_str = "_".join([str(s) for s in snaps])
    if snaps_str=="":
        snaps_str="all"
    # COLIBRE version
    kw_get_gallens = {"snaps":snaps,
                     "sim":sim,
                     "subsim":subsim,
                     "simsuite":simsuite,} 
    if correct_core:
        warnings.warn("Correcting for cored profile")
    
    if simsuite=="EAGLE":
        if sim=="RefL0050N0752":
            print("SETUP FOR SEAGLE_I")
            kwargs_query   = {"sim":sim,#"RefL0050N0752",
            "min_z":0.1,#0.49,
            "max_z":0.3,#0.51,
            "min_hmr":1,                                   
        
            "min_vel_disp":120,
            "min_mass_stars":1.76e10*.6777}
        else:
            raise RuntimeError("To update in a smart way")
        assert sim==kwargs_query["sim"]
    else:
        # not used
        kwargs_query = {}
    if reload:
        warnings.warn("Reloading prev. results")
    
    name_plot = f"{cat_dir}/tE_sis_vs_comp.png"
    out_dll = f"{cat_dir}/theta_sis.dll"
    title = r"Compare $\theta_E$"
    
    comp_tE_vs_tEsis(reload=reload,
                     title=title,
                    kw_get_gallens=kw_get_gallens,
                    kwargs_query=kwargs_query,
                    out_dll = out_dll,
                    name_plot=name_plot,
                    correct_core=correct_core,
                    compute_q=compute_q
                    )
