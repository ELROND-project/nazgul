# copy from plot_one_gal
# the aim is to reproduce Fig.5 of SEAGLE I 
# the  Surface density profiles of DM, stars, gas and the total mass of a typical ETG from EAGLE.
import os
import sys,gc
import copy
import dill
import warnings
import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const

from python_tools.get_res import load_whatever
from python_tools.tools import mkdir,ensure_unit
from nazgul.project_gal import project_kw_parts,dens_map_AMR,cells2SigEnclRad,cells2MRad,ProjGal

from nazgul.lib_cosmo import SigCrit
from nazgul.lib_plot import base_colors
from nazgul.AMR2D_PLL import plot_AMR_cells
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.Translator.translator import PartGal,Gal2kwMXYZ,Gal2kwMXYZ_part,get_sim_func

def get_kw_extents_RE(Gal,RE,tE,scale_tE_cutout=10):
    cutout_arcs = tE*scale_tE_cutout
    cutout_kpc  = RE*scale_tE_cutout
    kw_extents = {"extent_arcsec":[-cutout_arcs.value,cutout_arcs.value,
                                   -cutout_arcs.value,cutout_arcs.value],
                  "extent_kpc":[-cutout_kpc.value,cutout_kpc.value,
                                -cutout_kpc.value,cutout_kpc.value]}
    return kw_extents

default_kw_densmap={"max_particles":100,
                    "min_area": 0.1*u.kpc*u.kpc,
                    "dens_thresh": 0.*u.Msun/(u.kpc**2),
                    "clip":True}

def _get_kw_1D_density_enclosed(Gal,kw_1D_dens,kw_2Ddens_all,plot_figall=True):
    r_all,Sigma_encl_all   = cells2SigEnclRad(kw_2Ddens_all)
    kw_1D_dens = _get_kw_1D_density_gen(kw_1D_dens,Gal,r,Sigma_encl_all,Sigma_encl_all,Sigma_crit,plot_figall)
    kw_1D_dens["Sigma_encl_all"] = Sigma_encl_all # Msun/kpc^2
    return kw_1D_dens

def _get_kw_1D_density_not_enclosed(Gal,kw_1D_dens,kw_2Ddens_all,plot_figall=True):
    r_all,M_all,area_all   = cells2MRad(kw_2Ddens_all)
    Sigma_all = M_all/area_all
    
    # still need to compute this for the theta_E
    # Cumulative sum
    cumulative_mass = np.cumsum(M_all)
    cumulative_area = np.cumsum(area_all)
    
    # Compute enclosed density Sigma(<r)
    Sigma_encl_all = cumulative_mass/cumulative_area

    kw_1D_dens = _get_kw_1D_density_gen(kw_1D_dens,Gal,r,Sigma_all,Sigma_encl_all,Sigma_crit,plot_figall)
    kw_1D_dens["Sigma_all"] = Sigma_all # Msun/kpc^2
    return kw_1D_dens

def _get_kw_1D_density_gen(kw_1D_dens,Gal,r,Sigma,Sigma_encl,plot_figall=True):
    r_all = r.to("kpc")
    
    Sigma = Sigma.to("Msun/kpc^2")
    Sigma_crit     = ensure_unit(Sigma_crit,Sigma.unit)
    
    kw_1D_dens["r_all"] = r #kpc
    Sigma_crit = kw_1D_dens["Sigma_crit"] # Msun/kpc^2
    
    MD_coord_all = copy.copy(kw_2Ddens_all["MD_coords"])
    kw_1D_dens["MD_coords_all"] = MD_coord_all #
    
    RE         = np.interp(Sigma_crit.value, 
                           Sigma_encl.value[::-1], 
                           r[::-1].value)*r.unit # kpc
    arcXkpc    = Gal.cosmo.arcsec_per_kpc_proper(Gal.z).to("arcsec/kpc") # arcsec/kpc
    tE         = RE*arcXkpc # arcsec
    kw_extents = get_kw_extents_RE(Gal,RE,tE,scale_tE_cutout)
    
    kw_1D_dens["tE"]         = tE
    kw_1D_dens["arcXkpc"]    = arcXkpc
    kw_1D_dens["kw_extents"] = kw_extents
    
        
    if plot_figall:
        savedir = get_savedir_plots(Gal,savedir=savedir)
        figall,axall = plot_AMR_cells(kw_2Ddens_all,kw_extents=kw_extents)        
        figall.suptitle("AMR of total mass projection")
        nm_AMR = f"{savedir}/AMR_full_proj{proj_index}.png"
        figall.savefig(nm_AMR)
        print(f"Saved {nm_AMR}") 
        plt.close(figall)
    # free memory
    del kw_2Ddens_all
    return kw_1D_dens
    
def get_kw_1D_density(Gal,
                      proj_index = 0,
                      scale_tE_cutout = 10,
                      enclosed=True,
                      reload=True,
                      kw_densmap = default_kw_densmap,
                      savedir=None,
                      plot_figall=True):
    PrjGal = ProjGal(Gal,proj_index)
    if enclosed:
        nm_kw_1D_dens = PrjGal.proj_dir/f"kw_1D_dens_{proj_index}.dll"
    else:
        nm_kw_1D_dens = PrjGal.proj_dir/f"kw_1D_dens_{proj_index}_not_enclosed.dll"

    try:
        assert reload
        kw_1D_dens = load_whatever(nm_kw_1D_dens)
        print(f"Reloaded {nm_kw_1D_dens}")
        return kw_1D_dens
    except:
        kw_1D_dens        = {}
        Sigma_crit        = get_Sigma_crit(PrjGal) 
        kw_parts_all      = Gal2kwMXYZ(Gal)
        kw_parts_all_proj = project_kw_parts(kw_parts_all,proj_index)
        kw_2Ddens_all     = dens_map_AMR(kw_parts_proj=kw_parts_all_proj,
                                         **kw_densmap) 
        # Tacun
        if Gal.simsuite== "COLIBRE":
            warnings.warn("This is a tacun, a monkey patch")
            from nazgul.Translator.COLIBRE.particle_galaxy import get_CoM
            XYZM   = kw_parts_all["Xs"],kw_parts_all["Ys"],kw_parts_all["Zs"],kw_parts_all["Ms"]
            CM_all = get_CoM(Gal,XYZM=XYZM)
            kw_1D_dens["CM_all"] = CM_all
            
        # free memory
        del kw_parts_all,kw_parts_all_proj
        kw_1D_dens["Sigma_crit"] = Sigma_crit
        if enclosed:
            kw_1D_dens = _get_kw_1D_density_enclosed(Gal,kw_1D_dens,kw_2Ddens_all,plot_figall=plot_figall)
        else:
            kw_1D_dens = _get_kw_1D_density_not_enclosed(Gal,kw_1D_dens,kw_2Ddens_all,plot_figall=plot_figall)
    
    with open(nm_kw_1D_dens,"wb") as f:
        dill.dump(kw_1D_dens,f)
    print(f"Saved {nm_kw_1D_dens}")

    return kw_1D_dens

def get_savedir_plots(Gal,savedir=None):
    if not savedir:
        savedir  = f"{Gal.gal_dir}/Plots/"
    mkdir(savedir)
    return savedir

def get_Sigma_crit(PrjGal):
    z_source = retrieve_z_source(PrjGal)
    Sigma_crit = SigCrit(z_lens   = PrjGal.z,
                         z_source = z_source,
                         cosmo    = PrjGal.cosmo)
    Sigma_crit = Sigma_crit.to("Msun/kpc^2")
    return Sigma_crit
    
def retrieve_z_source(PrjGal,z_source_def=2):
    try:
        kw_proj_res = load_whatever(PrjGal.projection_path)
        z_source = kw_proj_res["z_source"]
    except:
        warnings.warn(f"Prev. z_source not found - Assuming z_source={z_source_def}")
        z_source = z_source_def
    return z_source
    
# Plot AMR density for different particle species
def plot_AMR_densityXpart(Gal,
                     proj_index    = 0,
                     max_particles = default_kw_densmap["max_particles"],
                     min_area      = default_kw_densmap["min_area"],
                     dens_thresh   = default_kw_densmap["dens_thresh"],
                     scale_tE_cutout = 10,
                     savedir       = None,
                     part_thresh   = 1e4, # min n* of particles to be plotted
                     rerun         = False,
                     verbose       = True):
    """ 
    Compute and plot density Adaptive Mesh Refinement map split into the various particles
    """
    savedir = get_savedir_plots(Gal,savedir=savedir)
    nm_Sgm = f"{savedir}/Sigma_decomposed_proj{proj_index}.png"

    if not rerun:
        # Check if last plot is already present
        # if so, skip plotting them again
        if os.path.isfile(nm_Sgm):
            warnings.warn(f"Plot {nm_Sgm} already present - skipping.")
            return None

    kw_densmap     = {"max_particles":max_particles,
                      "min_area":min_area,
                      "dens_thresh":dens_thresh,
                      "clip":True}
    kw_1D_dens     = get_kw_1D_density(Gal,
                                       proj_index,
                                       kw_densmap=kw_densmap,
                                       reload=True,
                                       savedir=savedir,
                                       enclosed=True,
                                       plot_figall=True)
    
    tE             = kw_1D_dens["tE"]
    RE             = kw_1D_dens["RE"]
    r_all          = kw_1D_dens["r_all"]
    arcXkpc        = kw_1D_dens["arcXkpc"]
    kw_extents     = kw_1D_dens["kw_extents"]
    Sigma_crit     = kw_1D_dens["Sigma_crit"]
    MD_coord_all   = kw_1D_dens["MD_coords_all"] 
    Sigma_encl_all = kw_1D_dens["Sigma_encl_all"]  # Msun/kpc^2
    
    # Plotting all the density
    
    fig2,ax2 = plt.subplots(1)
    ax2.plot(r_all,Sigma_encl_all/1e9,color='cyan',label="Total")

    ax_tmp = ax2.twiny() 
    part_types = get_sim_func(Gal.simsuite,"part_type_list")
    i = 0
    for tp in part_types:
        if "bh" in tp.lower() or "hole" in tp.lower():
            warnings.warn("Ignoring BH particles- too few to make an AMR")
            continue
        if Gal.simsuite== "COLIBRE":
            warnings.warn("This is a tacun, a monkey patch")
            kw_parts      = Gal2kwMXYZ_part(Gal,part_type=tp,CM=kw_1D_dens["CM_all"])
        else:
            kw_parts      = Gal2kwMXYZ_part(Gal,part_type=tp)
        # consider the case that there are too few particles and skip it
        if len(kw_parts["Xs"])<part_thresh:
            warnings.warn(f"Less than {part_thresh} {tp} particles - skipping")
            continue
        
        kw_parts_proj = project_kw_parts(kw_parts,proj_index)
        kw_2D_dens    = dens_map_AMR(kw_parts_proj=kw_parts_proj,**kw_densmap)
        # free memory
        del kw_parts,kw_parts_proj
        del kw_2D_dens["MD_coords"]
        kw_2D_dens["MD_coords"] = MD_coord_all
        # plot 2d dens distr.
        fig1,ax1 = plot_AMR_cells(kw_2D_dens,kw_extents=kw_extents)
        fig1.suptitle(f"AMR of mass projection of {tp} particles")
        nm_amr_dec = f"{savedir}/AMR_{tp}_proj{proj_index}.png"
        fig1.savefig(nm_amr_dec)
        plt.close(fig1)
        del fig1,ax1
        gc.collect()
        print(f"Saved {nm_amr_dec}") 

        # plot 1D Sigma
        r,Sigma_encl   = cells2SigEnclRad(kw_2D_dens)
        # free memory
        del kw_2D_dens
        
        r = r.to("kpc")
        Sigma_encl = Sigma_encl.to("Msun/kpc^2")
        ax2.plot(r,Sigma_encl/1e9,color=base_colors[i],label=tp)
        i+=1
    ax_tmp.plot(r_all.to("kpc")*arcXkpc, np.zeros_like(r_all),alpha=0.)
    ax_tmp.tick_params(axis='x')

    ax2.set_title("Surface density profile (cnf. Fig 5 SEAGLE I)")
    ax2.axvline(RE.value,ls="--",c="k",label=r"R$_{\rm{Ein}}$="+f'{np.round(RE.value,2)} {RE.unit}={np.round(tE.value,2)}"')
    ax2.set_xlabel(f"Radius (R) [{r.unit}]")
    ax2.set_ylabel(r"$\Sigma$(R) [10$^9$"+str(Sigma_encl.unit)+"]")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax_tmp.set_xscale('log')

    ax_tmp.set_xlabel(f"Radius (R) ['']")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(nm_Sgm)
    print(f"Saved {nm_Sgm}") 
    plt.close(fig2)
    plt.close("all")
    Gal.slim_down()
    gc.collect()
    
    
    
if __name__=="__main__": 
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot 2D mass distribution of the galaxy and Fig 5 SEAGLE_1")
    parser.add_argument('-snap','--snap',type=str,dest="snap",default="27",help=f"Snap to consider")
    parser.add_argument('-Gn',type=int,dest="Gn",default=8, help=f"Galaxy Number (Gn) to consider")
    parser.add_argument('-SGn',type=int,dest="SGn",default=0,help=f"Sub-Galaxy Number (SGn) to consider")
    parser.add_argument('-prj','--proj_index',type=int,dest="proj_index",default=0,help=f"Projection index")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")

    #sim,Gn,SGn,snap = "RefL0025N0752",13,0,"25"
    args       = parser.parse_args()
    snap       = str(args.snap)
    Gn         = args.Gn
    SGn        = args.SGn
    proj_index = args.proj_index
    sim        = args.sim
    subsim     = args.subsim
    simsuite   = args.simsuite

    print("Using simulation: "+sim)
    print("Snap: "+snap)
    print("Galaxy: Gn",Gn,"SGn",SGn)

    warnings.warn("Patchy solution to finding a galaxy")
    if simsuite=="EAGLE":
        Gal    = PartGal({"Gn":Gn,"SGn":SGn},
                         simsuite=simsuite,
                         sim=sim,
                         subsim=subsim,
                         z=None,snap=snap,    # redshift or snap
                         M=None,Centre=None,
                         reload=False)
    elif simsuite=="COLIBRE":
        from nazgul.pathfinder import std_data_dir
        gal_path = std_data_dir/simsuite/sim/subsim/f"snap_{snap}"/f"Gn{Gn}"/"Gal"/f"G{Gn}.dll"
        Gal = load_whatever(gal_path)
        
    Gal.run()
    plot_AMR_densityXpart(Gal,proj_index=proj_index)
