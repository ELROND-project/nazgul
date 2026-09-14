# count effective lenses
import dill,sys,os
import argparse
import warnings
import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable

from python_tools.tools import mkdir,silencer
from python_tools.get_res import load_whatever
from nazgul.pathfinder import get_catlensdir
from nazgul.project_gal import ProjectionError

from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.pathfinder import get_sim_dir,get_snap_dir,std_data_dir


def get_all_gallens_gen_paths(snaps=[27],sim=std_sim,simsuite=std_simsuite,subsim=std_subsim,data_dir=std_data_dir):
    """
    Loacate general directory for all computed lenses
    """
    if len(snaps)!=0:
        gen_paths = []
        for snap in snaps:
            
            snap_dir = get_snap_dir(snap,sim=sim,subsim=subsim,simsuite=simsuite,data_dir=data_dir)
            gen_paths.append(snap_dir)
    else:
        sim_dir = get_sim_dir(sim=sim,subsim=subsim,
                    simsuite=simsuite,data_dir=data_dir)
        gen_paths = [sim_dir]
        
    if len(gen_paths)==0:
        print(f"Snaps : {snaps}")
        print(f"Sim : {sim}")
        print(f"SimSuite : {simsuite}")
        print(f"SubSim: {subsim}")
        print(f"Data dir: {data_dir}")
        raise RuntimeError("No computed gallenses found")
    return gen_paths


def get_all_gallens_paths(snaps=[27],sim=std_sim,simsuite=std_simsuite,subsim=std_subsim,data_dir=std_data_dir,
                         how2deal_with_doubles="take_latest"):
    """
    Loacate position of all computed lenses
    """
    gen_paths = get_all_gallens_gen_paths(snaps=snaps,sim=sim,subsim=subsim,
                                          simsuite=simsuite,data_dir=data_dir)
    
    if len(snaps)!=0:
        computed_gallenses = []
        for snap_dir in gen_paths:
            computed_gallenses = glob(f"{snap_dir}/Gn*/Sub/Sub_*Prj?_*.pkl")
    else:
        sim_dir = gen_paths[0]
        computed_gallenses = glob(f"{sim_dir}/*/Gn*/Sub/Sub_*Prj?_*.pkl")
    # there might be doubles, we have to deal with them in a way or another
    computed_gallenses = deal_with_doubles(computed_gallenses,how2deal_with_doubles)
    if len(computed_gallenses)==0:
        print(f"Snaps : {snaps}")
        print(f"Sim : {sim}")
        print(f"SimSuite : {simsuite}")
        print(f"SubSim: {subsim}")
        print(f"Data dir: {data_dir}")
        raise RuntimeError("No computed gallenses found")
    return computed_gallenses


def deduplicate_lens_paths(paths):
    """
    For paths that share the same name except the hash suffix (last _XXXXXX.pkl),
    keep only the most recently modified file.
    
    Returns a deduplicated list of PosixPaths.
    """
    # group paths by their "base name" = everything before the last underscore
    groups = defaultdict(list)
    for p in paths:
        p        = Path(p)
        stem     = p.stem                        # e.g. Sub_Lens_Gn31SGn1_Prj1_iegUjX
        base_key = "_".join(stem.split("_")[:-1]) # e.g. Sub_Lens_Gn31SGn1_Prj1
        # include parent dir so paths from different galaxies don't collide
        full_key = (p.parent, base_key)
        groups[full_key].append(p)

    result = []
    for (parent, base_key), group in groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # keep the most recently modified file
            newest = max(group, key=lambda p: p.stat().st_mtime)
            discarded = [p for p in group if p != newest]
            #print(f"Duplicates found for {base_key}:")
            #for p in discarded:
            #    print(f"  discarding: {p.name}")
            #print(f"  keeping:    {newest.name}")
            result.append(newest)

    return result
    
def deal_with_doubles(gallenses_paths,how2deal_with_doubles="take_latest"):

    if how2deal_with_doubles=="take_all":
        return gallenses_paths
    elif how2deal_with_doubles =="take_latest":
        return deduplicate_lens_paths(gallenses_paths)
    else:
        raise RuntimeError(f"how2deal_with_doubles {how2deal_with_doubles} not implemented")

@silencer
def get_all_gallens(snaps=[27],sim=std_sim,simsuite=std_simsuite,subsim=None,data_dir=std_data_dir,n_lenses=np.nan,verbose=True):
    lenses= []
    computed_gallenses = get_all_gallens_paths(snaps=snaps,sim=sim,simsuite=simsuite,subsim=subsim,data_dir=data_dir)
    if not np.isnan(n_lenses):
        print(f"Taking only the first {n_lenses} lenses") 
        computed_gallenses = computed_gallenses[:n_lenses]

    for gal_lns in computed_gallenses:
        ln = load_whatever(gal_lns)
        ln.unpack()
        monkey_patch_naming(ln,gal_lns)
        try:
            ln.run()
            lenses.append(ln)
        except ProjectionError as PE:
            # ignore galaxies which are not lenses
            # should already have been discarded a priori
            pass
    return lenses
    
def monkey_patch_naming(lnsgal,lnsgal_path):
    if str(lnsgal.pkl_path)!=str(lnsgal_path) and not "results" in str(lnsgal_path):
        warnings.warn("MONKEY-PATCH:\nUpdating name of stored instance")
        os.rename(lnsgal_path,lnsgal.pkl_path)
    return 0
        
def get_catdir_stat(snaps=[],sim=std_sim,subsim=std_subsim,
                    simsuite=std_simsuite):
    snaps_str = "_".join([str(s) for s in snaps])
    if snaps_str=="":
        snaps_str="all"
    catdir = get_catlensdir(sim=sim,
                            subsim=subsim,
                            simsuite=simsuite)
    catdir = catdir.with_name(f"CatGal_snap_{snaps_str}")
    mkdir(catdir)
    return catdir
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Compute and plot some useful statistic on the computed lenses")
    parser.add_argument('-snap','--snap',nargs="+",dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    
    args      = parser.parse_args()
    snaps     = args.snaps #[25,26,27]
    sim       = args.sim
    subsim    = args.subsim
    simsuite  = args.simsuite

    lenses =  get_all_gallens(sim=sim,
                              subsim=subsim,
                              simsuite=simsuite,
                              snaps=snaps)

    catdir = get_catdir_stat(snaps=snaps,sim=sim,
                            subsim=subsim,
                            simsuite=simsuite)
    
    lens_paths= []
    N_lenses= len(lenses)
    # Study thetaE distribution
    thetaEs = []
    z_source = []
    z_source_min = []
    z_lens = []
    masses = []
    gals   = []
    for ln in lenses:
        gals.append(ln.Gal)
        lens_paths.append(ln.pkl_path) # gal-lens
        thetaEs.append(ln.thetaE.value)
        z_source.append(ln.z_source)
        
        z_source_min.append(ln.z_source_min)
        z_lens.append(ln.z_lens)
        masses.append(ln.Gal.M_tot)
        
    print("\n\n")
    print("Actual lenses:"+str(N_lenses))
    if N_lenses==0:
        raise RuntimeWarning("No lenses found")
    thetaEs  = np.array(thetaEs)
    masses   = np.array(masses)
    z_lens   = np.array(z_lens)
    z_source = np.array(z_source)
    z_source_min = np.array(z_source_min)
    cat_lens = {"lens_path":lens_paths,
                "thetaE":thetaEs}
    cat_file = catdir/"LensCat.pkl"
    with open(cat_file,"wb") as f:
        dill.dump(cat_lens,f)
    print(f"Saved {cat_file}")
    
    
    plt.title(r"$\theta_E$ of Lenses")
    plt.hist(thetaEs,bins=20)
    plt.xlabel(r"$\theta_E$ ['']")
    plt.ylabel("N (tot="+str(N_lenses)+")")
    fig_tE = str(catdir)+"/Distr_thetaE.png"
    plt.savefig(fig_tE)
    print(f"Saving {fig_tE}")
    plt.close()
    plt.title(r"$\theta_E^2$ of Lenses")
    plt.hist(thetaEs**2,bins=20)
    plt.xlabel(r"$\theta_E^2$ [arcsec^2]")
    plt.ylabel("N (tot="+str(N_lenses)+")")
    fig_tE = str(catdir)+"/Distr_thetaE2.png"
    plt.savefig(fig_tE)
    print(f"Saving {fig_tE}")
    plt.close()
    
    plt.title(r"z source")
    plt.hist(z_source,bins=20)
    plt.xlabel(r"z source")
    plt.ylabel("N (tot="+str(N_lenses)+")")
    fig_tE = str(catdir)+"/Distr_zsource.png"
    plt.savefig(fig_tE)
    print(f"Saving {fig_tE}")
    
    plt.close()
    
    
    plt.title(r"M lens [solar masses]")
    plt.hist(masses,bins=20)
    plt.xlabel(r"M [M$_\odot$]")
    plt.ylabel("N (tot="+str(N_lenses)+")")
    plt.title("Total galaxy mass")
    fig_tE = str(catdir)+"/Distr_mlens.png"
    plt.savefig(fig_tE)
    print(f"Saving {fig_tE}")
    plt.close() 
    
    plt.scatter(masses,thetaEs,c=z_lens,cmap="viridis")
    plt.colorbar(label=r"z$_{lens}$")
    plt.xlabel(r"M [M$_\odot$]")
    plt.ylabel(r"$\theta_E$ ['']")
    plt.title("Relation between total galaxy masses and Einstein radius (N="+str(N_lenses)+")")
    plt.tight_layout()
    fig_MtE = str(catdir)+"/MvstE.png"
    plt.savefig(fig_MtE)
    print(f"Saving {fig_MtE}")
    plt.close()

        
    plt.scatter(masses,thetaEs**2,c=z_lens,cmap="viridis")
    plt.colorbar(label=r"z$_{lens}$")
    plt.xlabel(r"M [M$_\odot$]")
    plt.ylabel(r"$\theta_E^2$ [arcsec^2]")
    plt.title("Relation between total galaxy masses and Einstein radius^2 (N="+str(N_lenses)+")")
    plt.tight_layout()
    fig_MtE = str(catdir)+"/MvstE2.png"
    plt.savefig(fig_MtE)
    print(f"Saving {fig_MtE}")
    plt.close()
    
    #Cropping mass outliers -> forget about it
    i_crop = masses>0 #<np.median(masses)+3*np.std(masses)
    plt.scatter(masses[i_crop],thetaEs[i_crop],c=z_source_min[i_crop],cmap="viridis")
    plt.colorbar(label=r"z$_{source,min}$")
    plt.xlabel(r"M [M$_\odot$]")
    plt.ylabel(r"$\theta_E$ ['']")
    plt.title("Relation between total galaxy masses and Einstein radius (N="+str(len(thetaEs[i_crop]))+")")
    plt.tight_layout()
    fig_MtEzs = str(catdir)+"/MvstE_zs.png"
    plt.savefig(fig_MtEzs)
    print(f"Saving {fig_MtEzs}")
    plt.close()
    
    
    # considering the star masses 
    # (to comp. w SEAGLE selection M_* > 1.76 *1e10 Msun)
    m_s = []
    for g in gals:
        g.setup()
        ms = float(g.M_stars)  #float(str(g).split("and Mass in ")[1].split("Stars:")[1].split(" ")[0] )
        m_s.append(ms)
    m_s = np.array(m_s)
    plt.hist(m_s,bins=15)
    plt.title(r"Star mass of galaxy [solar masses]")
    plt.axvline(1.76 *1e10,label=r"m$_{\rm{SEAGLE min}} = 1.76e10 M$_{\odot}$")
    plt.xlabel(r"M$_{*}$ [M$_\odot$]")
    fig_ms = str(catdir)+"/Distr_Mstars.png"
    plt.legend(loc="upper right")
    plt.savefig(fig_ms)
    plt.close()



    # "corner plot" of theta_E^2 and mass 
    fig,axes = plt.subplots(2,2,figsize=(10,10))
    
    plt.suptitle(r"Total galaxy masses vs $\theta_E^2$ (N="+str(N_lenses)+")")
    ax = axes[0][1]
    ax.remove()

    
    ax = axes[1][1]
    ax.hist(thetaEs**2,bins=20)
    ax.set_xlabel(r"$\theta_E^2$ [arcsec^2]")
    ax.set_ylabel("N (tot="+str(N_lenses)+")")
        
    ax = axes[0][0]
    ax.set_title(r"M$_{tot\,lens}$ [M$_\odot$]")
    ax.hist(masses,bins=20)
    ax.set_xlabel(r"M")
    ax.set_ylabel("N (tot="+str(N_lenses)+")")

    ax = axes[1][0]
    im0 = ax.scatter(masses,thetaEs**2,c=z_lens,cmap="viridis")
    ax.set_xlabel(r"M [M$_\odot$]")
    ax.set_ylabel(r"$\theta_E^2$ [arcsec^2]")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"z$_{lens}$")
    fig_MtE = str(catdir)+"/corner_MvstE2.png"
    plt.tight_layout()
    plt.savefig(fig_MtE)
    print(f"Saving {fig_MtE}")
    plt.close()
