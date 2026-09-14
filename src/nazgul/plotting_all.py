# Plot all lenses
import warnings
import numpy as np
import argparse,sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from stat_lenses import get_all_gallens,get_catdir_stat
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.mount_doom.lens_system import LensSystem

def auto_grid(n):
    ncols = np.ceil(np.sqrt(n))
    nrows = np.ceil(n / ncols)
    return int(nrows), int(ncols)


def plot_grid(images,savename,extents=None,cmap="hot",titles=None,label=None,suptitle=None,extent_type="arc"):
    n = len(images)  # or any list of things to plot
    
    nrows, ncols = auto_grid(n)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows+.2))
    
    axes = axes.ravel()  # flatten for easy iteration
    
    for i, img in enumerate(images):
        if extents is not None:
            extent = extents[i]
        im0 = axes[i].imshow(img,origin="lower",extent=extent,cmap=cmap)
        if extent_type=="kpc":
            axes[i].set_xlabel("X [kpc]")
            axes[i].set_ylabel("Y [kpc]")
        elif extent_type=="arc":
            axes[i].set_xlabel('RA ["]')
            axes[i].set_ylabel('DEC ["]')

        if titles is not None:
            axes[i].set_title(titles[i])
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical',label=label)

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(savename)
    print(f"Saved {savename}")
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot a grid of give precomputed lenses")
    parser.add_argument('-snap','--snap',nargs="+",type=int,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ext','--extent_type',type=str,dest="extent_type",default="arc",help=f"Extent type: either 'arc' (arcseconds, default) or 'kpc'")
    parser.add_argument('-nl','--n_lenses',type=int,dest="n_lenses",default=np.nan,help=f"Number of lenses to model")

    args      = parser.parse_args()
    snaps     = args.snaps #[25,26,27]
    sim       = args.sim
    subsim    = args.subsim
    simsuite  = args.simsuite
    extent_type = args.extent_type 
    n_lenses   = args.n_lenses

    if extent_type not in ["arc","kpc"]:
        raise ValueError(f'extent_type must be either "arc" or "kpc", not {extent_type}')
    lenses  =  get_all_gallens(sim=sim,
                               subsim=subsim,
                              simsuite=simsuite,
                              snaps=snaps,
                              n_lenses=n_lenses)
    savedir = get_catdir_stat(sim=sim,
                              subsim=subsim,
                              simsuite=simsuite,
                              snaps=snaps)
    kappas  = []
    images  = []
    extents_arc = []
    extents_kpc = []
    gal_names   = []
    z_str       = ""
    zls         = []
    for l in lenses:
        kappas.append(l.kappa_map)
        kw_extents = l.kw_extents
        extents_kpc.append(kw_extents["extent_kpc"])
        extents_arc.append(kw_extents["extent_arcsec"])
        lns = LensSystem.from_GalLens(l)
        lns.setup()
        Sim = lns.get_Sim()
        lns.image_sim = lns.get_lensed_image(Sim=Sim,kwargs_source=None, unconvolved=False)
        images.append(lns.image_sim)
        zl = np.round(lns.gallens.z_lens,2)
        zls.append(zl)
        if len(snaps)!=1:
            z_str = f" z:{zl:0.3}"  
        gal_names.append(f"{l.Gal.name} prj:{l.proj_index}{z_str}")
        
    if len(snaps)!=1:
        zls = np.sort(list(set(zls)))
        if len(zls)==1:
            z_str = f"z = {np.round(zls[0],2)}"
        else:
            z_str_range = f"{np.round(zls[0],2)}<z<{np.round(zls[1],2)}"
            if snaps==[]:
                # all snaps
                z_str = z_str_range
            elif np.diff(np.sort(snaps))==np.ones(len(snaps)-1):
                # snaps are sequential -> it's a range of z
                z_str = z_str_range
            else:
                z_str = f"z = {[float(np.round(z,2)) for z in zls]}"
    else:
        z_str = f"z = {np.round(np.mean(zls),2)}"
    log_images = np.log10(np.array(images))
    if extent_type=="arc":
        extents = extents_arc
    elif extent_type=="kpc":
        extents = extents_kpc
    
    plot_grid(log_images,savedir/"grid_gal_images.png",extents=extents,cmap="winter",titles=gal_names,label=r"log$_{10}$ flux [arbitrary]",suptitle=f"Sim {sim}: Lensed Images {z_str}",extent_type=extent_type)
    plot_grid(kappas,savedir/"grid_gal_kappas.png",extents=extents,titles=gal_names,label=r"$\kappa$",suptitle=f"Sim {sim}: Convergence Map {z_str}",extent_type=extent_type)
    