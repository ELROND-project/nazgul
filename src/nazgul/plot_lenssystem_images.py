# Model all lenses where LOS is not simulated but w. LOS in the model
# to study the internal shear (á la Etherington)
import os,gc
import argparse
import warnings
import numpy as np
import sys,dill
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nazgul.plot_PL import plot_kappamap
from nazgul.MountDoom.lens_system import LensSystem

from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.Modelling.lib_models import setup_lens,setup_sim_obs,get_kwargs_likelihood,get_lens_mask#,get_lenses2model
from nazgul.stat_lenses import get_all_gallens
from nazgul.Modelling.lib_models import model_res_base

from nazgul.plot_image_pair import _limits,plot_image_pairs_pdf

res_dir_base      = model_res_base/"simNoShear/"



if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot the observed input to the model")
    parser.add_argument('-nl','--n_lenses',type=int,dest="n_lenses",default=np.nan,help=f"Number of lenses to model")
    parser.add_argument('-mtE','--min_thetaE',type=float,dest="min_thetaE",default=None,help=f"Min theta_E for the gal to be considered a lens")
    parser.add_argument('-snap','--snap',nargs="+",type=str,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")

    args       = parser.parse_args()
    n_lenses   = args.n_lenses
    min_thetaE = args.min_thetaE 
    snaps      = args.snaps #[25,26,27]
    sim        = args.sim
    subsim     = args.subsim
    simsuite   = args.simsuite

    lenses2skip = []
    
    kw_get_all_gallens = {"sim":sim,
                          "subsim":subsim,
                           "simsuite":simsuite,
                            "snaps":snaps}

    res_dir = res_dir_base
    gal_lenses  = get_all_gallens(**kw_get_all_gallens)
    """get_lenses2model(res_dir=res_dir,
                                   reload=False,
                                   kw_get_all_gallens=kw_get_all_gallens,
                                   n_lenses=n_lenses,
                                   min_thetaE=min_thetaE,
                                   skip_lenses=lenses2skip)"""

    sim_images = []
    obs_images = []
    nms_lenses = []
    extents    = []
    limits     = []
    err_fail   = []
    N_gallenses = len(gal_lenses)
    if np.isnan(n_lenses):
        n_lenses=N_gallenses
    
    for i,gal_lens in enumerate(gal_lenses[:n_lenses]): 
        print("\n     Loading lens "+gal_lens.name+\
              "\n####################################################\n")
        lens = LensSystem.from_GalLens(gal_lens)
        try:
            lens = setup_lens(lens,res_dir=res_dir,check_if_workin_on_it=False)
        except Exception as e:
            warnings.warn(f"DEBUG\nlens {lens} has failed due to:\n{e}\n")
            err_fail.append({"lens":{lens.name},"error":e})
            continue
        """plot_kappamap(lens.gallens.kappa_map, 
                      extent_kpc=lens.gallens.kw_extents["extent_kpc"],
                      savename=f"{lens.model_res_dir}/kappa_gal.png")"""

        multi_band_list = setup_sim_obs(lens)
        Sim = lens.get_Sim() 
        image_orig = lens.get_lensed_image(Sim=Sim,unconvolved=True)
        image_obs = multi_band_list[0][0]["image_data"]
        mask = get_lens_mask(lens,image_obs,plot_mask=False)
        mask[mask==0] = np.nan
        nms_lenses.append(lens.name)
        sim_images.append(image_orig)
        obs_images.append(image_obs*mask)
        limits.append(_limits(np.log10(np.where(image_obs > 0, image_obs, np.nan))))
        ext = lens.gallens.kw_extents["extent_arcsec"]
        extents.append(ext)
        del lens
        gc.collect()
        
        """
        fig,axis = plt.subplots(1,1)
        ax = axis[0]
        im0 =ax.imshow(image_orig,origin="lower",extent=ext,cmap="hot")
        ax.set_xlabel('RA ["]')
        ax.set_ylabel('DEC ["]')
        ax.set_title("Image")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical',label=label_clb)
        ax = axis[1]
        im0 =ax.imshow(image_obs*mask,origin="lower",extent=ext,cmap="hot")
        ax.set_xlabel('RA ["]')
        ax.set_ylabel('DEC ["]')
        ax.set_title("Image observed masked")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical',label=label_clb)
        fig.tight_layout()
        nm = f"{lens.model_res_dir}/kappa_gal.png"
        fig.savefig(nm)
        print(f"Saving {nm}")
        """
    del gal_lenses
    with open("tmp/del_fail.dll","wb") as f:
        dill.dump(err_fail,f)
    print("N fails:",len(err_fail),"\n% fails:",np.round((len(err_fail)/N_gallenses)*100,1))
    sim_subsim = sim
    if subsim is not None:
        sim_subsim += "_"+subsim
    if len(snaps)==1:
        snap_str = "snap_"+str(snaps[0])
    else:
        snap_str = f"snaps{'_'.join(snaps)}"
    output_pdf = f"tmp/all_sim_lenses_{sim_subsim}_{simsuite}_{snap_str}.pdf"
    plot_image_pairs_pdf(images1 = sim_images,
                         images2 = obs_images,
                         names   = nms_lenses,
                         extents = extents,
                         limits  = limits,
                         output_pdf = output_pdf,
                         cmap1 = "gist_heat",
                         cmap2 = "gist_heat",
                         log_scale1 = True,
                         log_scale2 = True,
                         label1 = "Simulated images",
                         label2 = "Sim. Observed (+mask)")
