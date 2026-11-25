# Model and plot lenses with varying source radius to check how it changes
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nazgul.plot_PL import plot_kappamap
from nazgul.MountDoom.lens_system import LensSystem

from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.Modelling.lib_models import setup_lens,setup_sim_obs,get_kwargs_likelihood,get_lenses2model,get_lens_mask
from nazgul.Modelling.lib_models import model_res_base
from nazgul.MountDoom.cracks_of_doom import kwargs_source_default,get_kwargs_sourceSim
from nazgul.plot_image_pair import _limits,plot_image_pairs_pdf
from python_tools.get_res import load_whatever


if __name__=="__main__":
    
    gal_lens = load_whatever(load_whatever("/pbs/home/g/gqueirolo/nazgul/src/nazgul/tmp/models/simNoShear/cat_lens2model.dll")["lens_cat"][0])
    
    print("\n     Loading lens "+gal_lens.name+\
          "\n####################################################\n")
    lens = LensSystem.from_GalLens(gal_lens)
    Sim  = lens.get_Sim()
    ext = lens.gallens.kw_extents["extent_arcsec"]
    print("lens tE=",np.round(lens.gallens.thetaE,2))
    nx,ny = 5,6
    fig,axis = plt.subplots(nx,ny,figsize=(20,25))
    for j,Rsersic_i in enumerate(np.linspace(.01,1,ny)):
        for i,mag_i in enumerate(np.linspace(30,23,nx)):
            kwargs_source = copy.deepcopy(kwargs_source_default)
            kwargs_source["R_sersic"] = Rsersic_i
            kwargs_source["magnitude"] = mag_i
            kwargs_source  = get_kwargs_sourceSim(Sim,kwargs_source)
            lens.kwargs_source = kwargs_source
            image_orig = lens.get_lensed_image(unconvolved=False,kwargs_source=kwargs_source)
            ax = axis[i][j]
            im0 =ax.imshow(np.log10(image_orig),vmin=-3,vmax=-0.2,
                           origin="lower",extent=ext,cmap="hot")
            ax.set_xlabel('RA ["]')
            ax.set_ylabel('DEC ["]')
            ax.set_title(f"R_s={np.round(Rsersic_i,1)},mag_s={np.round(mag_i,1)}")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, cax=cax, orientation='vertical')
    plt.tight_layout()
    nm = "tmp/var_source_R.png"
    plt.savefig(nm)
    plt.close()
    
    fig,axis = plt.subplots(nx,ny,figsize=(20,25))
    for j,Rsersic_i in enumerate(np.linspace(.01,1,ny)):
        for i,n_ser_i in enumerate(np.linspace(1,4,nx)):
            kwargs_source = copy.deepcopy(kwargs_source_default)
            kwargs_source["R_sersic"] = Rsersic_i
            kwargs_source["n_sersic"] = n_ser_i
            kwargs_source  = get_kwargs_sourceSim(Sim,kwargs_source)
            lens.kwargs_source = kwargs_source
            image_orig = lens.get_lensed_image(unconvolved=False,kwargs_source=kwargs_source)
            ax = axis[i][j]
            im0 =ax.imshow(np.log10(image_orig),vmin=-3,vmax=-0.2,
                           origin="lower",extent=ext,cmap="hot")
            ax.set_xlabel('RA ["]')
            ax.set_ylabel('DEC ["]')
            ax.set_title(f"R_s={np.round(Rsersic_i,1)},n_ser={np.round(n_ser_i,1)}")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, cax=cax, orientation='vertical')
    plt.tight_layout()
    nm = "tmp/var_source_nSer.png"
    plt.savefig(nm)
    print(f"Saved {nm}")

    