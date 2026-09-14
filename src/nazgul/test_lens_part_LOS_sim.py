"""
Test the new code structure
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from python_tools.image_manipulation import plot_comp_two_images
from python_tools.read_fits import load_fits

from nazgul.Translator.translator import PartGal
from nazgul.mount_doom.generate_particle_lens_dom import LensPart,wrapper_get_rnd_lens

from nazgul.lens_part_LOS import get_kw_los

from lenstronomy.SimulationAPI.ObservationConfig.HST import HST

def plot_two_images(im1,im2,extent=None,xlbl=None,ylbl=None,ttl1=None,ttl2=None,colorbarlbl=None):
    fig, axis = plt.subplots(1,2,figsize=(15,8))
    
    ax  = axis[0]
    im0 = ax.matshow(im1,origin='lower',extent=extent,cmap="hot")
    if xlbl:
        ax.set_xlabel(xlbl)
    if ylbl:
        ax.set_ylabel(ylbl)
    if ttl1:
        ax.set_title(ttl1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=colorbarlbl)

    ax  = axis[1]
    im0 = ax.matshow(im2,origin='lower',extent=extent,cmap="hot")
    if xlbl:
        ax.set_xlabel(xlbl)
    if ylbl:
        ax.set_ylabel(ylbl)
    if ttl2:
        ax.set_title(ttl2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=colorbarlbl)
    fig.tight_layout()
    return fig

if __name__ == "__main__":

    Gal    = PartGal(5,0,
                 z=None,snap="20",    # redshift or snap
                 M=None,Centre=None,
                 reload=True)
    LP = LensPart(Gal)
    LP.run()
    imsim = LP.image_sim 
    kw_los = get_kw_los()
    kw_add_lenses = {"lens_model_list":["LOS"],
                    "kwargs_lens":[kw_los]}
    LP_LOS = LensPart(Gal,
                  kwargs_add_lenses=kw_add_lenses)
    LP_LOS.run()
    imsimLOS = LP_LOS.image_sim 
    fig = plot_comp_two_images(imsim,imsimLOS,extent=LP.kw_extents["extent_arcsec"],
                         ttl1="Sim Image",ttl2="Sim Image +LOS")
    nm = f"tmp/imsim_w_LOS_{Gal}.png"
    fig.savefig(nm)
    print("Saving "+nm)

    band_HST = HST(band='WFC3_F160W', psf_type="PIXEL")
    psf_path = Path(f"./ObsData/HST/WFC3/F160W/PSFSTD_WFC3IR_F160W.fits")
    psf = load_fits(psf_path)[-2]
    # we can supersample it
    pssf    = 3
    psf_ss  = zoom(psf,pssf)
    kwargs_psf_HST = {"kernel_point_source":psf_ss,
                      "point_source_supersampling_factor":pssf}

    multi_band_list    = LP.sim_multi_band_list(band=band_HST,
                                               kwargs_psf=kwargs_psf_HST)    
    multi_band_listLOS = LP_LOS.sim_multi_band_list(band=band_HST,
                                               kwargs_psf=kwargs_psf_HST)
    extent=LP.kw_extents["extent_arcsec"]
    image_sim     = multi_band_list[0][0]["image_data"]
    image_sim_LOS = multi_band_listLOS[0][0]["image_data"]

    error     = multi_band_list[0][0]["noise_map"]
    error_LOS = multi_band_listLOS[0][0]["noise_map"]

    im1 = image_sim
    im2 = image_sim_LOS
    xlbl,ylbl ="RA","DEC"
    ttl1 = "image 1"
    ttl2 = "image 1 + LOS"
    colorbarlbl  = None
    fig, axis = plt.subplots(1,3,figsize=(18,8))
    
    ax  = axis[0]
    im0 = ax.matshow(im1,origin='lower',extent=extent,cmap="hot")
    if xlbl:
        ax.set_xlabel(xlbl)
    if ylbl:
        ax.set_ylabel(ylbl)
    if ttl1:
        ax.set_title(ttl1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    ax  = axis[1]
    im0 = ax.matshow(im2,origin='lower',extent=extent,cmap="hot")
    if xlbl:
        ax.set_xlabel(xlbl)
    if ylbl:
        ax.set_ylabel(ylbl)
    if ttl2:
        ax.set_title(ttl2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=colorbarlbl)

    ax  = axis[2]
    diff = (im1-im2)/np.hypot(error,error_LOS)
    # define vmin/vmax in order to have 0==white
    vm = 3*np.std(diff)

    im0 = ax.matshow(diff,origin='lower',extent=extent,cmap="bwr",vmax=vm,vmin=-vm)
    if xlbl:
        ax.set_xlabel(xlbl)
    if ylbl:
        ax.set_ylabel(ylbl)
    ax.set_title("Im - Im_LOS/sqrt(err^2 + err_los^2)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=colorbarlbl)

    fig.tight_layout()
    nm = f"tmp/sim_images_{Gal}.png"
    plt.savefig(nm)
    print(f"Saving {nm}")
    