"""
Test the new code structure
"""
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from python_tools.image_manipulation import plot_comp_two_images

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

    """Gal    = PartGal(5,0,
                 z=None,snap="20",    # redshift or snap
                 M=None,Centre=None,
                 reload=True)
                 
    LP = LensPart(Gal)
    """
    LP = wrapper_get_rnd_lens()
    Gal = LP.Gal
     
    LP.run()
    imsim = LP.image_sim 
    kappa = LP.kappa_map
    fig = plot_two_images(imsim,kappa,extent=LP.kw_extents["extent_arcsec"],
                         ttl1="Sim Image",ttl2="Kappa map")
    nm = "tmp/imsim_kappa.png"
    fig.savefig(nm)
    print("Saving "+nm)

    kw_los = get_kw_los()
    kw_add_lenses = {"lens_model_list":["LOS"],
                    "kwargs_lens":[kw_los]}
    
    
    LP_LOS = LensPart(Gal,
                  kwargs_add_lenses=kw_add_lenses)
    LP_LOS.run()
    imsimLOS = LP_LOS.image_sim 
    fig = plot_comp_two_images(imsim,imsimLOS,extent=LP.kw_extents["extent_arcsec"],
                         ttl1="Sim Image",ttl2="Sim Image +LOS")
    nm = "tmp/imsim_w_LOS.png"
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

    
    multi_band_list = LP_LOS.sim_multi_band_list(band=band_HST,
                                               kwargs_psf=kwargs_psf_HST)
    

"""
    
    LP = LensPart(Gal,kwargs_lensmodel={"z_lens":.1})
    LP.run()
    imsim = LP.image_sim 
    kappa = LP.kappa_map
    fig = plot_two_images(imsim,kappa,extent=LP.kw_extents["extent_arcsec"],
                         ttl1="Sim Image",ttl2="Kappa map")
    nm = "tmp/imsim_kappa_1.png"
    fig.savefig(nm)
    print("Saving "+nm)

    LP = LensPart(Gal,kwargs_lensmodel={"z_source":3})
    LP.run()
    imsim = LP.image_sim 
    kappa = LP.kappa_map
    fig = plot_two_images(imsim,kappa,extent=LP.kw_extents["extent_arcsec"],
                         ttl1="Sim Image",ttl2="Kappa map")
    nm = "tmp/imsim_kappa_2.png"
    fig.savefig(nm)
    print("Saving "+nm)
"""