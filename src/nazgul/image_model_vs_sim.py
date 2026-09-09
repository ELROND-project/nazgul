# compare output from model_sim image with the original simulated image
import glob,sys
import argparse
import warnings
import numpy as np

from copy import deepcopy
import matplotlib.pyplot as plt
import lenstronomy.Util.util as util
from pathlib import Path
from lenstronomy.LensModel.lens_model import LensModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from lenstronomy.Util.util import image2array,array2image

from nazgul.pathfinder import std_sim
from nazgul.mount_doom.cracks_of_doom import LoadLens
from nazgul.fit_iso_ell import get_kwisodens,_get_kwiso

from python_tools.get_res import load_whatever
from python_tools.tools import to_dimless
#from model_sim_lens import lens_model_list
# temp. modelling path

from nazgul.Modelling.lib_models import setup_lens,model_res_base,get_kw_lens_mask
from nazgul.mount_doom.lens_system import LensSystem
import nazgul.mount_doom.cracks_of_doom as cod
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from python_tools.tools_WOI import is_someone_workin_on_it
from nazgul.Translator import std_sim,std_simsuite,std_subsim

from nazgul.combined_modelling_results import get_full_chain,get_res_dir,name_models

columns_ttl=["Simulation","Model","Simulation-Model"]

def get_image_model(lens,im=None,scale_to_im=True):
    # get the image from the lens model
    # NOT the one obtained by the simulation
    _ra,_dec = lens.gallens._radec
    Sim = lens.get_Sim()

    kwargs_res = load_whatever(f"{lens.model_res_dir}/kw_res.dll")

    kwargs_lens = kwargs_res["kwargs_lens"]
    kwargs_source_list = kwargs_res["kwargs_source"]
    kw_input = load_whatever(f"{lens.model_res_dir}/kw_input.dll")
    data_class,psf_class,sourceModel,kwargs_numerics = cod.get_dataclasses(Sim)
    warnings.warn("Weak coding practice: adding 0 params bc Sim by def consider elliptical sersic")
    if "e1" not in kwargs_source_list[0].keys():
        kwargs_source_list[0]["e1"] = 0
        kwargs_source_list[0]["e2"] = 0
    imageNumerics = NumericsSubFrame(pixel_grid=data_class,
                                         psf=psf_class, 
                                         **kwargs_numerics)
    lens_model   = LensModel(lens_model_list=kw_input["kwargs_model"]["lens_model_list"])
    #RA,DEC          = cod.get_RADEC((_ra,_dec))
    alpha_x,alpha_y = lens_model.alpha(_ra,_dec, kwargs_lens) 
    x_source_plane, y_source_plane = _ra-alpha_x,_dec-alpha_y
    # the coords have to be given as flat
    #x_source_plane = image2array(x_source_plane)
    #y_source_plane = image2array(y_source_plane)

    source_light = sourceModel.surface_brightness(x_source_plane, y_source_plane,
                                                  kwargs_source_list, k=None)
    source_light = array2image(source_light)
    if scale_to_im:
        source_light *= np.sum(im)/np.sum(source_light)
    return source_light

def plot_model_sim_diff(Lens,image_sim,image_model,nm="image_simVsmodel.png"):
    fig,axes  = plt.subplots(1,3,figsize=(12,7))
    fig.suptitle(r"Lensed Image")

    _plot_model_sim_diff(axes,Lens,image_sim,image_model)

    nm_fig = f"{Lens.model_res_dir}/{nm}"
    print(f"Saving {nm_fig}")
    plt.tight_layout()
    plt.savefig(nm_fig)
    plt.close()
    
def _plot_model_sim_diff(axes,Lens,image_sim,image_model,i_row=0,
                         columns_ttl=columns_ttl):
    fig       = axes.flatten()[0].get_figure()
    lens_name = Lens.name.replace("Sub_","")

    try:
        axes[i_row][1]
        axes = axes[i_row]
    except:
        i_row =0 

    # the sim is over this
    kw_extents  = Lens.gallens.kw_extents
    extent_full = kw_extents["extent_arcsec"]
    kw_imshow = {"origin":"lower",
                 "cmap":plt.cm.inferno,
                 "extent":extent_full}
    
    cmap_mask = ListedColormap(['grey', (1, 1, 1, 0)]) 
    # White for 0, fully transparent for 1
    alpha_mask = .6

    ims     = []    
    kw_mask = get_kw_lens_mask(Lens,image_sim)
    mask    = kw_mask["mask_comb_HD"]
    ax      = axes[0]
    ax.set_ylabel(lens_name)
    ax.get_yaxis().set_ticks([])
    if i_row ==0:
        ax.set_title(columns_ttl[0])

    ims.append(ax.imshow(np.log10(image_sim),**kw_imshow))
    ax      = axes[1]
    if i_row ==0:
        ax.set_title(columns_ttl[1])
    #image_model*=np.sum(image_sim)/np.sum(image_model)
    ims.append(ax.imshow(np.log10(image_model),**kw_imshow))
    ax      = axes[2]
    if i_row ==0:
        ax.set_title(columns_ttl[2])
    kw_imshow["cmap"] = "seismic" #or bwr
    kw_imshow_mask = deepcopy(kw_imshow)
    kw_imshow_mask["cmap"]  = cmap_mask
    kw_imshow_mask["alpha"] = alpha_mask
    mask[np.where(mask==0)] = np.nan
    ims.append(ax.imshow((image_sim-image_model)*mask,vmin=-1,vmax=1,**kw_imshow))
    #ims.append(ax.imshow((image_sim-image_model),**kw_imshow))
    #ax.imshow(mask,**kw_imshow_mask)
    
    for i,axii in enumerate(axes):
        #axii.set_xlim(x_min,x_max)
        #axii.set_ylim(x_min,x_max)
        if i!=0:
            axii.set_ylabel("DEC")
        axii.set_xlabel("RA")
        divider = make_axes_locatable(axii)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        if i<2:
            fig.colorbar(ims[i], cax=cax, orientation='vertical',label=r"log$_{10}$ flux")
        else:
            fig.colorbar(ims[i], cax=cax, orientation='vertical',label=r"$\Delta$ flux") 
    return axes

from nazgul.combined_modelling_results import get_all_lens_model_paths,get_model_title
from matplotlib.backends.backend_pdf import PdfPages

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot isocontours of kappa map from simulation vs modelling")
    parser.add_argument('-m','--model',type=str,
                        dest="model",
                        help=f"Name of type of model - accepted: {name_models}")

    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-snap','--snap',dest="snaps",default=[],nargs="+",help="(Optional) Define a specific snap list")
    #parser.add_argument('-lp','--lens_path',type=str,
    #                   dest="lens_path",
    #                   help="Path to pre-computed LensPart class instance")
    args         = parser.parse_args()
    #lens_path    = args.lens_path
    model    = args.model
    snaps    = args.snaps # by def. take all snaps
    sim      = args.sim
    subsim   = args.subsim
    simsuite = args.simsuite
    lenses2ignore= [""]

    res_dir = get_res_dir(model,simsuite=simsuite,sim=sim,subsim=subsim)
    nm_combined = f"{res_dir}/combined_comp_image_mod_sim.pdf"

    lens_resdir_paths = get_all_lens_model_paths(res_dir,snaps=snaps)  # paths only, no loading
    n_lenses          = len(lens_resdir_paths)
    ncols = len(columns_ttl)
    lines_per_page = 5
    plt.rcParams.update({'font.size': 10})
    scale_fig = 7
    scale_to_im = True
    if scale_to_im:
        warnings.warn("Hacky way - we rescale the model to the simulation since the amplitude is not stored")
    with PdfPages(nm_combined) as pdf:
        # iterate over pages
        for page_start in range(0, n_lenses, lines_per_page):
            page_slice  = lens_resdir_paths[page_start : page_start + lines_per_page]
            nrows_page  = len(page_slice)
            fig, axes = plt.subplots(nrows_page, ncols,
                                     figsize=(scale_fig * ncols, scale_fig * nrows_page),
                                     squeeze=False)
            for i_row, model_res_dir in enumerate(page_slice):
                gallens = LoadLens(model_res_dir/"link_gallens.pkl")
                Lens    = LensSystem.from_GalLens(gallens)
                Lens = setup_lens(Lens,res_dir=res_dir,
                                  check_if_workin_on_it=False,workin_on_it=False)
                image_sim   = Lens.image_sim
                image_model = get_image_model(Lens,scale_to_im=scale_to_im,im=image_sim)
                _plot_model_sim_diff(axes,Lens,image_sim,image_model,i_row=i_row,
                                     columns_ttl=columns_ttl)
                fig.suptitle(get_model_title(model))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            del fig
            print(f"Saved page {page_start // lines_per_page + 1} to {nm_combined}")
           
    """
    path_lenses = str(res_dir/"snap_*")
    for path in glob.glob(path_lenses):
        for ln2i in lenses2ignore:
            if ln2i in str(path):
                continue
        else:
            model_res_dir = Path(path)
            if is_someone_workin_on_it(model_res_dir):
                # Still under work - results not updated
                continue
            # the following way to get the model name is very weak
            model   = model_res_dir.parent.parent.name
            gallens = LoadLens(model_res_dir/"link_gallens.pkl")
            Lens    = LensSystem.from_GalLens(gallens)
            Lens = setup_lens(Lens,res_dir=res_dir,
                              check_if_workin_on_it=True,workin_on_it=False)
            image_model = get_image_model(Lens)
            image_sim   = Lens.image_sim
            
            plot_model_sim_diff(Lens,image_sim,image_model,,nm="image_simVsmodel.png")


    """