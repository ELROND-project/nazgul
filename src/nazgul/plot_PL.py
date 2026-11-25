"""Plotting functions for the Particle Lenses
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nazgul.pathfinder import tmp_dir
from python_tools.fwhm import get_fwhm


def _plot_caustics(kw_crit,
                   kw_extents=None,
                   savename=tmp_dir/"test_caustics.png"):
 
    cl_rad_x,cl_rad_y   = kw_crit["critical_lines"]["radial"]
    cc_rad_x,cc_rad_y   = kw_crit["caustics"]["radial"]
    cl_tan_x,cl_tan_y   = kw_crit["critical_lines"]["tangential"]
    cc_tan_x,cc_tan_y   = kw_crit["caustics"]["tangential"]

    cent_caust_tan = np.mean(cc_tan_x),np.mean(cc_tan_y)
    fig,ax = plt.subplots()
    ax.scatter(cc_rad_x,cc_rad_y,c="b",marker=".",label="Radial Caustics")
    ax.scatter(cc_tan_x,cc_tan_y,c="r",marker=".",label="Tangential Caustics")
    ax.scatter(cl_rad_x,cl_rad_y,c="cyan",marker=".",label="Radial Crit. Curve")
    ax.scatter(cl_tan_x,cl_tan_y,c="darkorange",marker=".",label="Tangential Crit. Curve")
    if kw_extents:
        xmin,xmax,ymin,ymax = kw_extents["extent_arcsec"]
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
    plt.gca().set_aspect('equal')
    ax.set_xlabel('RA ["]')
    ax.set_ylabel('DEC ["]')
    ax.legend()
    ax.set_title("Caustics and Critical Curves") 
    plt.tight_layout()
    print(f"Saving {savename}") 
    plt.savefig(savename)
    plt.close(fig)
    
def plot_caustics(Model,savename=tmp_dir/"test_caustics.png",kw_extents=None):
    if kw_extents is None:
        kw_extents = Model.gallens.kw_extents
    kw_crit = LPClass.get_kw_critical_curve_caustics()
    return _plot_caustics(kw_crit,savename=savename,kw_extents=kw_extents)


def plot_kappamap(kappa_map,extent_kpc,title1="",savename="kappa.png",cmap="gist_heat",label_clb=r'$\kappa$',
                  to_sigma_clip=True,sigma=10):
    if to_sigma_clip:
        warnings.warn("Sigma clipping kappa map")
        sgc = sigma_clip(kappa_map,sigma=sigma)        
        msk_sgc = np.invert(sgc.mask)
        kappa_map *= msk_sgc
    fig,axes = plt.subplots(2,figsize=(8,16))

    ax  = axes[0]
    im0 = ax.imshow(kappa_map,origin="lower",extent=extent_kpc,cmap=cmap)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(title1) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=label_clb)


    # take advantage of the circular simmetry and obtain the projection
    k1_proj = kappa_map[int(len(kappa_map)/2)]
    x = np.linspace(extent_kpc[0],extent_kpc[1],len(k1_proj))
    _xcnt = np.median(x)
    ax = axes[1]
    ax.plot(x,k1_proj,c="k")
    fwhm_k1 = get_fwhm(k1_proj,x) 
    hmax  = max(k1_proj)/2.
    ax.axvline(_xcnt,c="g",alpha=.5)

    ax.plot([_xcnt-fwhm_k1/2,_xcnt+fwhm_k1/2],[hmax,hmax],ls="-.",c="r",label="FWHM="+str(np.round(fwhm_k1,3)))
    ax.legend()
    ax.set_title(title1 +" projection at x=0")
    plt.suptitle("Density distribution")
    print(f"Saving {savename}")
    plt.savefig(savename)
    plt.close()


def plot_lensed_im_and_kappa(Model,savename="lensed_im.pdf",kw_extents=None):
    kappa = Model.kappa_map
    if hasattr(Model,"Gal"):
        name = Model.Gal.name
    else:
        name = Model.name
    name = name.replace("Sub_","")
    name = name.replace("Lens_","")
    
    if kw_extents is None:
        kw_extents = Model.kw_extents
    fg,axes = plt.subplots(1,2,figsize=(10,5))
    ax = axes[0]

    extent_kpc    = kw_extents["extent_kpc"]
    extent_arcsec = kw_extents["extent_arcsec"]
    
    im0   = ax.matshow(np.log10(kappa),origin='lower',extent=extent_kpc,cmap="gist_heat")
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(r"Convergence "+name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label=r"log$_{10}\, \kappa$")

    lnsd_im  = Model.image_sim 
    ax = axes[1]
    im0   = ax.matshow(np.log10(lnsd_im),origin='lower',extent=extent_arcsec)
    ax.set_xlabel('X ["]')
    ax.set_ylabel('Y ["]')
    
    ax.set_title("Lensed image "+name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label=r"log$_{10}$ flux [arbitrary]")
    plt.suptitle(r"With z$_{\rm{lens}}$="+str(np.round(Model.z_lens,2))+r" z$_{\rm{source}}$="+str(np.round(Model.z_source,2)))
    plt.tight_layout()
    print(f"Saving {savename}") 
    plt.savefig(savename)
    plt.close("all")
    
def plot_all(Model,savename_lensed="lensed_im.pdf",savename_kappa="kappa.png",savename_caustics="caustics.png",fast_caustic=True,skip_caustic=False):
    kw_extents = Model.kw_extents
    plot_lensed_im_and_kappa(Model,savename=Model.savedir/savename_lensed,kw_extents=kw_extents)
    if not skip_caustic:
        plot_caustics(Model,fast_caustic=fast_caustic,savename=Model.savedir/savename_caustics,kw_extents=kw_extents)
    plt.close("all")
    return 0
