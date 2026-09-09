# for some reason it was not able to fit anymore- 
# corrected for it by pre-fitting, 
import dill
import warnings
import numpy as np
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter
from lenstronomy.Data.imaging_data import ImageData
import lenstronomy.Util.simulation_util as sim_util
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model

from python_tools.get_res import load_whatever
from python_tools.image_manipulation import mask_out
from python_tools.tools import ensure_unit,to_dimless

from nazgul.lib_plot import base_colors
from nazgul.masking import overplot_mask
from nazgul.fit_ellipses import get_initial_kwfit

def rescale_kappa(kappa,sigma_smooth=1.0,thrs_scale=3,add_k=1e-6):
    # smooth it
    kappa_smooth = gaussian_filter(kappa, sigma=sigma_smooth)
    # crop low 
    threshold_kappa_smooth = thrs_scale*np.min(kappa_smooth)
    kappa_masked = np.where(kappa_smooth >threshold_kappa_smooth, kappa_smooth, 0)
    # take the log
    kappa_masked_log = np.log10(kappa_masked+add_k)
    return kappa_masked_log

def rescale_pot(potential):
    # we rescale it in order to have it descending and positive
    # 1st we make it negative ie "flip it"
    flip_pot = -potential
    # 2nd we subtract the minum (ie shift it to be positive)
    shift_pot = flip_pot - np.min(flip_pot)
    return shift_pot

def linlaw(x, a, b) :
    return a + x * b

def get_radius2radecgrid(rad,pixel_num):
    deltaPix    = to_dimless(2*rad/pixel_num)
    kwargs_data = sim_util.data_configure_simple(pixel_num, deltaPix)
    dataclass   = ImageData(**kwargs_data)
    __radec     = dataclass.coordinate_grid(pixel_num,pixel_num)
    _radec      = __radec[0].flatten(),__radec[1].flatten()
    return _radec

def _err_map_type(map_type):
    if map_type not in ["kappa","psi"]:
        raise RuntimeError(f"map_type must be 'kappa' or 'psi', not {map_type}")
        
def get_maxsma(image):
    image = np.asarray(image)
    if not np.any(image):
        raise RuntimeError("Image is only 0.0")

    cx, cy = int(image.shape[0] / 2.), int(image.shape[1] / 2.)
    rows, cols = np.nonzero(image)          # only the nonzero pixels, not the whole grid
    max_sma = np.hypot(rows - cx, cols - cy).max()
    return max_sma
    
def _get_kwiso(map,optimise_init_prms=True,verbose=True,fflag=0.7):
    # Force map to be positive
    if np.any(map<0):
        warnings.warn(RuntimeWarning("Found negative values in map - masking them"))
        map[np.where(map<0)] = 0
    
    # x0, y0, sma(semimajor), eps(ellipticity=1-b/a), pa
    if optimise_init_prms:
        kw_init_prms = get_initial_kwfit(map)
        geom = EllipseGeometry(**kw_init_prms)
        print("Original guesstimate with fit_ellipse:", kw_init_prms["x0"], kw_init_prms["y0"])
    else:
        geom = EllipseGeometry(map.shape[0]/2., map.shape[1]/2., 10., 0.5, 0./180.*np.pi)
        print("Original rough guesstimate:", map.shape[0]/2., map.shape[1]/2.)
    geom.find_center(map)
    ellipse = Ellipse(map, geometry=geom)
    # since the map might reach a floor of ==0.0, this might trigger a bug
    # see https://github.com/astropy/photutils/issues/1174
    # thus we insert a maxsma to avoid this
    maxsma = get_maxsma(map)
    #set fflag to 0 to improve fitting for potential (to test!)
    isolist = ellipse.fit_image(maxsma=maxsma,fflag=fflag)
    if len(isolist.a3)==0:
        print("<<! DEBUG - no iso-fit successful !>>")
        print("map has negative:",np.any(map<0))
        print("map has nan:",np.any(map==np.nan))
        print("max sma:",maxsma)
        mask_sma = mask_out(kw_init_prms["x0"],kw_init_prms["y0"],maxsma,np.ones_like(map))
        
        _fig,_ax = plt.subplots()
        _ax.set_title("Log10(map)")
        _im0 = _ax.imshow(np.log10(np.abs(map)+1e-12),origin="lower",cmap="hot")
        _divider = make_axes_locatable(_ax)
        _cax = _divider.append_axes('right', size='5%', pad=0.05)
        _fig.colorbar(_im0, cax=_cax, orientation='vertical',label=r"log$_{10}$(map)")
        _ax.axvline(kw_init_prms["x0"],c="k",label="x-y guestimates")
        _ax.axhline(kw_init_prms["y0"],c="k")
        _ax = overplot_mask(_ax,mask_sma)
        _fig.legend()
        nm = "tmp/map.png"
        _fig.savefig(nm)
        plt.close(_fig)
        print(f"DEBUG - Saved {nm}")
        raise RuntimeError("<<! The isofit has failed !>>")
    model = build_ellipse_model(map.shape, isolist)
    if verbose:
        print("Isofit succesful")
    return {"isolist":isolist,"geom":geom,"map":map,"model":model}

def get_kwiso(Lens,cutoff_rad=None,verbose=True,map_type="kappa",
              _rescale=True,optimise_init_prms=True,
              n_level_isocont=10):
    if cutoff_rad is None:
        cutoff_rad = get_iso_cutoff(Lens)
    cutoff_rad = ensure_unit(cutoff_rad,u.kpc)
    cutoff_rad = to_dimless(cutoff_rad)
    image_rad  = ensure_unit(Lens.radius/Lens.arcXkpc,u.kpc)
    image_rad  = to_dimless(image_rad)

    # the maps are, ATM, the same resolution as the alpha map -> could be recomputed taking _kappa_map(x,y)
    if cutoff_rad<=image_rad:
        # if it's smaller, we don't care and we take the whole image 
        # (with original resolution)
        cutoff_rad = image_rad
        if map_type =="kappa":
            map  = Lens.kappa_map
            if _rescale:
                map = rescale_kappa(map)
            
        elif map_type =="psi":
            map = Lens.psi_map
            if _rescale:
                map = rescale_pot(map)
        else:
            _err_map_type(map_type)
    else:
        if verbose:
            print("Cutoff radius larger than pixel grid")
            print("cutoff_rad",cutoff_rad)
            print("image_rad",image_rad)
        # if it's larger, we expand the grid to it (giving up resolution in the way)
        _radec = get_radius2radecgrid(cutoff_rad*Lens.arcXkpc,Lens.pixel_num)
        if map_type =="kappa":
            map  = Lens._kappa_map(_radec=_radec)
            if _rescale_kappa:
                map = rescale_kappa(map)

        elif map_type =="psi":
            warnings.warn("This might take a while")
            map    = Lens.compute_psi_map(_radec=_radec)
        else:
            _err_map_type(map_type)
    kw_iso= _get_kwiso(map,optimise_init_prms=optimise_init_prms,verbose=verbose) 
    kw_iso["cutoff_rad"] = cutoff_rad
    kw_iso["map_type"] = map_type

    ## add a simple isocontours computation (not a fit!)
    kw_iso["isoconts"] = get_conts(map,n_levels=n_level_isocont)
    return kw_iso

def get_kwisodens(Lens,cutoff_rad=None,verbose=True):
    kwiso_kappa = get_kwiso(Lens,cutoff_rad=None,map_type="kappa",verbose=verbose)
    # renaming for simplicity/monkey-patching
    kwiso_kappa["kappa"] = kwiso_kappa.pop("map")
    del kwiso_kappa["map_type"]
    return kwiso_kappa

def get_kwisopotential(Lens,cutoff_rad=None,verbose=True):
    kwiso_psi = get_kwiso(Lens,cutoff_rad=None,map_type="psi",verbose=verbose)
    # renaming for simplicity/monkey-patching
    kwiso_psi["psi"] = kwiso_psi.pop("map")
    del kwiso_psi["map_type"]
    return kwiso_psi

def fit_iso(Lens,cutoff_rad=None,pixel_num=None,verbose=True,map_type="kappa",
            save=True,reload=True): 
    if map_type=="kappa":
        savename=f"kw_res_isodens_prj{Lens.proj_index}.dll"
    elif map_type=="psi":
        savename=f"kw_res_isopsi_prj{Lens.proj_index}.dll"
    else:
        _err_map_type(map_type)
    res_path = f"{Lens.savedir}/{savename}"
    if reload:
        try:
            kw_res = load_whatever(res_path)
            print(f"Previous isofit results found: {res_path}")
            return kw_res
        except FileNotFoundError:
            print(f"Previous results not found, fitting isocontours of {map_type}")
            reload=False
            pass
    if cutoff_rad is None:
        cutoff_rad = get_iso_cutoff(Lens)
    if pixel_num is None:
       pixel_num  = Lens.pixel_num # here in case I need to change it
    cutoff_rad = ensure_unit(cutoff_rad,u.kpc)
    cutoff_rad = to_dimless(cutoff_rad)


    kw_iso     = get_kwiso(Lens,cutoff_rad=cutoff_rad,verbose=verbose,map_type=map_type)
    isolist    = kw_iso["isolist"]
    cutoff_rad = kw_iso["cutoff_rad"]
    kpcPix     = 2*cutoff_rad/pixel_num # 2*rad = Diam /pixel_number of the image
    sma_kpc    = isolist.sma*kpcPix # semi-major axis in kcp

    # discard first point
    popt_log,pcov_log = curve_fit(linlaw,np.log10(sma_kpc[1:]),np.log10(isolist.intens[1:]))
    ydatafit          = linlaw(np.log10(sma_kpc[1:]), *popt_log)
    kw_loglogfit      = {"popt_log":popt_log,"pcov_log":pcov_log,"fity":ydatafit,"fitx":np.log10(sma_kpc[1:])}
    kw_res            = {"isofit":kw_iso,"loglogfit":kw_loglogfit,"cutoff_rad":cutoff_rad}
    if save:
        with open(res_path,"wb") as f:
            dill.dump(kw_res,f)
        print(f"Saving isofit: {res_path}")
    return kw_res

def fit_isodens(Lens,cutoff_rad=None,pixel_num=None,verbose=True,save=True,reload=True):
    # monkey-patching
    kw_res = fit_iso(Lens,cutoff_rad=cutoff_rad,pixel_num=pixel_num,verbose=verbose,save=save,reload=reload,
                    map_type="kappa")
    kw_res["isodens"] = kw_res.pop("isofit")
    kw_res["isodens"]["kappa"] = kw_res["isodens"].pop("map")
    del kw_res["isodens"]["map_type"]
    return kw_res
    
def fit_isopot(Lens,cutoff_rad=None,pixel_num=None,verbose=True,save=True,reload=True):
    # monkey-patching
    kw_res = fit_iso(Lens,cutoff_rad=cutoff_rad,pixel_num=pixel_num,verbose=verbose,save=save,reload=reload,
                    map_type="psi")
    kw_res["isopot"] = kw_res.pop("isofit")
    kw_res["isopot"]["psi"] = kw_res["isopot"].pop("map")
    del kw_res["isopot"]["map_type"]
    return kw_res

def get_conts(image,n_levels=10):
    conts = []
    lvls  = []
    for i in range(n_levels):
        # uniformily distributed levels
        lvl = np.min(image) + i*(np.max(image)-np.min(image))/n_levels
        lvls.append(lvl)
        conts.append(find_contours(image,level=lvl))
    kw_cont = {"contours":conts,"levels":lvls}
    return kw_cont
    
def plot_conts(fig,conts,colors=base_colors):
    ax = fig.axes[0]
    for i,c in enumerate(conts):
        for j,cc in enumerate(c):
            if j==0:
                col = colors[i%len(colors)]
                ax.plot(*cc.T,ls="--",label=i,c=col)
            else:
                ax.plot(*cc.T,ls="--",c=col)
    fig.legend(loc="lower right")
    return fig

def plot_isofit(Lens,map_type="kappa",savedir=None,cutoff_rad=None,pixel_num=None,
                verbose=True,kw_res=None,reload=True):
    if savedir is None:
        savedir = Lens.savedir
    if pixel_num is None:
        pixel_num  = Lens.pixel_num # here in case I need to change it
    if kw_res is None:
        if cutoff_rad is None:
            cutoff_rad = get_iso_cutoff(Lens)
        kw_res = fit_iso(Lens=Lens,map_type=map_type,
                         cutoff_rad=cutoff_rad,pixel_num=pixel_num,
                         reload=reload,verbose=verbose)
    cutoff_rad = kw_res["cutoff_rad"]

    # assuming x,y centred around 0
    xmin = -cutoff_rad
    ymin = -cutoff_rad
    xmax = +cutoff_rad
    ymax = +cutoff_rad
    extent = [xmin,xmax,ymin,ymax] 
    map    = kw_res["isofit"]["map"]
    model  = kw_res["isofit"]["model"]
    residual = map - model
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    im_i = ax1.imshow(np.log10(map),cmap=plt.cm.inferno,origin="lower",extent=extent)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    map_nm = fr"$\{map_type}"
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=map_nm+r"_{sim}$")
    ax1.set_title(r"log$_{10}$("+map_nm+r"$)")
    
    im_i = ax2.imshow(np.log10(model),cmap=plt.cm.inferno,origin="lower",extent=extent)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=map_nm+r"_{iso}$")
    ax2.set_title(r"log$_{10}$("+map_nm+r"_{Model}$)")

    vm = np.median(residual) +2*np.std(residual)
    #print("testing vm residual:",vm)
    im_i = ax3.imshow(residual,cmap="bwr",extent=extent,origin="lower",vmin=-vm,vmax=vm)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=map_nm+r"_{sim}$-"+map_nm+r"_{iso}$")

    ax3.set_title("Residual")
    for ax in ax1,ax2,ax3:
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel("kpc")
        ax.set_ylabel("kpc")

    # overplot a few isophotes on the residual map
    isolist = kw_res["isofit"]["isolist"]
    iso1 = isolist.get_closest(10.)
    iso2 = isolist.get_closest(40.)
    iso3 = isolist.get_closest(100.)
    
    x, y, = iso1.sampled_coordinates()
    Nx,Ny = map.shape
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    x, y, = iso2.sampled_coordinates()
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    x, y, = iso3.sampled_coordinates()
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    name_plot = f"{savedir}/{map_type}_model.png"
    print(f"Saving {name_plot}")
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    kpcPix  = cutoff_rad/pixel_num
    sma_kpc = isolist.sma*kpcPix # semi-major axis in kcp
    
    geom    = kw_res["isofit"]["geom"]

    fig = plt.figure(figsize=(12, 7))    
    ax1 = fig.add_subplot(221)
    ax1.errorbar(sma_kpc, 1-isolist.eps, yerr=isolist.ellip_err, fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor axis [kpc]')
    ax1.set_ylabel('Axis Ratio []')
    ax_twin = ax1.twiny() 
    ax_twin.set_xlim(ax1.get_xlim())
    ticks_loc = ax1.get_xticks()
    conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
    conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
    ax_twin.set_xticks(ticks_loc)
    ax_twin.set_xticklabels(conv_ticks)
    ax_twin.set_xlabel(r'$\theta/\theta_E$ []')
    
    ax1 = fig.add_subplot(222)
    ax1.errorbar(sma_kpc, isolist.pa/np.pi*180., yerr=isolist.pa_err/np.pi* 80., fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor axis [kpc]')
    ax1.set_ylabel('PA [deg]')
    ax_twin = ax1.twiny() 
    ax_twin.set_xlim(ax1.get_xlim())
    ticks_loc = ax1.get_xticks()
    conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
    conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
    ax_twin.set_xticks(ticks_loc)
    ax_twin.set_xticklabels(conv_ticks)
    ax_twin.set_xlabel(r'$\theta/\theta_E$ []')

    ax1 = fig.add_subplot(223)
    ax1.errorbar(sma_kpc, (isolist.x0-geom.x0)*kpcPix, yerr=isolist.x0_err, fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor axis [kpc]')
    ax1.set_ylabel('X0-Xcnt [kpc]')
    ax_twin = ax1.twiny() 
    ax_twin.set_xlim(ax1.get_xlim())
    ticks_loc = ax1.get_xticks()
    conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
    conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
    ax_twin.set_xticks(ticks_loc)
    ax_twin.set_xticklabels(conv_ticks)
    ax_twin.set_xlabel(r'$\theta/\theta_E$ []')

    
    ax1 = fig.add_subplot(224)
    ax1.errorbar(sma_kpc, (isolist.y0-geom.y0)*kpcPix, yerr=isolist.y0_err, fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor axis [kpc]')
    ax1.set_ylabel('Y0-Ycnt [kpc]')
    ax_twin = ax1.twiny() 
    ax_twin.set_xlim(ax1.get_xlim())
    ticks_loc = ax1.get_xticks()
    conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
    conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
    ax_twin.set_xticks(ticks_loc)
    ax_twin.set_xticklabels(conv_ticks)
    ax_twin.set_xlabel(r'$\theta/\theta_E$ []')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    name_plot = f"{savedir}/{map_type}_prms1.png"
    print(f"Saving {name_plot}")
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    #plt.figure(figsize=(10, 5))
    fig = plt.figure(figsize=(12, 7))    
    #limits = [0., 100., -0.1, 0.1]
    
    ax1 = fig.add_subplot(221)
    #plt.axis(limits)
    ax1.errorbar(sma_kpc, isolist.a3, yerr=isolist.a3_err, fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor axis [kpc]')
    ax1.set_ylabel('A3')
    ax_twin = ax1.twiny()
    ax_twin.set_xlim(ax1.get_xlim())
    ticks_loc = ax1.get_xticks()
    conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
    conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
    ax_twin.set_xticks(ticks_loc)
    ax_twin.set_xticklabels(conv_ticks)
    ax_twin.set_xlabel(r'$\theta/\theta_E$ []')

    ax1 = fig.add_subplot(222)
    #plt.axis(limits)
    ax1.errorbar(sma_kpc, isolist.b3, yerr=isolist.b3_err, fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor axis [kpc]')
    ax1.set_ylabel('B3')
    ax_twin = ax1.twiny()
    ax_twin.set_xlim(ax1.get_xlim())
    ticks_loc = ax1.get_xticks()
    conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
    conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
    ax_twin.set_xticks(ticks_loc)
    ax_twin.set_xticklabels(conv_ticks)
    ax_twin.set_xlabel(r'$\theta/\theta_E$ []')
    
    ax1 = fig.add_subplot(223)
    #plt.axis(limits)
    ax1.errorbar(sma_kpc, isolist.a4, yerr=isolist.a4_err, fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor axis [kpc]')
    ax1.set_ylabel('A4')
    ax_twin = ax1.twiny()     
    ax_twin.set_xlim(ax1.get_xlim())
    ticks_loc = ax1.get_xticks()
    conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
    conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
    ax_twin.set_xticks(ticks_loc)
    ax_twin.set_xticklabels(conv_ticks)
    ax_twin.set_xlabel(r'$\theta/\theta_E$ []')

    ax1 = fig.add_subplot(224)
    #plt.axis(limits)
    ax1.errorbar(sma_kpc, isolist.b4, fmt='o', yerr=isolist.b4_err, markersize=4)
    ax1.set_xlabel('Semimajor axis [kpc]')
    ax1.set_ylabel('B4')
    ax_twin = ax1.twiny()
    ax_twin.set_xlim(ax1.get_xlim())
    ticks_loc = ax1.get_xticks()
    conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
    conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
    ax_twin.set_xticks(ticks_loc)
    ax_twin.set_xticklabels(conv_ticks)
    ax_twin.set_xlabel(r'$\theta/\theta_E$ []')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    
    name_plot = f"{savedir}/{map_type}_prms2.png"
    print(f"Saving {name_plot}")
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    fig,axis = plt.subplots(2,figsize=(13,8))
    ax = axis[0]
    ax.set_title(r"Plot of "+map_nm+r"$")
    
    ax.scatter(sma_kpc,isolist.intens,c="k")
    ax.set_xlabel(r'Semimajor axis [kpc])')
    ax.set_ylabel(map_nm+r'$') 

    if map_type=="kappa":
        # fit as linear in log
        popt_log,pcov_log = kw_res["loglogfit"]["popt_log"],kw_res["loglogfit"]["pcov_log"]
        ydatafit          = kw_res["loglogfit"]["fity"]
        ax = axis[1]
        ax.set_title(r"LogLog plot of "+map_nm+r'$')
        str_fit = "log10("+map_nm+r"$) ="+str(np.round(popt_log[0],2))+"log10(sma)^"+str(np.round(popt_log[1],2))
        ax.plot(np.log10(sma_kpc[1:]),ydatafit, c="b",ls="--",label="Fit:"+str_fit)
        
        ax.scatter(np.log10(sma_kpc),np.log10(isolist.intens),c="k")
        ax.legend()
        ax.set_xlabel(r'log$_{10}$(Semimajor axis [kpc])')
        ax.set_ylabel(r'log$_{10}$('+map_nm+r'$)')

        ax_twin = ax.twiny() 
        ax_twin.set_xlim(ax.get_xlim())
        ticks_loc = ax.get_xticks()
        conv_ticks = ticks_loc*Lens.arcXkpc.value/Lens.thetaE.value #arcsec
        conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
        ax_twin.set_xticks(ticks_loc)
        ax_twin.set_xticklabels(conv_ticks)
        ax_twin.set_xlabel(r'$\theta/\theta_E$ []')

        name_plot = f"{savedir}/{map_type}_map.png"
        print(f"Saving {name_plot}")
        plt.tight_layout()
        plt.savefig(name_plot)
        plt.close()
        
        # Plot gamma(r)
        logr          = kw_res["loglogfit"]["fitx"]
        y             = np.log10(isolist.intens[1:])
        gamma_der     = -np.gradient(y,logr)
        gamma_fit_fix = -kw_res["loglogfit"]["popt_log"][1]
        plt.plot(logr,gamma_der,c="g",label=r"$\gamma=-\frac{\mathrm{d isodensity}}{\mathrm{d log(r)}}$") 
        plt.plot(logr,gamma_fit_fix*np.ones_like(logr),c="b",ls="--",label=r"$\gamma_{opt.}$="+str(np.round(gamma_fit_fix,2))) 
        plt.title(r"$\gamma$(r)")
        plt.xlabel(r"log$_{10}$r")
        plt.ylabel(r"$\gamma$(r)")
        ax_twin = plt.twiny() 
        ax_twin.set_xlim(plt.axes().get_xlim())
        ticks_loc = plt.axes().get_xticks()
        conv_ticks = ticks_loc + np.log10(Lens.arcXkpc.value/Lens.thetaE.value) 
        conv_ticks = [np.round(cvt,2) for cvt in conv_ticks]
        ax_twin.set_xticks(ticks_loc)
        ax_twin.set_xticklabels(conv_ticks)
        ax_twin.set_xlabel(r'$log_{10}(\theta/\theta_E$) []')

        plt.legend()
        namefig = f"{savedir}/gamma_r.png"
        print(f"Saving {namefig}")
        plt.savefig(namefig)
        plt.close()
    return kw_res
    
def plot_isodens(Lens,savedir=None,cutoff_rad=None,pixel_num=None,verbose=True,kw_res=None,reload=True):
    kw_res = plot_isofit(Lens=Lens,map_type="kappa",savedir=savedir,cutoff_rad=cutoff_rad,pixel_num=pixel_num,
                         verbose=verbose,kw_res=kw_res,reload=reload)
    kw_res["isodens"] = kw_res.pop("isofit")
    kw_res["isodens"]["kappa"] = kw_res["isodens"].pop("map")
    del kw_res["isodens"]["map_type"]
    return kw_res

def plot_isopot(Lens,savedir=None,cutoff_rad=None,pixel_num=None,verbose=True,kw_res=None,reload=True):
    kw_res = plot_isofit(Lens=Lens,map_type="psi",savedir=savedir,cutoff_rad=cutoff_rad,pixel_num=pixel_num,
                         verbose=verbose,kw_res=kw_res,reload=reload)
    kw_res["isopot"] = kw_res.pop("isofit")
    kw_res["isopot"]["psi"] = kw_res["isopot"].pop("map")
    del kw_res["isopot"]["map_type"]
    return kw_res

def get_iso_cutoff(Lens):
    cutoff_rad = Lens.radius/Lens.arcXkpc #kpc
    print("Cutting plot at "+str(np.round(cutoff_rad,3))+", same cutout as the lens image")
    return cutoff_rad

if __name__=="__main__":
    raise RuntimeError("Outdated")
    from nazgul.mount_doom.cracks_of_doom import LoadLens
    from nazgul.mount_doom.lens_system import LensSystem
    from nazgul.modelling_wLOS import default_lens_path as lens_path

    # for now applied to a "known" lens galaxy
    gal_lens = LoadLens(lens_path)
    Lens = LensSystem.from_GalLens(gal_lens)
    savedir = Path("tmp/")
    cutoff_rad = get_iso_cutoff(Lens)
    kw_res = plot_isodens(Lens,savedir,cutoff_rad=cutoff_rad,reload=False)
    kw_res = plot_isopot(Lens,savedir,cutoff_rad=cutoff_rad,reload=False)
