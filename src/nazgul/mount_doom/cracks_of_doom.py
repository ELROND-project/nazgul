"""
General script for helper functions used in the generation of lenses
"""

import dill
import warnings
import numpy as np
from copy import copy
from pathlib import Path
from functools import cached_property 

import astropy.units as u
from scipy.ndimage import zoom
from scipy.interpolate import splprep, splev, RectBivariateSpline

from lenstronomy.Util import util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.sim_api import SimAPI

# My libs
from python_tools.get_res import LoadClass
from python_tools.tools import mkdir,to_dimless,ensure_unit,convert_error_to_warning
# general path
from nazgul.pathfinder import path_nazgul, std_data_dir

def _resolve_gal_path(stored_path,data_dir=std_data_dir):
    """Translate a stored Gal_path to an absolute path on this machine.

    Handles three cases:
      - relative path (new pickles): prepend path_nazgul
      - absolute path, same machine (old same-machine pickles): unchanged via pathlib join
      - absolute path, different machine (old cross-machine pickles): strip everything
        before the RingBearer anchor and prepend the local path_nazgul
    """
    p = Path(stored_path)
    if not p.is_absolute():
        return path_nazgul / p
    try:
        return path_nazgul / p.relative_to(path_nazgul)
    except ValueError:
        parts = p.parts
        data_root = data_dir.name
        for i, part in enumerate(parts):
            if part == data_root:
                return path_nazgul / Path(*parts[i:])
        raise ValueError(
            f"Cannot resolve Gal_path '{p}': no '{data_root}' anchor found"
        )

# basic galaxy class
from nazgul.basic_gal import BasicGal,store_class
# cosmology
from nazgul.lib_cosmo import SigCrit
# Get particle from galaxy catalogue
from nazgul.Translator.translator import LoadGal
# particle lens class and params.
from nazgul.particle_lenses import PartLens
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
# likelihood class
from nazgul.likelihood import Likelihood
from nazgul.likelihood_z_source import kw_prior_z_source_zl
# project galaxy along various axis
from nazgul.project_gal import ProjGal,ProjectionError,project_Gal
#Default values
import nazgul.configurations as conf

# Path definitions:
# define where to store the obtained lenses classes
std_sim_lens_path   = path_nazgul/"sim_lens/"

def get_lens_dir(Gal,sim_lens_path=std_sim_lens_path):
    lens_dir = Path(sim_lens_path)/Gal.sim/f"snap{Gal.snap}_G{Gal.Gn}.{Gal.SGn}"
    mkdir(lens_dir)
    Gal.lens_dir = lens_dir
    return lens_dir

##########################
##########################
# Sampling of the Profiles
##########################
##########################

# source parametrisation
kwargs_sersic_ellipse_basic = {'R_sersic': .1, 'n_sersic': 3, 
                            'center_x': 0,
                            'center_y': 0, 
                            'e1': 0.0, 
                            'e2': 0.0}
kwargs_sersic_ellipse         = {'amp': 100.}|kwargs_sersic_ellipse_basic
kwargs_sersic_ellipse_mag     = {'magnitude':24.}|kwargs_sersic_ellipse_basic
kwargs_sersic_ellipse_abs_mag = {'abs_magnitude':-21}|kwargs_sersic_ellipse_basic

kwargs_source_default     = kwargs_sersic_ellipse_abs_mag
source_model_list         = ['SERSIC_ELLIPSE']

def get_kwargs_sourceSim(Sim,kwargs_source=None,lens=None):
    if kwargs_source is None:
        if lens is None:
            kwargs_source = kwargs_source_default
        else:
            kwargs_source = lens.kwargs_source_def
    if "abs_magnitude" in kwargs_source.keys():
        # convert absolute magnitude in apparent
        Mag = kwargs_source["abs_magnitude"]
        dL  = lens.cosmo.luminosity_distance(lens.gallens.z_source)
        m = Mag + 5*np.log10(dL.to("pc").value/10)
        del kwargs_source["abs_magnitude"]
        kwargs_source["magnitude"] = m
        
    if "magnitude" in kwargs_source.keys():
        kwargs_source_list       = [kwargs_source]
        # the following only depends on -kwargs_source_params, -magnitude_0_point -source_model_list
        _, kwargs_source_list, _ = Sim.magnitude2amplitude(kwargs_source_mag = kwargs_source_list)
        kwargs_source            = kwargs_source_list[0]
    return kwargs_source

def get_dataclasses(Sim):
    print("Pixel_num: ",  Sim.numpix)
    print("DeltaPix: ",   np.round(Sim.pixel_scale,3))
    data_class         = Sim.data_class
    psf_class          = Sim.psf_class
    kwargs_numerics    = {'supersampling_factor': 1, 'supersampling_convolution': False}
    source_model_class = Sim.source_model_class
    return data_class,psf_class,source_model_class,kwargs_numerics

##########################
# Model class for parts. #
##########################
# kwargs of ultra-performing band for default simulated images -> quite arbitrary, possibly to improve 
kwargs_band_sim = {'read_noise': 0, # no RN noise
 'pixel_scale': None,               # to update depending on the lens
 'ccd_gain': 2.5,             # standard gain for HST
 'exposure_time': 5400.0,     # very long exp time for HST
 'sky_brightness': 35,        #"dark" sky
 'magnitude_zero_point': 30,  # very deep 
 'num_exposures': 4,          # standard HST n exp.
 'psf_type': 'NONE'}          # "infinite" psf resolution 

kw_prior_z_source_minimal = {"z_source_max":conf.z_source_max}
kw_prior_z_source_stnd    = kw_prior_z_source_zl|kw_prior_z_source_minimal
                
def MAD_mask(values,v0=0,sigma_scale=3):
    # robust estimator of noise: Median Absolute Deviation    
    mad = np.median(np.abs(values - np.median(values)))

    sigma = 1.4826 * mad

    mask = np.abs(values-v0) < sigma_scale*sigma   # ~99.7% Gaussian confidence
    return mask

# optimised w. CGPT:
def fit_xy_spline_old(x, y,
    u=np.linspace(0, 1, 200),
    n_eval=150,           # points used for error estimation
    ):
    # --- angular ordering ---
    xc = np.median(x)
    yc = np.median(y)
    theta = np.arctan2(y - yc, x - xc)
    order = np.argsort(theta)
    
    x_ord = x[order]
    y_ord = y[order]
    n = len(x_ord)

    # fixed parameter grid
    u_fit = np.linspace(0, 1, n, endpoint=False)

    # subsampling indices for error evaluation
    idx = np.linspace(0, n - 1, n_eval).astype(int)
    u_sub = u_fit[idx]
    x_sub = x_ord[idx]
    y_sub = y_ord[idx]

    # --- coarse search ---
    s_vals = np.logspace(-5,-1, 12)
    errs = np.empty(len(s_vals))

    for i, s in enumerate(s_vals):
        tck, _ = splprep(
            [x_ord, y_ord],
            s=s * n,
            per=True,
            quiet=True
        )
        xs, ys = splev(u_sub, tck)
        errs[i] = np.sum(np.hypot(xs - x_sub, ys - y_sub))

    # --- refine around minimum ---
    i0 = np.argmin(errs)
    lo = max(i0 - 1, 0)
    hi = min(i0 + 1, len(s_vals) - 1)

    s_refined = np.logspace(
        np.log10(s_vals[lo]),
        np.log10(s_vals[hi]),
        10
    )

    best_err = np.inf
    best_tck = None

    for s in s_refined:
        tck, _ = splprep(
            [x_ord, y_ord],
            s=s * n,
            per=True,
            quiet=True
        )
        xs, ys = splev(u_sub, tck)
        err = np.sum(np.hypot(xs - x_sub, ys - y_sub))
        if err < best_err:
            best_err = err
            best_tck = tck

    # --- final evaluation ---
    xs, ys = splev(u, best_tck)
    return xs, ys

# this obtained via Claude
def fit_xy_spline(x, y,
                  u=np.linspace(0, 1, 200),
                  n_eval=150):   # ← expose this; set False for open arcs
    
    # --- order by arc length, not angle ---
    # Start from the point with the most extreme position (e.g. leftmost)
    # to get a consistent starting point
    i_start = np.argmin(x)   # or use np.argmin(y), depending on geometry
    
    # reorder: rotate array so i_start is first
    x_rot = np.roll(x, -i_start)
    y_rot = np.roll(y, -i_start)
    
    # compute cumulative arc length as parameter
    dx   = np.diff(x_rot)
    dy   = np.diff(y_rot)
    ds   = np.hypot(dx, dy)
    # argsort by arc length from the starting point would require 
    # nearest-neighbour chaining; simpler: use angular order only 
    # if periodic, otherwise sort by x or use a greedy chain
    xc    = np.median(x)
    yc    = np.median(y)
    for _ in range(3):
        xc = np.average(x, weights=1/np.hypot(x - xc, y - yc))
        yc = np.average(y, weights=1/np.hypot(x - xc, y - yc))

    theta = np.arctan2(y - yc, x - xc)
    order = np.argsort(theta)
    x_ord = x[order]
    y_ord = y[order]
    x_ord = np.append(x_ord, x_ord[0])
    y_ord = np.append(y_ord, y_ord[0])


    n     = len(x_ord)
    u_fit = np.linspace(0, 1, n)
    idx   = np.linspace(0, n - 1, n_eval).astype(int)
    u_sub = u_fit[idx]
    x_sub = x_ord[idx]
    y_sub = y_ord[idx]

    # --- coarse search ---
    s_vals = np.logspace(-5, -1, 12)
    errs   = np.empty(len(s_vals))
    for i, s in enumerate(s_vals):
        tck, _ = splprep([x_ord, y_ord], s=s * n,
                          per=True, quiet=True)
        xs, ys = splev(u_sub, tck)
        if i==0:
            x_rough,y_rough = copy(xs),copy(ys)
        errs[i] = np.sum(np.hypot(xs - x_sub, ys - y_sub))
    if np.std(xs)<(np.std(x_ord)/1e5):
        print("Rough fit seems to work best")
        return x_rough,y_rough
    # --- refine ---
    i0 = np.argmin(errs)
    lo = max(i0 - 1, 0)
    hi = min(i0 + 1, len(s_vals) - 1)
    s_refined = np.logspace(np.log10(s_vals[lo]), np.log10(s_vals[hi]), 10)
    best_err, best_tck = np.inf, None
    for s in s_refined:
        tck, _ = splprep([x_ord, y_ord], s=s * n,
                          per=True, quiet=True)
        xs, ys = splev(u_sub, tck)

        err    = np.sum(np.hypot(xs - x_sub, ys - y_sub))
        if err < best_err:
            best_err, best_tck = err, tck

    xs, ys = splev(u, best_tck)
    return xs, ys
#
# helper funct
#

def kw_prior2like_zs(kw_prior_z_source,z_lens):
    """
    Convert kw_prior_z_source into the kw_like 
    needed for the Likelihood class for the z_source sampling
    if kw_prior_z_source does not have the required function, returns None
    if kw_prior_z_source had "fixed_z_source", fix the z_source
    """
    kw_like_zs = None
    prior_keys = kw_prior_z_source.keys()
    if "f_lkl_z_source" in prior_keys and "fixed_z_source" in prior_keys:
        raise RuntimeError("Either sample z_source or fix it")
    elif "f_lkl_z_source" in prior_keys:
        kw_like_zs = {}
        # wrapper funct. to fix the z_lens
        def like_func_zs(z_source,*args):
            f_lkl_pr = kw_prior_z_source["f_lkl_z_source"]
            return f_lkl_pr(z_source=z_source,z_lens=z_lens,*args)
        kw_like_zs["like_func"] = like_func_zs
        kw_like_zs["like_prms"] = kw_prior_z_source.get("prms_lkl_z_source",[]) 
        return kw_like_zs
    elif "fixed_z_source" in prior_keys:
        def lkl(zs,*args):
            return 0 
        kw_like_zs = {"fixed":kw_prior_z_source["fixed_z_source"]}
    return kw_like_zs
        

# we override the standard loading function to recompute some large dataset that
# are deleted to save space
def LoadLens(LnsCl,verbose=True):
    LnsCl = LoadClass(LnsCl,verbose=verbose,path_base=path_nazgul)
    # has to consider the possibility it failed to load
    if LnsCl: 
        # recompute deleted components
        LnsCl.unpack()
    return LnsCl

def store_lens(lens_class):
    store_class(lens_class,path=lens_class.pkl_path,LoadClass=LoadLens)

def ReadLens(aClass,verbose=True):
    return LoadLens(aClass.pkl_path,verbose=verbose)


def get_extents(arcXkpc,_radec=None):
    """Returns the extent of the image in various units
    """
    _ra,_dec = _radec
    RA,DEC   = util.array2image(_ra),util.array2image(_dec)
    ra0,dec0 = RA[0]*u.arcsec,DEC.T[0]*u.arcsec # center of the bin

    Dra0  = np.diff(ra0)  
    Ddec0 = np.diff(dec0)

    ra_edges  = np.hstack([ra0[0]-(Dra0[0]/2.),ra0[1:]-(Dra0/2.),ra0[-1]+.5*Dra0[-1]])
    dec_edges = np.hstack([dec0[0]-(Ddec0[0]/2.),dec0[1:]-(Ddec0/2.),dec0[-1]+.5*Ddec0[-1]])
    
    # in kpc:
    xmin = ra_edges[0]/arcXkpc
    xmax = ra_edges[-1]/arcXkpc
    ymin = dec_edges[0]/arcXkpc
    ymax = dec_edges[-1]/arcXkpc

    extent_kpc    = [xmin.value, xmax.value,ymin.value, ymax.value] #kpc
    extent_arcsec = [ra_edges[0].value, ra_edges[-1].value,dec_edges[0].value, dec_edges[-1].value] #arcsec
    bins_arcsec   = [ra_edges,dec_edges]
    kw_extents = {"extent_kpc":extent_kpc,
              "extent_arcsec":extent_arcsec,
              "bins_arcsec":bins_arcsec}
    return kw_extents


def get_RADEC(_radec):
    _ra,_dec = _radec
    RA,DEC   = util.array2image(_ra),util.array2image(_dec)
    return RA,DEC


class BasicLensPart(BasicGal):
    _large_attributes_unpack = []
    _large_attributes_setup  = []

    def __init__(self,
                 Galaxy,      # class instance of PartGal
                 projection_index, # projection index
                 kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 pixel_num=conf.pixel_num, # number of pixels
                 min_thetaE = conf.min_thetaE, # minimum theta observable
                 scale_tE = conf.scale_tE,     # rescaling of theta E to get radius (ie size of image cutout) = tE*scale_tE 
                 reload=True # reload previous instance
                 ):

        self.proj_index   = projection_index
        # Wrapper class of PartGal to extend for projections
        Galaxy             = ProjGal(Gal=Galaxy,projection_index=self.proj_index) 
        # setup of data
        self.Gal           = Galaxy
        self.Gal_path      = Galaxy.dill_path
        self.Gal_name      = Galaxy.name # must be stored
        z_source_max       = kw_prior_z_source["z_source_max"]
        # if reload, check if Gal is a lens - if it isn't, raise error
        if reload:
            if not self.Gal.is_lens(z_source_max=z_source_max,
                                   min_thetaE=min_thetaE):
                raise ProjectionError(f"Previously defined as not a lens given z_(s, max)={z_source_max} and min_thetaE={min_thetaE}")

        self.reload   = reload
        ######
        # lensing params
        self.pixel_num     = pixel_num      
        self.kwlens_part   = kwlens_part
        self.PartLens      = PartLens(kwlens_part)
        self.PartLens_name = self.PartLens.name
        # cosmo params
        self.z_lens        = self.Gal.z
        self.cosmo         = self.Gal.cosmo
        self.arcXkpc       = self.cosmo.arcsec_per_kpc_proper(self.z_lens)

        # criteria for supercriticality of the lens
        self.z_source_max      = z_source_max
        self.kw_prior_z_source = kw_prior_z_source
        self.kw_like_zs        = kw_prior2like_zs(kw_prior_z_source=kw_prior_z_source,
                                              z_lens=self.z_lens)
        self.min_thetaE        = ensure_unit(min_thetaE,u.arcsec) #arcsec
        self.scale_tE          = scale_tE

    def __str__(self): 
        """Human-readable identifier.

        Lazily initializes the name if it has not been generated yet.
        """
        #return f'Name:{self.name}\nHash{self._hash_b64}'
        return self.name
    @property
    def name(self):
        # define name and path of savefile
        name= f"Lens_{self.Gal_name}_Prj{self.proj_index}"
        return name
        
    def get_kw_sublenspart(self):
        kw_sublenspart = {}
        kw_sublenspart["Galaxy"] = self.Gal
        kw_sublenspart["reload"] = self.reload
        kw_sublenspart["pixel_num"] = self.pixel_num
        kw_sublenspart["min_thetaE"] = self.min_thetaE
        kw_sublenspart["kwlens_part"] = self.kwlens_part
        kw_sublenspart["projection_index"] = self.proj_index
        kw_sublenspart["kw_prior_z_source"] = self.kw_prior_z_source
        return kw_sublenspart

    def ReadClass(self,cl):
        L = ReadLens(cl)
        if L:
            L._unpack_Gal()
            L._unpack_PartLens()
        return L

    def _unpack_Gal(self):
        # reload Galaxy and cosmology
        if not hasattr(self,"Gal"):
            Galaxy   = LoadGal(_resolve_gal_path(self.Gal_path))
            Galaxy.rebase()
            if not isinstance(Galaxy,ProjGal):
                Galaxy   = ProjGal(Gal=Galaxy,
                               projection_index=self.proj_index)
            self.Gal = Galaxy
            # verify that we load the correct galaxy
            assert self.Gal_name == Galaxy.name
        if not hasattr(self,"cosmo"):
            self.cosmo = self.Gal.cosmo
        return
    def _unpack_PartLens(self):
        if not hasattr(self,"PartLens"):
            self.PartLens = PartLens(self.kwlens_part)
        if np.all([hasattr(self,att) for att in ["z_lens","z_source","cosmo"]]):
            # the setup is very fast 
            self.PartLens.setup(self)
        return
    def _unpack(self,verbose=True):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        print("Unpacking basic lens...")
        # Gal & cosmo
        self._unpack_Gal()
        
        # re-define PartLens
        self._unpack_PartLens()

    def _setup(self):
        self._unpack_Gal()
        self.PartLens = PartLens(self.kwlens_part)
        
    def store(self):
        store_lens(self)
        
    @property
    def pkl_path(self):
        return _resolve_gal_path(self.savedir)/f"{self.name}_{self._hash_b64}.pkl"

    def is_precomputed(self):
        if self.pkl_path.exists():
            return True
        return False
        
    @property
    def deltaPix(self):
        Diam_arcsec = 2*self.radius #diameter in arcsec
        deltaPix    = Diam_arcsec/self.pixel_num # ''/pix
        return deltaPix
        
    #
    # Coordinates 
    # 
    @property
    def _radec(self):
        return util.make_grid(self.pixel_num,to_dimless(self.deltaPix))

    @property
    def kw_extents(self):
        kw_extents = get_extents(arcXkpc=self.arcXkpc,
                                     _radec=self._radec)
        return kw_extents
        
    def get_RADEC(self):
        return get_RADEC(self._radec)
    #
    # Lensing computations
    #
    @cached_property
    def alpha_map(self):
        return self._alpha_map(_radec=None)
    @cached_property
    def kappa_map(self):
        return self._kappa_map(_radec=None)
    @cached_property
    def hessian(self):
        return self._hessian(_radec=None)
    @cached_property
    def psi_map(self):
        return self._psi_map(_radec=None)

    def sample_z_source(self,z_source_min,z_source_max):
        # this is here to allow modularity 
        if self.kw_like_zs is None:
            # simple uniform sample btw the ranges
            z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        elif "fixed" in self.kw_like_zs.keys():
            # if fixed, we don't sample it
            z_source = self.kw_like_zs["fixed"]
            # ensure it is still acceptable
            assert z_source>=z_source_min and z_source<z_source_max
        else:
            # else we follow the given likelihood
            Lkl_source = Likelihood(var_range=[[z_source_min,z_source_max]],
                                    kw_like = self.kw_like_zs)
            # the following still has shape = n_walkers
            z_source_list = Lkl_source.sample(n_samples=1,progress=False)
            z_source      = np.random.choice(z_source_list)
        return z_source
    
    def galaxy_projection(self,verbose=True,recompute=False,**kwargs_proj):
        """
        Compute galaxy projection given a certain projection

        If not reload, then automatically recompute
        else, if not recompute, check if results are present - if not, recompute

        At the end, if not recompute (ie results are already present) return, 
        else recompute and return
        """
        if not self.reload:
            recompute = True
        else:
            if not recompute:
                list_att_gal_prj = ["z_source_min",
                                   "z_source",
                                   "MD_coords",
                                    "thetaE",
                                    "SigCrit"]
                for att_gal_prj in list_att_gal_prj:
                    # we should not recompute it if it already has the solutions 
                    if not hasattr(self,att_gal_prj):
                        recompute = True
                        break
        if not recompute:
            return
        # Compute projection
        kwres_proj_res    = project_Gal(GalProj=self.Gal,
                                        z_source_max=self.z_source_max,
                                        sample_z_source=self.sample_z_source,
                                        min_thetaE=self.min_thetaE,
                                        arcXkpc=self.arcXkpc,verbose=verbose,
                                        reload=not recompute,
                                       **kwargs_proj)
        # store results
        assert self.proj_index == kwres_proj_res["proj_index"]
        self.z_source_min = kwres_proj_res["z_source_min"]
        self.z_source     = kwres_proj_res["z_source"]
        self.MD_coords    = kwres_proj_res["MD_coords"]
        if verbose:
            print("Z source sampled:",np.round(self.z_source,2))
        self.thetaE    = kwres_proj_res["thetaE"]
        if verbose:
            print("Approx. thetaE:",np.round(self.thetaE,3))
   
        # the following can only be computed once we know the z_source:
        self.SigCrit       = SigCrit(cosmo=self.cosmo,
                                     z_lens=self.z_lens,
                                     z_source=self.z_source) # Msun/kpc^2
        return 