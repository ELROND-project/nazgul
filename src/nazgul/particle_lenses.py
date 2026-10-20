"""
Particle lenses function
Main Class: PartLens
From sample of particles, returns kwargs and model for lenstronomy
depending on the particle potential profile chosen (for now AS or PM)
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
from warnings import warn

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.arsinh_parall import ParallelArsinh
from lenstronomy.LensModel.Profiles.point_mass_parall import ParallelPointMass

from python_tools.tools import short_SciNot,to_dimless
# default params:
theta_c_AS     = 5e-3 
default_kwlens_part_AS = {"type":"AS","theta_cAS":theta_c_AS}
default_kwlens_part_PM = {"type":"PM"}
#
# cosmo from https://academic.oup.com/mnras/article/474/3/3391/4644836, Agnello 2017
# point mass theta_E (from eq.4.7 of Meneghetti's lecture note - and by memory)
# theta_E = \sqrt ( 4GM D_ls / (c^2 Ds Dl) )
# divide the computation such that it's done only once
def thetaE_PM_prefact(z_lens,z_source,cosmo):    
    cosmo_ds  = cosmo.angular_diameter_distance(z_source)
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens)
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source)
    pref      = 4*const.G*cosmo_dds/(const.c*const.c*cosmo_ds*cosmo_dd)
    return np.sqrt(pref) # 

@u.quantity_input
def thetaE_PM(M:u.g,theta_pref:u.g**-.5):
    thetaE_rad = np.sqrt(M)*theta_pref
    thetaE     = thetaE_rad.to("")*u.rad.to("arcsec")
    return thetaE.value #in arcsec
# ARCSINH thetaE is actually the same as PM
def thetaE_AS_prefact(z_lens,z_source,cosmo):    
    # is actually the same of PM, but it could be in principle different
    return thetaE_PM_prefact(z_lens,z_source,cosmo)
@u.quantity_input
def thetaE_AS(M:u.g,theta_pref:u.g**-.5):
    # is actually the same of PM, but it could be in principle different
    return thetaE_PM(M,theta_pref)
"""
# maybe useful funct:
def MfromtE(tE,theta_pref:u.g**-.5):
    tErad  = tE*u.arcsec.to("rad")
    M = (tErad/theta_pref)**2
    return M.to("Msun")
"""
def _build_kwargs_lens_AS(args):
        tE,tCAS, ra, dec = args
        return {
            "theta_E": tE,
            "theta_c": tCAS,
            "center_x": ra,
            "center_y": dec
        }
        
def _build_kwargs_lens_PM(args):
        tE, ra, dec = args
        return {
            "theta_E": tE,
            "center_x": ra,
            "center_y": dec
        }


# From EAGLE simulation

# Helper funct - create the kwargs_lens given the part. parameters, ie theta_E,x,y,(core if needed) 
# and the lens_model
#

def add_lenses(kwargs_lens,lens_model_list,kw_add_lenses=None):
    # add lenses models if present
    # useful for LOS
    if kw_add_lenses is not None:
        add_kwl         = kw_add_lenses["kwargs_lens"]
        add_lml         = kw_add_lenses["lens_model_list"]
        lens_model_list = [*add_lml,*lens_model_list]
        kwargs_lens     = [*add_kwl,*kwargs_lens]
    return kwargs_lens,lens_model_list
    
def get_lens_model_PM(thetaEs,samples,kw_add_lenses=None):
    kwargs_lens_PM  = [_build_kwargs_lens_PM((thetaEs, samples[0],samples[1]))]
    lens_model_list = ["POINT_MASS_PARALL"]
    kwargs_lens_PM,lens_model_list= add_lenses(kwargs_lens=kwargs_lens_PM,
                                               lens_model_list=lens_model_list,
                                               kw_add_lenses=kw_add_lenses)
    lens_model_PM   = LensModel(lens_model_list=lens_model_list)
    return kwargs_lens_PM,lens_model_PM 

def get_lens_model_AS(theta_cAS,thetaEs,samples,kw_add_lenses=None):
    try:
        len(theta_cAS)
    except TypeError:
        theta_cAS  *= np.ones_like(thetaEs)
    kwargs_lens_AS  = [_build_kwargs_lens_AS((thetaEs,theta_cAS, samples[0],samples[1]))]
    lens_model_list = ["ARSINH_PARALL"]
    kwargs_lens_AS,lens_model_list= add_lenses(kwargs_lens=kwargs_lens_AS,
                                               lens_model_list=lens_model_list,
                                               kw_add_lenses=kw_add_lenses)
    lens_model_AS   = LensModel(lens_model_list=lens_model_list)
    return kwargs_lens_AS,lens_model_AS

    
def get_lens_model_AS_adaptive(thetaEs,samples,kw_adaptive,kw_add_lenses=None):
    # adapt theta_cAS to the density of particles
    theta_cASs = adaptive_tcAS(thetaEs,samples,kw_adaptive=kw_adaptive)
    return get_lens_model_AS(thetaEs,theta_cASs,samples,kw_add_lenses=kw_add_lenses)

"""
class Prof2LensModel():
    raise RuntimeError("Not used")
    #Wrapper class used only here to rename the functions in order to be 
    #compatible with LensModel naming -> not sure it's how it should be done...
    def __init__(self,Prof):
        self.Prof = Prof
    def alpha(self,*args,**kwargs):
        return self.Prof.function(*args,**kwargs)
    def potential(self,*args,**kwargs):
        return self.Prof.function(*args,**kwargs)
    def kappa(self,*args,**kwargs):
        f_xx, f_xy, f_yx, f_yy = self.Prof.hessian(*args,**kwargs)
        kappa = 1.0 / 2 * (f_xx + f_yy)
        return kappa
"""

# Claude indicated me this optimised method - less precise but enough so and fast
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

def binned_density(pos,h=0.05,ngrid=1000):
    lo, hi = pos.min(axis=1), pos.max(axis=1)
    pad = 4 * h
    xe = np.linspace(lo[0] - pad, hi[0] + pad, ngrid + 1)
    ye = np.linspace(lo[1] - pad, hi[1] + pad, ngrid + 1)
    H, _, _ = np.histogram2d(pos[0], pos[1], bins=[xe, ye])
    dx, dy = xe[1] - xe[0], ye[1] - ye[0]
    sm = gaussian_filter(H / (dx * dy), sigma=(h / dx, h / dy), mode="constant")
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    interp = RegularGridInterpolator((xc, yc), sm, bounds_error=False, fill_value=None)
    #return interp(pos.T) - 1.0 / (2 * np.pi * h**2)
    # we ignore the subtraction of the term for the single particle in order not to have 0
    # density in low-density regions
    return interp(pos.T) 
    
def _get_dens(samples):
    """
    input: samples array (2,N) of particles
    output: number density at the position of each particle 
    """
    samples = np.array(samples) # this scrubs away the dimensions
    dens = binned_density(samples)
    # we cap a minimum density
    if dens.min()<=0:
        dens[dens<=0] = 1e-12 
    return dens
    
def adaptive_tcAS(thetaEs,samples,kw_adaptive):
    types = ["SIS","NDENS"]
    if kw_adaptive["type"]=="SIS":
        theta_max     = kw_adaptive["theta_max"]
        smooth_fact   = kw_adaptive["smooth_fact"] #=1
        theta         = np.hypot(*samples)
        n_smpl        = len(samples[0])
        theta_core_AS = smooth_fact*np.sqrt(2*np.pi*theta_max*theta/n_smpl)
    elif kw_adaptive["type"]=="NDENS":
        # we consider the number density and smoothing factor 
        # to adapt the theta_core
        smooth_fact   = kw_adaptive["smooth_fact"] #=1
        # the following must be a list of number density which follows from
        # the samples (ie not a simple density estimation)
        n_dens_arcsec2 = _get_dens(samples)
        theta_core_AS  = smooth_fact/np.sqrt(n_dens_arcsec2)
    else:
        raise RuntimeError(f"kw_adaptive type needs to be {types}, not {kw_adaptive['type']}")
    return theta_core_AS

def get_lens_prof_PM(thetaEs,samples):
    kwargs_lens_PM  = _build_kwargs_lens_PM((thetaEs, samples[0],samples[1]))
    lens_prof_PM    = ParallelPointMass()
    return kwargs_lens_PM,lens_prof_PM 

def get_lens_prof_AS(theta_cAS,thetaEs,samples):
    try:
        len(theta_cAS)
    except TypeError:
        theta_cAS *= np.ones_like(thetaEs)
    kwargs_lens_AS = _build_kwargs_lens_AS((thetaEs,theta_cAS, samples[0],samples[1]))
    lens_prof_AS   = ParallelArsinh()
    return kwargs_lens_AS,lens_prof_AS
    
def get_lens_prof_AS_adaptive(thetaEs,samples,kw_adaptive):
    theta_cASs = adaptive_tcAS(thetaEs=thetaEs,
                               samples=samples,
                               kw_adaptive=kw_adaptive)
    return get_lens_prof_AS(theta_cASs,thetaEs,samples)
    
#
# Particle functions
# -> not needed anymore due to structural changes
    
"""
def get_kwrg_PM(samples,Ms,
                    z_lens,z_source,
                    theta_E):     
    theta_pref = thetaE_PM_prefact(z_lens=z_lens,z_source=z_source)
    thetaEs    = thetaE_PM(M=Ms,theta_pref = theta_pref)

    kwargs_lens_PM,lens_model_PM = get_lens_model_PM(thetaEs,samples)
    return {"kwargs_lens":kwargs_lens_PM,"lens_model_PART":lens_model_PM}
                
def get_kwrg_AS(samples,Ms,theta_cAS
                    z_lens,z_source,
                    theta_E):
                    
    theta_pref = thetaE_AS_prefact(z_lens=z_lens,z_source=z_source)
    thetaEs    = thetaE_AS(M=Ms,theta_pref=theta_pref)
 
    kwargs_lens_AS,lens_model_AS = get_lens_model_AS(theta_cAS,thetaEs,samples)

    return {"kwargs_lens":kwargs_lens_AS,"lens_model_PART":lens_model_AS}
"""
#
# naming functions
#

def _get_tcAS_str(kwargs_lens):

    return tcAS_str 
    
def get_name_PM(kw_lens=None):
    """Get the name for Point Mass lenses
    """
    return f"PM"
    
def get_name_AS(kwargs_lens):
    """Get the name for Arsinh particle lenses
    """
    tcAS     = to_dimless(kwargs_lens["theta_cAS"])
    tcAS_str = short_SciNot(tcAS)
    tcAS_str = _get_tcAS_str(kwargs_lens)
    return f"AS_tc{tcAS_str}"

# Lens modelling 
#################

#
# Particle Lens computation class
#
class PartLens_basis():
    """
    Common class for all PM lens approach
    """
    def __init__(self,kwargs_lens_part):
        self.kwargs_lens = kwargs_lens_part
        type_part = kwargs_lens_part["type"]
        self.name = type_part
        
        if type_part=="PM":
            self.thetaE_prefact = thetaE_PM_prefact
            self.thetaE         = thetaE_PM

        elif type_part=="ARCSINH" or type_part=="AS":
            self.thetaE_prefact = thetaE_AS_prefact
            self.thetaE         = thetaE_AS
            
        elif type_part=="ARCSINH_TCADAPT" or type_part=="ASTA":
            self.thetaE_prefact = thetaE_AS_prefact
            self.thetaE         = thetaE_AS
            self.kw_adaptive    = kwargs_lens_part["kw_adaptive"]
        else:
            raise TypeError("This particle model is not known: "+type_part)

    
    def setup(self,Mod):
        """Define cosmological parameters from the LensPart class
        """
        self.z_lens   = Mod.z_lens
        self.z_source = Mod.z_source
        self.cosmo    = Mod.cosmo
                                          
    def get_lens_PART(self,samples,Ms):
        raise RuntimeError("To implement")
    
    ### Class Structure ####
    ########################
    def _identity(self):
        """Returns tuple to identify uniquely this galaxy
        convert kwargs in immutable tuple to be hashable"""
        Id = (self.name,
              tuple(sorted(self.kwargs_lens.items())))
        return Id
    
    def __hash__(self):
        """simplify the hash method"""
        # note: this works only if _identity does not contain
        # dictionary/kwargs
        return hash(self._identity())

    def __eq__(self, other):
        if not isinstance(other, PartLens_basis):
            return NotImplemented
        return self._identity() == other._identity()

    def __str__(self):
        if not getattr(self,"name",False):
            self._setup_names()
        return self.name



class PartLens(PartLens_basis):
    def __init__(self,kwargs_lens_part):
        super().__init__(kwargs_lens_part)
        type_part = kwargs_lens_part["type"]
        if type_part=="PM":
            self.get_lens_prof  = get_lens_prof_PM
        elif type_part=="ARCSINH" or type_part=="AS":
            self.get_lens_prof  = get_lens_prof_AS
        elif type_part=="ARCSINH_TCADAPT" or type_part=="ASTA":
            self.get_lens_prof  = get_lens_prof_AS_adaptive
        else:
            raise TypeError("This particle model is not known: "+type_part)

    def get_lens_PART(self,samples,Ms):
        """From the sample of particles (position and masses) return their model and parameters
        in lenstronomy format.
        If present, consider additional lenses profiles
        """
        theta_pref = self.thetaE_prefact(z_lens=self.z_lens,z_source=self.z_source,cosmo=self.cosmo)
        thetaEs    = self.thetaE(M=Ms,theta_pref = theta_pref)
        kw_lns_mod = {}
        if self.name =="ARSINH"  or self.name =="AS":
            kw_lns_mod = {"theta_cAS":self.kwargs_lens["theta_cAS"]}
        elif self.name=="ARCSINH_TCADAPT" or self.name=="ASTA":
            kw_lns_mod = {"kw_adaptive":self.kw_adaptive}
            
        kwargs_lens_PART,lens_profile_PART = self.get_lens_prof(thetaEs=thetaEs,
                                                               samples=samples,
                                                               **kw_lns_mod)
        return kwargs_lens_PART,lens_profile_PART

    
    ### Class Structure ####
    ########################
    def __eq__(self, other):
        if not isinstance(other, PartLens):
            return NotImplemented
        return self._identity() == other._identity()

        
class PartLensExpanded(PartLens_basis):
    """
    Add the possibility to consider ulterior lens profiles
    """
    def __init__(self,kwargs_lens_part,kw_add_lenses=None,kw_add_part=None):
        raise RuntimeError("Do not use this")
        super().__init__(kwargs_lens_part)
        self.kw_add_lenses = kw_add_lenses
        self.kw_add_part   = kw_add_part #additional kwargs for particle
        type_part = kwargs_lens_part["type"]
        if type_part=="PM":
            self.get_lens_model  = get_lens_model_PM
        elif type_part=="ARCSINH" or type_part=="AS":
            self.get_lens_model  = get_lens_model_AS
        else:
            raise TypeError("This particle model is not known: "+type_part)
               
    def get_lens_PART(self,samples,Ms):
        """From the sample of particles (position and masses) return their model and parameters
        in lenstronomy format.
        If present, consider additional lenses profiles
        """
        theta_pref = self.thetaE_prefact(z_lens=self.z_lens,z_source=self.z_source,cosmo=self.cosmo)
        thetaEs    = self.thetaE(M=Ms,theta_pref = theta_pref)
        kw_lns_mod = {}
        if self.name =="ARSINH"  or self.name =="AS":
            kw_lns_mod = {"theta_cAS":self.kwargs_lens["theta_cAS"]}
        elif self.name =="ARCSINH_TCADAPT"  or self.name =="ASTA":
            kw_lns_mod = {"kw_adaptive":self.kw_add_part}
            
        kwargs_lens_PART,lens_model_PART = self.get_lens_model(thetaEs=thetaEs,
                                                               samples=samples,
                                                               kw_add_lenses=self.kw_add_lenses,
                                                               **kw_lns_mod)
        return kwargs_lens_PART,lens_model_PART
    
    ### Class Structure ####
    ########################
    def _identity(self):
        """Returns tuple to identify uniquely this galaxy
        convert kwargs in immuatable tuple to be hashable"""
        Id = (self.name,
              "Expanded",
              tuple(sorted(self.kwargs_lens.items())))
        return Id

    def __eq__(self, other):
        if not isinstance(other, PartLensExpanded):
            return NotImplemented
        return self._identity() == other._identity()

########################
########################
