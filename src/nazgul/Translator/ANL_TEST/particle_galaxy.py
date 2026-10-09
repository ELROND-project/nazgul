import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck13 as default_cosmo
from lenstronomy.Util.param_util import ellipticity2phi_q, phi_q2_ellipticity

from python_tools.tools import to_dimless,short_SciNot,ensure_unit,mkdir

n_smpl   = int(5e5)
theta_E  = 0.8 #arcsec
z_lens   = 0.1
z_source = 1.2
cntx     = 0
cnty     = 0

q,phi    = .3,0
e1,e2    = phi_q2_ellipticity(phi=phi,q=q)


def SigCrit(cosmo=default_cosmo,z_lens=z_lens,z_source=z_source):
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens).to("kpc")   #kpc
    cosmo_ds  = cosmo.angular_diameter_distance(z_source).to("kpc") #kpc
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source).to("kpc") #kpc

    Sigma_Crit        = (cosmo_ds*const.c**2)/(4*np.pi*const.G*cosmo_dds*cosmo_dd) #
    return Sigma_Crit.to("Msun /(kpc kpc)")

def SigCrArc2(cosmo=default_cosmo,z_lens=z_lens,z_source=z_source):
    arcXkpc           = cosmo.arcsec_per_kpc_proper(z_lens) # ''/kpc
    Sigma_Crit        = SigCrit(cosmo=cosmo,z_lens=z_lens,z_source=z_source)
    Sigma_Crit_arcs2  = Sigma_Crit.to("Msun /(kpc kpc)")/(arcXkpc*arcXkpc)
    return Sigma_Crit_arcs2

def sample_SIS(n_smpl=n_smpl,
                theta_E=theta_E,
                z_lens=z_lens,
                z_source=z_source,
                theta_max=None,
                cosmo=default_cosmo,
                cntx=cntx,cnty=cnty):
    if theta_max is None:
        theta_max = 4*theta_E
    phi,theta    = np.random.uniform([0,0],[2*np.pi,to_dimless(theta_max)],size=(n_smpl,2)).T
    RAs          = theta*np.cos(phi) +cntx
    DECs         = theta*np.sin(phi) +cnty
    samples_arcs = np.array([RAs,DECs])*u.arcsec

    arcXkpc      = cosmo.arcsec_per_kpc_proper(z_lens) # ''/kpc
    samples      = samples_arcs/arcXkpc # kpc

    # theta_E must be connected to the mass scale, as it only scales up and down the kappa
    kwargs_lens_SIS  = [{'theta_E' : to_dimless(theta_E), 
                        'center_x': cntx, 
                        'center_y': cnty}] 

    # note the discussion on Notion: SIS sampling:easy!
    Sigma_Crit_arcs2 = SigCrArc2(cosmo=cosmo,z_lens=z_lens,z_source=z_source)

    Mmax       = Sigma_Crit_arcs2 *np.pi*ensure_unit(theta_E,u.arcsec)*ensure_unit(theta_max,u.arcsec)
    Mscale_sun = Mmax.to("Msun")/n_smpl

    Mscale_sun = ensure_unit(Mscale_sun,u.Msun)
    
    return kwargs_lens_SIS,Mscale_sun,samples

def sample_SIE(n_smpl=n_smpl,
                theta_E=theta_E,
                z_lens=z_lens,
                z_source=z_source,
                theta_max=None,
                cosmo=default_cosmo,
                cntx=cntx,cnty=cnty,
                e1=e1,e2=e2):
    """
    Sample a SIE distribution with particles with the same mass and returns inputs for Translator 
    """
    phi_ell,q_ell = ellipticity2phi_q(e1,e2)
    if theta_max is None:
        theta_max = 4*theta_E/q_ell
    phi,theta    = np.random.uniform([0,0],[2*np.pi,to_dimless(theta_max)],size=(n_smpl,2)).T
    RAs_sis       = theta*np.cos(phi) 
    DECs_sis      = theta*np.sin(phi)
    
    """
    Change coordinate by coordinate rescaling
    """
    x_ell = RAs_sis
    y_ell = DECs_sis * q_ell
    rot_m = np.array([[np.cos(phi_ell),-np.sin(phi_ell)],
                      [np.sin(phi_ell),np.cos(phi_ell)]])
    x,y = np.dot(np.array([x_ell,y_ell]).T,rot_m).T
    x =  np.cos(phi_ell) * x_ell - np.sin(phi_ell) * y_ell
    y =  np.sin(phi_ell) * x_ell + np.cos(phi_ell) * y_ell

    RAs_sie      = x + cntx
    DECs_sie     = y + cnty
    
    samples_arcs = np.array([RAs_sie,DECs_sie])*u.arcsec

    arcXkpc      = cosmo.arcsec_per_kpc_proper(z_lens) # ''/kpc
    samples      = samples_arcs/arcXkpc # kpc

    # theta_E must be connected to the mass scale, as it only scales up and down the kappa
    kwargs_lens_SIE  = [{'theta_E' : to_dimless(theta_E), 
                         'e1': e1,
                         'e2':e2,
                        'center_x': cntx, 
                        'center_y': cnty}] 

    # note the discussion on Notion: SIS sampling:easy!
    Sigma_Crit_arcs2 = SigCrArc2(cosmo=cosmo,z_lens=z_lens,z_source=z_source)

    Mmax       = Sigma_Crit_arcs2 *np.pi*ensure_unit(theta_E,u.arcsec)*ensure_unit(theta_max,u.arcsec)
    Mscale_sun = Mmax.to("Msun")/n_smpl
    # mass has to be weighted by axis ratio as well
    Mscale_sun /= q_ell
    Mscale_sun = ensure_unit(Mscale_sun,u.Msun)
    
    return kwargs_lens_SIE,Mscale_sun,samples

# From here on - adapted to nazgul pipeline
from pathlib import Path

from nazgul.pathfinder import get_gal_dir
from nazgul.Translator.particle_galaxy import BasicPartGal,store_class,clip_coord
from nazgul.Translator.ANL_TEST import simsuite_name,sim,part_type_list, check_part_type
from nazgul.Translator.ANL_TEST.pathfinder import get_galname

def get_kw_SimPartGal(kw_Gal,sim,simsuite,subsim,data_dir,z,snap,M,Centre,reload):
    # This is almost a fake function - we'll get all info from kw_Gal
    assert simsuite==simsuite_name
    return kw_Gal|{"sim":sim}

def get_z_snap(z,snap=None):
    snap = 0
    return z,snap

def Gal2MXYZ(Gal):
    samples = Gal.samples
    # Note: this is by def. in 2D - will add a fake dimension
    Xs,Ys = samples
    Zs = np.zeros_like(Xs)
    ms = Gal.Mscale_sun
    Ms = np.ones_like(to_dimless(Xs))*ms
    return Ms,Xs,Ys,Zs 

def gal_path2kwGal(gal_pkl_path):
    gal_pkl_path = Path(gal_pkl_path)
    galname =  gal_pkl_path.name
    theta_E,n_smpl = galname.split("tE")[1].split("_nS")
    theta_E,n_smpl = float(theta_E),int(n_smpl.split(".")[0])
    kw_Gal = {"theta_E":theta_E,"n_smpl":n_smpl}
    return kw_Gal

def get_rnd_SPG(**kw_galpart):
    raise NotImplementedError()
def get_all_SPG(**kw_galpart):
    raise NotImplementedError()


class SimPartGal(BasicPartGal):
    _large_attributes_setup  = ["samples"]
    _large_attributes_unpack = []
    # Fix random seed to obtain always the same sample
    np.random.seed(0)

    # fix all cosmo params
    cosmo    = default_cosmo
    z_source = z_source
    snap     = 0
    
    def __init__(self,sim=sim[0],n_smpl=n_smpl,theta_E=theta_E,z_lens=z_lens,e1=None,e2=None):
        #cosmo=default_cosmo,z_lens=z_lens,z_source=z_source
        self.n_smpl  = int(n_smpl)
        self.theta_E = theta_E
        self.profile = sim
        self.sim     = sim
        if self.profile=="SIE":
            self.e1  = e1
            self.e2  = e2
        self.gal_dir = get_gal_dir(kw_gal=self.kwargs_gal,
                                   snap=self.snap,
                                   sim=self.sim,
                                   subsim=None,
                                   simsuite=simsuite_name)
        mkdir(self.gal_dir)
        self.z_lens  = z_lens
        self.z       = z_lens

        self.N_part = n_smpl
        
        self.simsuite = simsuite_name
        self._type_id = "SimPartGal_"+simsuite_name+"_"+self.profile

    @property
    def name(self):
        return get_galname(self.kwargs_gal)

    @property
    def kwargs_gal(self):
        kwargs_gal = {"theta_E":self.theta_E,
                            "n_smpl":self.n_smpl,
                            "profile":self.profile}
        if self.profile=="SIE":
            kwargs_gal["e1"] =self.e1
            kwargs_gal["e2"] =self.e2
        return kwargs_gal
    def _identity(self):
        Id = (self._type_id,self.sim,self.profile,self.name)
        return Id
        
    def __str__(self):
        str_gal = "Simulated analytical galaxy with SIS profile "
        str_gal += f"with theta_E={short_SciNot(self.theta_E)}"
        str_gal += f", n_smpl={short_SciNot(self.n_smpl)}"
        str_gal += f", z_lens={short_SciNot(self.z_lens)}"
        str_gal += f", z_source={short_SciNot(self.z_source)}"
        str_gal += f", cosmo={self.cosmo.name}"
        return str_gal

    def run(self,reload=False,verbose=True):
        if self.profile =="SIS":
            self.kwargs_lens,self.Mscale_sun,self.samples = sample_SIS(n_smpl=self.n_smpl,
                                                                  theta_E = self.theta_E,
                                                                  z_lens = self.z_lens,
                                                                  z_source = self.z_source)
        elif self.profile=="SIE":
            self.kwargs_lens,self.Mscale_sun,self.samples = sample_SIE(n_smpl=self.n_smpl,
                                                                  theta_E = self.theta_E,
                                                                  z_lens = self.z_lens,
                                                                  z_source = self.z_source,
                                                                  e1 = self.e1,
                                                                  e2 = self.e2)
        self.M_tot  = self.n_smpl*self.Mscale_sun
    def store_gal(self,update=True):
        # store class instance 
        store_class(self,path=self.dill_path_abs(),update=update)


def Gal2MXYZ_part(Gal,part_type): 
    """Given the galaxy, return Masses (in Msun) and
    XY coords. of a specific particle type in kpc  centered around center
    -> given the sim, all parts are the same: return Gal2MXYZ if 
    """
    part_type =  check_part_type(part_type)
    # There is only 1 type of particle, so:
    return Gal2MXYZ(Gal)


def standardise_snap(snap,*args,**kwargs):
    # Snap should not be a factor in this Simulation
    return snap 