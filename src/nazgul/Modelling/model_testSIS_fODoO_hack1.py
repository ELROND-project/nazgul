"""
Simulate an image with SIS and try to model it - leave gamma OD and omega LOS free

-> assume optimistic HST observations 

-> hack1: this uses a "hack galaxy", ie one where the alpha map has been changed with the exact SIS alpha map
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from python_tools.tools import mkdir

from nazgul.MountDoom.lens_system import LensSystem
from nazgul.Modelling.pathfinder import model_res_base

# LOS simulated as dependent on gal name (not z)
from nazgul.MountDoom.fuel_tank import lensfuel_los_gal_optimistic_band

from nazgul.MountDoom.generate_gal_lens import GalLens
from nazgul.MountDoom.cracks_of_doom import kwargs_source_default

mask_center = True
gauss_tE_prior = True
overwrite_results = True
check_if_workin_on_it = False

res_dir_base      = model_res_base/"testSIS_fODoO_hack1/"
mkdir(res_dir_base)

description_model = "Try to model a single SIS w LOS, and model the lens with EPL +LOS minimal (with free gamma OD and omega LOS), and an elliptical sersic source - now with fixed lens pos - and Optimistic Observation (def. lnstr. HST band) - hacked gal: use analytical SIS alpha map instead of one from particles"

from python_tools.tools import mkdir,to_dimless
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.util import array2image,image2array,make_grid

from nazgul.Modelling.model_testSIS import n_smpl,tE_in,z_lens
from nazgul.Modelling.model_testSIS_freegOD import create_SIS_gallens,kw_model

from nazgul.MountDoom.cracks_of_doom import get_RADEC
from nazgul.Modelling.model_testSIS import tE_in



n_smpl_SIS = int(1e7)
gallens = create_SIS_gallens(tE_in=tE_in,n_smpl=n_smpl_SIS,reload=True)
gallens.unpack()

cosmo = gallens.cosmo
z_lens= gallens.z_lens
z_source = gallens.z_source
pixel_num = gallens.pixel_num
deltaPix = gallens.deltaPix
radius = gallens.radius
thetaE  = gallens.thetaE
arcXkpc = gallens.arcXkpc
z_source_min = gallens.z_source_min
z_source_max = gallens.z_source_max
pixel_num_padd = gallens.pixel_num_padd
kw_extents_padd = gallens.kw_extents_padd
kw_extents = gallens.kw_extents
_radec_padd = gallens._radec_padd
pad_pix = gallens.pad_pix #int((pixel_num_padd-pixel_num)/2.)

# this is the one w/o LOS
kw_lens_true_noLOS = [{'theta_E':     tE_in, 
                         'center_x':    0.0, 
                         'center_y':    0.0}]
class Hack_Gal():
    gal_dir = Path("/cosma/home/do019/dc-quei1/nazgul/src/nazgul/RingBearer/Hack/Gal/")
    mkdir(gal_dir)
    simsuite = "ANL_TEST"
    sim = "SIS"
    snap = "0"
    name = "HackGalSIS"
    
class Hack_Gallens():
    Gal = Hack_Gal()
    def __init__(self):
        self.cosmo = cosmo
        self.deltaPix = deltaPix
        self.arcXkpc = arcXkpc
        self.pixel_num = pixel_num
        self.radius  =  radius
        self.thetaE  = thetaE
        self.name = "hack_galaxy"
        self.proj_index = 0
        self.z_lens = z_lens
        self.z_source = z_source
        self.z_source_min = z_source_min
        self.z_source_max = z_source_max
        self.kw_extents = kw_extents
        
        self.pixel_num_padd = pixel_num_padd
        self.kw_extents_padd = kw_extents_padd

        self._kw_lens_true = kw_lens_true_noLOS
        self._lenslist = ["SIS"]
        self._lensModel = LensModel(self._lenslist)

        self.pad_pix = pad_pix
        self.alpha_map_padd = self._alpha_map(self._radec_padd)
        self.kappa_map_padd = self._kappa_map(self._radec_padd)
        self.psi_map_padd = self._psi_map(self._radec_padd)
        self.hessian_map_padd = self._hessian(self._radec_padd)

        self.savedir = self.Gal.gal_dir.parent/"Sub"
        mkdir(self.savedir)
        self.pkl_path = self.savedir/"Sub_HackGal.pkl"
    @property
    def _radec(self):
        ra, dec = self._radec_padd
        crop = lambda a: image2array(self._crop_padd(array2image(a)))
        return crop(ra), crop(dec)
        
    @property
    def _radec_padd(self):
        return make_grid(self.pixel_num_padd,to_dimless(self.deltaPix))

    def _identity(self):
        return ("hack")
    def setup(self):
        return None
    def _needs_setup(self):
        return False
    def unpack(self):
        return None
    def run(self):
        return None
    def _alpha_map(self,_radec):
        flat_ra, flat_dec = _radec
        _alpha_x,_alpha_y = self._lensModel.alpha(x=flat_ra,y=flat_dec,kwargs=self._kw_lens_true)
        alpha_x,alpha_y = array2image(_alpha_x),array2image(_alpha_y)
        return alpha_x,alpha_y  
        
    def _kappa_map(self,_radec):
        flat_ra, flat_dec = _radec
        _kappa = self._lensModel.kappa(x=flat_ra,y=flat_dec,kwargs=self._kw_lens_true)
        kappa = array2image(_kappa)
        return kappa
        
    def _psi_map(self,_radec):
        flat_ra, flat_dec = _radec
        _psi = self._lensModel.potential(x=flat_ra,y=flat_dec,kwargs=self._kw_lens_true)
        psi = array2image(_psi)
        return psi
    def _hessian(self,_radec):
        flat_ra, flat_dec = _radec
        _fxx,_fxy,_fyx,_fyy = self._lensModel.hessian(x=flat_ra,y=flat_dec,kwargs=self._kw_lens_true)
        fxx,fxy,fyx,fyy = array2image(_fxx),array2image(_fxy),array2image(_fyx),array2image(_fyy)
        return fxx,fxy,fyx,fyy  

    def get_RADEC(self):
        return get_RADEC(self._radec)
    def get_RADEC_padd(self):
        return get_RADEC(self._radec_padd)
        
    def _crop_padd(self, m):
        if isinstance(m, (tuple, list)):
            return type(m)(self._crop_padd(x) for x in m)
        n = self.pad_pix
        return m[n:m.shape[0] - n, n:m.shape[1] - n] # n=0 safe (m[0:-0] would be empty)
 
    @property
    def alpha_map(self):
        return self._crop_padd(self.alpha_map_padd)
    @property
    def kappa_map(self):
        return self._crop_padd(self.kappa_map_padd)
    @property
    def psi_map(self):
        return self._crop_padd(self.psi_map_padd)
    @property
    def hessian_map(self):
        return self._crop_padd(self.hessian_map_padd)

if __name__=="__main__":
    from nazgul.Modelling.lib_models import setup_lens,get_res_dir,model_lens,n_it_std,n_burn_std,n_run_std,n_part_std

    parser = argparse.ArgumentParser(prog=sys.argv[0],description=description_model)
    parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Type of run:
        0 = standard, PSO_it = {n_it_std} PSO_prt = {n_part_std} MCMCb = {n_burn_std} MCMCr = {n_run_std}  
        1 = test run  PSO_it = 3      PSO_prt = 3      MCMCb = 1     MCMCr = 2 
        2 = test run (longer)  PSO_it = 100      PSO_prt = 20      MCMCb = 100     MCMCr = 200 
        (PSO_it: PSO iterations, PSO_prt: PSO particles, MCMCr: MCMC run steps, MCMCb: MCMC burn in steps)\n""")
    
    args     = parser.parse_args()
    run_type = args.run_type
    simsuite = "ANL_TEST"
    sim      = "SIS"
    subsim   = None
    
    if run_type==0:
        n_it   = n_it_std
        n_part = n_part_std
        n_burn = n_burn_std
        n_run  = n_run_std
        n_smpl_SIS = n_smpl
    elif run_type == 1:
        n_it   = 3
        n_part = 3
        n_burn = 1
        n_run  = 2
        n_smpl_SIS = int(1e3)
        check_if_workin_on_it= False
    elif run_type==2:
        n_it   = 100
        n_part = 20
        n_burn = 100
        n_run  = 200
        check_if_workin_on_it = False
    else:
        raise ValueError(f"run_type must be 0,1 or 2, not {run_type}")
    
    res_dir = get_res_dir(res_dir_base,simsuite,sim,
                          subsim=subsim,run_type=run_type)
        
    gallens_hack = Hack_Gallens()
    lens_system_hack = LensSystem.from_GalLens(gallens_hack,lensFuel=lensfuel_los_gal_optimistic_band)
    # the SIS has no radial CC, so the sampling of the source pos fails
    # we can trick it by defining it a priori
    # can't be 0,0 exactly as it would trigger an error :
    kwargs_source_pos = {"center_x":1e-9,"center_y":1e-9}
    lens_system_hack.kwargs_source = kwargs_source_default|kwargs_source_pos
    lens_system_hack.setup(reload=False)
    
    lens = setup_lens(lens=lens_system_hack,
                      res_dir=res_dir,
                      check_if_workin_on_it=check_if_workin_on_it,
                      overwrite=overwrite_results)           
    
    model_lens(lens,kw_model,
               n_it   = n_it,
               n_part = n_part,
               n_burn = n_burn,
               n_run  = n_run,
               mask_center=mask_center,              
               gauss_tE_prior=gauss_tE_prior)
