import warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom

from lenstronomy.Util import util
from nazgul.pathfinder import path_nazgul
from python_tools.tools import to_dimless

from python_tools.read_fits import load_fits


######################################
# kwargs_of realistic HST observations used to simulate the "observed" images 
kwargs_band_HST_camera = {
    'read_noise': 2,                      # Readout noise
    'pixel_scale':0.065,                  # 0.065 F160W after drizzling (could also do 0.08 to be more conservative
    'ccd_gain': 2.35,                     # averaged over the 4 amplifier (does not matter)
}
# inspired by F160W taken from idgc07c[nlpq]q_flt.fits 
sky_count      = 0.11 # after drizzling, clip outliers and take median  (e-/sec)
exp_time_1exp  = 550 # ~average over 4 exposures
num_exposures  = 4   #  
# taken from https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
# the following ZP computation is also correct, returns 25.937 and the error is 0.008 so it's consistent
# PHOTFLAM is the inverse sensitivity at the infinite aperture, taken from
#PHOTFLAM_f160w = 1.9429e-20 
#PHOTPLAM_f160w = 15369.18
#ZP_AB_f160w = -2.5*np.log10(PHOTFLAM_f160w) - 21.1 - 5*np.log10(PHOTPLAM_f160w) + 18.6921
ZP_AB_f160w    = 25.941 

sky_brightness = -np.log10(sky_count) * 2.5 + ZP_AB_f160w
kwargs_band_HST_obs = {
    'sky_brightness':sky_brightness,      # ~21.5 mag
    'exposure_time':exp_time_1exp,        # average time for 1 exposure
    'magnitude_zero_point':ZP_AB_f160w,   # ~25.9 mag
    'num_exposures': num_exposures,       # stnd n* of exposures combined in drizzing
    'psf_type':'PIXEL'                    # kernel to be provided later on
}

class BasicBand():
    """
    Inspired by class HST in lenstronomy.SimulationAPI.ObservationConfig.py 
    """
    def __init__(self,
                 kwargs_camera ,
                 kwargs_obs,
                 kwargs_psf=None):
        self.camera = kwargs_camera
        self.obs = kwargs_obs
        # obtained from https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/psf
        self.kwargs_psf = kwargs_psf
    def kwargs_single_band(self):
        """
        :return: merged kwargs from camera and obs dicts
        # add kwargs_psf if present (if not, PSF=None)
        """
        kwargs = util.merge_dicts(self.camera, self.obs)
        
        if self.kwargs_psf is not None:
            kwargs.update(self.kwargs_psf)
        return kwargs
    @property
    def _psf_id(self):
        if self.kwargs_psf is None:
            psf_id = None
        elif self.kwargs_psf["psf_type"]=="PIXEL":
            if hasattr(self,"psf_path"):
                psf_id = {"psf_path": self.psf_path}
            else:
                raise NotImplementedError("Pixel-type PSF cannot be simply identified") 
        else:
            # this should be parametric and not too long
            psf_id = self.kwargs_psf
        return {"psf_id":psf_id}
        
    def _identity(self):
        return (self.camera,
                self.obs,
               self._psf_id)
    

class Band_HST_F160W(BasicBand):
    """
    Inspired by class HST in lenstronomy.SimulationAPI.ObservationConfig.py 
    """
    def __init__(self,
                 kwargs_camera = kwargs_band_HST_camera,
                 kwargs_obs    = kwargs_band_HST_obs,
                pssf_effective= 5):
        self.camera = kwargs_camera
        self.obs = kwargs_obs
        # obtained from https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/psf
        self.psf_path =  Path(f"{path_nazgul}/ObsData/HST/WFC3/F160W/PSFSTD_WFC3IR_F160W.fits")
        self.pssf_effective = pssf_effective
        self.kwargs_psf = self._get_kwargs_psf(pssf_effective=self.pssf_effective)
        
    def _get_kwargs_psf(self,pssf_effective=5):
        if np.abs(int(pssf_effective)-pssf_effective)>1e-7:
            raise RuntimeError("We should have an integer pssf_effective") 
        pssf_effective = int(pssf_effective)

        psf_path = self.psf_path
        
        delta_pix_native = 0.128          # arcsec/pix, native F160W
        pssf_orig        = 4              # STScI PSF supersampling vs native
        delta_pix_psf    = delta_pix_native / pssf_orig   # = 0.032 arcsec/pix

        delta_pix_band   = self.camera["pixel_scale"]     # = 0.08 arcsec/pix (lenstronomy target)
        pssf_band        = delta_pix_native / delta_pix_band # = 1.6
        # pssf is the ratio of image pixel scale to PSF pixel scale,
        # as lenstronomy expects. The PSF must be zoomed to achieve this.
        # Current PSF pixel scale: delta_pix_psf = 0.032 "/pix
        # Target PSF pixel scale for given pssf: delta_pix_band / pssf
        zoom_factor = pssf_effective*pssf_band/pssf_orig        
    
        psf = load_fits(psf_path)[-2]
        psf = _positivise_psf(psf)
        
        if zoom_factor<1:
            warnings.warn("PSSF should be set s.t. zoom_factor>1")
        
        if not np.isclose(zoom_factor, 1):
            psf = zoom(psf, zoom_factor, order=3)
            psf = _positivise_psf(psf)
            
        kwargs_psf = {
            "psf_type":"PIXEL",
            "kernel_point_source_normalisation":True,
            "kernel_point_source": psf,
            "point_source_supersampling_factor": pssf_effective
        }
        return kwargs_psf


def _positivise_psf(psf,verbose=False):
    if np.any(psf<0):
        if verbose:
            warnings.warn("Some negative pixels in the PSF")
        i_psf0,j_psf0 = np.where(psf<0)
        if i_psf0.shape[0]*100/psf.ravel().shape[0]>30:
            raise ValueError("PSF has more than 30% negative pixels, something is not right")
        if verbose:
            warnings.warn("Setting minimum value for negative PSF pixels")
        psf[psf<0] = np.min(psf[psf>0])/100
    # renormalise it aftwards
    psf /= psf.sum()
    return psf


# Std band:
std_band_obs = Band_HST_F160W()
