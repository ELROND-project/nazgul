import warnings
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from scipy.ndimage import map_coordinates
from scipy.interpolate import CubicSpline

from lenstronomy.Util import util

from nazgul.pathfinder import path_nazgul
from python_tools.tools import to_dimless

from python_tools.read_fits import load_fits

######################################
# kwargs_of realistic HST observations used to simulate the "observed" images 
kwargs_band_HST_camera = {
    'read_noise': 2,                      # Readout noise
    'pixel_scale':0.065,                  # 0.065 F160W after drizzling (TDCOSMOXXV: could also do 0.08 to be more conservative
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
sky_flux       = sky_count/(kwargs_band_HST_camera["pixel_scale"]**2) # e-/sec/arcsec^2 
sky_brightness = -np.log10(sky_flux) * 2.5 + ZP_AB_f160w
kwargs_band_HST_obs = {
    'psf_type':'PIXEL',                   # kernel to be provided later on
    'exposure_time':exp_time_1exp,        # average time for 1 exposure
    'num_exposures': num_exposures,       # stnd n* of exposures combined in drizzing
    'magnitude_zero_point':ZP_AB_f160w,   # ~25.9 mag
    'sky_brightness':np.round(sky_brightness,8) # ~22.4 mag - rounding to avoid floating error to affect naming
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
                psf_id = {"psf_path": self.psf_path,
                          "pssf_effective":self.pssf_effective}
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
        pssf_effective   = int(pssf_effective)
        self.kwargs_psf = self._get_kwargs_psf(pssf_effective=pssf_effective)
        self.pssf_effective = self.kwargs_psf["point_source_supersampling_factor"]
        
    def _get_kwargs_psf(self,pssf_effective=5):
        if np.abs(int(pssf_effective)-pssf_effective)>1e-7:
            raise RuntimeError("We should have an integer pssf_effective") 
        pssf_effective = int(pssf_effective)

        psf_path = self.psf_path
        
        delta_pix_native = 0.128          # arcsec/pix, native F160W
        pssf_orig        = 4              # STScI PSF supersampling vs native
        delta_pix_psf    = delta_pix_native / pssf_orig   # = 0.032 arcsec/pix

        delta_pix_band   = self.camera["pixel_scale"]     # = 0.065 arcsec/pix (lenstronomy target)
        pssf_band        = delta_pix_native / delta_pix_band # = 1.97
        # pssf is the ratio of image pixel scale to PSF pixel scale,
        # as lenstronomy expects. The PSF must be zoomed to achieve this.
        # Current PSF pixel scale: delta_pix_psf = 0.032 "/pix
        # Target PSF pixel scale for given pssf: delta_pix_band / pssf
    
        psf = load_fits(psf_path)[-2]
        """psf_res,info_psf_res = resample_psf_odd(psf,
                                                pix_in=delta_pix_psf,
                                                pix_band=delta_pix_band,
                                                pssf=pssf_effective)
        """
        psf_res = resample_psf_odd(psf,
                                   pssf_band=pssf_band,
                                   pssf_orig=pssf_orig,
                                   pssf=pssf_effective)
        kwargs_psf = {
            "psf_type":"PIXEL",
            "kernel_point_source": psf_res,
            "kernel_point_source_normalisation":True,
            "point_source_supersampling_factor": pssf_effective
        }
        return kwargs_psf


def resample_psf_odd(psf,pssf,pssf_band,pssf_orig=4):
    psf = np.asarray(psf, dtype=float)
    psf = _positivise_psf(psf)
    zoom_factor = pssf*pssf_band/pssf_orig        
    # zoom factor need to output odd psf
    assert int(np.round(zoom_factor*psf.shape[0],0))%2==1

    if zoom_factor<1:
        warnings.warn("PSSF should be set s.t. zoom_factor>1")
        
    if not np.isclose(zoom_factor, 1):
        psf = zoom(psf, zoom_factor, order=3)
        psf = _positivise_psf(psf)
    
    return psf

# resampling function implemented by Claude, avoding issues with the odd PSF size
def resample_psf_odd_discarded(psf, pix_in, pix_band, pssf, order=3):
    """
    Resample a PSF so that its pixel scale is exactly pix_band / pssf,
    with an odd, square output shape and the central pixel preserved.

    Unlike scipy.ndimage.zoom (which rounds the output shape and then uses the
    effective factor (n_out-1)/(n_in-1)), the sampling step is imposed directly,
    and the output size is chosen afterwards.

    :param psf: 2D square array with odd size, peak at the central pixel
    :param pix_in: pixel scale of the input PSF [arcsec/pix]  (e.g. 0.128/4)
    :param pix_band: pixel scale of the image [arcsec/pix]     (e.g. 0.065)
    :param pssf: integer supersampling factor wanted by lenstronomy
    :param order: spline order of the interpolation
    :return: psf_out (odd shape, sum=1), info dict
    """
    raise RuntimeError("Discarded because the final shape needs to be exactly proportional to pssf")
    psf = np.asarray(psf, dtype=float)
    # ensure that the input does not have too many negative pixels
    psf = _positivise_psf(psf)

    if psf.ndim != 2 or psf.shape[0] != psf.shape[1]:
        raise ValueError(f"PSF must be square 2D, got shape {psf.shape}")
    n_in = psf.shape[0]
    if n_in % 2 == 0:
        raise ValueError(f"Input PSF has even size {n_in}: its centre is not a pixel centre")
    if pssf < 1 or abs(int(pssf) - pssf) > 1e-7:
        raise ValueError(f"pssf must be a positive integer, got {pssf}")
    pssf = int(pssf)

    pix_out = pix_band / pssf
    z = pix_in / pix_out                      # output pixels per input pixel
    if z < 1:
        warnings.warn(f"Resampling factor {z:.3f}<1: the PSF is being downsampled")

    # largest odd n_out such that every output pixel centre falls inside the input grid
    n_out = 2 * int(np.floor((n_in - 1) * z / 2)) + 1
    c_in, c_out = (n_in - 1) / 2, (n_out - 1) / 2

    # input-pixel coordinate of each output pixel: central pixel -> central pixel
    x_in = c_in + (np.arange(n_out) - c_out) / z
    yy, xx = np.meshgrid(x_in, x_in, indexing="ij")
    out = map_coordinates(psf, [yy, xx], order=order, mode="nearest")

    # negatives from spline ringing, then renormalise (non-mutating)
    out = _positivise_psf(out)

    # what was actually done, measured from the sampling grid itself
    step_in = x_in[1] - x_in[0]               # input pixels per output pixel
    pix_out_real = pix_in * step_in
    # verify that pssf_effective is int:
    pssf_effective = pix_band / pix_out_real
    np.testing.assert_almost_equal(pssf_effective,int(pssf_effective),3)
    # verify that it is indeed = pssf
    if not np.isclose(pssf_effective, pssf, rtol=1e-10):
        raise RuntimeError(f"pssf mismatch: {pssf_effective} vs {pssf}")
    pssf_effective = int(pssf_effective)
        
    info = {
        "shape": out.shape,
        "zoom_requested": z,
        "pix_out": np.round(pix_out_real,8),# avoid rounding error
        "pssf_effective": pssf_effective, 
        "coverage_arcsec": n_out * pix_out_real,
    }
    return out, info


def _positivise_psf(psf, max_neg_frac=0.30):
    psf = psf.copy()
    neg = psf < 0
    if neg.any():
        if neg.sum() / psf.size > max_neg_frac:
            raise ValueError("PSF has more than 30% negative pixels, something is not right")
        pos = psf[psf > 0]
        psf[neg] = pos.min() / 100 if pos.size else 0.0
    # renormalise it aftwards
    psf /= psf.sum()
    return psf

class Band_HST_noPSF(BasicBand):
    """
    Inspired by class HST in lenstronomy.SimulationAPI.ObservationConfig.py 
    Same as Band_HST_F160W but with a point-like PSF
    """
    def __init__(self,
                 kwargs_camera = kwargs_band_HST_camera,
                 kwargs_obs    = kwargs_band_HST_obs,
                pssf_effective= 5):
        self.camera = kwargs_camera
        self.obs = kwargs_obs
        self.kwargs_psf = {"psf_type":"NONE"}


# Std band:
std_band_obs = Band_HST_F160W()
