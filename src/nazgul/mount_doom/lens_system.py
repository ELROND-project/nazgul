"""
Restructured from generate_particle_lens_dom with a different philosofy in mind:
This class should be as simple as possible, basically only a conveniency wrapper for LensModel, 
but geared toward the galaxy lens
"""
import dill
import warnings
import numpy as np
from pathlib import Path
# for debug plots
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage import zoom
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline,griddata

import astropy.units as u
from astropy.stats import sigma_clip

from lenstronomy.Util import util
from lenstronomy.Util.util import array2image,image2array 
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame

# My libs
from python_tools.tools import to_dimless,mkdir,to_uid,silencer
import nazgul.mount_doom.cracks_of_doom as cod
from nazgul.project_gal import get_2Dkappa_map
# Class structure
from nazgul.basic_gal import BasicGal
from nazgul.mount_doom.generate_gal_lens import GalLens
from nazgul.mount_doom.cracks_of_doom import store_lens,_resolve_gal_path,LoadLens

# Default values
# particle lens class and params.
from nazgul.mount_doom.cracks_of_doom import source_model_list
from nazgul.configurations import pixel_num,min_thetaE
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
from nazgul.mount_doom.cracks_of_doom import kwargs_band_sim,kw_prior_z_source_stnd
from nazgul.mount_doom.cracks_of_doom import kwargs_source_default,get_kwargs_sourceSim

verbose = True
empty_kwargs_add_lenses = {"lens_model_list":[],"kwargs_lens":[]}


class LensSystem(BasicGal):
    # Large attributes to not store and to recompute/reload BEFORE computation
    _large_attributes_setup  = []
    # Large attributes to not store and to recompute/reload AFTER computation
    _large_attributes_unpack = ["cosmo","lens_model"]
    
    def __init__(self,
                 kwargs_gallens,  #                 
                    #Galaxy,
                    #projection_index, # projection index of the galaxy 
                    #kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                    #pixel_num=pixel_num, # sim prms 
                    #kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                    #min_thetaE = min_thetaE,
                 kwargs_band_sim=kwargs_band_sim,
                 kwargs_add_lenses=empty_kwargs_add_lenses,
                 kwargs_lensmodel={}, 
                 ):
        """
        :param
            kwargs_gallens: Defining the Galaxy lens parameters:
                Galaxy :           PartGal instance
                kwlens_part:       kw_args of the PartLens class
                pixel_num:         int, number of pixel
                kw_prior_z_source: kwargs, define the prior distribution of the z_source to be sampled from
                min_thetaE:        float (ideally with units of arcsec), define the minumum thetaE for which the galaxy is considered a lens
            kwargs_band_sim: bandmodel used to define the Sim class
            kwargs_add_lenses: kwargs for additional lenses
            kwargs_lensmodel: additional kwargs to pass to LensModel (e.g. z_lens, as != z_galaxy)
        """ 

        # initialise it independently
        self.gallens            = GalLens(**kwargs_gallens)
        self.gallens.unpack()
        # LensModel
        self.kwargs_lensmodel    = kwargs_lensmodel
        self.kwargs_add_lenses   = kwargs_add_lenses
        # observational params
        self.kwargs_band_sim     = kwargs_band_sim
        # We assume that the cosmology has to be the same as the galaxy:
        self.cosmo               = self.gallens.cosmo
        self.kwargs_source_def   = kwargs_source_default
        mkdir(self.savedir)


    def _identity(self):
        return (
            self.gallens._identity(),
            self.kwargs_lensmodel,
            self.kwargs_add_lenses,
            self.kwargs_band_sim)
        
    def setup(self,
                Sim=None,
                update_source_pos=True,
                reload=True,
                rnd_seed=None, # to set differently to sample new source pos
                verbose=verbose):
        upload_successful = False
        if reload:
            upload_successful = self.upload_prev()
        if not upload_successful:
            self.gallens.unpack()
            self.gallens.run()
            self.create_lens(kwargs_add_lenses=self.kwargs_add_lenses, # these are given here as the self, but in principle with the freedom to re-define them
                             Sim=Sim,verbose=verbose)
            if update_source_pos:
                self.sample_source_pos(update=update_source_pos,rnd_seed=rnd_seed)
            elif self.kwargs_source_def["center_x"]==0 and self.kwargs_source_def["center_y"]==0:
                warnings.warn("Source is still positioned at 0,0 but I was instructed not to resample its position.")
            # store the results
            self.store()
    #############################
    @property
    def savedir(self):
        return Path(self.gallens.Gal.gal_dir).parent/"LensSystem"
    @property
    def pkl_path(self):
        return _resolve_gal_path(self.savedir)/f"{self.name}_{self._hash_b64}.pkl"
    @property
    def _rnd_seed(self):
        """
        Define a random seed unique for this specific lens. 
        Used for reproducibility  
        """
        _str = self.pkl_path
        # have to cut it if not it's too long
        seed = int(str(to_uid(_str))[:9])
        return seed

    def ReadClass(self,cl,verbose=True):
        LS = LoadLens(cl.pkl_path,verbose=verbose)
        return LS
    
    @silencer
    def _unpack(self,verbose=True):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        print("Unpacking system lens...")
        self.gallens.unpack()
        self.cosmo = self.gallens.cosmo

    def __str__(self):
        """Human-readable identifier.

        Lazily initializes the name if it has not been generated yet.
        """
        return f'Name:{self.name}\nHash:{self._hash_b64}'

    @property
    def name(self):
        # define name and path of savefile
        name_lnsp = self.gallens.name
        # to correct asap
        name= name_lnsp.replace("Sub_","LS_")
        return name
        
    @classmethod
    def from_GalLens(cls, GalLens,**kwargs_lenssystem):
        """Construct from an existing GalLens instance."""
        obj = cls.__new__(cls)
        obj.gallens = GalLens
        obj.gallens.unpack()
        # We assume the cosmology has to be the same as the Galaxy:
        obj.cosmo               = obj.gallens.cosmo        
        mkdir(obj.savedir)

        # LensModel
        obj.kwargs_lensmodel    = kwargs_lenssystem.get("kwargs_lensmodel",{})
        obj.kwargs_add_lenses   = kwargs_lenssystem.get("kwargs_add_lenses",empty_kwargs_add_lenses)
        # observational params
        obj.kwargs_band_sim     = kwargs_lenssystem.get("kwargs_band_sim",kwargs_band_sim)
        obj.kwargs_source_def   = kwargs_lenssystem.get("kwargs_source",kwargs_source_default)
        return obj
        
    #############################

    ###############
    ###############
    

    def create_lens(self,kwargs_add_lenses=empty_kwargs_add_lenses,
                    Sim=None,kwargs_source=None,verbose=verbose):
        # Define the radius based on ~ theta_E
        self.radius    = self.gallens.radius
        if verbose:
            print("Image radius:",np.round(self.radius,3))        
        # setup dataclasses (dataclass,psf_class,sourcemodel and some helper kwargs):
        self.setup_dataclasses(Sim=Sim, verbose=verbose)
        # setup lenses 
        self.setup_lenses(kwargs_add_lenses=kwargs_add_lenses,
                        verbose=verbose)        
    def store(self):
        store_lens(self)
        
    def setup_dataclasses(self,Sim=None,verbose=True):
        """
        Define all classes not dependent on lensing
        Handled by SimAPI
        """
        if Sim is None:
            Sim = self.get_Sim()
        if verbose:
            print("Setting up data classes ...")
        self.data_class,self.psf_class,self.source_model_class,self.kwargs_numerics = cod.get_dataclasses(Sim)
        if verbose:
            print("... Data classes set up")
        return 0

    def _reformat_kwargs_lensmodel(self):
        # ensure that kwargs_lensmodel has at least 
        # the default z_lens and z_source (if not input differently)
        default_kw_lm = {"z_lens":self.gallens.z_lens,
                         "z_source":self.gallens.z_source}
        
        kwargs_lensmodel = self.kwargs_lensmodel
        if kwargs_lensmodel is None or kwargs_lensmodel=={}:
            # for consistency, we set z_lens==z_gal and z_source == z_source_sampled
            kwargs_lensmodel     = default_kw_lm
        
        # update default_kw_lm with input kwargs_lensmodel if present
        default_kw_lm |= kwargs_lensmodel
        self.kwargs_lensmodel = default_kw_lm
        
    def setup_lenses(self,kwargs_add_lenses=empty_kwargs_add_lenses,verbose=verbose):
        """
        Setup lensing parameters tailored for lenstronomy
        """
        if hasattr(self,"lens_model") and hasattr(self,"kwargs_lens"):
            if verbose:
                print("Lens model already setup")
            return 0
        if verbose:
            print("Setting up lensing parameters...")
        add_lens_model_list    = kwargs_add_lenses["lens_model_list"]
        add_kwargs_lens        = kwargs_add_lenses["kwargs_lens"]
        lens_model_list        = ["PART_GAL",*add_lens_model_list]
        self.kwargs_lens       = [{},*add_kwargs_lens]
        pkwl_part_lens         = {"lenspart":self.gallens}
        self._reformat_kwargs_lensmodel()
        # the following in principle it is not necessary anymore...?
        if getattr(self.kwargs_lensmodel,"z_lens",self.gallens.z_lens)!=self.gallens.z_lens:
            pkwl_part_lens["z_lens"] = self.kwargs_lensmodel["z_lens"]
        if getattr(self.kwargs_lensmodel,"z_source",self.gallens.z_source)!=self.gallens.z_source:
            pkwl_part_lens["z_source"] = self.kwargs_lensmodel["z_source"]
        profile_kwargs_list    = [pkwl_part_lens]
        for adlml in add_lens_model_list:
            profile_kwargs_list.append({})
        self.lens_model = LensModel(lens_model_list=lens_model_list,
                                    profile_kwargs_list = profile_kwargs_list,
                                    **self.kwargs_lensmodel)
        if verbose:
            print("... Lensing parameters set up")
    
    ########################
    # Lensing Computations #
    ########################
    
    def alpha_map(self,_radec=None):
        if self.kwargs_add_lenses == empty_kwargs_add_lenses:
            return self.gallens._alpha_map(_radec=_radec)
        if _radec is None:
            _radec = self.gallens._radec
        _ra,_dec = _radec
        alpha_x,alpha_y = self.lens_model.alpha(_ra, _dec, self.kwargs_lens)
        alpha_x,alpha_y = array2image(alpha_x),array2image(alpha_y)
        return alpha_x,alpha_y
        
    def psi_map(self,_radec=None):
        print("Computing lensing PM potential...")
        if self.kwargs_add_lenses == empty_kwargs_add_lenses:
            return self.gallens._psi_map(_radec=_radec)
        if _radec is None:
            _radec = self.gallens._radec #arcsecs  
        _ra,_dec = _radec
        psi = self.lens_model.potential(_ra, _dec, self.kwargs_lens)
        psi = array2image(psi)
        return psi
        
    def _kappa_map_from_lens(self,_radec=None,exact=False):
        # compute analytically from the particles -> actually should not be the way to do it !
        print("Computing kappa map from PM...")
        # the following should prob. be in the unpack function
        self.gallens.setup_lenses()
        if _radec is None:
            _radec = self.gallens._radec
        _ra,_dec = _radec
        kappa = self.lens_model.kappa(_ra, _dec, self.kwargs_lens)
        kappa = array2image(kappa)
        return kappa
        
    def kappa_map(self,_radec=None):
        # compute from density map
        # actually better bc does not depend on the particle profile
        print("Computing kappa map from density map...")
        kappa = self.gallens._kappa_map(_radec=_radec)
        if self.kwargs_add_lenses["lens_model_list"]!=[]:
            # Add kappa of the additional profiles
            if _radec is None:
                _radec = self.gallens._radec
            _ra,_dec   = _radec
            lens_model_only_add = LensModel(lens_model_list=self.kwargs_add_lenses["lens_model_list"],
                                           **self.kwargs_lensmodel)
            kappa_add  = lens_model_only_add.kappa(_ra, _dec, self.kwargs_add_lenses["kwargs_lens"])
            kappa     += array2image(kappa_add)
        return kappa
            
    # Shear components and caustics/CL
    def hessian(self,_radec=None):
        """Computes the hessian matrix on the grid by taking the gradient 
        of the alpha map
        """
        print("Computing hessian as gradient of the deflection map...")
        # Can be now computed also beyond the cutout
        if self.kwargs_add_lenses == empty_kwargs_add_lenses:
            return self.gallens._hessian(_radec=_radec)
        if _radec is None:
            _radec          = self.gallens._radec
        alpha_x,alpha_y = self.alpha_map(_radec=_radec)
        _ra,_dec = _radec
        RA0,DEC0 = array2image(_ra)[0],array2image(_dec)[:,0]
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, RA0,DEC0)
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, RA0,DEC0)
        f_xx,f_xy,f_yx,f_yy  = dalpha_x_dx,dalpha_x_dy,dalpha_y_dx,dalpha_y_dy
        return f_xx,f_xy,f_yx,f_yy

    def get_kw_shear(self,_radec=None):
        """From the hessian matrix compute the shear components
        """
        f_xx,f_xy,f_yx,f_yy = self.hessian(_radec=_radec)
        # derived shear1,shear2 and shear
        shear1 = 1./2 * (f_xx - f_yy)
        shear2 = f_xy
        shear  = np.hypot(shear1,shear2)
        kw_shear = {"shear1":shear1,"shear2":shear2,"shear":shear}
        return kw_shear
        
    def shear_map(self,_radec=None):
        return self.get_kw_shear(_radec=_radec)["shear"]

    # Caustics
    ##########
    
    def get_kw_critical_curve_caustics(self,
                                       _radec=None,
                                      reload=True):
        """ Fit the critical curve and map it to the caustic
        """
        if hasattr(self,"kw_crit"):
            print("Reloading critical lines")
            return self.kw_crit
        # note: alpha is computed from the particle (and shear from alpha)
        # thus depends on the particle lens model chosen, while kappa
        # is obtained directly as a density map + cosmological scaling
        print("Computing critical lines...")        
        alpha_x,alpha_y = self.alpha_map(_radec=_radec)
        kappa           = self.kappa_map(_radec=_radec)
        shear           = self.shear_map(_radec=_radec)

        # in principle, the _radec would give the number of pixel
        if _radec:
            pixel_num = int(np.sqrt(_radec[0].shape))
        else:
            pixel_num = self.gallens.pixel_num
        # coords
        if _radec is None:
            _radec          = self.gallens._radec
        RA,DEC      = cod.get_RADEC(_radec)
        ra0,dec0    = RA[0],DEC.T[0]

        eigen_rad_orig = 1 - kappa + shear
        eigen_tan_orig = 1 - kappa - shear

        if not np.any(eigen_rad_orig<0):
            raise RuntimeError("No negative radial eigenvalue - something is off")
        if not np.any(eigen_tan_orig<0):
            raise RuntimeError("No negative radial eigenvalue - something is off")
        # it's not enough that there is 1 point<0, there have to be at least a few
        min_points = 10
        # clipping outliers and smoothing with a guassian filter (wrapper try to optimise the smoothing)
        eigen_rad,is_rad_smooth = wrapper_clip_smooth_map(eigen_rad_orig,name="Eigen radial",min_points=min_points)
        eigen_tan,is_tan_smooth = wrapper_clip_smooth_map(eigen_tan_orig,name="Eigen tangetial",
                                                          min_points=min_points)

        iery,ierx = get0_index(eigen_rad,pixel_num)
        iety,ietx = get0_index(eigen_rad,pixel_num)
        
        if len(iery)<min_points:            
            ####DEBUG####
            nm = "tmp/eigen_rad.dll"
            with open(nm,"wb") as f:
                dill.dump([eigen_rad_orig,eigen_rad,is_rad_smooth],
                          f)
            print(f"Check {nm}")
            ###############
            dbg_plot = "tmp/eigen_rad.png"
            fig,axis = plt.subplots(1,2)
            ext = self.gallens.kw_extents["extent_arcsec"]
            im0 = axis[0].imshow(np.abs(eigen_rad_orig),
                                 extent=ext,origin="lower")
            axis[0].set_title("|Eigen rad|")
            divider = make_axes_locatable(axis[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, cax=cax, orientation='vertical')
            
            im0 = axis[1].imshow(np.abs(eigen_rad),
                                 extent=ext,origin="lower")
            str_smooth=""
            if is_rad_smooth:
                str_smooth = " and smoothed"
            axis[1].set_title("|Eigen rad| (clipped"+str_smooth+")")
            divider = make_axes_locatable(axis[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, cax=cax, orientation='vertical')
            fig.tight_layout()
            fig.savefig(dbg_plot)
            print(f"Saving {dbg_plot}")
            plt.close(fig)
            raise RuntimeError(f"Radial critical curve not found.\nCheck {dbg_plot}")

        if len(iety)<min_points:
            dbg_plot = "tmp/eigen_tan.png"
            ext = self.gallens.kw_extents["extent_arcsec"]

            fig,axis = plt.subplots(1,2)
            im0 = axis[0].imshow(np.abs(eigen_tan_orig),
                                 extent=ext,origin="lower")
            axis[0].set_title("|Eigen tan|")
            divider = make_axes_locatable(axis[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, cax=cax, orientation='vertical')
            
            im0 = axis[1].imshow(np.abs(eigen_tan_smooth),
                                 extent=ext,origin="lower")
            axis[1].set_title("|Eigen tan| (clipped and smoothed)")
            divider = make_axes_locatable(axis[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, cax=cax, orientation='vertical')
            fig.tight_layout()
            fig.savefig(dbg_plot)
            print(f"Saving {dbg_plot}")
            plt.close(fig)
            raise RuntimeError(f"Tangential critical curve not found.\nCheck {dbg_plot}")

        # critical and caustics divided in tangential and radial
        cl_rad_x_noisy,cl_rad_y_noisy   = ra0[ierx],dec0[iery]    
        cl_tan_x_noisy,cl_tan_y_noisy   = ra0[ietx],dec0[iety]
        # fit them with splines  
        cl_rad_x,cl_rad_y = cod.fit_xy_spline(cl_rad_x_noisy,cl_rad_y_noisy)
        cl_tan_x,cl_tan_y = cod.fit_xy_spline(cl_tan_x_noisy,cl_tan_y_noisy)

        # fit alpha in 2D
        # TODO: verify that it is indeed dec0,ra0 and not the other way out ->correct, see TEST
        alpha_x_spline = RectBivariateSpline(dec0,ra0, alpha_x)
        alpha_y_spline = RectBivariateSpline(dec0,ra0, alpha_y)

        cc_rad_x_noisy = cl_rad_x_noisy -alpha_x_spline.ev(cl_rad_x_noisy,cl_rad_y_noisy)
        cc_rad_y_noisy = cl_rad_y_noisy -alpha_y_spline.ev(cl_rad_x_noisy,cl_rad_y_noisy)

        cc_tan_x_noisy = cl_tan_x_noisy -alpha_x_spline.ev(cl_tan_x_noisy,cl_tan_y_noisy)
        cc_tan_y_noisy = cl_tan_y_noisy -alpha_y_spline.ev(cl_tan_x_noisy,cl_tan_y_noisy)

        

        cc_rad_x = cl_rad_x-alpha_x_spline.ev(cl_rad_y,cl_rad_x)
        cc_rad_y = cl_rad_y-alpha_y_spline.ev(cl_rad_y,cl_rad_x)

        cc_tan_x  = cl_tan_x-alpha_x_spline.ev(cl_tan_y,cl_tan_x)
        cc_tan_y  = cl_tan_y-alpha_y_spline.ev(cl_tan_y,cl_tan_x)

        """
        #DEBUG plot and store
        plt.close()
        plt.scatter(cc_rad_x_noisy,cc_rad_y_noisy,label="cc rad,noisy",marker=".")
        plt.scatter(cc_rad_x,cc_rad_y,label="cc rad",marker=".")
        plt.scatter(cc_tan_x_noisy,cc_tan_y_noisy,label="cc tan,noisy",marker=".")
        plt.scatter(cc_tan_x,cc_tan_y,label="cc tan",marker=".")
    
        plt.scatter(cl_rad_x_noisy,cl_rad_y_noisy,label="cl rad,noisy",marker=".")
        plt.scatter(cl_rad_x,cl_rad_y,label="cl rad",marker=".")
        plt.scatter(cl_tan_x_noisy,cl_tan_y_noisy,label="cl tan,noisy",marker=".")
        plt.scatter(cl_tan_x,cl_tan_y,label="cl tan",marker=".")
        plt.legend()
        
        with open("tmp/del.dll","wb") as f :
            dill.dump([[[cl_rad_x,cl_rad_y],[cl_rad_x_noisy,cl_rad_y_noisy]],
                       [[cl_tan_x,cl_tan_y],[cl_tan_x_noisy,cl_tan_y_noisy]]],f)
        plt.savefig("tmp/tmp1.png")
        plt.close()
        """
        kw_crit = {"caustics":{"radial":[cc_rad_x,cc_rad_y],
                               "tangential":[cc_tan_x,cc_tan_y]},
                   "critical_lines":{"radial":[cl_rad_x,cl_rad_y],
                                     "tangential":[cl_tan_x,cl_tan_y]},
                   "rad_smooth":is_rad_smooth,
                    "tan_smooth":is_tan_smooth,
                  }
        self.kw_crit = kw_crit
        return kw_crit
    ########################
    ########################
    
    ########################
    # Creating Lensed image:
    ########################
    
    # Source
    #########
    def update_source_position(self,ra_source,dec_source):
        # useful if we want to put it in the center of the caustic
        self.kwargs_source_def["center_x"] = ra_source
        self.kwargs_source_def["center_y"] = dec_source
        return 0
        
    def sample_source_pos(self,update=False,_radec=None,rnd_seed=None,recompute=False):
        """Sample the source position within the tangential
        critical caustic
        """
        print("Sampling source position within tangential caustic")
        
        if _radec is None:
            _radec = self.gallens._radec
        if rnd_seed is None:
            rnd_seed = self._rnd_seed
        # fixing seed for reproducibility            
        print(f"Fixing seed to {rnd_seed}")
        np.random.seed(rnd_seed)
        res_path = self.savedir/"kw_sampled_source_pos.dll"
        try:
            assert not recompute
            kw_sampled_source_pos = load_whatever(res_path)
            print("Loaded pre-computed source position")
            ra_source = kw_sampled_source_pos["ra_source"]
            dec_source = kw_sampled_source_pos["dec_source"]
        except:
            ra_source,dec_source = _sample_source_pos(self,_radec=_radec)
            kw_sampled_source_pos = {"ra_source":ra_source,"dec_source":dec_source}
            with open(res_path,"wb") as f:
                dill.dump(kw_sampled_source_pos,f)
            print("Stored computed source position")
            
        if self.kwargs_source_def["center_x"]==0 and self.kwargs_source_def["center_x"]==0 and not update:
            print("Source position has to be sampled a first time")
            update=True
        if update:
            self.update_source_position(ra_source,dec_source)
        else:
            print("Source position not updated")
        return ra_source,dec_source

    def get_xy_source_plane(self,alpha_map=None,_radec=None):
        """Map the x,y grid into the source plane
        (used to fit the light of the source to the image)
        """
        # coords
        if _radec is None:
            _radec  = self.gallens._radec
        RA,DEC      = cod.get_RADEC(_radec)
        # if not given as input, reads the computed deflection map
        if alpha_map is None:
            # if not already, compute alpha_map
            alpha_map   = self.alpha_map(_radec=_radec)
        alpha_x,alpha_y = alpha_map        
        x_source_plane, y_source_plane = RA-alpha_x,DEC-alpha_y
        # the coords have to be given as flat
        x_source_plane = image2array(x_source_plane)
        y_source_plane = image2array(y_source_plane)
        return x_source_plane,y_source_plane
        
    def get_lensed_image(self,
                         Sim,
                         kwargs_source=None,
                         _radec=None, # 
                         unconvolved=True):
        imageNumerics,sourceModel = self.get_imageNumerics(Sim=Sim,return_sourceModel=True)
        alpha_map_HR = self.alpha_map(_radec=_radec)
        # Downsample alpha_map to the imagenumerics grid resolution by interpolation
        coord = _radec
        if coord is None:
            coord = self.gallens._radec
        # the following are "flat" coordinates
        coord_out   = imageNumerics.coordinates_evaluate
        alpha_map_x = interpolate_map(map=alpha_map_HR[0],coord=coord,coord_out=coord_out)
        alpha_map_y = interpolate_map(map=alpha_map_HR[1],coord=coord,coord_out=coord_out)
        alpha_map   = alpha_map_x,alpha_map_y
        x_source_plane,y_source_plane = self.get_xy_source_plane(_radec=coord_out,alpha_map=alpha_map)
        
        if kwargs_source is None:
            #note: the following should be the standard use
            kwargs_source = self.kwargs_source_def
        _kwargs_source = get_kwargs_sourceSim(Sim,kwargs_source,lens=self)
        kwargs_source_list            = [_kwargs_source]
        # note: following is in flux/arcsec^2 -> has to be converted into flux/pix eventually
        source_light                  = sourceModel.surface_brightness(x_source_plane, y_source_plane, 
                                                                       kwargs_source_list, k=None)
        # the following is to compare the Sim to the "native" resolution of the gallens image
        sim_deltapix  = imageNumerics.grid_class.pixel_width
        gal_deltapix  = to_dimless(self.gallens.deltaPix)
        if sim_deltapix+1e15<gal_deltapix:
            raise RuntimeError("Simulated observatory can not have higher resolution than baseline simulation") 
        image_sim = imageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
        return image_sim

    # Simulating observations
    #########################
    def get_imageNumerics(self,Sim,return_sourceModel=False):    
        data_class,psf_class,sourceModel,kwargs_numerics = cod.get_dataclasses(Sim)
        imageNumerics = NumericsSubFrame(pixel_grid=data_class,
                                         psf=psf_class, 
                                         **kwargs_numerics)
        if return_sourceModel:
            return imageNumerics,sourceModel
        return imageNumerics
 
    
    def get_Sim(self,band=None,
                    kwargs_psf=None,# add psf and pssf
                    source_model_list=source_model_list):
        kwargs_source_model = {"source_light_model_list":source_model_list,
                                   "cosmo":self.cosmo
                                   }
        kwargs_model = {"z_source":self.gallens.z_source} | kwargs_source_model
        if not "cosmo" in kwargs_model.keys():
            # cosmology should in principle not be used, but better be consistent
            kwargs_model["cosmo"] = self.cosmo
        if band is None:
            # This is the resolution of the simulation itself
            kwargs_single_band = self.kwargs_band_sim
            kwargs_single_band["pixel_scale"] = to_dimless(self.gallens.deltaPix)            
            # keep the pixel number as high as the simulation
            pixel_num = self.gallens.pixel_num
            # assumes the PSF is either NONE (default) or already defined
        else:
            kwargs_single_band = band.kwargs_single_band()
            if kwargs_psf is not None:
                kwargs_single_band.update(kwargs_psf)
            # must recompute pixel_num in order to covert to ~ the same aperture,
            # but with the new resolution 
            # -> round down to be sure we are within the bounds
            pixel_num = int(to_dimless(2*self.gallens.radius)/kwargs_single_band["pixel_scale"])
        # instantiate simulation API class
        Sim = SimAPI(numpix = pixel_num, # N of pixels in "observed" image
                 kwargs_single_band = kwargs_single_band, # telescope specific keyword arguments (eg HST, see above)
                 kwargs_model = kwargs_model,# kwargs source model (in principle kw lens as well)
                )
        #Sim.arcXkpc = self.gallens.arcXkpc
        def _map_coord2pix(coord1,coord2):
            return map_coord2pix(Sim.data_class,coord1,coord2,self.gallens.arcXkpc)
        setattr(Sim, 'map_coord2pix', _map_coord2pix)
        return Sim

    def sim_image(self,SimObs,kwargs_source=None,noisy=False,rnd_seed=None):
        """Obtain simulated images given SimObs
        """
        image_SimObs     = self.get_lensed_image(Sim=SimObs,
                                                 kwargs_source=kwargs_source,
                                                 unconvolved=False)
        if noisy:
            if rnd_seed is None:
                rnd_seed = self._rnd_seed
            # fixing seed for reproducibility            
            print(f"Fixing seed to {rnd_seed}")
            np.random.seed(rnd_seed)
            image_SimObsnoisy  = image_SimObs + SimObs.noise_for_model(model=image_SimObs)
            error_SimObs       = SimObs.estimate_noise(image_SimObsnoisy) # NOT variance
            return image_SimObsnoisy,error_SimObs
        return image_SimObs

    def sim_multi_band_list(self,band,kwargs_psf=None,source_model_list=source_model_list):
        """Setup Simulation given band specific, its psf and source_model_list 
        """
        SimObs                         = self.get_Sim(band=band,kwargs_psf=kwargs_psf,
                                                         source_model_list=source_model_list)
        image_SimObsnoisy,error_SimObs = self.sim_image(SimObs,noisy=True)
        kw_data_sim               = SimObs.kwargs_data
        kw_data_sim["image_data"] = image_SimObsnoisy
        kw_data_sim["noise_map"]  = error_SimObs
        kw_psf_sim                = SimObs.kwargs_psf
        # consider if to add psf error - depends on observations
        if "point_source_supersampling_factor" in kw_psf_sim.keys():
            kw_numerics_sim = {'point_source_supersampling_factor':kw_psf_sim["point_source_supersampling_factor"]} 
        else:
            kw_numerics_sim = {'point_source_supersampling_factor':1}
        image_band      = [kw_data_sim, kw_psf_sim, kw_numerics_sim]
        multi_band_list = [image_band]
        return multi_band_list
    

def clip_map(map,sigma2clip=8,verbose=True):
    if verbose:
        print(f"Clipping map")
    sgc = sigma_clip(map,sigma=sigma2clip)
    msk_map = np.invert(sgc.mask)
    return map*msk_map

def clip_smooth_map(map,sigma2clip=8,sigma2smooth=3,verbose=True):
    msked_map = clip_map(map,sigma2clip=sigma2clip,verbose=verbose)
    if verbose:
        print(f"Smoothing map")
    map_smooth = gaussian_filter(msked_map,sigma=sigma2smooth)
    return map_smooth


def wrapper_clip_smooth_map(map,min_points=10,sigma2clip=8,
                            sigma2smooth_min=1,sigma2smooth_max=10,name=None,verbose=False):
    is_smooth = False
    for s2s in np.linspace(sigma2smooth_max,sigma2smooth_min,20):
        map_smooth = clip_smooth_map(map,sigma2clip=sigma2clip,sigma2smooth=s2s,verbose=verbose)
        points = len(np.where(map_smooth<0)[0])
        if points>min_points:
            is_smooth = True
            return map_smooth,is_smooth
    if not is_smooth:            
        warnings.warn(f"{name} could not be smoothed - using un-smoothed map (increase noise)")
    return map_smooth,is_smooth


def get0_index(map,pixel_num):
    """
    # have to find when those are ~0
    mintv = np.min(np.abs(eigen_rad))
    Dv    = np.max(np.abs(eigen_rad)) - mintv
    maxtv = mintv + 0.1*Dv
    test_values = np.linspace(mintv,maxtv,20)
    """
    test_values = np.linspace(.1,4)
    ix,iy = [],[] #placeholders 
    # radial
    for tv in test_values:
        if len(ix)/(pixel_num**2) >0.001 :
            break
        else:
            iy,ix = np.where(cod.MAD_mask(np.abs(map),0,tv)) 
    # not the inverted order
    return ix,iy

def _sample_source_pos(lens,_radec=None):
    """
    Sample the source position within the tangential and radial caustics
    Note: we could restrict to within only tang. 
    """
    kw_caustics  = lens.get_kw_critical_curve_caustics(_radec=_radec)
    ra_ct,dec_ct = kw_caustics["caustics"]["tangential"]
    ra_cr,dec_cr = kw_caustics["caustics"]["radial"]
    # We compute the convex hull defined by the tangential caustic
    # sample uniformily from max to min, and accept only if
    # within the convex hull
    # not exact but fairly precise nonetheless
    if kw_caustics["tan_smooth"]:
        hull_tan  = Delaunay(np.array([ra_ct,dec_ct]).T)
        def accept_tan(x0,y0):
            return hull_tan.find_simplex([x0,y0])!=-1
    else:
        r0ct,d0ct = np.mean(ra_ct),np.mean(dec_ct)
        r0st,d0st = np.std(ra_ct),np.std(dec_ct)
        sigt = np.hypot(r0st,d0st)
        def accept_tan(x0,y0):
            # within 1 sigma from the center
            return np.hypot(x0-r0ct,y0-d0ct) < sigt

    
    if kw_caustics["rad_smooth"]:
        hull_rad  = Delaunay(np.array([ra_cr,dec_cr]).T)
        def accept_rad(x0,y0):
            return hull_rad.find_simplex([x_cnd,y_cnd])!=-1
    else:
        r0cr,d0cr = np.mean(ra_cr),np.mean(dec_cr)
        r0sr,d0sr = np.std(ra_cr),np.std(dec_cr)
        sigr = np.hypot(r0sr,d0sr)
        def accept_rad(x0,y0):
            # within 1 sigma from the center
            return np.hypot(x0-r0cr,y0-d0cr) < sigr
    
    x_bounds = np.array([np.min([ra_ct.min(),ra_cr.min()]),
                         np.max([ra_ct.max(),ra_cr.max()]) 
                        ])
    y_bounds = np.array([np.min([dec_ct.min(),dec_cr.min()]),
                         np.max([dec_ct.max(),dec_cr.max()])
                        ])
    
    #x_bounds = np.array([ra_ct.min(),ra_ct.max()])
    #y_bounds = np.array([dec_ct.min(),dec_ct.max()])
    
    ra_source,dec_source = None,None
    for i in range(1000):
        # sample within range
        x_cnd = np.random.uniform(*x_bounds)
        y_cnd = np.random.uniform(*y_bounds)
        
        # accept if inside the hull (if smooth) or 
        # within a sigma of the mean if not smooth
        if accept_rad(x_cnd,y_cnd) and accept_tan(x_cnd,y_cnd):
            ra_source,dec_source = x_cnd,y_cnd
            break
    if ra_source is None:
        print(x_bounds)
        print(y_bounds)
        plt.close()
        #ra_cr,dec_cr = kw_caustics["caustics"]["radial"]
        plt.scatter(ra_cr,dec_cr,marker=".",label="radial",color="b")
        plt.scatter(ra_ct,dec_ct,marker=".",label="tang.",color="r")
        plt.legend()
        plt.savefig("tmp/radec_ct_cr_debug.png")
        plt.close()
        raise RuntimeError("Failed to sample a source position")
    return ra_source,dec_source

def interpolate_map(map,coord,coord_out):
    map_flat = image2array(map)
    map_out = griddata(coord,map_flat,coord_out)
    if len(np.shape(coord_out))==2:
        map_out = array2image(map_out)
    return map_out

# add a map coordinate to pixel that can take either kpc or arcsec
# add to Sim, as this is the one we are interested on
def map_coord2pix(data_class,coord1,coord2,arcXkpc):
    print("to test this conversion")
    if type(coord1)==u.quantity.Quantity:
        assert type(arcXkpc)==u.quantity.Quantity
        assert coord1.unit == coord2.unit
        if coord1.unit == u.kpc:
            coord1 = coord1*arcXkpc
            coord2 = coord2*arcXkpc
            coord1 = coord1*arcXkpc
        elif coord1.unit!=u.arcsec:
            raise ValueError(f"Only accepeted units are kpc or arcsec, not {coord1.unit}")
        coord1 = coord1.value
        coord2 = coord2.value    
    return data_class.map_coord2pix(coord1,coord2)