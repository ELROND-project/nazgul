from pathlib import Path
import astropy.units as u
from lenstronomy.SimulationAPI.sim_api import SimAPI

from python_tools.get_res import LoadClass
from python_tools.tools import to_dimless,mkdir

from nazgul.pathfinder import path_nazgul
from nazgul.ObsData.band import std_band_obs
from nazgul.basic_gal import BasicClass,store_class
from nazgul.mount_doom.cracks_of_doom import band_true as std_band_true
from nazgul.mount_doom.cracks_of_doom import source_model_list as std_source_model_list
from nazgul.mount_doom.cracks_of_doom import kwargs_source_default as std_kwargs_source_init


# find better name
# lensSap, lenssim,lenstorch,lens...
str_class = "LF_"

empty_kwargs_add_lenses = {"lens_model_list":[],"kwargs_lens":[]}

def empty_get_kw_add_lenses(gallens,kwargs_add_lenses=None):
    return empty_kwargs_add_lenses

# eg: lens_fuel_LOS = LensFuel("LOS",get_kw_add_lenses=los_kw_add)
class LensFuel(BasicClass):
    def __init__(self,
                 name, # it has to define uniquely the type of "fuel" used
                 band_obs=std_band_obs,
                 band_true=std_band_true,
                 source_model_list=std_source_model_list,
                 kwargs_source_init=std_kwargs_source_init, 
                 kwargs_add_lenses=empty_kwargs_add_lenses,
                 get_kw_add_lenses=empty_get_kw_add_lenses
                 ):
        self.name               = str_class+str(name)
        self.source_model_list  = source_model_list
        # these are the initial values of the kwargs_source, its position (and maybe amplitude) 
        # will change depending on the lens system 
        self.kwargs_source_init = kwargs_source_init
        # these are the initial values before getting updated w get_kw_add_lenses 
        # wich might depend on the lens system
        self.kwargs_add_lenses_init  = kwargs_add_lenses
        self.get_kw_add_lenses = get_kw_add_lenses
        # Observations properties tied to the 
        self.band_true         = band_true  #this need to be updated with the correct pixel scale dep. on lens
        self.band_obs          = band_obs
        
    def setup(self,gallens):
        # add some gallens properties without storing the whole gallens
        for k in ("name","cosmo", "z_source","z_lens","arcXkpc"):
            setattr(self, "gallens_"+k, getattr(gallens, k))  
        # Update band true w. Correct pixel scale
        self.band_true.camera["pixel_scale"] = to_dimless(gallens.deltaPix)
        #The actual sim used for the simulated lens
        pixel_num_True = gallens.pixel_num
        self.SimTrue  = self._get_Sim(band=self.band_true,pixel_num=pixel_num_True)
        # The sim used for creating the observed images
        # must recompute pixel_num in order to covert to ~ the same aperture,
        # but with the new resolution 
        # -> round down to be sure we are within the bounds
        pixel_num_Obs = int(to_dimless(2*gallens.radius)/self.band_obs.camera["pixel_scale"])
        self.SimObs   = self._get_Sim(band=self.band_obs,pixel_num=pixel_num_Obs)
        # instatiate the kwargs_add_lenses
        self.kwargs_add_lenses = self.get_kw_add_lenses(gallens,self.kwargs_add_lenses_init)
        self.savedir = Path(gallens.Gal.gal_dir).parent/f"LensSystem/{self.subdir}/"
        mkdir(self.savedir)
        # store it in the model_res_dir (not in the lenssystem
        store_class(self,path=self.pkl_path,overwrite=False,update=False)
        return 0
        
    @property
    def subdir(self):
        return str(self.name)+"_"+str(self._hash_b64)
    
    @property
    def pkl_path(self):
        # savedir has to be introduced by 
        return self.savedir/f"{self.name}_{self._hash_b64}.pkl"

    def _identity(self):
        return (
            self.name,
            self.band_obs._identity(),
            self.band_true._identity(),
            self.kwargs_source_init, # the actual source will dep. on the lens
            self.kwargs_add_lenses,
            self.source_model_list,
            self.kwargs_add_lenses_init, # initial values
            self.get_kw_add_lenses.__name__) # we at least save the name of the function
        
    # Lens Add: 
    #def get_kw_add_lenses(self,gallens):
    #    # Each instance of this class should define it depending on the model
    #    raise NotImplementedError
    #    # basic idea: return self.kwargs_add_lenses_init
    #def set_kw_add_lenses(self,get_kw_add_lenses):
    #    setattr(self,"get_kw_add_lenses",get_kw_add_lenses)
        
    def _get_Sim(self,
                band,
                pixel_num,
                ):
        
        kwargs_source_model = {"source_light_model_list":self.source_model_list,
                                "cosmo":self.gallens_cosmo}
        kwargs_model = {"z_source":self.gallens_z_source} | kwargs_source_model
        
        kwargs_single_band = band.kwargs_single_band()
        
        # instantiate simulation API class
        Sim = SimAPI(num_pix = pixel_num, # N of pixels in "observed" image
                 kwargs_single_band = kwargs_single_band, # telescope specific keyword arguments (eg HST, see above)
                 kwargs_model = kwargs_model,# kwargs source model (in principle kw lens as well)
                )
        #Sim.arcXkpc = self.gallens.arcXkpc
        def _map_coord2pix(coord1,coord2):
            return map_coord2pix(Sim.data_class,coord1,coord2,self.gallens_arcXkpc)
        setattr(Sim, 'map_coord2pix', _map_coord2pix)
        return Sim

    def ReadClass(self,cl):
        LF = LoadClass(cl.pkl_path,path_base=path_nazgul)
        return LF
    def __str__(self):
        """Human-readable identifier.
        """
        return f'Name:{self.name}\nHash:{self._hash_b64}'

        

# add a map coordinate to pixel that can take either kpc or arcsec
# add to Sim, as this is the one we are interested on
def map_coord2pix(data_class,coord1,coord2,arcXkpc):
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

