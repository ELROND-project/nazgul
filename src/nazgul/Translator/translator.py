# interface between nazgul and the simulation-specific "translators"
import numpy as np
from pathlib import Path
from importlib import import_module

from python_tools.get_res import LoadClass
from python_tools.get_res import load_whatever

from nazgul.pathfinder import std_data_dir,path_nazgul
# allowed simulations suites
from nazgul.configurations import SimSuiteNames,min_z,max_z,min_mass
from nazgul.Translator import std_simsuite,std_sim
from nazgul.pathfinder import get_part_dir,get_gal_dir
from nazgul.basic_gal import store_class

def check_simsuite(simsuite):
    if simsuite not in SimSuiteNames:
        raise RuntimeError(f"The simsuite {simsuite} is not yet implemented, allowed only {SimSuiteNames}") 
        
def get_sim_func(simsuite,func_name):
    check_simsuite(simsuite)
    module_path = f"nazgul.Translator.{simsuite}.particle_galaxy"
    module = import_module(module_path)
    func   = getattr(module, func_name)
    return func

# useful function (but find better place)
def get_CM(Ms,Xs,Ys,Zs=None):
    """Get Center of Mass (CM)
    """
    X_cm = np.sum(Xs*Ms)/np.sum(Ms)
    Y_cm = np.sum(Ys*Ms)/np.sum(Ms)
    if Zs is None:
        return X_cm,Y_cm
    else:
        Z_cm = np.sum(Zs*Ms)/np.sum(Ms)
        return X_cm,Y_cm,Z_cm


def get_rnd_kw_gal(simsuite=std_simsuite,sim=std_sim,subsim=None,
                   min_mass = str(min_mass),
                   min_z    = str(min_z),
                   max_z    = str(max_z),
                   check_prev=True,save_pkl=True):
    """Given the simulation, the range of redshift and minimum mass required, 
        returns a kwargs of a random galaxy from the simulation
    """
    check_simsuite(simsuite)
    if simsuite=="EAGLE":
        from nazgul.Translator.EAGLE.particle_galaxy import get_rnd_gal_indexes as get_rnd_kw_gal
        min_mass = str(min_mass)
        min_z    = str(min_z)
        max_z    = str(max_z)
        kw     = get_rnd_kw_gal(sim=sim,min_mass=min_mass,max_z=max_z,min_z=min_z,\
                        check_prev=check_prev,plot=False,save_pkl=save_pkl)
    elif simsuite=="COLIBRE":
        from nazgul.Translator.COLIBRE.particle_galaxy import get_rnd_kw_gal
        simulation_dir = Path(sim)/subsim
        kw             = get_rnd_kw_gal(simulation_dir=simulation_dir,
                                min_mass=min_mass,max_z=max_z,min_z=min_z)
    return kw

def get_z_snap(simsuite,z=None,snap=None,*args,**kwargs):
    get_z_snap = get_sim_func(simsuite,"get_z_snap")
    return get_z_snap(z=z,snap=snap,*args,**kwargs)

def Gal2MXYZ(Gal):
    print("Running Gal2MXYZ...")
    Gal.run()
    simsuite = Gal.simsuite
    Gal2MXYZ = get_sim_func(simsuite,"Gal2MXYZ")
    return Gal2MXYZ(Gal) 

def Gal2MXYZ_part(Gal,part_type,*args,**kwargs):
    Gal.run()
    simsuite = Gal.simsuite
    Gal2MXYZ_part = get_sim_func(simsuite,"Gal2MXYZ_part")
    return Gal2MXYZ_part(Gal,part_type,*args,**kwargs)

def Gal2kwMXYZ(Gal): 
    Ms,Xs,Ys,Zs = Gal2MXYZ(Gal)
    return {"Ms":Ms,"Xs":Xs,"Ys":Ys,"Zs":Zs}

def Gal2kwMXYZ_part(Gal,part_type,*args,**kwargs): 
    Ms,Xs,Ys,Zs = Gal2MXYZ_part(Gal,part_type=part_type,*args,**kwargs)
    return {"Ms":Ms,"Xs":Xs,"Ys":Ys,"Zs":Zs}

# from path to kw of Gal

def gal_path2simsuite(gal_dill_path,data_dir=std_data_dir):
    path_split       = np.array(Path(gal_dill_path).parts)
    data_dir_name    = Path(data_dir).name
    index_data_dir   = np.where(path_split==data_dir_name)[0][0]
    index_simsuite   = index_data_dir+1
    simsuite         = path_split[index_simsuite]
    check_simsuite(simsuite)
    return simsuite
    
def gal_path2kwGal(gal_dill_path,data_dir=std_data_dir):
    """From path extract the required inputs for PartGal class
    """
    gal_dill_path   = Path(gal_dill_path)
    simsuite        = gal_path2simsuite(gal_dill_path,data_dir)
    gal_path2kwGal  = get_sim_func(simsuite,"gal_path2kwGal")
    kw_Gal_full     = gal_path2kwGal(gal_dill_path)
    return kw_Gal_full

def get_rnd_PG(simsuite,**kw_galpart):
    get_rnd_SPG = get_sim_func(simsuite,"get_rnd_SPG")
    SPG = get_rnd_SPG(**kw_galpart)
    PG  = PartGal.SPG2PG(SPG)
    return PG

def get_all_PG(simsuite,**kw_galpart):
    get_all_SPG = get_sim_func(simsuite,"get_all_SPG")
    all_SPG = get_all_SPG(**kw_galpart)
    all_PG  = []
    for SPG in all_SPG:
        PG  = PartGal.SPG2PG(SPG)
        all_PG.append(PG)
    return all_PG

def get_vdisp(simsuite,simpartgal,**kw_veldisp):
    get_vdisp_sim = get_sim_func(simsuite,"get_vdisp")
    return get_vdisp_sim(simpartgal,**kw_veldisp)
    
def LoadGal(path,if_fail_recompute=True,verbose=True):
    # Try loading galaxy - if fail and fail_recompute==True, try recomputing it
    Gal = LoadClass(path=path,verbose=verbose,path_base=path_nazgul)
    if not Gal and if_fail_recompute:
        if verbose:
            print("Recomputing Galaxy ...")
        simsuite    = gal_path2simsuite(path)
        kw_Gal_full = gal_path2kwGal(path)
        Gal         = PartGal(simsuite=simsuite,**kw_Gal_full)
        if verbose:
            print("... done computing Galaxy")
    if Gal:
        # now this basically does nothing
        Gal.unpack()
    return Gal


class PartGal():
    """Given the simulation, snap (or z) and galaxy numbers, set up a class
    with all the needed particle properties converted in physical units

    -> a wrapper of the individual SimPartGal of the specific ones for each simulation suite
    """
    def __init__(self, 
                 kw_Gal,# way to ID the galaxy (sim-dependent)
                 simsuite=std_simsuite, # identity of the simulation
                 sim=std_sim, # which sim. between the sim. suite
                 subsim=None, # optional sub. dir of simulation
                 data_dir=std_data_dir, # where is the particle data stored
                 z=None,snap=None,    # redshift or snap
                 M=None,Centre=None, # these can be recovered
                 reload=True,     # set to false only for debug
                ):
        
        check_simsuite(simsuite)
        SimPartGal        = get_sim_func(simsuite,"SimPartGal")
        get_kw_SimPartGal = get_sim_func(simsuite,"get_kw_SimPartGal")
        kw_SimPartGal     = get_kw_SimPartGal(kw_Gal=kw_Gal,simsuite=simsuite, 
                                          sim=sim,subsim=subsim,
                                          data_dir=data_dir,
                                          z=z,snap=snap,
                                          M=M,Centre=Centre,
                                          reload=reload)
        self._SimPartGal = SimPartGal(**kw_SimPartGal)
        #self._SimPartGal.run(reload=reload)

    @property
    def simsuite(self):
        return self._SimPartGal.simsuite
        
    @classmethod
    def SPG2PG(cls, SPG):
        """Construct from an existing SimPartGal instance."""
        obj = cls.__new__(cls)  # bypass __init__
        obj._SimPartGal = SPG
        return obj

    def __getstate__(self):
        return {"_SimPartGal": self._SimPartGal}
        
    def __setstate__(self, state):
        self._SimPartGal = state["_SimPartGal"]
        
    def store_gal(self):
        # store class instance 
        store_class(self._SimPartGal,path=self.dill_path_abs())
        
    def __str__(self):
        return self._SimPartGal.__str__()
        
    def __getattr__(self, name):
        try:
            return getattr(self._SimPartGal, name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")
#   def run(self,reload=True):
#    #    self._SimPartGal.run(reload=reload)
#    #    #self.store_gal()
