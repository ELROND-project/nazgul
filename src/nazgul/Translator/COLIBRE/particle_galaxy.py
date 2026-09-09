# copied from remade_gal.py
# adapted for COLIBRE sim.

# get random swift galaxy from get_rand_gal.py

import dill
import warnings
import unyt as u  # package used by swiftsimio to provide physical units
import numpy as np
from pathlib import Path
from functools import cached_property

from python_tools.tools import mkdir
from python_tools.get_res import LoadClass,load_whatever

from nazgul.pathfinder import get_gal_dir,path_nazgul
from nazgul.Translator.particle_galaxy import BasicPartGal,store_class
from nazgul.Translator.COLIBRE import simsuite_name,simsuite_short_name,part_type_list,check_part_type

from nazgul.Translator.COLIBRE.get_Gal import get_swiftgal,get_snap,get_z_snap
from nazgul.Translator.COLIBRE.get_Gal import std_sim,std_subsim,colibre_base_path
from nazgul.Translator.COLIBRE.get_Gal import min_z,max_z,min_mass,get_rnd_kw_gal,get_all_kw_gal

def gal_path2kwGal(gal_pkl_path):
    gal_pkl_path = Path(gal_pkl_path)
    Gn_dir       = gal_pkl_path.parent.parent
    snap_dir     = Gn_dir.parent
    subsim_dir   = snap_dir.parent   
    sim_dir      = subsim_dir.parent
    Gn           = Gn_dir.name.replace("Gn","")
    snap         = snap_dir.name.replace("snap_","")
    kw_gal_full  = {}
    kw_gal_full["sim"]    = str(sim_dir.name)
    kw_gal_full["subsim"] = str(subsim_dir.name)
    
    kw_gal_full["kw_Gal"] = {"snap":str(snap),
                             "soap_index": int(Gn)}
    # M,center not necessary
    return kw_gal_full


def get_masses_part(Gal,part_type):
    "Return masses in [Msun] of given particle type"
    part_type = check_part_type(part_type)
    part  = getattr(Gal,part_type)
    Mpart = _get_masses_part(part)
    return Mpart

def _get_coord_part(part):
    "Return coordinates in [kpc] of given particle instance"
    coords = part.coordinates
    coords_phys = coords.to_physical().in_units(u.kpc)
    Xpart = coords_phys[:, 0] # kpc
    Ypart = coords_phys[:, 1] # kpc
    Zpart = coords_phys[:, 2] # kpc
    return Xpart,Ypart,Zpart
    
def _get_masses_part(part):
    part_name = part.group_name
    if part_name!="black_holes":
        masses = part.masses
    else:
        #warnings.warn("Using dynamical mass for BH, verify that it make sense")
        # the first is done to compute the potential, the other is updated w. the accretion and used for feedback calc.
        # -> should be correct to use dynamical mass
        masses = part.dynamical_masses
    Mpart = masses.to_physical().in_units(u.Msun)  # Msun
    return Mpart

def get_coord_part(Gal,part_type):
    "Return coordinates in [kpc] of given particle type"
    part_type = check_part_type(part_type)
    part  = getattr(Gal,part_type)
    Xpart,Ypart,Zpart =  _get_coord_part(part)
    return Xpart,Ypart,Zpart
        
def get_masses(Gal):
    "Get masses of all particles"
    masses = []
    for part_type in part_type_list:
        masses.append(get_masses_part(Gal,part_type))    
    Ms    = np.concatenate(masses) #Msun
    return Ms

def get_coords(Gal):
    "Get coords of all particles"
    _Xs,_Ys,_Zs = [],[],[]
    for part_type in part_type_list:
        xs,ys,zs  = get_coord_part(Gal,part_type)
        _Xs.append(xs)
        _Ys.append(ys)
        _Zs.append(zs)
    Xs = np.concatenate(_Xs) #kpc
    Ys = np.concatenate(_Ys) #kpc
    Zs = np.concatenate(_Zs) #kpc
    return Xs,Ys,Zs
    
# adapted from wip_select_swiftgal
def Gal2MXYZ(ColGal):
    Gal     = ColGal.swift_gal
    # Given a ColibreGal galaxy, which then plot to as swift galaxy, return Masses (in Msun) and
    # XY coords. of particles in kpc  centered around center of mass
    Ms       = get_masses(Gal)
    # Particle pos
    Xs,Ys,Zs = get_coords(Gal)
    
    # recenter around CM
    X_cm,Y_cm,Z_cm = get_CoM(Gal,XYZM=[Xs,Ys,Zs,Ms])
    Xs-=X_cm
    Ys-=Y_cm
    Zs-=Z_cm
    
    #Convert all to astropy for convenience
    Ms = Ms.to_astropy()
    Xs = Xs.to_astropy()
    Ys = Ys.to_astropy()
    Zs = Zs.to_astropy()
    
    return Ms, Xs,Ys,Zs

def get_CoM(Gal,XYZM=None):
    if XYZM is None:
        # Given a ColibreGal galaxy, which then plot to as swift galaxy, return Masses (in Msun) and
        # XY coords. of particles in kpc  centered around center of mass
        Ms       = get_masses(Gal)
        # Particle pos
        Xs,Ys,Zs = get_coords(Gal)
        XYZM = Xs,Ys,Zs,Ms
    Xs,Ys,Zs,Ms = XYZM
    # Centre of Mass
    X_cm = np.sum(Xs*Ms)/np.sum(Ms)
    Y_cm = np.sum(Ys*Ms)/np.sum(Ms)
    Z_cm = np.sum(Zs*Ms)/np.sum(Ms)
    return X_cm,Y_cm,Z_cm
    
def Gal2MXYZ_part(Gal,part_type,CM=None): 
    """Given the galaxy, return Masses (in Msun) and
    XY coords. of a specific particle type in kpc centered around center
    """
    part_type = check_part_type(part_type)
    part = getattr(Gal,part_type) 
    
    # Particle masses
    Ms       = _get_masses_part(part)
    # Particle pos
    Xs,Ys,Zs = _get_coord_part(part)
    
    # center around the center of the galaxy -> this has already been subtracted by COLIBRE 
    # but do re-center around the CoM
    """Cx,Cy,Cz  = Gal.centre*1e3*u.kpc # Mpc
        
    Xs -= Cx
    Ys -= Cy
    Zs -= Cz
    """
    # NOTE: however to compare them to the output of Gal2MXYZ
    # we should rescale them by the CoM -> 
    if CM is None:
        CM = get_CoM(Gal)
    X_cm,Y_cm,Z_cm = CM
    Xs -= X_cm
    Ys -= Y_cm
    Zs -=Z_cm
    
    #Convert all to astropy for convenience
    Ms = Ms.to_astropy()
    Xs = Xs.to_astropy()
    Ys = Ys.to_astropy()
    Zs = Zs.to_astropy()
    return Ms,Xs,Ys,Zs

def get_kw_SimPartGal(kw_Gal,sim,simsuite,subsim,data_dir,z,snap,M,Centre,reload):
    assert simsuite==simsuite_name
    return {"kw_Gal":kw_Gal,"sim":sim,"subsim":subsim}

# basically a wrapper for swift galaxies
class SimPartGal(BasicPartGal):
    # what we need is
    # z
    # cosmology
    # particles (get from Gal2MXYZ)
    # identity
    # name
    
    # define name to verify identity
    _type_id = "SimPartGal_"+simsuite_name
    _large_attributes_setup  = ["stars","gas","dark_matter","black_holes","swift_gal","_swift_gal"]
    _large_attributes_unpack = []
    
    simsuite = simsuite_name
    simsuite_code = simsuite_short_name
 
    def __init__(self,kw_Gal,sim=std_sim,subsim=std_subsim):
        #kw_Gal: soap_index,snap and/or z
        #self.kw_Gal     = kw_Gal
        self.soap_index  = kw_Gal["soap_index"] #self.swift_gal.halo_catalogue.soap_index
        self.z,self.snap = get_z_snap(z=kw_Gal.get("z",None),
                            snap=kw_Gal.get("snap",None),
                                     sim=sim,subsim=subsim)
        
        self.sim         = Path(sim)
        self.subsim      = Path(subsim)
        
        # here we load but not store the swift galaxy to avoid 
        # increasing the memory load
        sg              = self.swift_gal
        self.soap_file  = Path(sg.halo_catalogue.soap_file)
        #'/cosma8/data/dp004/colibre/Runs/L0025N0752/THERMAL_AGN_m5/SOAP-HBT/halo_properties_0127.hdf5'
        
        self.a =  sg.metadata.a
        self.verbose_assert_almost_equal((1/self.a)-1,self.z,msg="Redshifts")
        self.verify_snap()

        self.gal_dir  = get_gal_dir(kw_Gal,snap=self.snap,
                                    sim=self.sim,subsim=self.subsim,
                                    simsuite=self.simsuite)
        mkdir(self.gal_dir)
        
        # total mass
        #SphOverDens = self.swift_gal.halo_catalogue.spherical_overdensity_500_crit
        #self.M_tot  = SphOverDens.total_mass.to_physical_value("Msun")[0] #Msun
        BoundSubHalo = self.swift_gal.halo_catalogue.bound_subhalo
        self.M_tot  = BoundSubHalo.total_mass.to_physical_value("Msun")[0] # Msun
        # coord of the centre
        self.centre = self.swift_gal.centre.to_physical_value("Mpc") 
        #self.part_dir = get_part_dir(self.snap,data_dir=data_dir,**kw_sim)
        # cosmo is a bottleneck and light to store
        # we compute it once and then it's done
        self.cosmo
    
    @property
    def swift_gal(self):
        try:
            return self._swift_gal
        except AttributeError:
            # only compute it once
            swift_gal = get_swiftgal(sim=self.sim,
                                     subsim=self.subsim,
                                     snap=self.snap,
                                     soap_index=self.soap_index)
            self._swift_gal = swift_gal
            return swift_gal
            
    @swift_gal.deleter
    def swift_gal(self):
        del self._swift_gal
    
    def initialise_parts(self):
        # heavy -> avoid until necessary and do not store
        sg               = self.swift_gal
        self.stars       = sg.stars
        self.gas         = sg.gas
        self.dark_matter = sg.dark_matter
        self.black_holes = sg.black_holes
        # The following are very inefficient
        if not hasattr(self,"M_stars"):
            self.M_stars     = np.sum(self.stars.masses.to_physical().in_units(u.Msun))
        if not hasattr(self,"M_gas"):
            self.M_gas       = np.sum(self.gas.masses.to_physical().in_units(u.Msun))
        if not hasattr(self,"M_dm"):
            self.M_dm        = np.sum(self.dark_matter.masses.to_physical().in_units(u.Msun))
        # again using dynamical masses for BH
        if not hasattr(self,"M_bh"):
            self.M_bh        = np.sum(self.black_holes.dynamical_masses.to_physical().in_units(u.Msun))
        if not hasattr(self,"N_stars"):
            self.N_stars = len(sg.stars.particle_ids)
        if not hasattr(self,"N_gas"):
            self.N_gas = len(sg.gas.particle_ids)
        if not hasattr(self,"N_dm"):
            self.N_dm = len(sg.dark_matter.particle_ids)
        if not hasattr(self,"N_bh"):
            self.N_bh = len(sg.black_holes.particle_ids)
        if not hasattr(self,"N_part"):
            self.N_part = self.N_stars + self.N_gas + self.N_dm + self.N_bh
        if not hasattr(self,"M"):
            self.M = self.M_stars + self.M_gas + self.M_dm + self.M_bh 
        return 0
        
    @cached_property
    def cosmo(self):
        return self.swift_gal.metadata.cosmology
    
    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        Id = (self._type_id,self.sim,self.subsim,
            self.snap,self.soap_index)
        return Id 
        
    def upload_prev(self,verbose=True):
        prev_Gal = ReadGal(self)
        if prev_Gal is False:
            if verbose:
                print("Failed loading of prev. gal.")
            return False
        if prev_Gal != self:
            if verbose:
                print(f"Prev. Gal not equal to self: {prev_Gal._identity()==self._identity()}")
                print(f"Prev. Gal: {prev_Gal._identity()}")
                print(f"Self:      {self._identity()}")
            return False
        # if common attribute, they are overwritten by previous:
        self.__dict__ = {**self.__dict__,**prev_Gal.__dict__}
        if verbose:
            print("Loaded prev. gal.")
        return True
    
    def store_gal(self):
        # store class instance 
        store_class(self,path=self.dill_path_abs())

    # ------------------------------------------------------------------
    # Lazy reconstruction logic
    # ------------------------------------------------------------------
    def _setup(self):
        """Setup all attributes NEEDED FOR COMPUTATION
        that were intentionally removed before serialization.
        """
        print("Unpacking Particle Galaxy ...")
        self.swift_gal
        self.initialise_parts()
        print("... unpacked Particle Galaxy")
        return 
        
    def _unpack(self,verbose=True):
        """Reconstruct attributes AFTER COMPUTATION
        that were intentionally removed before serialization.
        """
        # there is nothing to do for this class
        return 

    ########################     
    @property
    def name(self):
        # arbitrary funct to give name to gal
        # assuming that the simulation stays ~ const
        nm = f"G{self.soap_index}"
        return nm
        
    def verify_snap(self):
        # quick validity check that the snap is correct
        nm_file        = str(self.soap_file.name)
        snap_from_file = nm_file.split("_")[-1].split(".")[0]
        assert self.snap==snap_from_file
    
    def run(self,reload=True):
        upload_successful = False
        if reload:
            upload_successful = self.upload_prev(verbose=True)
        self.setup()
        if not upload_successful:
            self.store_gal()
            
# this function is a wrapper for convenience - it takes the class itself as input
def ReadGal(Gal,verbose=True):
    if not Gal.dill_path_abs().is_file():
        return False
    other_Gal = LoadClass(path=Gal.dill_path_abs(),verbose=verbose,path_base=path_nazgul)
    # If failed, return False
    if not other_Gal: 
        if verbose:
            print("Failed loading of prev.")       
        return False
    # check that loaded Gal would be indeed the same
    if Gal!=other_Gal:
        if verbose:
                print(f"Prev. Gal not equal to self: {other_Gal._identity()==Gal._identity()}")
                print(f"Prev. Gal: {other_Gal._identity()}")
                print(f"Self:      {Gal._identity()}")
        return False
    return other_Gal

def LoadGal(path,if_fail_recompute=True,verbose=True):
    # Try loading galaxy - if fail and fail_recompute==True, try recomputing it
    Gal = LoadClass(path=path,verbose=verbose,path_base=path_nazgul)
    if not Gal and if_fail_recompute:
        full_kwgal = gal_path2kwGal(path)
        Gal        = SimPartGal(**full_kwgal)
    if Gal:
        Gal.unpack()
    return Gal
    
def get_rnd_SPG(sim=std_sim,subsim=std_subsim,
                kw_criteria={"min_mass":min_mass},
               min_z=min_z,
               max_z=max_z,
               colibre_base_path=colibre_base_path
              ):
    """Randomly sample a galaxy from the simulation 
    
    kw_swiftgal = get_rnd_kw_swiftgal(colibre_base_path=colibre_base_path,
                            sim=sim,
                            subsim=subsim,
                            max_z=max_z,
                            min_z=min_z,
                            min_mass=min_mass)
    kw_Gal = {"soap_index":kw_swiftgal["soap_index"],
              "snap":kw_swiftgal["snap"]}
    """
    kw_Gal = get_rnd_kw_gal(sim=sim,subsim=subsim,
                           kw_criteria= kw_criteria,
                           min_z=min_z,
                           max_z=max_z,
                           colibre_base_path=colibre_base_path)
    SPG    = SimPartGal(kw_Gal=kw_Gal,
                       sim=sim,
                       subsim=subsim)
    return SPG

def get_all_SPG(sim=std_sim,subsim=std_subsim,
               colibre_base_path=colibre_base_path,
               kw_criteria= {"min_mass":min_mass},
               min_z=min_z,
               max_z=max_z,
               limit_n=1e3
               ):
    """Get all possible galaxies in the range"""
    all_SPG = []
    all_kw_Gal = get_all_kw_gal(sim=sim,subsim=subsim,
                           kw_criteria= kw_criteria,
                           min_z=min_z,
                           max_z=max_z,
                           colibre_base_path=colibre_base_path)
    
    for kw_Gal in all_kw_Gal:
        SPG    = SimPartGal(kw_Gal=kw_Gal,
                       sim=sim,
                       subsim=subsim)
        all_SPG.append(SPG)
    return all_SPG

def get_vdisp(simpartgal,
              verbose=True,
             **kw_other       # ignored
             ):
    # Get velocity dispersion for a given galaxy 
    # Note: in principle we should recover it similarly as how it's done in get_Gal
    # but I couldn't find a way to do it that way. Instead I re-computed it from the star velocities
    
    #selection_criteria = part_gal.swift_gal.bound_subhalo
    #if verbose:
    #    print(f"As selection criteria taking {selection_criteria.group_name}, ie {selection_criteria.group}")
    #return _get_vdisp(selection_criteria,unit="km/s")
    
    swfg = simpartgal.swift_gal
    
    # Follows from equation 15 of Vandenbroucke et al., 2024
    # https://ftp.strw.leidenuniv.nl/mcgibbon/SOAP.pdf
    # recenter velocities wrt velocity of center of mass OF THE COMPONENT:
    strs_vel = swfg.stars.velocities
    # to do so we take the masses  
    # ~and do not convert in phys. coord.~ No, it's very inefficient
    # -> doens't matter as it would cancel each other
    masses = swfg.stars.masses.value
    # broadcast them to match the 3D velocity matrix
    masses_broad = np.broadcast_to(masses,(3,len(masses))).T
    svc = np.sum(strs_vel*masses_broad,axis=0)/np.sum(masses)
    dv =  strs_vel - svc # swfg.velocity_centre
    """
    # reattach the correct unit -> not needed as we just take the value of masses
    unit_mass  = cosmo_quantity(1,masses.unit_quantity,comoving=masses.comoving,
                               scale_factor=swfg.metadata.a,scale_exponent=0)
    masses_broad = masses_broad*unit_mass"""
    # eq. 15: (note we only consider the diagonal i=i, ie vxx**2,vyy**2,vzz**2)
    vdisp2 = np.sum(dv*dv*masses_broad,axis=0)/np.sum(masses)
    # we get 1D vel disp from eq. 17: 
    vdisp = np.sqrt(np.sum(vdisp2)/3)
    # convert in physical coordinates and km/s
    vdisp_ph = vdisp.to_physical_value("km/s")
    return vdisp_ph   # km/s