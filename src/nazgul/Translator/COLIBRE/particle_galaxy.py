# copied from remade_gal.py
# adapted for COLIBRE sim.

# get random swift galaxy from get_rand_gal.py

import dill
import warnings
import unyt as u  # package used by swiftsimio to provide physical units
import numpy as np
from pathlib import Path
from functools import cached_property

from python_tools.tools import mkdir,ensure_unit
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
    return Xpart.to_astropy(),Ypart.to_astropy(),Zpart.to_astropy()
    
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
    return Mpart.to_astropy()

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
    
    #Convert all to astropy for convenience -> done in advance
    
    #Ms = Ms.to_astropy()
    #Xs = Xs.to_astropy()
    #Ys = Ys.to_astropy()
    #Zs = Zs.to_astropy()
    
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
    try:
        part = getattr(Gal,part_type) 
    except:
        Gal.run()
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
    
    Xs -= ensure_unit(X_cm,Xs.unit)
    Ys -= ensure_unit(Y_cm,Ys.unit)
    Zs -= ensure_unit(Z_cm,Zs.unit)
    
    #Convert all to astropy for convenience -> done in advance
    #Ms = Ms.to_astropy()
    #Xs = Xs.to_astropy()
    #Ys = Ys.to_astropy()
    #Zs = Zs.to_astropy()
    return Ms,Xs,Ys,Zs

def get_kw_SimPartGal(kw_Gal,sim,simsuite,subsim,data_dir,z,snap,M,Centre,reload):
    assert simsuite==simsuite_name
    return {"kw_Gal":kw_Gal,"sim":sim,"subsim":subsim}

# basically a wrapper for swift galaxies
class SimPartGal(BasicPartGal):
    """Particle-based galaxy extracted from a hydrodynamical simulation snapshot.

    Wraps a swiftgalaxy object identified by (sim, subsim, snap, soap_index) and
    exposes its particle species (stars, gas, dark matter, black holes) plus their
    aggregate masses/counts. Particle data is heavy, so it is loaded lazily via
    `run` / `setup`, stripped before serialization (see `_large_attributes_setup`),
    and reloaded on demand.
    """

    # define name to verify identity
    _type_id = f"SimPartGal_{simsuite_name}"

    # Heavy, I/O-backed attributes: dropped before pickling, reloaded by _setup().
    # NOTE: only the backing field "_swift_gal" belongs here, not the "swift_gal"
    # property itself -- hasattr() on a property always succeeds (it just runs the
    # getter), so listing "swift_gal" here made every _needs_setup() check
    # silently trigger a full reload as a side effect.
    _large_attributes_setup = ["_swift_gal", "stars", "gas", "dark_matter", "black_holes"]
    _large_attributes_unpack = []

    simsuite = simsuite_name
    simsuite_code = simsuite_short_name

    # species -> where to read counts/masses from on the swift galaxy object
    _SPECIES_CONFIG = {
        "stars":       dict(count_attr="N_stars", mass_attr="M_stars", mass_field="masses"),
        "gas":         dict(count_attr="N_gas",   mass_attr="M_gas",   mass_field="masses"),
        "dark_matter": dict(count_attr="N_dm",    mass_attr="M_dm",    mass_field="masses"),
        # black holes: dynamical mass is used rather than particle mass
        "black_holes": dict(count_attr="N_bh",    mass_attr="M_bh",    mass_field="dynamical_masses"),
    }

    def __init__(self, kw_Gal, sim=std_sim, subsim=std_subsim):
        # kw_Gal: soap_index, and either snap or z
        self.soap_index  = kw_Gal["soap_index"] #self.swift_gal.halo_catalogue.soap_index
        self.z,self.snap = get_z_snap(z=kw_Gal.get("z",None),
                            snap=kw_Gal.get("snap",None),
                                     sim=sim,subsim=subsim)
        
        self.sim         = Path(sim)
        self.subsim      = Path(subsim)

        # Access (but don't store separately) the swift galaxy just to pull
        # small metadata now; particle arrays themselves are loaded later,
        # lazily, via initialise_parts().
        sg = self.swift_gal
        self.soap_file = Path(sg.halo_catalogue.soap_file)
        # e.g. '/cosma8/data/dp004/colibre/Runs/L0025N0752/THERMAL_AGN_m5/SOAP-HBT/halo_properties_0127.hdf5'

        self.a = sg.metadata.a
        self.verbose_assert_almost_equal((1 / self.a) - 1, self.z, msg="Redshifts")
        self.verify_snap()

        self.gal_dir  = get_gal_dir(kw_Gal,snap=self.snap,
                                    sim=self.sim,subsim=self.subsim,
                                    simsuite=self.simsuite)
        mkdir(self.gal_dir)

        # total mass (bound-subhalo definition)
        #SphOverDens = self.swift_gal.halo_catalogue.spherical_overdensity_500_crit
        #self.M_tot  = SphOverDens.total_mass.to_physical_value("Msun")[0] #Msun
        bound_subhalo = sg.halo_catalogue.bound_subhalo
        self.M_tot    = bound_subhalo.total_mass.to_physical_value("Msun")[0]  # Msun

        # coordinates of the centre
        self.centre = sg.centre.to_physical_value("Mpc")

        # cosmo is a bottleneck and light to store: compute (and cache) it once
        self.cosmo

    # ------------------------------------------------------------------
    # Swift galaxy access
    # ------------------------------------------------------------------
    @property
    def swift_gal(self):
        try:
            return self._swift_gal
        except AttributeError:
            # only fetched the first time it's accessed
            self._swift_gal = get_swiftgal(sim=self.sim,
                                     subsim=self.subsim,
                                     snap=self.snap,
                                     soap_index=self.soap_index)
            return self._swift_gal

    @swift_gal.deleter
    def swift_gal(self):
        del self._swift_gal

    @cached_property
    def cosmo(self):
        return self.swift_gal.metadata.cosmology

    def initialise_parts(self):
        """Load particle species and their aggregate mass/count.

        Idempotent: anything already present (e.g. reloaded from a previous
        run) is left untouched rather than recomputed.
        """
        sg = self.swift_gal

        for species in self._SPECIES_CONFIG:
            if not hasattr(self, species):
                setattr(self, species, getattr(sg, species))

        for species, cnfg in self._SPECIES_CONFIG.items():
            particles = getattr(self, species)
            if not hasattr(self, cnfg["mass_attr"]):
                masses = getattr(particles, cnfg["mass_field"])
                setattr(self, cnfg["mass_attr"], np.sum(masses.to_physical().in_units(u.Msun)))
            if not hasattr(self, cnfg["count_attr"]):
                setattr(self, cnfg["count_attr"], len(particles.particle_ids))

        if not hasattr(self, "N_part"):
            self.N_part = sum(getattr(self, cnfg["count_attr"]) for cnfg in self._SPECIES_CONFIG.values())
        if not hasattr(self, "M"):
            self.M = sum(getattr(self, cnfg["mass_attr"]) for cnfg in self._SPECIES_CONFIG.values())
        return 0

    # ------------------------------------------------------------------
    # Class structure
    # ------------------------------------------------------------------
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        # Returns tuple to identify uniquely this galaxy
        Id = (self._type_id,self.sim,self.subsim,
            self.snap,self.soap_index)
        return Id

    def ReadClass(self, cl):
        return ReadGal(cl)

    def upload_prev(self, verbose=True):
        prev_gal = self.ReadClass(self)
        if prev_gal is False:
            if verbose:
                print("Failed loading of prev. gal.")
            return False
        if prev_gal != self:
            if verbose:
                print(f"Prev. Gal not equal to self: {prev_gal._identity() == self._identity()}")
                print(f"Prev. Gal: {prev_gal._identity()}")
                print(f"Self:      {self._identity()}")
            return False
        # common attributes are overwritten by the previous, already-computed version
        self.__dict__ = {**self.__dict__, **prev_gal.__dict__}
        if verbose:
            print("Loaded prev. gal.")
        return True

    def store_gal(self):
        # store class instance
        store_class(self, path=self.dill_path_abs())

    # ------------------------------------------------------------------
    # Lazy reconstruction logic
    # ------------------------------------------------------------------
    def _setup(self):
        """Load everything needed for computation that was stripped before
        serialization: the swift galaxy and its particle species.
        """
        print("Setting up Particle Galaxy ...")
        self.swift_gal
        self.initialise_parts()
        print("... Particle Galaxy set up")
        return

    def _unpack(self,verbose=True):
        """Reconstruct attributes AFTER COMPUTATION
        that were intentionally removed before serialization.
        """
        # nothing to do: this class has no post-computation attributes to restore
        return

    @property
    def name(self):
        # arbitrary function to give a name to the galaxy
        # assuming that the simulation stays ~constant
        return  _get_gal_name(self.soap_index)

    def verify_snap(self):
        # quick validity check that the snap is correct
        nm_file        = str(self.soap_file.name)
        snap_from_file = nm_file.split("_")[-1].split(".")[0]
        err_msg        = f"snap mismatch: expected {self.snap}, soap file suggests {snap_from_file}"
        assert self.snap == snap_from_file, (err_msg)

    def run(self, reload=True):
        """Ensure the galaxy is fully set up, reusing prior work whenever possible.

        - If this instance already has everything setup needs (e.g. `run` was
          already called earlier in this session), this is a no-op.
        - Otherwise, optionally reload a previously stored instance from disk
          (`reload=True`), then perform (only) the missing setup, and store the
          result if it wasn't already on disk.
        """
        if not self._needs_setup():
            return self

        upload_successful = False
        if reload:
            upload_successful = self.upload_prev(verbose=True)

        self.setup()

        if not upload_successful:
            self.store_gal()

def get_gal_name(kw_gal):
    soap_index = kw_gal["soap_index"]
    return  _get_gal_name(soap_index)
    
def _get_gal_name(soap_index):
    return f"G{soap_index}"
            
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