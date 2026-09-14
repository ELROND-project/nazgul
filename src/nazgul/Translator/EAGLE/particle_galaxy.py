"""
-> this has to be generalised for any simulation, for now only adapted to EAGLE
Define the galaxy class PartGal (storing all particle data), as well as related helper functions, e.g. to sample galaxies 
"""

import glob
import dill
import h5py
import warnings
import numpy as np
from copy import copy
from pathlib import Path
import astropy.units as u
from decimal import Decimal
import matplotlib.pyplot as plt
#from astropy.stats import sigma_clip
#from functools import cached_property
from multiprocessing import Pool,cpu_count
from astropy.cosmology import Planck13

from python_tools.tools import mkdir
from python_tools.get_res import LoadClass
from python_tools.get_res import load_whatever

from nazgul.Translator.EAGLE.get_gal_indexes import get_gals,get_catpath,get_query
from nazgul.pathfinder import get_gal_dir,get_part_dir,std_sim,std_data_dir,path_nazgul
from nazgul.Translator.EAGLE.fnct import _count_part,_mass_part
from nazgul.Translator.EAGLE.fnct import get_snap,get_z_snap,read_snap_header,get_nfiles

from nazgul.Translator.EAGLE import simsuite_name,simsuite_short_name,part_type_list,check_part_type
from nazgul.Translator.translator import min_z,max_z,min_mass
from nazgul.Translator.particle_galaxy import BasicPartGal,store_class,clip_coord,compute_principal_axis_gen

def gal_path2kwGal(gal_path):
    """From path extract the ALL required inputs for SimPartGal class
    """
    gal_path     = Path(gal_path)

    GnSGn_dir    = gal_path.parent.parent
    snap_dir     = GnSGn_dir.parent
    sim_dir      = snap_dir.parent
    #simsuite_dir = sim_dir.parent
    _Gn,SGn      = GnSGn_dir.name.split("SGn")
    Gn           = _Gn.replace("Gn","")
    snap         = snap_dir.name.replace("snap_","").lstrip("0")
    kw_gal_full  = {}
    kw_gal_full["kw_Gal"] = {"Gn":int(Gn),
                             "SGn": int(SGn)}
    kw_gal_full["snap"]   = str(snap)
    kw_gal_full["sim"]    = str(sim_dir.name)
    # M,center not necessary
    return kw_gal_full

def get_rnd_gal_indexes(sim=std_sim,
                        min_mass = str(min_mass),
                        min_z=str(min_z),
                        max_z=str(max_z),
                        check_prev=True,save_pkl=True,**kwargs_query):
    """Given the simulation, the range of redshift and minimum mass required, 
        returns a random galaxy from the simulation
    """
    min_mass = str(min_mass)
    min_z    = str(min_z)
    max_z    = str(max_z)
    data     = get_gals(sim=sim,min_mass=min_mass,max_z=max_z,min_z=min_z,\
                        check_prev=check_prev,plot=False,save_pkl=save_pkl,
                        **kwargs_query)
    index = np.arange(len(data["z"]))
    rnd_i = np.random.choice(index)
    kw_gal = {}
    for k in data.keys():
        if k=="query" or k=="sim":
            kw_gal[k] = data[k]
        else:
            kw_gal[k] = data[k][rnd_i]

    z        = kw_gal["z"]
    M        = kw_gal["M"]
    kw_Gal   = {"Gn":kw_gal["Gn"],"SGn":kw_gal["SGn"]}
    Centre   = np.array([kw_gal["CMx"],kw_gal["CMy"],kw_gal["CMz"]])

    kwGal    = {"z":z,"kw_Gal":kw_Gal,"sim":sim,"M":M,"Centre":Centre}
    return kwGal

def get_vdisp(simpargal,**kwargs_query):
    snap,Gn,SGn = simpargal.snap,simpargal.Gn,simpargal.SGn
    cat_path = get_catpath(**kwargs_query)
    try:
        query_out = load_whatever(cat_path)
        if not "SVD" in query_out.keys():
            if "Vdisp" in query_out.keys():
                query_out["SVD"] = query_out["Vdisp"]
            else:
                raise RuntimeError(f"Query results do not contain velocity dispersion. Keys:{query_out.keys()}")
    except Exception as e:
        print("DEBUG",kwargs_query)
        print(f"Failed to recover previous cat due to Error: {e}\nRerunning query...")
        if "simsuite" in kwargs_query:
            del kwargs_query["simsuite"]
        myQuery = get_query(**kwargs_query)
        from nazgul.Translator.EAGLE.sql_connect import exec_query
        query_out = exec_query(myQuery)
        with open(cat_path, "wb") as f:
            dill.dump(query_out, f)
        print(f"Saving {cat_path}")

    vdisp_stars_all = query_out["SVD"]*u.km/u.s
    list_Gn   = query_out["Gn"]
    list_SGn  = query_out["SGn"]
    list_z    = query_out["z"]
    list_snap = np.array([int(get_snap(zi)) for zi in list_z])
    vdisp_stars = vdisp_stars_all[(list_Gn==Gn) & (list_SGn==SGn) & (list_snap==int(snap))]
    assert len(vdisp_stars)==1
    return vdisp_stars[0] # km/s

def get_kw_SimPartGal(kw_Gal,sim,simsuite,subsim,data_dir,z,snap,M,Centre,reload):
    
    assert simsuite==simsuite_name

    return {"kw_Gal":kw_Gal,"sim":sim,"snap":snap,"z":z,"M":M,"Centre":Centre}

# index for particle types:
# gas,dm, stars,bh : 0,1,4,5
_kw_ind_type = {0:"gas",
                1:"dm",
                4:"stars",
                5:"bh"}
_kw_type_ind = {}
for k in _kw_ind_type:
    _kw_type_ind[_kw_ind_type[k]] = k

    
class SimPartGal(BasicPartGal):
    """Given the simulation, snap (or z) and galaxy numbers, set up a class
    with all the needed particle properties converted in physical units
    """
    # define name to verify identity
    _type_id = "SimPartGal_"+simsuite_name
    _large_attributes_setup  = ["stars","gas","dm","bh"]
    _large_attributes_unpack = []
    # indexes of particles: gas,dm,stars,bh:
    indexes       = [0,1,4,5]
    simsuite      = simsuite_name
    simsuite_code = simsuite_short_name
    def __init__(self, 
                 kw_Gal, # identity of the galaxy: Gn,SGn
                 sim=std_sim, 
                 data_dir=std_data_dir,
                 z=None,snap=None, # define redshift bin
                 M=None,Centre=None # these can be recovered
                 ):   
        self.sim      = sim
        z,snap        = get_z_snap(z,snap)
        self.snap     = snap
        self.z        = z
        self.Gn       = kw_Gal["Gn"]
        self.SGn      = kw_Gal["SGn"]
        # Input Dir:
        self.part_dir = get_part_dir(snap,sim=sim,simsuite=self.simsuite,data_dir=data_dir)
        # Output dir:
        self.gal_dir  = get_gal_dir(kw_gal=kw_Gal,snap=snap,sim=sim,
                                    simsuite=self.simsuite,data_dir=data_dir)# data and simsuite are for now fixed
        # Mass and Centre can be recovered
        kw_MCntr      = get_kwMCntr(self.Gn,self.SGn,z=z,sim=sim)
        _M            = kw_MCntr["M"]
        _Centre       = kw_MCntr["Centre"]
        if M is not None:
            assert M  == _M
        if Centre is not None:
            assert np.all(Centre == _Centre)
        self.M        = _M
        self.centre   = _Centre
        self.a,self.h,self.boxsize = self.read_snap_header()

        # all paths are dealt as properties
        mkdir(self.gal_dir)        
        #self.run(reload=reload)
        
    @property
    def name(self):
        #Note: this is unique only within the snap
        return f"Gn{self.Gn}SGn{self.SGn}"
        
    @property
    def nfiles(self):
        # n* of files per snapshot:
        return get_nfiles(self.sim)

    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        Id = (self._type_id,self.sim,self.snap,self.name)
        return Id
        
    def __str__(self):
        str_gal = f"Gal {self.Gn}.{self.SGn}"
        str_gal += f", at z={str(np.round(self.z,3))}/snap={self.snap},"
        str_gal += f" with \nN={'%.1E'%Decimal(int(self.N_part))} part.\nof \ntot Mass={'%.1E'%Decimal(float(self.M))} [M_sun]\n"
        str_gal +=f" divided in N \n\
                Stars:{'%.1E'%Decimal(int(self.N_stars))}\n\
                Gas:{'%.1E'%Decimal(int(self.N_gas))}\n\
                DM:{'%.1E'%Decimal(int(self.N_dm))}\n\
                BH:{'%.1E'%Decimal(int(self.N_bh))}\n"
        str_gal +=f"and Mass in \n\
                    Stars:{'%.1E'%Decimal(float(self.M_stars))} [M_sun]\n\
                    Gas:{'%.1E'%Decimal(float(self.M_gas))} [M_sun]\n\
                    DM:{'%.1E'%Decimal(float(self.M_dm))} [M_sun]\n\
                    BH:{'%.1E'%Decimal(float(self.M_bh))} [M_sun]\n"
        return str_gal 

    # ------------------------------------------------------------------
    # Lazy reconstruction logic
    # ------------------------------------------------------------------

    def _setup(self):
        """Setup all attributes NEEDED FOR COMPUTATION
        that were intentionally removed before serialization.
        """
        print("Setting up Particle Galaxy ...")
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
    ########################
    
    @property 
    def cosmo(self):
        # from McAlpine'16:
        # flat ΛCDM cosmology with parameters taken from the Planck mission (Planck Collaboration et al., 2014)
        cosmo = Planck13
        assert cosmo.H0.value/100 == self.h
        return  cosmo
        
    def store_gal(self,update=True):
        # store class instance 
        store_class(self,path=self.dill_path_abs(),update=update)

    def run(self,reload=True,verbose=True):
        if reload:
            self.upload_prev(verbose=verbose)
        # actually store the gal
        self.setup()
        self._count_tot_part()
        self._mass_tot_part()
        self._verify_cnt()
        self.compute_principal_axes()
        self.store_gal(update=True)

    def upload_prev(self,verbose=True):
        prev_Gal = ReadGal(self,verbose=False)
        if prev_Gal is False:
            if verbose:
                print("Failed loading of prev. gal.")
            return
        if prev_Gal != self:
            if verbose:
                print(f"Prev. Gal not equal to self: {not prev_Gal._identity()==self._identity()}")
                print(f"Prev. Gal: {prev_Gal._identity()}")
                print(f"Self:      {self._identity()}")
            return
        k0 = copy(list(self.__dict__.keys()))
        # if common attribute, they are overwritten by previous:
        self.__dict__ = {**self.__dict__,**prev_Gal.__dict__}
        k1 = copy(list(self.__dict__.keys()))
        if k1!=k0 and verbose:
            print("Loaded previosly computed galaxy")
        return
    
    def compute_principal_axes(self):
        self.initialise_parts()
        if not hasattr(self,"principal_axes"):
            self.principal_axes = compute_principal_axes(self)
        return self.principal_axes
        
    @property
    def xy_propr2comov(self):
        # useful to check Center of Mass
        return self.prop2comov("Coordinates") 

    @property
    def m_propr2comov(self):
        # useful to check Center of Mass
        return self.prop2comov("Mass") 

    def prop2comov(self,varType):
        """Proper coordinate/mass to comoving
        physically motivated 
        """
        if varType=="Coordinates":
            aexp = 1
        elif varType=="Mass":
            aexp = 0
        return  1/(self.a**aexp)

    def read_snap_header(self):
        return read_snap_header(z=self.z,snap=self.snap,sim=self.sim)

    def initialise_parts(self):
        part_loaded = True
        if not hasattr(self,"gas"):
            part_loaded = False
            print("Loading particles...")
        self._init_gas()
        self._init_dm()
        self._init_stars()
        self._init_bh()
        if not part_loaded:
            print("... particles loaded")
            part_loaded = True
        return 0

    def _init_gas(self):
        if not hasattr(self,"gas"):
            self.gas = self.read_part(_kw_type_ind["gas"])
    def _init_dm(self):
        if not hasattr(self,"dm"):
            self.dm  = self.read_part(_kw_type_ind["dm"])
    def _init_stars(self):
        if not hasattr(self,"stars"):
            self.stars = self.read_part(_kw_type_ind["stars"])
    def _init_bh(self):
        if not hasattr(self,"bh"):
            self.bh = self.read_part(_kw_type_ind["bh"])
    ## Loading particles #
    ######################
    
    def _get_files(self):
        files = glob.glob(f"{self.part_dir}/snap_*.hdf5")
        if len(files) != self.nfiles:
            raise RuntimeError(f"Expected {self.nfiles} files, got {len(files)}")
        return files

    def build_index(self, itype):
        """
        Return mapping: {file_id: particle_indices}
        """
        files = self._get_files()
        index_map = {}

        for i, fl in enumerate(files):
            with h5py.File(fl, "r") as f:
                grp = f[f"PartType{itype}/GroupNumber"][:]
                sub = f[f"PartType{itype}/SubGroupNumber"][:]
                
                mask = np.logical_and(grp==self.Gn,sub==self.SGn)
                idx  = np.where(mask)[0]
                
                if len(idx) > 0:
                    # sort them to speed up the hdf5 readout:
                    idx = np.sort(idx)
                    index_map[i] = idx
        if index_map == {}:
            # This is possible (although odd) and could be real - if it's an error of the reading of files
            # it will appear when we compare the mass with the exepcted mass. Thus we do not raise an error
            warnings.warn(f"Files do not contain any particle of type {_kw_ind_type[itype]} for this galaxy: {self._identity()}")
        return index_map
        
    @property
    def kw_index_maps(self):
        index_maps = {f'{itype}':self.build_index(itype) for itype in self.indexes}
        return index_maps
        
    def read_part(self, itype):
        """
        Load only selected particles using index_map
        """
        files = self._get_files()
        tasks = [(files[i], idx, itype,self.boxsize,self.centre) for i, idx in self.kw_index_maps[str(itype)].items()]

        nproc = min(cpu_count(), len(files))

        with Pool(nproc) as pool:
            results = pool.map(_load_one_file, tasks)

        # ---- merge ----
        output = {"coords":[],"mass":[]}
        if itype!=1:
            output["smooth"] = []
        if results==[]:
            # create an empty output for missing particles
            output["coords"] = np.array([[],[],[]]).T
            output["mass"] = np.array([])
            if itype!=1:
                output["smooth"] = np.array([])
            warnings.warn(f"This galaxy do not contain particle of type {itype}, i.e. {_kw_ind_type[itype]}")
            return output

        for r in results:
            output["coords"].append(r["coords"])
            output["mass"].append(r["mass"])
            if itype!=1 and "smooth" in r:
                output["smooth"].append(r["smooth"])
        
        coords = np.vstack(output["coords"])
        # Raise error if the particles are "split"
        if np.any(np.std(coords,axis=0)>1.5):
            raise RuntimeError("The galaxy is split, to investigate - likely a conversion error (by h or a, or both)")
            
        output["coords"] = coords
        output["mass"]   = np.hstack(output["mass"])
        if itype != 1:
            output["smooth"] = np.hstack(output["smooth"])
        return output
        
    def _count_tot_part(self):
        """Count total particle number"""
        if not hasattr(self,"N_gas"):
            self.N_gas      = _count_part(self.gas)
        if not hasattr(self,"N_dm"):
            self.N_dm       = _count_part(self.dm)
        if not hasattr(self,"N_stars"):
            self.N_stars    = _count_part(self.stars)
        if not hasattr(self,"N_bh"):
            self.N_bh       = _count_part(self.bh)
        if not hasattr(self,"N_part"):
            self.N_part     =  self.N_gas+self.N_dm+self.N_stars+self.N_bh
        return self.N_part

    def _mass_tot_part(self):
        """Count total particle mass and verify it's the same as 
        the input one
        """
        if not hasattr(self,"M_gas"):
            self.M_gas   = _mass_part(self.gas)
        if not hasattr(self,"M_dm"):
            self.M_dm    = _mass_part(self.dm)
        if not hasattr(self,"M_stars"):
            self.M_stars = _mass_part(self.stars)
        if not hasattr(self,"M_bh"):
            self.M_bh    = _mass_part(self.bh)
        
        if not hasattr(self,"M_tot"):
            self.M_tot   =  self.M_gas+self.M_dm+self.M_stars +self.M_bh
        self.verbose_assert_almost_equal(float(self.M_tot)/float(self.M),1,decimal=3,msg="The summed and the total expected mass differ:")
            
        return self.M_tot
                
    def _verify_cnt(self):
        """verify that the center of mass is indeed correct
        """
        if not hasattr(self,"M_tot"):
            self._mass_tot_part()

        star_wcnt = _weighted_center(self.stars)
        dm_wcnt   = _weighted_center(self.dm)
        gas_wcnt  = _weighted_center(self.gas)
        bh_wcnt   = _weighted_center(self.bh)
        
        center_actual = (
            star_wcnt +
            gas_wcnt  +
            dm_wcnt   +
            bh_wcnt ) / self.M_tot

        center_actual  = np.array(center_actual)
        center_desired = self.centre/self.xy_propr2comov
        self.verbose_assert_almost_equal(center_actual,center_desired,decimal=3,msg="The expected and measured CM centre differ:") 

def _weighted_center(part_type):
    # note: not CoM!
    if part_type["mass"].sum()!=0:
        w_cnt = np.average(part_type["coords"], weights=part_type["mass"], axis=0)*np.sum(part_type["mass"]) 
    else:
        if len(part_type["coords"])!=0:
            raise ValueError("Particle has 0 total mass but !=0 number of particles")
        # there are no particle of this species in this gal
        w_cnt = np.zeros(3)
    return w_cnt

        
def _load_one_file(args):
    fl, indices, itype,boxsize,centre = args
    results = {}
    with h5py.File(fl, "r") as f:
        # Cosmological params (fixed for all type of particles)
        a    = f["Header"].attrs["Time"]
        h    = f["Header"].attrs["HubbleParam"]

        # Coordinates
        #############
        splits = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        
        _coords = []
        for chunk in splits:
            start, end = chunk[0], chunk[-1] + 1
            data = f[f"PartType{itype}/Coordinates"][start:end]
            _coords.append(data[chunk - start])
        coords2scale = np.vstack(_coords)
        # conversion
        cgs_coords  = f[f"PartType{itype}/Coordinates"].attrs["CGSConversionFactor"]
        aexp_coords = f[f"PartType{itype}/Coordinates"].attrs["aexp-scale-exponent"]
        hexp_coords = f[f"PartType{itype}/Coordinates"].attrs["h-scale-exponent"]

        coords = coords2scale * cgs_coords * (a ** aexp_coords) * (h ** hexp_coords)*u.cm.to(u.Mpc)
        # Periodic wrap coordinates around centre.
        # -> boxsize is given in cMpc/h, must correct for both scaling factors
        boxsize           = boxsize*(a**aexp_coords)*(h**hexp_coords)
        centre            = centre*(a**aexp_coords) # given in comoving (but not 1/h)
        results['coords'] = np.mod(coords-centre+0.5*boxsize,boxsize)+centre-0.5*boxsize       
        # Mass
        ######
        if itype != 1:
            _mass2scale = []
            for chunk in splits:
                start, end = chunk[0], chunk[-1] + 1
                data = f[f"PartType{itype}/Mass"][start:end]
                _mass2scale.append(data[chunk - start])
            mass2scale = np.hstack(_mass2scale)
            cgs_mass  = f[f"PartType{itype}/Mass"].attrs["CGSConversionFactor"]
            aexp_mass = f[f"PartType{itype}/Mass"].attrs["aexp-scale-exponent"]
            hexp_mass = f[f"PartType{itype}/Mass"].attrs["h-scale-exponent"]
            mass = mass2scale * cgs_mass * (a ** aexp_mass) * (h ** hexp_mass)*u.g.to(u.Msun)
            results["mass"] = mass
            # Smoothness
            ############
            _smooth2scale = []
            for chunk in splits:
                start, end = chunk[0], chunk[-1] + 1
                data = f[f"PartType{itype}/SmoothingLength"][start:end]
                _smooth2scale.append(data[chunk - start])
            smooth2scale = np.hstack(_smooth2scale)
            # correct smoothing factor
            cgs_smooth  = f[f"PartType{itype}/SmoothingLength"].attrs["CGSConversionFactor"]
            aexp_smooth = f[f"PartType{itype}/SmoothingLength"].attrs["aexp-scale-exponent"]
            hexp_smooth = f[f"PartType{itype}/SmoothingLength"].attrs["h-scale-exponent"]

            smooth = smooth2scale * cgs_smooth * (a ** aexp_smooth) * (h ** hexp_smooth)*u.cm.to(u.Mpc)
            results["smooth"] = smooth
        else:
            dm_mass = f['Header'].attrs.get('MassTable')[1]
            #n_particles = f['Header'].attrs.get('NumPart_Total')[1]
            # Create an array of lenght n_particles each set to dm_mass
            mass2scale = np.ones(len(indices), dtype='f8') * dm_mass
            # Use the conversion factors from the mass entry in the gas particles.
            cgs_massdm  = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
            aexp_massdm = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
            hexp_massdm = f['PartType0/Mass'].attrs.get('h-scale-exponent')
            # Convert to proper/physical mass 
            dm_scale        = cgs_massdm*(a**aexp_massdm)*(h**hexp_massdm)
            results["mass"] = np.multiply(mass2scale, dm_scale, dtype='f8')*u.g.to(u.Msun)

            # DM has no smoothing scale
            results["smooth"] = None
    print("DEBUG - particle_galaxy, _load_one_file")
    # add this inside _load_one_file before return:
    result_size = sum(v.nbytes for v in results.values() if hasattr(v,"nbytes"))
    print(f"[WORKER] returning {result_size/1024**2:.1f} MB", flush=True)
    return results

# this function is a wrapper for convenience - it takes the class itself as input
def ReadGal(Gal,verbose=True):
    other_Gal = ReadGalNoUnpack(Gal,verbose=verbose)
    if other_Gal:
        other_Gal.unpack()
    return other_Gal

def ReadGalNoUnpack(Gal,verbose=True):
    "This Reads store galaxy but doesn't unpack it"
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
            print(f"Prev. Gal not equal to self: {not other_Gal._identity()==Gal._identity()}")
            print(f"Prev. Gal: {other_Gal._identity()}")
            print(f"Self:      {Gal._identity()}")
        return False
    return other_Gal
    
def LoadGal(path,if_fail_recompute=True,verbose=True):
    # Try loading galaxy - if fail and fail_recompute==True, try recomputing it
    Gal = LoadClass(path=path,verbose=verbose,path_base=path_nazgul)
    if not Gal and if_fail_recompute:
        kwGal   = gal_path2kwGal(path)
        Gal     = SimPartGal(**kwGal)
    if Gal:
        Gal.unpack()
    return Gal

# to simplify the input: given the sim, z, and GnSgn, 
# we get the mass and center of the galaxy for input of SimPartGal 
def get_myCat(Gn,SGn,z,sim,min_mass=min_mass,dz=0.05,**kwargs_query):
    min_z=str(z-0.05)
    max_z=str(z+0.05)

    # try first finding the exact one
    cat_path = get_catpath(min_mass=min_mass,\
                           min_z=min_z,max_z=max_z,
                           sim=sim)
    if Path.is_file(cat_path):
        myCat = load_whatever(cat_path)
    else:
        # try all the other
        found = False
        # now quite a desperate attempt
        for pkl in Path(cat_path).parent.glob("*.pkl"):
            cat = load_whatever(pkl)
            if np.any(np.abs(cat["z"]-z)<dz):
                zz      = min(cat["z"], key=lambda x:abs(x-z))
                indexx  = np.where((cat["Gn"]==Gn) & (cat["SGn"]==SGn) & (cat["z"]==zz))[0]
                if len(indexx)==1:
                    found  = True
                    myCat = cat
                    index  = indexx[0]
                    break
        if not found:
            # get_gals is slow-ish but safe (require internet access!)
            # safe-ish since we now hard write the min_mass - but should be fine
            # TODO: find a safer approach
    
            myCat   = get_gals(sim=sim,
                                min_mass=min_mass,min_z=min_z,max_z=max_z,
                                save_pkl=False,check_prev=True,verbose=False,plot=False,**kwargs_query)
    
    z        = min(myCat["z"], key=lambda x:abs(x-z))
    index    = np.where((myCat["Gn"]==Gn) & (myCat["SGn"]==SGn) & (myCat["z"]==z))[0]
    if len(index)>1:
        raise RuntimeError(f"Found multiple galaxies with same z, G and SGn")
    index    = index[0] 
    return myCat,index
    
def get_kwMCntr(Gn,SGn,sim=std_sim,
                z=None,snap=None):
    z,snap       = get_z_snap(z,snap)
    myCat,index  = get_myCat(Gn,SGn,z,sim)
    Centre       = np.array([myCat["CMx"],myCat["CMy"],myCat["CMz"]]).T[index]
    kwMCntr      = {"M":myCat["M"][index],
                    "Centre":Centre}
    return kwMCntr

def get_rnd_SPG(sim=std_sim,min_mass=min_mass,min_z=min_z,max_z=max_z,
               check_prev=True,save_pkl=True):
    """Randomly sample a galaxy from the simulation 
    """

    kwGal   = get_rnd_gal_indexes(sim=sim,min_mass=min_mass,
                                   min_z=min_z,max_z=max_z,
                                   check_prev=check_prev,save_pkl=save_pkl)

    SPG      = SimPartGal(**kwGal)
    return SPG

def get_all_SPG(sim=std_sim,min_mass=min_mass,min_z=min_z,max_z=max_z,
               check_prev=True,save_pkl=True,limit_n=1e3,**kwargs_query):
    """Get all possible galaxies in the range"""
    min_mass = str(min_mass)
    min_z    = str(min_z)
    max_z    = str(max_z)
    data     = get_gals(sim=sim,min_mass=min_mass,max_z=max_z,min_z=min_z,
                        check_prev=check_prev,plot=False,save_pkl=save_pkl,**kwargs_query)
    n_data   = len(data["z"])
    if n_data>limit_n:
        raise RuntimeError(f"Too many galaxy loaded (N={n_data}). Aborted")
    all_SPG  = []
    for i in range(n_data):
        kw_gal = {}
        for k in data.keys():
            if k=="query" or k=="sim":
                kw_gal[k] = data[k]
            else:
                kw_gal[k] = data[k][i]
        z        = kw_gal["z"] 
        M        = kw_gal["M"] 
        kw_Gal   = {"Gn":kw_gal["Gn"],"SGn":kw_gal["SGn"]}
        Centre   = np.array([kw_gal["CMx"],kw_gal["CMy"],kw_gal["CMz"]]) 
        
        kwGal    = {"z":z,"kw_Gal":kw_Gal,"sim":sim,"M":M,"Centre":Centre}
        SPG      = SimPartGal(**kwGal)
        all_SPG.append(SPG)
    return all_SPG
    
def Gal2MXYZ(Gal): 
    """Given the galaxy, return Masses (in Msun) and
    XY coords. of particles in kpc  centered around center
    """
    # Particle masses
    Mstar = Gal.stars["mass"] # Msun
    Mgas  = Gal.gas["mass"]   # Msun
    Mdm   = Gal.dm["mass"]    # Msun
    Mbh   = Gal.bh["mass"]    # Msun
    Ms    = np.concatenate([Mstar,Mgas,Mdm,Mbh])*u.Msun #Msun

    # Particle pos
    Xstar,Ystar,Zstar =  np.transpose(Gal.stars["coords"]) # Mpc
    Xgas,Ygas,Zgas    =  np.transpose(Gal.gas["coords"])   # Mpc
    Xdm,Ydm,Zdm       =  np.transpose(Gal.dm["coords"])    # Mpc
    Xbh,Ybh,Zbh       =  np.transpose(Gal.bh["coords"])    # Mpc
    Xs = np.concatenate([Xstar,Xgas,Xdm,Xbh])*u.Mpc.to("kpc")*u.kpc #kpc
    Ys = np.concatenate([Ystar,Ygas,Ydm,Ybh])*u.Mpc.to("kpc")*u.kpc #kpc
    Zs = np.concatenate([Zstar,Zgas,Zdm,Zbh])*u.Mpc.to("kpc")*u.kpc #kpc

    # center around the center of the galaxy
    # center of mass is given in Comoving coord 
    # see https://arxiv.org/pdf/1510.01320 D.23 
    # ->  it's given in cMpc (not cMpc/h) fsr
    Cx,Cy,Cz = Gal.centre*u.Mpc.to("kpc")*u.kpc/(Gal.xy_propr2comov) # (now) kpc 
    
    Xs -= Cx
    Ys -= Cy
    Zs -= Cz
    
    # clip particle outliers AFTER recentering
    Ms,Xs,Ys,Zs = clip_coord(Ms,Xs,Ys,Zs)
    
    return Ms, Xs,Ys,Zs

def Gal2MXYZ_part(Gal,part_type): 
    """Given the galaxy, return Masses (in Msun) and
    XY coords. of a specific particle type in kpc  centered around center
    """
    part_type = check_part_type(part_type)
    part = getattr(Gal,part_type) 
    # Particle masses
    
    Ms = part["mass"]*u.Msun #Msun
    
    # Particle pos
    Xs,Ys,Zs =  np.transpose(part["coords"]) *u.Mpc.to("kpc")*u.kpc #kpc

    # center around the center of the galaxy
    # center of mass is given in Comiving coord 
    # see https://arxiv.org/pdf/1510.01320 D.23 
    # ->  it's given in cMpc (not cMpc/h) fsr
    Cx,Cy,Cz = Gal.centre*u.Mpc.to("kpc")*u.kpc/(Gal.xy_propr2comov) # (now) kpc 
    
    Xs -= Cx
    Ys -= Cy
    Zs -= Cz
    # clip particle outliers AFTER recentering
    Ms,Xs,Ys,Zs = clip_coord(Ms,Xs,Ys,Zs)
    
    return Ms,Xs,Ys,Zs
    
def compute_principal_axes(Gal):
    """
    Compute the principal axes from inertial tensor.
    The ratio c/b is used by Vyvere et al. '22 to discard elliptical or lenticular galaxies
    """
    
    Mstar,Xstar,Ystar,Zstar = Gal2MXYZ_part(Gal,part_type = "stars")
    principal_axes = compute_principal_axis_gen(m=Mstar.value,
                                                x = Xstar.value,
                                                y = Ystar.value,
                                                z = Zstar.value)
    # ensure the memory is freed
    del Mstar,Xstar,Ystar,Zstar
    return principal_axes

        
# The following should be done in the test_particle_galaxy
"""    
# for debug:
if __name__=="__main__":

    SPG = get_rnd_SPG()
    fig, ax = plt.subplots(3)
    nx = 100
    for name,part in zip(["stars","dm","gas"],[SPG.stars,SPG.dm,SPG.gas]):
        coords = part["coords"]
        x,y,z  = coords.T
        print(np.std(coords,axis=0))
        ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
        ax[1].hist(y,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
        ax[2].hist(z,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
    ax[0].set_xlabel("X [kpc]")
    ax[2].set_xlabel("Z [kpc]")
    ax[1].set_xlabel("Y [kpc]")
    ax[2].legend()
    namefig = f"tmp/hist_by_hand_parts.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close(fig)
    print(f"Saved {namefig}") 
    
    print("Testing dens map")
    from test_proj_part_hist import get_dens_map_rotate_hist
    NG.proj_dir = "./tmp/proj_part"
    mkdir(NG.proj_dir)
    dens_Ms_kpc2,radius,dP,cosmo = get_dens_map_rotate_hist(Gal=NG,pixel_num=100j,
                                                            z_source_max=5,verbose=True,plot=True)
"""
