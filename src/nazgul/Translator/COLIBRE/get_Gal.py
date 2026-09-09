# extract from a given simulation suite a set of plausible lenses
# and select randomly one of these
import os
import gc
import numpy as np
from pathlib import Path

import unyt as u  # package used by swiftsimio to provide physical units
from swiftgalaxy import SWIFTGalaxy, SOAP
from swiftsimio import SWIFTDataset, cosmo_quantity

from nazgul.configurations import min_z,max_z,min_mass
from nazgul.Translator.COLIBRE import simsuite_name,sim,subsim

# the following is, so far, not a variable
colibre_base_path = Path("/cosma8/data/dp004/colibre/Runs/")

# later on consider if this is the one we want:
std_sim = Path(sim[0])
std_subsim = Path(subsim[str(std_sim)][0])
# note: the following is for now used both as the path for the output list in the colibre dataset, as well as in the Ringbearer data structure.
def find_simulation_dir(sim=std_sim,
                        subsim=std_subsim):
    # allows for more flexibility
    simulation_dir =Path(str(sim)+"/"+str(subsim))
    return simulation_dir
#std_simulation_dir = std_sim/std_subsim
#print("Todo: check if this is the correct simulation suite:",std_simulation_dir)

##### Snap/z map ####
def get_snap_z_map(sim=std_sim,
                    subsim=std_subsim):
    simulation_dir = find_simulation_dir(sim=sim,subsim=subsim)
    snap_z_map=colibre_base_path/simulation_dir/"output_list.txt"
    return snap_z_map
    
def Read_ZSnap(file_name):
    z = []
    snap_snip = []
    with open(file_name, 'r') as data:
        for line in data.readlines():
            if line[0]== "#":
                continue
            p = line.split(", ")
            z.append(float(p[0]))
            snap_snip.append(str(p[1]).replace("\n",""))
    return np.array(z),snap_snip

def _get_snap(snap):
    if type(snap)!=str:
        snap = f'{int(snap):04d}'
    assert len(snap)==4
    return snap
    
def get_kw_snap_z(sim=std_sim,
                  subsim=std_subsim):
    snap_z_map = get_snap_z_map(sim=sim,subsim=subsim)
    z_all,snap_snip = Read_ZSnap(snap_z_map)
    # we want to ignore snipshots (don't remember why, but most likely there are no soap gals)
    z,snap= [],[]
    kw_snap_z = {}
    for i in range(len(z_all)):
        if snap_snip[i]=="Snapshot":
            z.append(z_all[i])
            snap.append(_get_snap(i))
            kw_snap_z[snap[-1]] = z[-1]
        elif snap_snip[i]=="Snipshot":
            _ = "Ignore Snipshots"
            #print("Ignore Snipshots")
    return kw_snap_z

def get_kw_z_snap(sim=std_sim,
                    subsim=std_subsim):
    kw_snap_z = get_kw_snap_z(sim=sim,subsim=subsim)
    #inverted kw
    kw_z_snap = {}
    for k in kw_snap_z:
        kw_z_snap[kw_snap_z[k]] = k
    return kw_z_snap
    
def get_z(snap,sim=std_sim,subsim=std_subsim):
    kw_snap_z = get_kw_snap_z(sim=sim,subsim=subsim)
    # format snap to integer
    int_snap = int(str(snap).lstrip("0"))
    # format it back to str w. leading 0 to match kw_snap_z
    snap = _get_snap(int_snap)
    return kw_snap_z[snap]
    
def get_snap(z,sim=std_sim,subsim=std_subsim):
    kw_z_snap = get_kw_z_snap(sim=sim,subsim=subsim)
    # consider a continous z instead of the discreet version
    # works for discreet z as well
    key_z = min(kw_z_snap.keys(),key=lambda k:np.abs(k-float(z)))
    snap  = f'{kw_z_snap[key_z]}'
    return snap

def get_z_snap(z=None,snap=None,sim=std_sim,subsim=std_subsim):
    if z is None and snap is None:
        raise UserWarning("Give either z or snap")
    if z is None:
        z = get_z(snap,sim=sim,subsim=subsim)
    else:
        snap = get_snap(z,sim=sim,subsim=subsim)
    return z,snap
#####################       

def get_soap_cat(snap,
                 sim=std_sim,
                 subsim=std_subsim,
                 colibre_base_path=colibre_base_path):
    simulation_dir = find_simulation_dir(sim=sim,subsim=subsim)
    snap = _get_snap(snap)
    soap_catalogue_file = os.path.join(
        colibre_base_path,
        simulation_dir,
        f"SOAP-HBT/halo_properties_{snap}.hdf5",
    )
    #print("soap_catalogue_file",soap_catalogue_file)
    return soap_catalogue_file

def get_virtual_snapfile(snap,
                         sim=std_sim,
                         subsim=std_subsim,
                         colibre_base_path=colibre_base_path,
                         **kwargs):
    virtual_snapshot_file = os.path.join(
        str(colibre_base_path),
        str(sim),
        str(subsim),
        f"SOAP-HBT/colibre_with_SOAP_membership_{snap}.hdf5")
    #print("virtual_snapshot_file",virtual_snapshot_file)
    return virtual_snapshot_file


def get_rnd_snap(sim=std_sim,
                 subsim=std_subsim,
                 max_z=max_z,
                 min_z=min_z):
    """By sampling uniformily z"""
    max_z = float(max_z)
    min_z = float(min_z)
    z_rnd = np.random.uniform(min_z,max_z)
    snap  = get_snap(z_rnd,sim=sim,subsim=subsim)
    #z     = get_z(snap)
    return snap

def get_all_snap(sim=std_sim,
                 subsim=std_subsim,
                 max_z=max_z,
                 min_z=min_z):
    max_z = float(max_z)
    min_z = float(min_z)
    #note: snaps are "inverted" wrt to z (z>>, snap<<)
    min_snap  = get_snap(min_z,sim=sim,subsim=subsim)
    max_snap  = get_snap(max_z,sim=sim,subsim=subsim)
    # the following ignored the fact that we also have snipshot which
    # are ignored
    #all_snap_int  = np.arange(int(max_snap),int(min_snap)+1)
    # snap has to be a string
    #all_snap   = [_get_snap(snap) for snap in all_snap_int]
    kw_z_snap = get_kw_z_snap(sim=sim,subsim=subsim)
    zs        = np.array(list(kw_z_snap.keys()))
    i_zs      = np.logical_and(zs>min_z,zs<max_z)
    all_snap  = [kw_z_snap[zi] for zi in zs[i_zs]]
    if all_snap==[]:
        raise ValueError(f"min_z={min_z} and max_z={max_z} do not correspond to any valid snapshot")
    return all_snap
    
# Selection functions:
######################
def select_minmass(selection_criteria,min_mass,scale_factor,verbose=True):
    """
    Return index list of candidates galaxies with mass (bound_halo tot mass) higher than min_mass
    """
    # We have to define based on what we select the mass - or if we want other criteria
    #selection_criteria = sd.spherical_overdensity_200_crit
    #m200c = selection_criteria.total_mass.to_physical_value("Msun")
    unit       = "Msun"
    Mboundhalo = selection_criteria.total_mass.to_physical_value(unit)
    #if verbose:
    #    print("And considering its total_mass")
    #    print(m200c)
    min_mass = cosmo_quantity(min_mass,u.Msun, comoving=False, scale_factor=scale_factor, scale_exponent=0).to_physical_value(unit)
    if verbose:
        print(f"and selecting only galaxies w. mass more than {min_mass} {unit}")
    #candidates_gal = np.argwhere(Mboundhalo> min_mass).squeeze()
    candidates_gal = Mboundhalo> min_mass
    return candidates_gal
    
def select_minmass_stars(selection_criteria,min_mass_stars,scale_factor,verbose=True):
    """
    Return index list of candidates galaxies with stellar mass (bound_halo stellar mass) higher than min_mass_stars
    """
    unit = "Msun"
    Mboundhalo_stars = selection_criteria.stellar_mass.to_physical_value(unit)
    #if verbose:
    #    print("And considering its total_mass")
    #    print(m200c)
    min_mass_stars = cosmo_quantity(min_mass_stars,u.Msun, comoving=False, scale_factor=scale_factor, scale_exponent=0).to_physical_value(unit)
    if verbose:
        print(f"and selecting only galaxies w. stellar mass more than {min_mass_stars} {unit}")
    candidates_gal = Mboundhalo_stars> min_mass_stars
    return candidates_gal

def _get_vdisp(selection_criteria,unit = "km/s"):
    if unit=="km/s":
        unit2 = "km**2/s**2"
    else:
        raise NotImplementedError(f"To implement the square of the unit {unit}") 
    # veldisp2_matrix = 6 dim matrix of vel disp - sigma_xx^2,sigma_yy^2,sigma_zz^2,sigma_xy^2,sigma_xz^2,sigma_yz^2 (note the square!)
    veldisp2_matrix = selection_criteria.stellar_velocity_dispersion_matrix.to_physical_value(unit2)
    # our vel disp is expected to be sigma_v = sqrt( (sigma_xx^2+sigma_yy^2+sigma_zz^2)/3):
    # see eq 17 Vandenbroucke et al
    # https://ftp.strw.leidenuniv.nl/mcgibbon/SOAP.pdf
    vel_disp = (np.sum(veldisp2_matrix[:,:3],axis=1)/3)**.5
    return vel_disp
    
def select_minveldisp(selection_criteria,min_vel_disp,scale_factor,verbose=True):
    """
    Return index list of candidates galaxies with velocity dispersion (bound_halo tot mass) higher than min_mass
    """
    unit = "km/s"
    vel_disp = _get_vdisp(selection_criteria,unit)
    min_vel_disp = cosmo_quantity(min_vel_disp,u.km/u.s,comoving=False,scale_factor=scale_factor, scale_exponent=0).to_physical_value(unit)
    if verbose:
        print(f"and selecting only galaxies w. velocity dispersion more than {min_vel_disp} {unit}")
    candidates_gal = vel_disp>min_vel_disp
    return candidates_gal
    
def select_half_mass_radius(selection_criteria,min_hmr,scale_factor,verbose=True):
    """
    Return index list of candidates galaxies with half (total) mass radius (bound_halo half_mass_radius_total) higher than min_hmr
    """
    unit ="kpc"
    half_mass_radius = selection_criteria.half_mass_radius_total.to_physical_value(unit)
    min_hmr = cosmo_quantity(min_hmr,u.kpc,comoving=False,scale_factor=scale_factor, scale_exponent=0).to_physical_value(unit)
    if verbose:
        print(f"and selecting only galaxies w. half-mass radius more than {min_hmr} {unit}")
    candidates_gal = half_mass_radius>min_hmr 
    return candidates_gal
    
######################

def get_gal_candidates(snap,
                       sim=std_sim,
                       subsim=std_subsim,
                        max_z=max_z,
                        min_z=min_z,
                        kw_criteria={"min_mass":float(min_mass)},
                        colibre_base_path=colibre_base_path,
                        soap_catalogue_file=None,
                        verbose=True):


    if soap_catalogue_file is None:
        soap_catalogue_file = get_soap_cat(snap=snap,
                                           sim=sim,
                                           subsim=subsim,
                                           colibre_base_path=colibre_base_path)
    swift_dataset = SWIFTDataset(soap_catalogue_file)
    selection_criteria = swift_dataset.bound_subhalo
    scale_factor       = swift_dataset.metadata.a
    # free memory
    del swift_dataset
    gc.collect()
    if verbose:
        print(f"As selection criteria taking {selection_criteria.group_name}, ie {selection_criteria.group}")
    # We have to define based on what we select - or if we want other criteria
    #selection_criteria = sd.spherical_overdensity_200_crit
    #m200c = selection_criteria.total_mass.to_physical_value("Msun")

    list_candidates_gal = []
    if "min_mass" in kw_criteria.keys():
        candidates_gal = select_minmass(selection_criteria=selection_criteria,
                                        min_mass = kw_criteria["min_mass"],
                                        scale_factor=scale_factor,
                                        verbose=verbose)
        list_candidates_gal.append(candidates_gal)
    if 'min_mass_stars' in kw_criteria.keys():
        candidates_gal = select_minmass_stars(selection_criteria=selection_criteria,
                                        min_mass_stars = kw_criteria["min_mass_stars"],
                                        scale_factor=scale_factor,
                                        verbose=verbose)
        list_candidates_gal.append(candidates_gal)
    if 'min_vel_disp' in kw_criteria.keys():
        candidates_gal = select_minveldisp(selection_criteria=selection_criteria,
                                           min_vel_disp= kw_criteria["min_vel_disp"],
                                           scale_factor=scale_factor,
                                           verbose=verbose)
        list_candidates_gal.append(candidates_gal)
    if 'min_hmr' in kw_criteria.keys():
        candidates_gal = select_half_mass_radius(selection_criteria=selection_criteria,
                                        min_hmr = kw_criteria["min_hmr"],
                                        scale_factor=scale_factor,
                                        verbose=verbose)
        list_candidates_gal.append(candidates_gal)
    if len(list_candidates_gal)>1:
        comb_candidates_gal = np.logical_and(*list_candidates_gal)
        comb_candidates_gal_index = np.argwhere(comb_candidates_gal).squeeze()
    else:
        comb_candidates_gal_index = np.argwhere(list_candidates_gal[0])

    if verbose:
        print(f"Found N={len(comb_candidates_gal_index)} candidates") 
    return comb_candidates_gal_index
    
def get_rnd_kw_gal(sim=std_sim,
                        subsim=std_subsim,
                        max_z=max_z,
                        min_z=min_z,
                        kw_criteria={"min_mass":float(min_mass)},
                        colibre_base_path=colibre_base_path,
                        soap_catalogue_file=None,
                        verbose=True):
    """ Return inputs for the SimPartGal of COLIBRE for a random
    galaxy within a range of redshift and respecting the criteria
    """
    min_z = float(min_z)
    max_z = float(max_z)
    if verbose:
        print("We randomly select a redshift/snap:")
    snap = get_rnd_snap(sim=sim,
                            subsim=subsim,
                            max_z=max_z,
                            min_z=min_z)
    if verbose:
        print(f"Snap selected:{snap}")
        
    z = get_z(snap,sim=sim,subsim=subsim)
    if verbose:
        print(f"z={z}")
        
    if soap_catalogue_file is None:
        soap_catalogue_file = get_soap_cat(snap=snap,
                                           sim=sim,
                                           subsim=subsim,
                                           colibre_base_path=colibre_base_path)

    candidates_gal = get_gal_candidates(snap=snap,
                                        sim=sim,
                                        subsim=subsim,
                                        soap_catalogue_file=soap_catalogue_file,
                                        kw_criteria=kw_criteria,
                                        colibre_base_path=colibre_base_path,
                                        verbose=verbose)
                                        
    chosen_halo_index = np.random.choice(candidates_gal)

        
    # for reproducibility
    inputs     = {"sim":sim,
                  "subsim":subsim,
                  "max_z":max_z,
                  "min_z":min_z,
                  "min_mass":min_mass,
                  "colibre_base_path":colibre_base_path,
                  "soap_catalogue_file":soap_catalogue_file}
    
    kw_rnd_gal = {"soap_catalogue_file":soap_catalogue_file,
                  "soap_index":chosen_halo_index,
                  #"candidates":candidates,
                  "snap":snap,
                  "inputs":inputs}
    
    # add virtual snapshot file
    virtual_snapshot_file = get_virtual_snapfile(**kw_rnd_gal)
    kw_rnd_gal["virtual_snapshot_file"] = virtual_snapshot_file
    
    return kw_rnd_gal

def get_all_kw_gal(sim=std_sim,
                    subsim=std_subsim,
                    max_z=max_z,
                    min_z=min_z,
                    kw_criteria={"min_mass":float(min_mass)},
                    colibre_base_path=colibre_base_path,
                    verbose=True):
    """ Return list of kw inputs for the SimPartGal of COLIBRE for ALL
    galaxies within a range of redshift and respecting the criteria
    """
    min_z = float(min_z)
    max_z = float(max_z)
    all_snap = get_all_snap(sim=sim,
                            subsim=subsim,
                            max_z=max_z,
                            min_z=min_z)
    if verbose:
        print(f"Snaps selected: {all_snap}")
    # for reproducibility
    inputs     = {"sim":sim,
                  "subsim":subsim,
                  "max_z":max_z,
                  "min_z":min_z,
                  "min_mass":min_mass,
                  "colibre_base_path": colibre_base_path}
    kw_all_gal = []
    for snap in all_snap:
        z = get_z(snap,sim=sim,subsim=subsim)
        if verbose:
            print(f"z={z}")
        
        soap_catalogue_file = get_soap_cat(snap=snap,
                                           sim=sim,
                                           subsim=subsim,
                                           colibre_base_path=colibre_base_path)
        inputs["soap_catalogue_file"]=soap_catalogue_file

        candidates_gal = get_gal_candidates(snap=snap,
                                            sim=sim,
                                            subsim=subsim,
                                            soap_catalogue_file=soap_catalogue_file,
                                            kw_criteria=kw_criteria,
                                            colibre_base_path=colibre_base_path,
                                            verbose=verbose)
        kw_gal_snap = {"soap_catalogue_file":soap_catalogue_file,
                  "snap":snap,
                  "inputs":inputs}
        # flat structure
        for halo_index in candidates_gal:
            kw_all_gal.append({"soap_index":halo_index}|kw_gal_snap)
    if verbose:
        print(f"Found {len(kw_all_gal)} total galaxies")
    return kw_all_gal
    
def get_swiftgal(soap_index,
                 sim=std_sim,subsim=std_subsim,
                z=None,snap=None,
                colibre_base_path=colibre_base_path
                ):
    z,snap_ = get_z_snap(z=z,snap=snap,sim=sim,subsim=subsim)
    
    soap_catalogue_file = get_soap_cat(snap=snap,
                                           sim=sim,
                                           subsim=subsim,
                                           colibre_base_path=colibre_base_path)
    virtual_snapshot_file = get_virtual_snapfile(snap,
                         sim=sim,
                         subsim=subsim,
                         colibre_base_path=colibre_base_path)

    swift_gal = SWIFTGalaxy(
        virtual_snapshot_file,
        SOAP(
            soap_catalogue_file,
            soap_index=soap_index,
        ),
    )

    return swift_gal



"""
from nazgul.pathfinder import get_catdir

def _get_swiftgal(chosen_halo_index, 
                 virtual_snapshot_file,
                 soap_catalogue_file,
                 **kwargs # to ignore
                ):
    swift_gal = SWIFTGalaxy(
        virtual_snapshot_file,
        SOAP(
            soap_catalogue_file,
            soap_index=chosen_halo_index,
        ),
    )
    return swift_gal

def _from_simdir_to_sim_subsim(simulation_dir=std_simulation_dir):
    simulation_dir= Path(simulation_dir)
    sim = simulation_dir.parent
    subsim = simulation_dir.name
    return sim,subsim
    
def get_rnd_kw_gal(simulation_dir=simulation_dir,
                          max_z=max_z,
                          min_z=min_z,
                          min_mass=min_mass,
                          colibre_base_path=colibre_base_path,
                         soap_catalogue_file=None,
                  check_prev=True,save_pkl=True):
    sim,subsim = _from_simdir_to_sim_subsim(simulation_dir)
    catdir = get_catdir(sim=sim,subsim=subsim,simsuite=simsuite)
    catfile = catdir/"galcat.dll"
    if check_prev:
        try:
            with open(catfile,"rb") as f:
                dill.load(f)
        except FileNotFoundError:
            print("Previous catalogue not found,

# not really used:
def _get_random_swiftgal(max_z=max_z,
                        min_z=min_z,
                        min_mass=min_mass,
                        simulation_dir=simulation_dir,
                        colibre_base_path=colibre_base_path):
    
    kw_rnd_gal    =  get_rnd_kw_swiftgal(max_z=max_z,
                        min_z=min_z,
                        min_mass=min_mass,
                        simulation_dir=simulation_dir,
                        colibre_base_path=colibre_base_path)
    rnd_swift_gal = _get_swiftgal(**kw_rnd_gal)
    
    return rnd_swift_gal,kw_rnd_gal
"""                
