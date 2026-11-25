"""
Random bazar of useful functions for the reading of particles for galaxies
-> EAGLE specific
"""
import glob
import h5py
import pickle
import numpy as np
# Implement path handling
from pathlib import Path

from python_tools.get_res import load_whatever
from python_tools.tools import mkdir,to_dimless

from nazgul.pathfinder import get_part_dir,get_sim_dir,std_sim,std_simsuite,std_data_dir
from nazgul.pathfinder import std_sim,test_sim,tutorial_sim,path_nazgul
#########
# Setup #
#########

# from https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots/
# valid fo all sims apart the variable IMF runs
#-> corrected by hand to higher precision
kw_snap_z = {
    "28": 2.2204460e-16,
    "27":1.0063854e-01,
    "26":1.8270987e-01,
    "25":2.7090108e-01,
    "24":3.6566857e-01,
    "23":5.0310731e-01,
    "22":6.1518979e-01,
    "21":7.3562962e-01,
    "20":8.6505055e-01,
    "19":1.0041217e+00,
    "18":1.2593315e+00,
    "17":1.4867073,
    "16":1.7369658,
    "15":2.0124102,
    "14":2.237037,
    "13":2.4784133,
    "12":3.0165045,
    "11":3.5279765,
    "10":3.9836636,
    "9":4.4852138,
    "8":5.0372367,
    "7":5.4874153,
    "6":5.9711623,
    "5":7.0495663,
    "4":8.074616,
    "3":8.987875,
    "2":9.993033,
    "1":15.132311,
    "0":20.000021}
#inverted kw
kw_z_snap = {}
for k in kw_snap_z:
    kw_z_snap[kw_snap_z[k]] = k
# z indexes
z_index = np.array([float(f) for f in list(kw_z_snap.keys())])

# Useful functions:
###################

def get_z(snap):
    # strip all leading zeros
    snap = str(snap).lstrip("0")
    if snap=="":
        snap="0"
    return kw_snap_z[snap]

def get_snap(z,_ln_snap=None):
    # consider a continous z instead of the discreet version
    # works for discreet z as well
    key_z = min(kw_z_snap.keys(),key=lambda k:np.abs(k-float(z)))
    snap  = str(kw_z_snap[key_z])
    if _ln_snap:
        snap  = snap.zfill(_ln_snap)
    return snap

def get_z_snap(z=None,snap=None):
    """Given either z or snap, return both
    """
    if z is None and snap is None:
        raise UserWarning("Give either z or snap")
    if z is None:
        z = get_z(snap)
    else:
        snap = get_snap(z)
    snap = str(snap)
    return z,snap

def verify_z_snap(z,snap):
    if z is not None and snap is not None:
        assert int(get_snap(z))==int(snap)

def standardise_snap(snap,*args,**kwargs):
    z = get_z(snap)
    # return it as a 3-character string
    snap = get_snap(z,3)
    return snap 
    
def get_nfiles(sim):
    # n* of files per snapshot:
    if sim=="RefL0025N0752":
        nfiles = 16
    elif sim == "RefL0050N0752":
        nfiles = 128
    else:
        raise NotImplementedError(f"The n* of files in the snapshot directory for simulation {sim} was not hardcoded")
    return nfiles

def get_partfiles(sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir,
                  z=None,snap=None,_i_="*"):
    """
    Find the files
    If _i_ is specified, only that specific subsection of the snapshot (useful for DM)
    If no redshift/snapshots are defined, take all of them
    """
    verify_z_snap(z,snap)
    sim_dir = get_sim_dir(sim=sim,simsuite=simsuite,data_dir=data_dir)

    if z is not None and snap is not None:
        # verify that they are compatible:
        assert int(get_snap(z))==int(snap)
    elif z is not None:
        zstr = str(int(z))
        snap = get_snap(z)
    # if snap
    part_dir = get_part_dir(snap=snap,sim=sim,
                     simsuite=simsuite,data_dir=data_dir)

    if snap is None:
        snap = "???"
    else:
        snap = str(snap)

    file_string = part_dir/f"snap_{snap.zfill(3)}_*.{_i_}.hdf5"
    files = glob.glob(str(file_string))
    # checking that the files are not empty
    if files == []:
        raise RuntimeError(f"Files not found: {file_string}")
    return files


def read_snap_header(z=None,snap=None,sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """Read various attributes from the header group.
    """
    verify_z_snap(z,snap)
    file    = get_partfiles(sim=sim,z=z,snap=snap,simsuite=simsuite,data_dir=data_dir,_i_=0)
    if len(file)!=1:
        print("file=",file)
        raise RuntimeError("Warning: define only one snapshot")
    file      = file[0]
    with h5py.File(file, 'r') as f:
        a       = f['Header'].attrs.get('Time')                # Scale factor.
        h       = f['Header'].attrs.get('HubbleParam')         # h = H0/(100km/s/Mpc)
        boxsize = f['Header'].attrs.get('BoxSize')             # L [cMph/h].
    return a,h,boxsize

def _count_part(part):
    return len(part["mass"])

def _mass_part(part):
    return np.sum(part["mass"])