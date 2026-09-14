from pathlib import Path
from python_tools.tools import mkdir

import nazgul.configurations as conf
from   nazgul.Translator import std_sim,std_subsim,std_simsuite,test_sim,tutorial_sim
"""
Data structure:
---------------
RingBearer
    |_ Sims Suite  (e.g EAGLE)
        |_ Specific Res. Sim  (e.g RefL0025N0752 -> eventual subdirectory if subsim exist, e.g. COLIBRE)
            |_Snapshots   (e.g snap1)                                                  |_ CatGal                   |_CatLens
                |_GnSgn                                    |_ParticleData                   |_pkl                     |_pkl
                     |_ Gal |_Proj |_Lens                         |_hdf5 (link)
                         |     |        |_ Sub |_Dom
                         |     |           |     |_pkl
                         |     |_pkl       |_pkl    
                         |_pkl                  
"""
path_nazgul        = conf.nazgul_path
path_nazgul_origin = conf.nazgul_path_origin if conf.nazgul_path_origin is not None else conf.nazgul_path

std_data_dir = path_nazgul/"RingBearer"
# (base)/tmp will be a collector of intermediate, mildly useful plots/results, with the advantage of being easily accessible
tmp_dir = path_nazgul/"tmp"
mkdir(tmp_dir)

# result directory
results_dir = path_nazgul/"results"
mkdir(results_dir)

std_simsuite_dir = std_data_dir/std_simsuite # which simulation suite
std_sim_dir      = std_simsuite_dir/std_sim  # which simulation

# path to LensPop directory
LensPop_dir = path_nazgul/"LensPop"
def get_simsuite_dir(simsuite=std_simsuite,data_dir=std_data_dir):
    data_dir     = Path(data_dir)
    simsuite_dir = data_dir/simsuite # which simulation suite
    return simsuite_dir

def get_sim_dir(sim=std_sim,subsim=std_subsim,
                simsuite=std_simsuite,data_dir=std_data_dir):
    simsuite_dir = get_simsuite_dir(simsuite=simsuite,
                                    data_dir=data_dir)
    sim_dir      = simsuite_dir/sim # which simulation
    if subsim:
        # this is the case for COLIBRE, ulterior subdir
        sim_dir  = sim_dir/str(subsim)
    return sim_dir

def get_catdir(sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    sim_path = get_sim_dir(sim=sim,subsim=subsim,simsuite=simsuite,data_dir=data_dir)
    catdir = sim_path/"CatGal"
    mkdir(catdir)
    return catdir
    
def get_catlensdir(sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    # not sure if needed
    sim_path = get_sim_dir(sim=sim,subsim=subsim,simsuite=std_simsuite,data_dir=data_dir)
    catdir = sim_path/"CatLens"
    mkdir(catdir)
    return catdir

def get_snap_dir(snap,sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where the simulation particle data is stored
    """
    sim_dir = get_sim_dir(sim=sim,subsim=subsim,
                          simsuite=simsuite,data_dir=data_dir)
    snap     = str(snap).zfill(3)
    snap_dir = sim_dir/f"snap_{snap}"
    return snap_dir

def get_part_dir(snap=None,sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where the simulation particle data is stored
    """
    part_dir_name = "ParticleData"
    if snap is not None:
        snap_dir = get_snap_dir(snap=snap,
                                sim=sim,subsim=subsim,simsuite=simsuite,
                                data_dir=data_dir)
        part_dir = snap_dir/part_dir_name
    else:
        sim_dir  = get_sim_dir(sim=sim,subsim=subsim,
                               simsuite=simsuite,data_dir=data_dir)
        
        part_dir = sim_dir/f"snap*/{part_dir_name}"
    return part_dir


#######
# After this it becomes sim dependent -> refer to the translator
from nazgul.Translator.pathfinder import translate_galname

def get_gal_maindir(kw_gal,snap,sim=std_sim,subsim=std_subsim,
                    simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where all galaxy results are stored
    """
    snap_dir = get_snap_dir(snap=snap,
                            sim=sim,subsim=subsim,simsuite=simsuite,
                            data_dir=data_dir)
    galname  = translate_galname(kw_gal,simsuite=simsuite)
    #galname  = f"Gn{int(Gn)}SGn{int(SGn)}"
    gal_dir  = snap_dir/galname
    return gal_dir

def get_gal_dir(kw_gal,snap,sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where particle galaxy is stored
    """
    gal_maindir = get_gal_maindir(kw_gal=kw_gal,snap=snap,
                                  sim=sim,subsim=subsim,simsuite=simsuite,data_dir=data_dir)
    gal_dir     = gal_maindir/"Gal"
    return gal_dir


nm_proj_dir = "Projection"
def get_proj_dir(kw_gal,snap,sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where projection of the galaxy is stored
    """
    gal_maindir = get_gal_maindir(kw_gal=kw_gal,snap=snap,
                                  sim=sim,subsim=subsim,simsuite=simsuite,data_dir=data_dir)
    proj_dir     = gal_maindir/nm_proj_dir
    return proj_dir

def get_proj_dir_from_galdir(galdir):
    galdir      = Path(galdir)
    gal_maindir = galdir.parent
    proj_dir    = gal_maindir/nm_proj_dir
    return proj_dir   

def get_lens_maindir(kw_gal,snap,sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where lens computations of the galaxy are stored (main dir)
    """
    gal_maindir = get_gal_maindir(kw_gal=kw_gal,snap=snap,
                                  sim=sim,subsim=subsim,simsuite=simsuite,data_dir=data_dir)
    lens_dir     = gal_maindir/"LensSystem"
    return lens_dir

def get_lens_subdir(kw_gal,snap,
                    subdir="./",
                    sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where lens computations of the galaxy are stored 
    (sub dir - dep. on algorithm used, by default == main lens dir)
    """
    lens_maindir = get_gal_maindir(kw_gal=kw_gal,snap=snap,
                                  sim=sim,subsim=subsim,simsuite=simsuite,data_dir=data_dir)
    lens_subdir  = lens_maindir/subdir 
    return lens_subdir

nm_lowdir = "Sub"
def get_lens_lowdir(kw_gal,snap,subdir="./",
                    sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    lens computation dir (low-level, only particles)
    """
    lens_subdir = get_lens_subdir(kw_gal=kw_gal,snap=snap,
                    subdir=subdir,
                    sim=sim,subsim=subsim,simsuite=simsuite,data_dir=data_dir)
    return  lens_subdir/nm_lowdir

def get_lens_lowdir_from_galdir(galdir):
    galdir      = Path(galdir)
    gal_maindir = galdir.parent
    low_dir     = gal_maindir/nm_lowdir
    return low_dir     

nm_highdir = "Dom"
def get_lens_highdir(kw_gal,snap,subdir="./",
                    sim=std_sim,subsim=std_subsim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    lens computation dir (high-level, lens model) 
    """
    lens_subdir = get_lens_subdir(kw_gal=kw_gal,snap=snap,
                    subdir=subdir,subsim=subsim,
                    sim=sim,simsuite=simsuite,data_dir=data_dir)
    return  lens_subdir/nm_highdir
    
def get_lens_highdir_from_galdir(galdir):
    galdir      = Path(galdir)
    gal_maindir = galdir.parent
    high_dir    = gal_maindir/nm_highdir
    return high_dir   
    
