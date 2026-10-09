"""
Script to simplify loading galaxy and lenses - quite fragile right now, possible to improve
"""
import warnings
import numpy as np
from glob import glob
from pathlib import Path

from python_tools.get_res import load_whatever

from nazgul.pathfinder import get_gal_dir
from nazgul.Translator import std_kw_sim
from nazgul.stat_lenses import deal_with_doubles
from nazgul.mount_doom.lens_system import LensSystem
from nazgul.Translator.translator import PartGal,get_z_snap

from nazgul.combined_modelling_results import get_res_dir,get_all_lens_model_paths


#################
# Loader of Gal #
#################
def get_soap_Gal_colibre(name):
    try:
        # if already soap index
        soap_index = int(name)
    except ValueError:
        # else we extract it brute-force
        name = str(name)
        if "/" in name:
            # simplify the presence of multiple consecutive /: 
            name = str(Path(name))
            
        if not "_G" in name:
            if "G" in name:
                spl_name = name[name.index("G")+1:]
            else:
                raise RuntimeError(f"unexpected name {name}")
        else:
            if "/" in name:                
                _name = name.split("/")
                _name_G = [ _n for _n in _name if "_G" in _n]
                if len(_name_G)>1:
                    # if they are all the same:
                    list_soap = [get_soap_Gal_colibre(_nm) for _nm in _name_G]
                    if all([list_soap[0] ==l for l in list_soap]):
                        return list_soap[0]
                    else:
                        raise RuntimeError(f"unexpected name {name}")
                else:
                    name = _name_G[0]
            spl_name = str(name).split("_G")[1].split("/")[0]

        # hail mary catch all:
        if "/" in spl_name:
            spl_name = spl_name.split("/")[0]
        if "_" in spl_name:
            spl_name = spl_name.split("_")[0]
        if "." in spl_name:
            spl_name = spl_name.split(".")[0]

        soap_index = int(spl_name)
    return soap_index

def get_kw_Gal_colibre(name,snap=None,z=None,kw_sim=std_kw_sim):
    z,snap = get_z_snap(snap=snap,z=z,**kw_sim) # standardise the snap
    kw_Gal = {"snap":snap}
    kw_Gal["soap_index"] = get_soap_Gal_colibre(name)
    return kw_Gal
    

def ultimate_gal_loader(name, #this could be anything as long as it has a number (for Colibre
                    snap,
                    kw_sim=std_kw_sim,
                    verbose=True):
    """General loader for galaxy"""
    
    if kw_sim["simsuite"].upper()=="COLIBRE":
        # very "adhoc" solution
        kw_Gal = get_kw_Gal_colibre(name,snap,kw_sim=kw_sim)
        snap   = kw_Gal["snap"] #
        from nazgul.Translator.COLIBRE.particle_galaxy import get_gal_name
        name = get_gal_name(kw_Gal)
    else:
        raise RuntimeError("pragma: no cover")  
    #gal = PartGal(kw_Gal,snap=snap,**kw_sim)
    #gal.setup(verbose=verbose)
    gal_dir = get_gal_dir(kw_Gal,snap=snap,**kw_sim)
    path = gal_dir/f"{name}.dll"
    gal = load_whatever(path)
    gal.unpack(verbose=verbose)
    return gal

#####################
# Loader of SubLens #
#####################

def extract_prj(name):
    name = str(Path(name)).lower()
    if not "prj" in name and not "proj" in name:
        raise ValueError(f"name {name} seems to not contain the projection index - either give it in the name or explicitely")
    if "prj" in name:
        prj = "prj"
    else:
        prj = "proj"
    return int(name.split(prj)[1])

def get_prj(name,prj=None):
    if prj is None:
        prj = extract_prj(name)
    else:
        prj = int(prj)
        try:
            prj_nm = extract_prj(name)
            if prj!=prj_nm:
                raise RuntimeError(f"Projection index in the name {prj_nm} is not compatible with the one given {prj}")
        except ValueError:
            pass
    if prj not in [0,1,2]:
        raise RuntimeError(f"Projection index must be 0,1 or 2, not {prj}")
    return prj

    
def ultimate_gallens_loader(name,
                           snap,
                           prj=None,
                           kw_sim=std_kw_sim,
                           verbose=False):
    
    gal = ultimate_gal_loader(name,snap,kw_sim=kw_sim)
    prj = get_prj(name,prj)

    list_lens_gal_paths = glob(str(gal.gal_dir.parent/f"Sub/Sub*Prj{prj}*"))
    path_list = deal_with_doubles(list_lens_gal_paths)
    if len(path_list)!=1:
        raise RuntimeError(f"Failed to find a single res:\n{path_list}")
    else:
        lens = load_whatever(path_list[0])
        lens.unpack(verbose=verbose)
        return lens
#######################
# Loader of Lens Fuel #
#######################
# for now there is no easier way than just instantiate it


#########################
# Loader of Lens System #
#########################

def get_lens_resdir_path(model,snaps,kw_sim=std_kw_sim,check_if_workin_on_it=True):
    res_dir = get_res_dir(model,kw_sim=kw_sim)
    lens_resdir_paths = get_all_lens_model_paths(res_dir,snaps=snaps,
                                                 kw_sim=kw_sim,
                                                 check_if_workin_on_it=check_if_workin_on_it)  
    return lens_resdir_paths
    
#e.g.
#model = "SNS_DCE"
#snaps = ["0110"]
#get_lens_resdir_path(model,snaps)[0]
    
def _get_gallens_path(lens_name,model,snaps,kw_sim=std_kw_sim,check_if_workin_on_it=True):
    if len(snaps)==1:
        snap = snaps[0]
        gallens = ultimate_gallens_loader(lens_name,snap=snap,kw_sim=kw_sim,verbose=False) 
    else:
        raise RuntimeError("Pragma no cover")
    gallens_name = gallens.name
    gallens_name = gallens_name.replace("Sub_","")
    gallens_resdir_path = get_lens_resdir_path(model=model,snaps=snaps,
                                               kw_sim=kw_sim,
                                               check_if_workin_on_it=check_if_workin_on_it)
    lrd_pths = []
    for lrd in gallens_resdir_path:
        if gallens_name in str(lrd):
            lrd_pths.append(lrd)
    if len(lrd_pths)==0:
        raise RuntimeError(f"Lens {gallens_name} not found in gallens_resdir_path list")
    elif len(lrd_pths)>1:
        raise RuntimeError(f"Found more than 1 solution for lens {gallens_name} in gallens_resdir_path list:\n{gallens_resdir_path}")
    else:
        return Path(lrd_pths[0])


def ultimate_lenssys_loader(name,
                            snap,
                            lensFuel, # this for now has to be added like this - not great but it works
                            prj=None,
                            kw_sim=std_kw_sim,
                            kwargs_lenssystem={},
                            verbose=False):
    galLens = ultimate_gallens_loader(name,snap=snap,kw_sim=kw_sim,verbose=verbose)
    lensSys = LensSystem.from_GalLens(lensFuel = lensFuel,galLens=galLens,**kwargs_lenssystem)
    lensSys.unpack(verbose=verbose)
    lensSys.setup(verbose=verbose)
    return lensSys
