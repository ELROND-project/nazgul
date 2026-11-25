"""
Define the basic galaxy class PartGal (storing all particle data), as well as related helper functions, e.g. to sample galaxies 
-> generalised for COLIBRE and EAGLE simulations
"""
import dill
import numpy as np
from decimal import Decimal
from pathlib import Path
from astropy.stats import sigma_clip

from python_tools.tools import mkdir
from python_tools.get_res import LoadClass

from nazgul.basic_gal import BasicGal,store_class
from nazgul.std_path import path_nazgul, std_data_dir

class BasicPartGal(BasicGal):
    """Given the simulation, snap (or z) and galaxy numbers, set up a class
    with all the needed particle properties converted in physical units
    """
    _large_attributes_setup = []
    _large_attributes_unpack = []
    
    @property
    def name(self):
        """Define name of instance
        """
        raise NotImplementedError
        
    @property
    def dill_path(self):
        """Relative path (relative to nazgul_path) for the dill file.

        Uses the RingBearer anchor as a fallback so this works even when the
        object was pickled on a different host with a different nazgul_path.
        """
        full_path = self.gal_dir / f"{self.name}.dll"
        try:
            return full_path.relative_to(path_nazgul)
        except ValueError:
            data_root = std_data_dir.name
            parts = full_path.parts
            for i, part in enumerate(parts):
                if part == data_root:
                    return Path(*parts[i:])
            raise ValueError(
                f"Cannot make dill_path relative: '{full_path}' contains no "
                f"'{data_root}' component"
            )

    def rebase(self, new_nazgul_path=None):
        """Update gal_dir (and part_dir if present) to use the local (or given) nazgul_path.

        Call this after loading a pickle that was created on a different machine
        so that derived paths (proj_dir, etc.) resolve correctly on this host.
        """
        base = Path(new_nazgul_path) if new_nazgul_path is not None else path_nazgul
        self.gal_dir = base / self.dill_path.parent
        if hasattr(self, "part_dir"):
            self.part_dir = Path(self.gal_dir).parent.parent / "ParticleData"

    def dill_path_abs(self, path_nazgul_load=None):
        """Absolute path for file I/O.

        Resolves the relative dill_path against the local path_nazgul by default.
        Pass path_nazgul_load to override (e.g. to reference the original
        remote location given by configurations.nazgul_path_origin).
        """
        base = Path(path_nazgul_load) if path_nazgul_load is not None else path_nazgul
        return base / self.dill_path

    def slim_down(self):
        # delete all heavy attributes from class to reduce memory load
        for la in self._large_attributes_setup:
            if hasattr(self, la):
                delattr(self,la)
    ### Class Structure ####
    ########################
    def __str__(self):
        str_gal = f"Sim {self.sim}"
        str_gal = f"Gal {self.name}"
        str_gal += f", at z={str(np.round(self.z,3))}/snap={self.snap},"
        str_gal += f" with \nN={'%.1E'%Decimal(int(self.N_part))} part.\nof \ntot Mass={'%.1E'%Decimal(float(self.M))} [M_sun]\n"
        return str_gal 
    ########################
    ########################
    
    @property 
    def cosmo(self):
        raise NotImplementedError
        
    def run(self,reload=True):
        raise NotImplementedError
            
    def initialise_parts(self):
        raise NotImplementedError
        
    def _verify_cnt(self):
        """verify that the center of mass is indeed correct
        """
        raise NotImplementedError

    def store_gal(self):
        raise NotImplementedError

def clip_coord(m,x,y,z,sigma=10,perc_thresh=99):
    _verify_mxyz_input(m,x,y,z)
    if len(x)==0:
        # No particles to start with, no particles to clip
        return m,x,y,z
    # clip coordinates outliers
    dists = np.sum(np.array([x,y,z])**2,axis=0)
    mask = np.invert(sigma_clip(dists,sigma=sigma).mask)
    #mask = np.ones(len(x),dtype=bool)
    #for coord in x,y,z:
    #    mask *= np.invert(sigma_clip(coord,sigma=sigma).mask)
    perc_final = np.round(len(m[mask])*100/len(m),3)
    if perc_final<perc_thresh:
        print(np.round(100-perc_final,2),"% of particle discarded")
        raise RuntimeError("Too many particles discarded")
        
    return m[mask],x[mask],y[mask],z[mask]
    


def compute_principal_axis_gen(m,x,y,z):
    """
    Compute eigenvalues of the mass-weighted inertia tensor, which can be used to infer the 3D axis of the particles
    """
    _verify_mxyz_input(m,x,y,z)
    # We have interest in the shape, not absolute value
    x -= np.mean(x)
    y -= np.mean(y)
    z -= np.mean(z)
    
    pos = np.transpose([x,y,z])
    m   = np.array(m) # ensures no issues with dimensions
    I   = np.zeros((3,3))
    for i in range(len(pos)):
        r = pos[i]
        I += m[i] * np.outer(r, r)

    # eigenvalues
    eigvals = np.linalg.eigvalsh(I)
    eigvals = np.sort(eigvals)[::-1]  # λ1 ≥ λ2 ≥ λ3

    a = np.sqrt(eigvals[0])
    b = np.sqrt(eigvals[1])
    c = np.sqrt(eigvals[2])
    
    principal_axes = {"a":a,"b":b,"c":c,"I3D":I}

    return principal_axes

def _verify_mxyz_input(m,x,y,z):
    assert len(m)==len(x)
    assert len(y)==len(x)
    assert len(z)==len(x)
