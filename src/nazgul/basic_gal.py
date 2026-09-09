import dill
import warnings
import numpy as np
from pathlib import Path

from python_tools.get_res import LoadClass
from python_tools.tools import to_uid,to_uid_base64,silencer

def store_class(ist_class,path=None,overwrite=True,update=True,LoadClass=LoadClass):
    """If overwrite=True or not present
    serialize the current object to disk using dill.
    If present and update=True, update the previous instance
    """
    if path is None:
        path = ist_class.pkl_path
    if Path(path).is_file and not overwrite:
        warnings.warn(RuntimeWarning("Previous instance exists and we are not overwriting it"))
        return False
    while update:
        # check that if we can update the class
        prev_class = LoadClass(path)
        if not prev_class or prev_class != ist_class:
            break
        # if common attribute, they are overwritten by new instance
        ist_class.__dict__ = {**prev_class.__dict__,**ist_class.__dict__}
        break
    _store_class(ist_class,path=path)
    return 

def _store_class(ist_class,path=None):
    """Serialize the current object to disk using dill.
    """
    if path is None:
        path = ist_class.pkl_path
    with open(path, "wb") as f:
        dill.dump(ist_class, f)
    print(f"Saved {path}")

    
class BasicGal:
    """General useful class for galaxies (being ensemble of particles or already lenses)
    """
    # Large attributes to not store and to recompute/reload BEFORE computation
    _large_attributes_setup = []
    # Large attributes to not store and to recompute/reload AFTER computation
    _large_attributes_unpack = []
    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this instance
        raise NotImplementedError
        
    def __hash__(self):
        # simplify the hash method
        #return hash(self._identity())
        # implement better hashing to consider identity with dictionaries
        return to_uid(self._identity())

    @property
    def _hash_b64(self):
        return to_uid_base64(self._identity(),n_chars=6)
        
    def __eq__(self, other):
        return self._identity() == other._identity()

    def __str__(self):
        raise NotImplementedError 
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # combine the large attrs to delete
        _large_attributes = self._large_attributes_setup + self._large_attributes_unpack
        # remove large attributes (if present, can be loaded again)
        if _large_attributes==[]:
            raise NotImplementedError("Implement a list of _large_attributes to delete before storing")
        for lg_att in _large_attributes:
            state.pop(lg_att, None)
        return state
        
    def __setstate__(self, state):
        # Optional: restore defaults or trigger rebuild of heavy attributes
        self.__dict__.update(state)
    
    # Unpacking : this is to be called
    # after loading a pre-computed class instance
    def _needs_unpacking(self):
        """Check whether the object is missing large attributes to unpack.
        """
        return not all(
            hasattr(self, attr)
            for attr in self._large_attributes_unpack
        )
    @silencer
    def unpack(self,verbose=True):
        """Public wrapper for lazy reconstruction.
        """
        if self._needs_unpacking():
            self._unpack(verbose=verbose)
        return self

    def _unpack(self,verbose=True):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        print("Unpacking basic gal ...")
        raise NotImplementedError
        
    # Setup: to be called
    # Before computation
    def _needs_setup(self):
        """Check whether the object is missing required attributes.
        """
        return not all(
            hasattr(self, attr)
            for attr in self._large_attributes_setup
        )
        
    def setup(self):
        """Public wrapper for lazy reconstruction.
        """
        if self._needs_setup():
            self._setup()
        return self

    def _setup(self):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        print("Setting up basic gal ...")
        raise NotImplementedError
        
    ########################
    ########################
    def ReadClass(self,cl):
        # e.g. return ReadGal(cl)
        raise NotImplementedError
        pre_cl = ReadClass
        pre_cl.unpack()
        return pre_cl
        
    def upload_prev(self):
        prev_Class = self.ReadClass(self)
        if prev_Class is False or prev_Class != self:
            return False
        # if common attribute, they are overwritten by previous:
        self.__dict__ = {**self.__dict__,**prev_Class.__dict__}
        return True
        
    def verbose_assert_almost_equal(self,value1,value2=1,decimal=3,msg=None):
        # a verbose way of giving info if if fails
        try:
            np.testing.assert_almost_equal(value1,value2,decimal=decimal)
        except AssertionError as AssErr:
            if msg:
                AssErr.add_note(msg)
            #print("Error for \n"+str(self))
            raise AssErr
        return 0