"""
Dumb af name for a simple list of standard way to define "lens_fuel" classes
so that they are standardised
"""
from copy import deepcopy
from python_tools.tools import to_uid
from nazgul.lens_part_LOS import get_kw_los
from nazgul.mount_doom.lib_LensFuel import LensFuel,empty_kwargs_add_lenses
# No LOS (or any other lens), HST F160W observations, Spherical Sersic source 
# the vanilla version of lensfuel, the diesel
std_lensfuel = LensFuel("standard")

# same as std, but now with LOS lens - constant values!
#kwargs_add_lenses = {"lens_model_list":[],"kwargs_lens":[]}

def _get_kw_los(kw_los,kwargs_add_lenses=empty_kwargs_add_lenses):
    if not kwargs_add_lenses:
        kwargs_add_lenses = empty_kwargs_add_lenses
    kwargs_add_lenses["lens_model_list"].append("LOS")
    kwargs_add_lenses["kwargs_lens"].append(kw_los)
    return kwargs_add_lenses

def get_kw_los_fixed(gallens,kwargs_add_lenses=empty_kwargs_add_lenses):
    index  = 42
    kw_lns = deepcopy(kwargs_add_lenses)
    kw_los = get_kw_los(index=index)
    kw_lns = _get_kw_los(kw_los,kw_lns)
    return kw_lns

lensfuel_los_fix = LensFuel("LOSFix",
                            get_kw_add_lenses=get_kw_los_fixed)

# same as std, but now with LOS lens - variable LOS prms, but fixed given the gallens
def get_kw_los_gal(gallens,kwargs_add_lenses=empty_kwargs_add_lenses):
    # deterministic way to select the kwargs_los,
    # for now independent on the lens properties (z_s,z_l,cosmo)
    # but future improvements will account for that

    # "rnd" but deterministic index selection
    kw_lns    = deepcopy(kwargs_add_lenses)
    _str      = gallens._identity()
    # have to cut it if not it's too long
    rnd_index = int(str(to_uid(_str))[:4])
    # rnd_index will be very large, but with wrap around index it shouldn't be an issue
    kw_los = get_kw_los(index=rnd_index,wrap_around=True)
    kw_lns = _get_kw_los(kw_los,kw_lns)
    return kw_lns

 
lensfuel_los_gal = LensFuel("LOSGal",
                            get_kw_add_lenses=get_kw_los_gal)

