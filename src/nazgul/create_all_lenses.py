# Script to sistematically create all available GL 
import numpy as np
import warnings
from nazgul.mount_doom.generate_gal_lens import wrapper_forge_all_lenses

run_profiler = False
if run_profiler:
    warnings.warn("Running profiler might increase memory use and lead to a OOM kill")
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
#_list=["Gn20SGn0","Gn27SGn0","Gn17SGn0","Gn24SGn0","Gn26SGn0","Gn25SGn0","Gn4SGn0","Gn28SGn0","Gn3SGn0","Gn1SGn0","Gn15SGn0","Gn33SGn0","Gn19SGn0","Gn41SGn0","Gn13SGn0","Gn7SGn0","Gn29SGn0","Gn2SGn0","Gn10SGn0","Gn34SGn0","Gn5SGn0","Gn11SGn0","Gn23SGn0","Gn31SGn0","Gn14SGn0","Gn8SGn0","Gn18SGn0","Gn12SGn0","Gn21SGn0"]

"""all_lenses = wrapper_forge_all_lenses(kw_galpart={"min_z":0.269,#0.49,
                                       "max_z":0.272,#0.51,
                                       "min_mass":1e12},
                           reload=False,
                            _test=False,
                            _list_of_skippable_gals=[])#_list)
"""
# Following SEAGLE_I
print("Following SEAGLE_I")
all_lenses = wrapper_forge_all_lenses(kw_galpart={
    "sim":"RefL0050N0752",
    "min_hmr":1,                                   
    "min_z":0.1,#0.49,
    "max_z":0.3,#0.51,
    "min_vel_disp":120,
    "min_mass_stars":1.76e10*.6777},
    kw_lenspart={"ignore_OoBErr":True}, # test
   reload=True,
    _test=False,
    _list_of_skippable_gals=[])#_list)

if run_profiler:
    profiler.stop()
    print(profiler.output_text(color=True,show_all=False))

print("Stat: \n")
print("Lenses: "+str(len(all_lenses)))
print("Success!")
