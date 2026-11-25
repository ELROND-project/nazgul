import numpy as np
import matplotlib.pyplot as plt

from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
from nazgul.MountDoom.generate_particle_lens_sub import SubLensPart
from nazgul.configurations import pixel_num,min_thetaE
from nazgul.MountDoom.cracks_of_doom import kw_prior_z_source_stnd

from nazgul.Translator.translator import PartGal


Gal    = PartGal({"Gn":5,"SGn":0},simsuite="EAGLE",
             z=None,snap="20",    # redshift or snap
             M=None,Centre=None,
             reload=True)

lens = SubLensPart(Gal,
                 kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # sim prms 
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE,
                 subdir="./",
                 reload=True # reload previous lens
                 ) 

lens.run()