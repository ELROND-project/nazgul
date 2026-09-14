import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from nazgul.masking import mask_SEAGLE,mask_center
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
from nazgul.mount_doom.generate_particle_lens_dom import LensPart
from nazgul.configurations import pixel_num,min_thetaE
from nazgul.mount_doom.cracks_of_doom import source_model_list
from nazgul.mount_doom.cracks_of_doom import kwargs_band_sim,kw_prior_z_source_stnd
from nazgul.pathfinder import get_lens_highdir

import nazgul.mount_doom.cracks_of_doom as cod

verbose = True
from nazgul.Translator.translator import PartGal


Gal    = PartGal(5,0,
             z=None,snap="20",    # redshift or snap
             M=None,Centre=None,
             reload=True)
lens = LensPart(Gal,
                 kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # sim prms 
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE,
                 subdir="./",
                 reload=True # reload previous lens
                 ) 

lens.run()

mask_HD =  mask_SEAGLE(lens)*mask_center(lens)

plt.close("all")
kw_extents = lens.kw_extents
extent_arcsec = kw_extents["extent_arcsec"]
kw_plot = {"cmap":"hot","extent":extent_arcsec,"origin":"lower"}
fig,axes = plt.subplots(1,2, figsize=(10, 5))
ax =axes[0]
im0 = ax.imshow(np.log10(lens.image_sim),**kw_plot)
ax.set_title("Image")
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical')   

ax =axes[1]
ax.set_title("Masked Image")
mask_nan = copy(mask_HD)
mask_nan[np.where(mask_HD==0)] = np.nan
im0 = ax.imshow(np.log10(mask_nan*lens.image_sim),**kw_plot)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical')   

nm = f"{lens.savedir}/{lens.name}_masked_im.png"
print(f"Saving {nm}")
plt.savefig(nm)
print("Success")