import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nazgul.Translator.translator import PartGal 
from nazgul.Translator.ANL_TEST.particle_galaxy import e1,e2 
from nazgul.MountDoom.generate_gal_lens import GalLens
from nazgul.MountDoom.lens_system import LensSystem
from nazgul.plot_AMRxpart import plot_AMR_densityXpart

from lenstronomy.LensModel.lens_model import LensModel

from pyinstrument import Profiler
profiler = Profiler()
profiler.start()

read_prev=True

pji = 0 # project index #
pg = PartGal({"n_smpl":1e6,"e1":0.25,"e2":0},
             sim="SIE",simsuite="ANL_TEST") 
pg.run() 
pg.store_gal()


profiler.stop()
print(profiler.output_text(color=True,show_all=False))
profiler.start()

plot_AMR_densityXpart(Gal=pg,proj_index=pji,
                      savedir="tmp/")
lensgal = GalLens(pg,pji)
lensgal.run(read_prev=read_prev)


lensgal.unpack()

lenssystem = LensSystem.from_GalLens(lensgal)

############### Alpha map ##################
fig,axes = plt.subplots(3,2,figsize=(6,8))
plt.suptitle(r"Comparison of $\alpha$ map")

axis = axes[0]
axis[0].set_title(r"$\alpha_x$")
axis[0].imshow(lensgal.alpha_map[0],origin="lower")
axis[0].set_ylabel("NAZGUL")
axis[1].set_title(r"$\alpha_y$")
axis[1].imshow(lensgal.alpha_map[1],origin="lower")
    

lens_anl  = LensModel(lens_model_list=['SIE'])
kw_sie    = pg.kwargs_lens
alpha_map_anl = lens_anl.alpha(*lensgal.get_RADEC(),kwargs=kw_sie)
a_anl_x,a_anl_y = alpha_map_anl 
axis = axes[1]
axis[0].set_ylabel("Analytical")
im0= axis[0].imshow(a_anl_x,origin="lower")
divider = make_axes_locatable(axis[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\alpha_{\rm{x}}$")
    
im0=axis[1].imshow(a_anl_y,origin="lower")
divider = make_axes_locatable(axis[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\alpha_{\rm{y}}$")
    
axis = axes[2]
ax1 = axis[0]
ax1.set_ylabel("Residual")
im0=ax1.imshow(a_anl_x-lensgal.alpha_map[0],cmap="bwr",origin="lower")
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\alpha_{\rm{Anl.-Comp.,x}}$")
    
ax2 = axis[1]
im0 = ax2.imshow(a_anl_y-lensgal.alpha_map[1],cmap="bwr",origin="lower")
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\alpha_{\rm{Anl.-Comp.,y}}$")
for axi in axes:
    for ax in axi:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
plt.tight_layout()

plt.savefig("tmp/alpha_.png")
plt.close()
#####################################################

############### Magnification map ##################
extents = lensgal.kw_extents["extent_arcsec"]
shear_map  = lenssystem.shear_map() 
lensgal.mu = np.ones_like(lensgal.alpha_map[0]) - lensgal.kappa_map**2 - shear_map**2

mu_anl = lens_anl.magnification(*lensgal.get_RADEC(),kwargs=kw_sie)
g1_anl,g2_anl = lens_anl.gamma(*lensgal.get_RADEC(),kwargs=kw_sie)
shear_anl = np.hypot(g1_anl,g2_anl)
kappa_anl = lens_anl.kappa(*lensgal.get_RADEC(),kwargs=kw_sie)
mu_anl_2 = np.ones_like(kappa_anl) - kappa_anl**2 - shear_anl**2
print("<mu anl - mu anl2>: ",np.median(mu_anl - mu_anl_2))


fig,axes = plt.subplots(3,3,figsize=(10,10))
plt.suptitle(r"Comparison of $\mu$ map")

axis = axes[0][0]
axis.set_title(r"NAZGUL - $\kappa^2$")
im0=axis.imshow(lensgal.kappa_map**2,origin="lower",extent=extents)
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\kappa^2_{\rm{Comp.}}$")


axis = axes[0][1]
axis.set_title(r"NAZGUL - $\gamma^2$")
im0=axis.imshow(shear_map**2,origin="lower",extent=extents)
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\gamma^2_{\rm{Comp.}}$")


axis = axes[0][2]
axis.set_title(r"NAZGUL - $\mu$")
im0=axis.imshow(lensgal.mu,origin="lower",extent=extents)
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\mu_{\rm{Comp.}}$")




axis = axes[1][0]
axis.set_title(r"Analytical - $\kappa^2$")
im0=axis.imshow(kappa_anl**2,origin="lower",extent=extents)
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\kappa^2_{\rm{Anl.}}$")


axis = axes[1][1]
axis.set_title(r"Analytical - $\gamma^2$")
im0=axis.imshow(shear_anl**2,origin="lower",extent=extents)
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\gamma^2_{\rm{Anl.}}$")


axis = axes[1][2]
axis.set_title(r"Analytical - $\mu$")
im0=axis.imshow(mu_anl,origin="lower",extent=extents)
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\mu_{\rm{Anl.}}$")


axis = axes[2][0]
axis.set_title(r"Residual - $\kappa^2$")
im0=axis.imshow(kappa_anl - lensgal.kappa_map,origin="lower",extent=extents,cmap="bwr")
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\kappa_{\rm{Anl.-Comp.}}$")

axis = axes[2][1]
axis.set_title(r"Residual - $\gamma^2$")
im0=axis.imshow(shear_anl-shear_map,origin="lower",extent=extents,cmap="bwr")
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\gamma_{\rm{Anl.-Comp.}}$")

axis = axes[2][2]
axis.set_title(r"Residual - $\mu$")
im0=axis.imshow(mu_anl-lensgal.mu,origin="lower",extent=extents,cmap="bwr")
divider = make_axes_locatable(axis)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"$\mu_{\rm{Anl.-Comp.}}$")

for axis in axes:
    for ax in axis:
        ax.set_ylabel("DEC")
        ax.set_xlabel("RA")


plt.tight_layout()

plt.savefig("tmp/mu_sie.png")
#####################################################
lensgal._unpack_Gal()

cosmo = lensgal.Gal.cosmo
Ds_p = cosmo.angular_diameter_distance(lensgal.Gal.z_source)
Ds = cosmo.angular_diameter_distance(lensgal.z_source)
Dls = cosmo.angular_diameter_distance_z1z2(lensgal.z_lens,lensgal.z_source)
Dls_p = cosmo.angular_diameter_distance_z1z2(lensgal.z_lens,lensgal.Gal.z_source)

rescale_fact = Ds*Dls_p/(Ds_p*Dls)

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as default_cosmo

from python_tools.tools import to_dimless,short_SciNot,ensure_unit,mkdir
from nazgul.pathfinder  import get_sim_dir

from nazgul.lib_cosmo import SigCrit
SigCrit_gal = SigCrit(cosmo =lensgal.cosmo,z_lens=lensgal.z_lens,z_source=lensgal.Gal.z_source)
SigCrit_rescaled = lensgal.SigCrit/rescale_fact
print(f"% error in SigCrit : {100*np.abs(SigCrit_gal-SigCrit_rescaled)/SigCrit_gal}")

tE_rescaled = (lensgal.thetaE*rescale_fact) #/lensgal.Gal.theta_E
print(f"% error in theta_E : {100*abs(lensgal.Gal.theta_E-tE_rescaled.value)/lensgal.Gal.theta_E}")


#####################################################
# simulate lensed image
from lenstronomy.Util.util import array2image,image2array 

RA,DEC = lensgal.get_RADEC()
def get_xy_sp(alpha_map):
    alpha_x,alpha_y = alpha_map        
    x_source_plane, y_source_plane = RA-alpha_x,DEC-alpha_y
    # the coords have to be given as flat
    x_source_plane = image2array(x_source_plane)
    y_source_plane = image2array(y_source_plane)
    return x_source_plane,y_source_plane
def get_source_light(alpha_map):
    imageNumerics,sourceModel     = lenssystem.get_imageNumerics(return_sourceModel=True)
    x_source_plane,y_source_plane = get_xy_sp(alpha_map=alpha_map)
    kwargs_source_list            = [imageNumerics.kwargs_source]
    source_light                  = sourceModel.surface_brightness(x_source_plane, y_source_plane, kwargs_source_list, k=None)
    return array2image(source_light)

source_anl = get_source_light(alpha_map_anl)
source_comp = get_source_light(lensgal.alpha_map)



fig,axes = plt.subplots(1,3,figsize=(8,6))
plt.suptitle(r"Comparison of source")

axis = axes
axis[0].imshow(source_comp,origin="lower",extent=extents)
axis[0].set_title("NAZGUL")
axis = axes
axis[1].set_title("Analytical")
axis[1].imshow(source_anl,origin="lower",extent=extents)

ax1 = axes[2]
ax1.set_title("Residual")
im0=ax1.imshow(source_anl-source_comp,cmap="bwr",origin="lower",extent=extents)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical',label=r"source$_{\rm{Anl.-Comp.}}$")

for ax in axes:
    ax.set_ylabel("DEC")
    ax.set_xlabel("RA")
plt.tight_layout()

plt.savefig("tmp/source_sie.png")
plt.close()

profiler.stop()
print(profiler.output_text(color=True,show_all=False))
