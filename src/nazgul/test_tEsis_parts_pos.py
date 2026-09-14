# From v_dist -> tE_SIS -> SIS gal -> sample parts positions -> compare with gal
import glob
import h5py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const

from nazgul.Translator.translator import PartGal,Gal2kwMXYZ
#from nazgul.Translator.EAGLE.particle_galaxy import 
from nazgul.mount_doom.generate_gal_lens import GalLens

# prev test
from nazgul.test_vdisp import get_tE_SIS
from nazgul.test_rad_gal import half_mass_radius

#sim,Gn,SGn,snap = "RefL0012N0188",1,0,"23"
sim,Gn,SGn,snap = "RefL0025N0752",3,0,"23"
print("Using simulation: "+sim)
print("Snap: "+snap)
print("Galaxy: Gn",Gn,"SGn",SGn)

Gal    = PartGal({"Gn":Gn,"SGn":SGn},simsuite="EAGLE",
             sim=sim,
             z=None,snap=snap,    # redshift or snap
             M=None,Centre=None,
             reload=False)

# Obtain thetaE
lensgal = GalLens(Gal,2)
lensgal.unpack()
lensgal.run()

tEsis = get_tE_SIS(lensgal)

# SIS gal with comp. thetaT
SISGal = PartGal({"n_smpl":1e6,"theta_E":tEsis,"z_lens":Gal.z},sim="SIS",simsuite="ANL_TEST") 

Gal.run()
SISGal.run() 

# here coords are in Mpc
gas    = Gal._SimPartGal.gas
dm     = Gal._SimPartGal.dm
stars  = Gal._SimPartGal.stars
bh     = Gal._SimPartGal.bh

# but the following convert them in kpc
kw_mxyz_sis = Gal2kwMXYZ(SISGal)
kw_mxyz_gal = Gal2kwMXYZ(Gal)


def kw_mxyz2mass_coords(kw_mxyz):
    kw_m_c = {}
    ms = np.array(kw_mxyz["Ms"])
    coords = np.array([kw_mxyz["Xs"],kw_mxyz["Ys"],kw_mxyz["Zs"]]).T
    kw_m_c["coords"] = coords*u.kpc.to("Mpc")
    kw_m_c["mass"] = ms
    return kw_m_c
    

kw_mc_sis = kw_mxyz2mass_coords(kw_mxyz_sis)
kw_mc_gal = kw_mxyz2mass_coords(kw_mxyz_gal)

# but the following assumes they are in Mpc, reason why we modify them in the kw_mxyz2mass_coords function
hmr_sis,_ = half_mass_radius(kw_mc_sis)
hmr_gal,_ = half_mass_radius(kw_mc_gal)
arcXkpc = lensgal.cosmo.arcsec_per_kpc_proper(lensgal.z_lens).to("arcsec/kpc").value
print(r"$\theta_{E,sis}$ in kpc",tEsis/arcXkpc)
print(r"$\theta_{E}$ in kpc",lensgal.thetaE/arcXkpc)
print("SIS half mass radius",hmr_sis)
print("Gal half mass radius",hmr_gal)
print("HMR SIS/GAL:",hmr_sis*100/hmr_gal,"%")


plt.hist(kw_mxyz_gal["Xs"],label="X Gal",alpha=.5)
print("the center of these is approx")
dm_coords = dm["coords"].T[0]*u.Mpc.to("kpc")-np.mean(dm["coords"].T[0]*u.Mpc.to("kpc")) 
plt.hist(dm_coords,label="X DM gal",alpha=.5)
gas_coords = gas["coords"].T[0]*u.Mpc.to("kpc")-np.mean(gas["coords"].T[0]*u.Mpc.to("kpc")) 
plt.hist(gas_coords,label="X Gas gal",alpha=.5)
stars_coords = stars["coords"].T[0]*u.Mpc.to("kpc")-np.mean(stars["coords"].T[0]*u.Mpc.to("kpc")) 
plt.hist(stars_coords,label="X Stars gal",alpha=.5)
plt.hist(kw_mxyz_sis["Xs"],label="X SIS",alpha=.5)
plt.legend()
nm = "tmp/hist_sisVsgal.png"
plt.savefig(nm)
print(f"Saved {nm}")

def sigma(kw_data, radius,proj_index=0,centre=None):
    """
    Compute the 2D density within a radius enclosing half the total kw_data mass.
 
    Parameters
    ----------
    kw_data : dict
        Must contain:
          'coords' : (N, 3) array  – particle positions in proper Mpc
          'mass'   : (N,)   array  – particle masses (any consistent unit)
        Optionally:
          'smooth' : (N,)   array  – smoothing lengths (unused here, kept for
                                     interface compatibility)
    centre : array-like of shape (3,), optional
        Galaxy centre in proper Mpc.  If None, the mass-weighted centre of the
        provided particles is used.
 
    Returns
    -------
    r_half : float
        Half-mass radius in proper Mpc.
    centre : ndarray, shape (3,)
        The centre used for the computation.
    """
    Xs,Ys,Zs = np.asarray(kw_data["coords"]).T
    if proj_index==0:
        Xs,Ys = Xs,Ys
    elif proj_index==1:
        Xs,Ys = Xs,Zs 
    elif proj_index==2:
        Xs,Ys = Ys,Zs 
    coords = np.asarray([Xs,Ys]).T*u.Mpc   # (N, 2)  proper Mpc
    mass   = np.asarray(kw_data["mass"])*u.Msun     # (N,)
 
    if centre is None:
        centre = np.average(coords, axis=0, weights=mass)
    centre = np.asarray(centre)*centre.unit
 
    # 2D radii from centre
    r = np.linalg.norm(coords - centre, axis=1)   # (N,)
 
    # Sort particles by radius and accumulate mass
    sort_idx    = np.argsort(r)
    r_sorted    = r[sort_idx]
    mass_sorted = mass[sort_idx]
    dr = r_sorted-radius
    idx_r = np.where(dr==np.min(dr))
    cum_mass    = np.cumsum(mass_sorted)
    mass_r =  cum_mass[idx_r]
    sigma = mass_r/(np.pi*radius*radius)
    return sigma
    
    # Find the radius where cumulative mass first reaches half_mass
    # Use linear interpolation between the two bracketing particles
    idx = np.searchsorted(cum_mass, half_mass)
 
    if idx == 0:
        r_half = r_sorted[0]
    else:
        # Interpolate between particle idx-1 and idx
        m_lo, m_hi = cum_mass[idx - 1], cum_mass[idx]
        r_lo, r_hi = r_sorted[idx - 1], r_sorted[idx]
        frac  = (half_mass - m_lo) / (m_hi - m_lo)
        r_half = r_lo + frac * (r_hi - r_lo)
 
    return r_half.to("kpc"), centre

rad = np.linspace(0.1,20,100)*u.kpc
sig_sis,sig_gal = [],[]
for r in rad:
    s_sis = sigma(kw_mc_sis,r)
    s_gal = sigma(kw_mc_gal,r)
    sig_sis.append(s_sis.value) #Msun/kpc^2
    sig_gal.append(s_gal.value)   #Msun/kpc2
    sig_sis = np.array(sig_sis)
sig_gal = np.array(sig_gal)

import matplotlib.pyplot as plt
plt.close()
plt.scatter(rad,np.log10(sig_sis),label="log Sigma SIS",marker=".")
plt.scatter(rad,np.log10(sig_gal),label="log Sigma Gal",marker="x")
plt.legend()
plt.savefig("tmp/sigma_SISvsGal.png")