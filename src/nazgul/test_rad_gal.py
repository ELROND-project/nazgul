import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from nazgul.Translator.translator import PartGal
from nazgul.MountDoom.generate_gal_lens import GalLens
from nazgul.Translator.EAGLE.sql_connect import exec_query
def half_mass_radius(kw_data, centre=None):
    """
    Compute the 3D physical radius enclosing half the total kw_data mass.
 
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
    coords = np.asarray(kw_data["coords"])*u.Mpc   # (N, 3)  proper Mpc
    mass   = np.asarray(kw_data["mass"])*u.Msun     # (N,)
 
    if centre is None:
        centre = np.average(coords, axis=0, weights=mass)
    centre = np.asarray(centre)*centre.unit
 
    # 3D radii from centre
    r = np.linalg.norm(coords - centre, axis=1)   # (N,)
 
    # Sort particles by radius and accumulate mass
    sort_idx    = np.argsort(r)
    r_sorted    = r[sort_idx]
    mass_sorted = mass[sort_idx]
    cum_mass    = np.cumsum(mass_sorted)
    half_mass   = 0.5 * cum_mass[-1]
 
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
    
if __name__=="__main__":
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
    
    
     
    
    myquery = "SELECT \
        gal.GroupNumber as Gn, \
        gal.SubGroupNumber as SGn, \
        gal.Redshift as z, \
        gal.Mass as M, \
        gal.CentreOfMass_x as CMx, \
        gal.CentreOfMass_y as CMy, \
        gal.CentreOfMass_z as CMz, \
        gal.HalfMassRad_DM as HMRD_dm, \
        gal.HalfMassRad_Gas as HMRD_gas, \
        gal.HalfMassRad_Star as HMRD_star, \
        gal.HalfMassRad_BH as HMRD_bh, \
        gal.MassType_DM as Mdm, \
        gal.MassType_Gas as Mgas, \
        gal.MassType_Star as Mstars, \
        gal.MassType_BH as Mbh \
    FROM \
        %s_Subhalo as gal \
    WHERE \
        gal.Snapnum = %s and \
        gal.Mass > 1e12 and \
        gal.GroupNumber = %s and \
        gal.SubGroupNumber = %s"%(sim,snap,Gn,SGn)
    query_out = exec_query(myquery)
    """
    From running "by hand" the following SQL script in 
    https://virgodb.cosma.dur.ac.uk:8443/Eagle/
    
    SELECT
        gal.GroupNumber as Gn, 
        gal.SubGroupNumber as SGn, 
        gal.Redshift as z, 
        gal.Mass as M, 
        gal.CentreOfMass_x as CMx, 
        gal.CentreOfMass_y as CMy, 
        gal.CentreOfMass_z as CMz,
        gal.HalfMassRad_DM as HMRD_dm,
        gal.HalfMassRad_Gas as HMRD_gas,
        gal.HalfMassRad_Star as HMRD_star,
        gal.HalfMassRad_BH as HMRD_bh,
        gal.MassType_DM as Mdm,
        gal.MassType_Gas as Mgas,
        gal.MassType_Star as Mstars,
        gal.MassType_BH as Mbh
    FROM
        RefL0012N0188_Subhalo as gal 
    WHERE
        gal.Snapnum = 23 and
        gal.Mass > 1e12 and
        gal.GroupNumber = 1 and
        gal.SubGroupNumber = 0
        
    I get the following HalfMassRad, which are allegedly in pkpc (Physical radius) - see McAlpine '16
    """
    #HMRD_dm,HMRD_gas,HMRD_star,HMRD_bh = np.array([137.22798,204.77927,3.6330612,0.97685826])*u.kpc
    HMRD_dm,HMRD_gas,HMRD_star,HMRD_bh  = np.array([query_out["HMRD_dm"],query_out["HMRD_gas"],query_out["HMRD_star"],query_out["HMRD_bh"]])*u.kpc
    #Mdm,Mgas,Mstars,Mbh = 5.1723753E12,2.16032395E11,7.43226E10,2.8830864E8
    Mdm,Mgas,Mstars,Mbh =  np.array([query_out["Mdm"],query_out["Mgas"],query_out["Mstars"],query_out["Mbh"]])*u.Msun
    """
    and the following center, in cMpc
    """
    center = np.array([query_out["CMx"],query_out["CMy"],query_out["CMz"]])*u.Mpc/Gal.xy_propr2comov
    # from Claude:
    
    gas    = Gal._SimPartGal.read_part(0)
    dm     = Gal._SimPartGal.read_part(1)
    stars  = Gal._SimPartGal.read_part(4)
    bh     = Gal._SimPartGal.read_part(5)
    center = None
    print("I get better results where re-computing the center - they are taken at the center of mass of each individual particle type")
    r_half,cnt = half_mass_radius(gas,centre=center)
    ratio_r = r_half/HMRD_gas
    print("Mass ratio for gas:",gas["mass"].sum()/Mgas*100,"%")
    print("HMR ratio for gas:",ratio_r.to("").value*100,"%")
    
    r_half,cnt = half_mass_radius(dm,centre=center)
    ratio_r = r_half/HMRD_dm
    print("Mass ratio for dm:",dm["mass"].sum()/Mdm*100,"%")
    print("HMR ratio for dm:",ratio_r.to("").value*100,"%")
    
    r_half,cnt = half_mass_radius(stars,centre=center)
    ratio_r = r_half/HMRD_star
    print("Mass ratio for star:",stars["mass"].sum()/Mstars*100,"%")
    print("HMR ratio for stars:",ratio_r.to("").value*100,"%")
    
    r_half,cnt = half_mass_radius(bh,centre=center)
    ratio_r = r_half/HMRD_bh
    print("Mass ratio for bh:",bh["mass"].sum()/Mbh*100,"%")
    print("HMR ratio for BH:",ratio_r.to("").value*100,"%")
    
