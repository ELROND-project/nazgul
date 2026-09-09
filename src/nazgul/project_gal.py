"""
Project the particles along different axis and verify if that produces a supercritical surface, given 
a maximum source redshift and a minimum Einstein angle
Uses Adaptime Mesh Refinement for the estimation of the density map
"""
import gc
import dill
import warnings
import numpy as np
from copy import copy
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d

from python_tools.get_res import load_whatever,get_path_str
from python_tools.tools import mkdir,to_dimless,ensure_unit,short_SciNot

import nazgul.pathfinder as pthf 
from nazgul.pathfinder import get_proj_dir_from_galdir, path_nazgul, tmp_dir
from nazgul.pathfinder import nm_proj_dir as dir_name
from nazgul.lib_cosmo import SigCrit,DsDds
from nazgul.AMR2D_PLL import AMR_density_PLL,plot_AMR_cells,get_MDfromAMRcells_PLL
from nazgul.Translator.translator import Gal2kwMXYZ,get_CM
# standard directory 
# Wrapper class of PartGal that extend it to
# deal with the projection components
class ProjGal:
    def __new__(cls,Gal,projection_index):
        # this allow for the wrapper to be applied over itself safely
        if isinstance(Gal,cls):
            if Gal.proj_index!=projection_index:
                raise RuntimeError("The given galaxy is already projected along a different axis")
            return Gal
        return super().__new__(cls)
                
    def __init__(self,Gal,projection_index):
        if not isinstance(Gal,ProjGal):
            self._gal            = Gal
            self._gal_id         = Gal._identity()
            self.proj_index      = projection_index
            mkdir(self.proj_dir)

    @property
    def projection_path(self):
        return self.proj_dir/f"projection_{self.proj_index}.pkl"

    @property
    def proj_dir(self):
        return get_proj_dir_from_galdir(self._gal.gal_dir)

    def __getstate__(self):
        return {"_gal": self._gal,
                "proj_index":self.proj_index}
        
    def __setstate__(self, state):
        self._gal       = state["_gal"]
        self.proj_index = state["proj_index"]
        
    def __str__(self):
        return self._gal.__str__()
        
    def __getattr__(self,name):
        return getattr(self._gal,name)
        
    ########################
    def _identity(self):
        """Return an immutable tuple uniquely identifying this galaxy.

        The identity is used for hashing, equality, and cache keys.
        """
        return (
            self._gal_id,
            self.proj_index
            )
    def store_gal(self):
        self._gal.store_gal()
        
    ########################
    # useful check if it is a lens:
    def is_lens(self,z_source_max,min_thetaE):
        try:
            kw_res_proj = load_whatever(self.projection_path)
            is_lens = False
            # simply check if projection is supercritical
            verify_if_lens = kw_res_proj["verify_lens"]
            if verify_if_lens(self,z_source_max=z_source_max,
                              min_thetaE=min_thetaE) is not np.nan:
                is_lens = True                    
        except FileNotFoundError:
            # If file is not there, we assume it is 
            # (rather, could be) a lens
            is_lens = True
        return is_lens      


def proj_parts(kw_parts,proj_index):    
    Xs,Ys,Zs = kw_parts["Xs"],kw_parts["Ys"],kw_parts["Zs"]
    if proj_index==0:
        return Xs,Ys
    elif proj_index==1:
        return Xs,Zs 
    elif proj_index==2:
        return Ys,Zs 
    else:
        raise RuntimeError("Projection index can only be 1,2 or 3, not "+str(proj_index))

def project_kw_parts(kw_parts,proj_index):    
    Xs,Ys = proj_parts(kw_parts,proj_index) 
    kw_parts_proj = {"Xs":Xs,"Ys":Ys,"Ms":kw_parts["Ms"]} 
    return kw_parts_proj

def kwparts2arcsec(kw_parts,arcXkpc):
    if "Zs" in kw_parts:
        # have no sense to have 3D distr in arcsec
        raise RuntimeError("The kw_parts should already be projected")
    RAs = kw_parts["Xs"]*arcXkpc
    DECs = kw_parts["Ys"]*arcXkpc
    return {"RAs":RAs,"DECs":DECs,"Ms":kw_parts["Ms"]}

class ProjectionError(Exception):
    # very specific error: raise if there is no projection s.t. 
    def __init__(self, error):
        self.error   = str(error)
        self.message = f"Projection Error: {self.error}"
        super().__init__(self.message)
    def __str__(self):
        return self.message
        
def project_Gal(GalProj,z_source_max,sample_z_source,min_thetaE,
                    arcXkpc=None,plot_2Ddens=False,verbose=True,reload=True,**kwargs):
    """
    Main projection function:
        - project particles on plane
        - create AMR
        - find center (coord of densest cell):
               - MD (mode/maximum density) coord
               - MD value
               - AMR cells
    - Input: 
        - GalProj: ProjGal instance, Galaxy projection
        - z_source_max: float, maximum redshift allowed for the source (z_source)
        - sample_z_source: func, given the range, will sample the z_source
        - min_thetaE: arcsec, min. threshold theta_E for the gal to be considered a lens
        - arcXkpc: arcsec/kpc, physical conversion scale (can be recomputed from the Gal)
        - plot_2Ddens: bool, if True plot the 2D density map from the AMR
        - reload: bool, if True tries to reload previous results and return them
        - verbose: bool
        
   - return:
       - kw_proj_res: kwargs, contains:
          - proj_index: int, projection index
          - z_source_min: float, minimum z source
          - verify_lens: func, function to verify if the gal is lens given min_thetaE and z_source_max
          - MD coord: array, coordinates of Maximum Density 
          - theta_E (approx) centered around MD
    """
    # if present and reload:
    if reload:
        try:
            kw_proj_res = load_whatever(GalProj.projection_path)
            print(f"Found and loaded projection from : {get_path_str(GalProj.projection_path,path_nazgul)}")
            return kw_proj_res
        except Exception as e :
            if verbose:
                print("Failed to load because "+str(e))
                print("Recomputing projection ...")
            pass
    # else compute it
    if arcXkpc is None:
        arcXkpc = GalProj.cosmo.arcsec_per_kpc_proper(Gal.z)
  
    # Read particles ONCE
    # kwargs of Msun, XYZ in kpc (explicitely) centered around Centre of Mass (CM)
    kw_parts       = Gal2kwMXYZ(GalProj)

    min_thetaE_kpc = min_thetaE/arcXkpc 
    proj_supercrit = False
    proj_index = GalProj.proj_index
    kw_proj = {"proj_index":proj_index}

    # Project the particles 
    kw_parts_proj = project_kw_parts(kw_parts=kw_parts,proj_index=proj_index)

    principal_axes_2D = get_principal_axis_2D(kw_parts_proj)

    # compute 2D density AMR density map (parallelised)
    kw_2Ddens = dens_map_AMR(kw_parts_proj=kw_parts_proj,
                              verbose=verbose)

    # From this point on kw_parts and kw_parts_proj are not used anymore
    # explicitely free memory
    del kw_parts
    del kw_parts_proj
    
    savenameSigmaEnc =GalProj.proj_dir/f"Sigma_enc_proj{proj_index}.png"

    # get range of source redshift
    kw_z_min = get_min_z_source(GalProj=GalProj,min_thetaE_kpc=min_thetaE_kpc,
                              kw_2Ddens=kw_2Ddens,
                              z_source_max=z_source_max,
                              savenameSigmaEnc=savenameSigmaEnc,verbose=verbose)
    if plot_2Ddens:
        fig,ax = plot_AMR_cells(kw_2Ddens)
        nm = f"{GalProj.proj_dir}/AMR_2DDens_{GalProj.name}_prj{proj_index}.png"
        fig.savefig(nm)
        print(f"Saved {nm}") 
        plt.close(fig)
        
    if kw_z_min["z_source_min"] is np.nan:
        if verbose:
            print("This projection of the galaxy does not lead to a supercritical lens. \
Rerun trying different projection")
        # store the kw_z_min
        del kw_z_min["fig_Sig"]
        kw_proj_res = kw_proj | kw_z_min
        pass
    else:
        # sample z_source
        z_source = sample_z_source(z_source_min = kw_z_min["z_source_min"],z_source_max=z_source_max)
        kw_z_min["z_source"] = z_source
        
        # get an estimate of theta_E 
        nm_sigmaplot = GalProj.proj_dir/f"Sigma_AMR_proj_{GalProj.proj_index}.png"
        thetaE = get_rough_thetaE(kw_2Ddens,GalProj.cosmo,GalProj.z,z_source,
                                  nm_sigmaplot=nm_sigmaplot,
                                  fig_Sig=kw_z_min["fig_Sig"])

        # AMR is not stored bc fairly large and not too long to compute
        del kw_2Ddens["AMR_cells"] 
        del kw_z_min["fig_Sig"]
        kw_thetaE = {"thetaE":thetaE} 
                    
        kw_proj_res  = kw_proj|kw_2Ddens|kw_z_min|kw_thetaE
        proj_supercrit = True
    # explicitely free memory
    del kw_2Ddens
    del kw_z_min
    gc.collect()
    kw_proj_res["principal_axes_2D"] = principal_axes_2D
    
    with open(GalProj.projection_path,"wb") as f:
        dill.dump(kw_proj_res,f)
    print(f"Saved {GalProj.projection_path}")

    if proj_supercrit is False:
        if verbose:
            print("M(gal)",short_SciNot(GalProj.M_tot))
            print("z_gal",GalProj.z)
        proj_message = "\nThis projection of the galaxy does not create a lens given the constraints\n"
        raise ProjectionError(proj_message)
    return kw_proj_res

def get_principal_axis_2D(kw_parts_proj):
    Ms = np.asarray(kw_parts_proj["Ms"].to("Msun"))*u.Msun
    Xs = np.asarray(kw_parts_proj["Xs"].to("kpc"))*u.kpc
    Ys = np.asarray(kw_parts_proj["Ys"].to("kpc"))*u.kpc
    return compute_principal_axis_2D(m=Ms,x=Xs,y=Ys)
    
def dens_map_AMR(kw_parts_proj,
                  max_particles=100,
                  min_area=0.1*u.kpc*u.kpc,
                  dens_thresh = 0.*u.Msun/(u.kpc**2),
                  verbose=True,
                  *args,**kwargs):
    """ 
    Compute density Adaptive Mesh Refinement map
    input  :  
    returns: kw_2Ddens["MD_value"][u.Msun/(u.kpc**2),1] 
             kw_2Ddens["MD_coord"][arcsec,2]
             kw_2Ddens["AMR_cells"][cells,N]
    """
    Ms = np.asarray(kw_parts_proj["Ms"].to("Msun"))*u.Msun
    Xs = np.asarray(kw_parts_proj["Xs"].to("kpc"))*u.kpc
    Ys = np.asarray(kw_parts_proj["Ys"].to("kpc"))*u.kpc

    # units are stripped by numba - have to "reattach" them "by hand"
    AMR_cells = AMR_density_PLL(Xs,Ys,Ms,
                                max_particles=max_particles, min_area=min_area,
                                dens_thresh=dens_thresh,*args,**kwargs)
    # use parallelised version - faster 
    MD_coords,MD_value = get_MDfromAMRcells_PLL(AMR_cells) 
    # Note: all inputs are still in kpc
    kw_2Ddens = {"MD_value":MD_value,"MD_coords":MD_coords,"AMR_cells":AMR_cells}
    return kw_2Ddens

def get_interpSigEncArc2(theta,Sigma_encl_arc):
    # intepolate it
    _interpSigEncArc2   = interp1d(theta,Sigma_encl_arc)
    Sigma_encl_arc_unit = Sigma_encl_arc.unit
    # define it such that it preserves the units
    def interpSigEncArc2(thetaE):
        thetaE = ensure_unit(thetaE,u.arcsec)
        return _interpSigEncArc2(thetaE)*Sigma_encl_arc_unit
    return interpSigEncArc2

def get_min_z_source(GalProj,kw_2Ddens,z_source_max,min_thetaE_kpc,verbose=True,savenameSigmaEnc = tmp_dir/"Sigma_enc.png"):
    """Given a projection, return the minimal z_source
    fails if it can't produce a supercritical lens w. z_source<z_source_max and 
    Sigma(theta_min)>Sigma_crit
    -> also return a function to verify if, given a min theta_E and max z_source, the
    galaxy is supercritical
    """   
    
    # compute surface density within minimum theta_E 
    # add plot Sigma_encl vs theta
    Sigma_crit_min   = SigCrit(z_lens=GalProj.z,z_source=z_source_max,cosmo=GalProj.cosmo)
    r,Sigma_encl     = cells2SigEnclRad(kw_2Ddens)
    dens_at_thetamin = getEnclDensAtRad(min_thetaE_kpc,r,Sigma_encl)

    arcXkpc = GalProj.cosmo.arcsec_per_kpc_proper(GalProj.z)
    theta = r*arcXkpc
    Sigma_encl_arc = Sigma_encl/(arcXkpc**2)

    
    # create a function to verify that the given lens is is indeed a lens
    interpSigEncArc2 = get_interpSigEncArc2(theta,Sigma_encl_arc)
    verify_lens  = create_verify_lens_fnc(interpSigEncArc2)
    
    Sigma_crit_min_arc = Sigma_crit_min/(arcXkpc**2)
    fig,ax = plt.subplots(1)
    ax.plot(theta,Sigma_encl_arc,color="k")
    ax.axhline(Sigma_crit_min_arc.value,ls="--",c="r",label=r"$\Sigma_{crit}^{min}=\Sigma_{crit}(z_{source,max}$="+str(z_source_max)+")="+str(short_SciNot(Sigma_crit_min_arc)))
    min_thetaE = min_thetaE_kpc*arcXkpc
    ax.axvline(to_dimless(min_thetaE),label=r"$\theta_{min}$="+str(short_SciNot(min_thetaE)),ls="-",c="grey")
    dens_at_thetamin_arc = dens_at_thetamin/(arcXkpc**2)
    ax.axhline(to_dimless(dens_at_thetamin_arc),label=r"$\Sigma(<\theta_{min})$="+str(short_SciNot(dens_at_thetamin_arc)),ls="--",c="g")
    if np.any(Sigma_crit_min_arc<Sigma_encl_arc):
        theta_E_max = theta[np.argmin(np.abs(Sigma_crit_min_arc-Sigma_encl_arc))]
        ax.axvline(to_dimless(theta_E_max),label=r"$\theta_E(z_{s,max})$="+str(short_SciNot(theta_E_max)),ls="--",c="b")
    # set limit to 5*min_thetaE
    ax.set_xlim(0,5*to_dimless(min_thetaE))
    ax.set_xlabel(r'$\theta$ ["]')
    ax.set_ylabel(r"$\Sigma$ ["+str(Sigma_encl_arc.unit)+"]")
    ax.set_title(r"$\Sigma_{encl}$")
    ax.legend()
    plt.close(fig)
    # Obtain the z_source_min:        
    ##########################
    # to be considered a lens, the dens. threshold has to be larger than the critical density 

    # convert it into a ratio of angular diameter distances Ds / Dds
    thresh_DsDds = dens_at_thetamin*4*np.pi*const.G*GalProj.cosmo.angular_diameter_distance(GalProj.z)/(const.c**2) 

    z_source_min = _get_min_z_source(cosmo=GalProj.cosmo,z_lens=GalProj.z,
                                    thresh_DsDds=thresh_DsDds,
                                    z_source_max=z_source_max,verbose=verbose)

    kw_zs_min = {"z_source_min":z_source_min,"fig_Sig":fig,"verify_lens":verify_lens}
    return kw_zs_min

def create_verify_lens_fnc(interpSigEncArc2):
    """
    Create function to verify that the galaxy is supercritical given the chosen 
    conditions: max z, min thetaE
    """
    # extract just the underlying data arrays from the interpolator
    Sigma_encl_arc_unit = interpSigEncArc2.__closure__[0].cell_contents
    _interpolator = interpSigEncArc2.__closure__[1].cell_contents
    x_data  = _interpolator.x
    y_data  = _interpolator.y
    kind    = _interpolator._kind
    _cache  = {}   # mutable object in closure, not serialised heavily

    def verify_lens(gal_class,min_thetaE=None,z_source_max=None):
        if "interp" not in _cache:
            from scipy.interpolate import interp1d
            _cache["interp"] =  interp1d(x_data, y_data, kind=kind)
        def rec_interpSigEncArc2(theta):
            theta = ensure_unit(theta,u.arcsec)
            return  _cache["interp"](theta)*Sigma_encl_arc_unit
        z_lens  = gal_class.z 
        cosmo   = gal_class.cosmo 
        arcXkpc = cosmo.arcsec_per_kpc_proper(z_lens)
        if min_thetaE is None:
            min_thetaE = gal_class.min_thetaE
        min_thetaE = ensure_unit(min_thetaE,u.arcsec)
        if z_source_max is None:
            z_source_max = gal_class.z_source_max
            
        minSigEncArc2  = rec_interpSigEncArc2(min_thetaE)
        minSigEnc      = minSigEncArc2*(arcXkpc**2)
        Dd             = cosmo.angular_diameter_distance(z_lens)
        thresh_DsDds   = minSigEnc*(4*np.pi*const.G*Dd)/(const.c**2)
        ensure_unit(thresh_DsDds,u.dimensionless_unscaled)
        
        if thresh_DsDds>=DsDds(cosmo=cosmo,z_d=z_lens,z_s=z_source_max):
             # can be a lens
            z_range      = _get_z_source_range(z_lens,z_source_max,n=1000)
            z_source_min = _get_min_z_source_thresh_DsDds(z_range,thresh_DsDds,cosmo=cosmo,z_d=z_lens)
            return z_source_min
        else:
            return np.nan
    return verify_lens

def getEnclDensAtRad(rad,radii,Sigma_encl):
    # get ENCLOSED density within radius
    rad = ensure_unit(rad,radii.unit)
    i_r = np.argmin(np.abs(radii-rad))
    return Sigma_encl[i_r]
    

def  _get_min_z_source_thresh_DsDds(z_source_range,thresh_DsDds,cosmo,z_d):
    DsDds_range  = np.array([DsDds(cosmo=cosmo,z_d=z_d,z_s=z_s) for z_s in z_source_range])
    diff         = thresh_DsDds-DsDds_range
    # the difference has to change sign at least in one point
    if len(np.where(diff>=0)[0])==0 or len(np.where(diff<0)[0])==0:
        print("Warning: the minimum z_source needed to have a lens is higher than the maximum allowed z_source")
        return np.nan
    abs_diff     = np.abs(diff)
    z_source_min = z_source_range[abs_diff.argmin()]
    return z_source_min    

def _get_z_source_range(z_lens,z_source_max,n=100,dz=0.01):
    if np.isinf(z_source_max):
        print("Edge case - z_source_max = inf, we set the range of redshift to be = [np.inf]")
        return [np.inf]
    return np.linspace(z_lens+dz,z_source_max,n)

def _get_min_z_source(cosmo,z_lens,thresh_DsDds,z_source_max,verbose=True):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")

    thresh_DsDds = ensure_unit(thresh_DsDds, u.dimensionless_unscaled)
    thresh_DsDds = thresh_DsDds.to("").value
    # since DsDds is a very smooth function, we just need to find if and where these meet
    z_source_range = _get_z_source_range(z_lens,z_source_max)
    DsDds_range    = np.array([DsDds(cosmo=cosmo,z_d=z_lens,z_s=z_s).value for z_s in z_source_range])

        
    min_DsDds = DsDds_range[-1]
    # the minimum should correspond to the highest redshift source:
    assert min_DsDds == np.min(DsDds_range)
    
    if not min_DsDds<thresh_DsDds:
        # to do: deal with this kind of output
        if verbose:
            print("Warning: the minimum z_source needed to have a lens is higher than the maximum allowed z_source")
            plt.close()
            fig_dsdds,ax = plt.subplots()
            ax.plot(z_source_range,DsDds_range,ls="-",c="k",label=r"D$_{\text{s}}$/D$_{\text{ds}}$(z$_{source}$)")
            ax.set_xlabel(r"z$_{\text{source}}$")
            ax.axhline(thresh_DsDds,ls="--",c="r",label=r"thr(dens)*4$\pi$*G*$D_{\text{l}}$/c$^2$="+str( short_SciNot(thresh_DsDds)))
            ax.set_title("Comparison between Distance ratio and threshold density")
            ax.legend()
            name = tmp_dir/"DsDds.pdf"
            fig_dsdds.savefig(name)
            plt.close(fig_dsdds)
            print("threshold density",short_SciNot(thresh_DsDds))
            print(f"Saved {name}")
        return np.nan
    else:
        z_source_min = _get_min_z_source_thresh_DsDds(z_source_range,thresh_DsDds,cosmo,z_d=z_lens)
        return z_source_min

def get_rough_thetaE(kw_2Ddens,cosmo,z_lens,z_source,
                     nm_sigmaplot="Sigma_AMR.png",fig_Sig=None):
    # approximate theta_E of the galaxy

    Dd      = cosmo.angular_diameter_distance(z_lens).to("Mpc")
    Ds      = cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds     = cosmo.angular_diameter_distance_z1z2(z_lens,z_source).to("Mpc") 
    kw_Ddds = {"Dd":Dd,"Dds":Dds,"Ds":Ds}
    return theta_E_from_AMR_densitymap(kw_2Ddens=kw_2Ddens,
                                       nm_sigmaplot=nm_sigmaplot,
                                       fig_Sig=fig_Sig,**kw_Ddds)

def cells2SigEnclRad(kw_2Ddens):
    
    r_sorted,m_sorted,area_sorted = cells2MRad(kw_2Ddens)
    
    # Cumulative sum
    cumulative_mass = np.cumsum(m_sorted)
    cumulative_area = np.cumsum(area_sorted)
    
    # Compute enclosed density Sigma(<r)
    Sigma_encl = cumulative_mass/cumulative_area
    
    return r_sorted,Sigma_encl

def cells2SigRad(kw_2Ddens):
    r_sorted,m_sorted,area_sorted = cells2MRad(kw_2Ddens)
    Sigma = m_sorted/area_sorted # for each "pixel" compute the density
    return r_sorted,Sigma

def cells2MRad(kw_2Ddens):
    """
    From the 2D AMR cells, order them in growing radii and compute return their radius, mass and area 
    """
    xc,yc = kw_2Ddens["MD_coords"] #kpc
    # to speed up the code I need to vectorise it -
    # but then I need to ingore the units
    #x0,x1,y0,y1,mass  = np.array([[c[0].value,c[1].value,c[2].value,c[3].value,c[4].value] for c in kw_2Ddens["AMR_cells"]]).T
    x0,x1,y0,y1,mass  = np.array([[cc.value for cc in c[:-1]] for c in kw_2Ddens["AMR_cells"]]).T
    x0_unit,x1_unit,y0_unit,y1_unit,mass_unit  = [c.unit for c in kw_2Ddens["AMR_cells"][0][:-1]]
    # verify that the units are consistent
    assert x0_unit==x1_unit
    assert x0_unit==y0_unit
    assert x0_unit==y1_unit
    assert x0_unit==xc.unit
    assert x0_unit==yc.unit
    length_unit = x0_unit
    x0 *=length_unit
    x1 *=length_unit
    y0 *=length_unit
    y1 *=length_unit
    mass *=mass_unit

    # locate center of the cells
    xc_cell = 0.5 * (x0 + x1)
    yc_cell = 0.5 * (y0 + y1)

    # compute their radius wrt MD
    dx = xc_cell - xc
    dy = yc_cell - yc
    r = np.sqrt(dx*dx + dy*dy)
    # area of the pixels
    area = (x1-x0)*(y1-y0)
    
    # Sort by radius
    idx = np.argsort(r)
    r_sorted = r[idx]
    m_sorted = mass[idx]
    area_sorted = area[idx] 
    return r_sorted,m_sorted,area_sorted
    
    
def theta_E_from_AMR_densitymap(kw_2Ddens, Dd, Ds, Dds,fig_Sig=None,nm_sigmaplot="Sigma_AMR.png"):
    # Critical density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # Physical scale of 1 arcsec at Dd
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc") # arcsec/kpc (on the lens plane)

    r_sorted,Sigma_encl = cells2SigEnclRad(kw_2Ddens)
    # theta
    theta = r_sorted*arcXkpc

    Sigma_crit_arcsec2 = Sigma_crit/(arcXkpc**2)
    Sigma_encl_arc2    = Sigma_encl/(arcXkpc**2)

    assert Sigma_crit_arcsec2.unit==Sigma_encl_arc2.unit
    
    thetaE = np.interp(Sigma_crit_arcsec2.value, Sigma_encl_arc2.value[::-1], theta[::-1].value)*theta.unit
    print("theta_E_arcsec found",short_SciNot(np.round(thetaE,2)))
    if fig_Sig is None:
        fig,ax = plt.subplots(1)
        ax.set_xlabel(r'$\theta$ ["]')
        ax.set_ylabel(r"$\Sigma$ ["+str(Sigma_encl.unit)+"]")
        ax.set_title(r"$\Sigma_{encl}$")
        ax.plot(theta,Sigma_encl,c="k")
    else:
        fig = fig_Sig
    fig.axes[0].axhline(Sigma_crit_arcsec2.value,ls="-.",c="r",label=r"$\Sigma_{crit}$ "+" ["+str(Sigma_crit_arcsec2.unit)+"]= "+ str(short_SciNot(Sigma_crit_arcsec2.value)))
    fig.axes[0].axvline(to_dimless(thetaE),label=r"$\theta_E$="+str(short_SciNot(thetaE)),ls="-",c="b")
    fig.axes[0].legend()
    # enforce a cutout of 5 thetaE
    fig.axes[0].set_xlim(0,5*to_dimless(thetaE))
    
    print(f"Saving {nm_sigmaplot}")
    fig.savefig(nm_sigmaplot)
    plt.close(fig)
    return thetaE
    
def Gal2MRADEC(Gal,proj_index,arcXkpc):
    kw_parts = Gal2kwMXYZ(Gal)
    Xs,Ys    = proj_parts(kw_parts,proj_index)
    RAs,DECs = Xs.to("kpc")*arcXkpc,Ys.to("kpc")*arcXkpc
    return kw_parts["Ms"],RAs,DECs

def Gal2kw_samples(Gal,proj_index,MD_coords,arcXkpc,dist_thresh=50*u.arcsec):
    # we scale the radius by a factor to be sure to include the center
    Ms,RAs,DECs = Gal2MRADEC(Gal,proj_index,arcXkpc=arcXkpc)
    # RA,DEC= arcsec, Ms = Msun
    #print("Some galaxy have a 'shifted' CM")
    RA_cm,DEC_cm = get_CM(Ms=Ms,Xs=RAs,Ys=DECs)
    print(f"Recentering around the maximum density point (MD) obtained with AMR")
    RA_MD,DEC_MD = MD_coords.to("kpc")*arcXkpc
    print("Info:  CM vs MD ")
    print("CM:",np.round(RA_cm,2),np.round(DEC_cm,2))
    print("MD:",np.round(RA_MD,2),np.round(DEC_MD,2))
    dist = np.sqrt((RA_cm-RA_MD)**2+(DEC_cm-DEC_MD)**2)
    print("Dist:",np.round(dist,2))
    dist = ensure_unit(dist,u.arcsec)
    if dist>dist_thresh:
        warnings.warn(RuntimeWarning(f"The distance between MD and CM {np.round(dist,2)} is larger than {np.round(dist_thresh,2)}."))
    kw_samples   = {}
    kw_samples["RAs"]  = RAs-RA_MD   #arcsec
    kw_samples["DECs"] = DECs-DEC_MD  #arcsec
    
    kw_samples["Ms"]   = Ms    #Msun
    kw_samples["cm"]   = RA_cm-RA_MD,DEC_cm-DEC_MD  # 
    return kw_samples

def get_2Dkappa_map(Gal,proj_index,MD_coords,SigCrit,kwargs_extents,arcXkpc=None):
    if arcXkpc is None:
        arcXkpc = Gal.cosmo.arcsec_per_kpc_proper(Gal.z) 
    kw_samples = Gal2kw_samples(Gal=Gal,proj_index=proj_index,
                                MD_coords=MD_coords,arcXkpc=arcXkpc)
    Ms       = kw_samples["Ms"]
    RAs,DECs = kw_samples["RAs"],kw_samples["DECs"]
    
    mass_grid, xedges, yedges   = np.histogram2d(RAs,DECs,
                                       bins=kwargs_extents["bins_arcsec"],
                                       weights=Ms,
                                       density=False) 
    # mass_grid shape: (nx, ny) -> transpose to (ny, nx) -> given the circular simmetry, doesn't really matter
    ra_edges,dec_edges = kwargs_extents["bins_arcsec"]
    Dra01   = np.diff(ra_edges) 
    Ddec01  = np.diff(dec_edges)
    
    # density_ij = M_ij/(Area_bin_ij)
    density    = mass_grid.T / (Dra01*Ddec01/(arcXkpc**2)) # Msun/kpc^2
    kappa = density/SigCrit
    kappa = kappa.to("").value
    return kappa


def compute_principal_axis_2D(m,x,y):
    """
    Similarly to compute_principal_axis_gen of Translator:
    Compute eigenvalues of the mass-weighted inertia tensor, which can be used to infer the axis of the particles, but now in 2D given a projection
    """
    cnt2 = np.mean(x)**2 + np.mean(y)**2 
    sz2  = np.std(x)**2 + np.std(y)**2 
    if cnt2>sz2:
        raise RuntimeError("Particles position must be centered around 0!")
    pos = np.transpose([x,y])
    m = np.array(m) # ensures no issues with dimensions
    I = np.zeros((2,2))
    for i in range(len(pos)):
        r = pos[i]
        I += m[i] * np.outer(r, r)

    # eigenvalues
    eigvals = np.linalg.eigvalsh(I)
    eigvals = np.sort(eigvals)[::-1]  # λ1 ≥ λ2 ≥ λ3

    a = np.sqrt(eigvals[0])
    b = np.sqrt(eigvals[1])
    
    principal_axes_2D = {"a":a,"b":b,"I2D":I}

    return principal_axes_2D