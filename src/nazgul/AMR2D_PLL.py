# updated from AMR2D - now parallelised and much faster and lighter for large particle numbers
# heavily helped by chatgpt

import warnings
import numpy as np
import astropy.units as u
from astropy.stats import sigma_clip

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection

import numba
from time  import time
from numba import njit, prange

@njit
def split_indices(x, y, pts, x0, x1, y0, y1):
    xm = 0.5*(x0 + x1)
    ym = 0.5*(y0 + y1)
    
    # temporary lists
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    
    for i in pts:
        xi = x[i]
        yi = y[i]

        # particles exactly on xm go to the right half; on ym go to upper half
        in_left = xi < xm
        in_lower = yi < ym
        if in_left:
            if in_lower: c0.append(i)   # SW
            else:        c2.append(i)   # NW
        else:
            if in_lower: c1.append(i)   # SE
            else:        c3.append(i)   # NE
    return c0, c1, c2, c3

def test_split_indices(x, y, m):
    """unit test function for split_indices"""
    pts = np.arange(len(x))

    c0,c1,c2,c3 = split_indices(x, y, pts, x.min(), x.max(), y.min(), y.max())
    
    total  = len(c0) + len(c1) + len(c2) + len(c3)
    unique = len(set(np.concatenate((c0,c1,c2,c3))))

    print("Total:", total)
    print("Unique:", unique)
    print("Expected:", len(pts))
    assert total == unique
    assert len(pts)==total
    


@njit
def compute_mass(pts, m):
    acc = 0.0
    for i in pts:
        acc += m[i]
    return acc

@njit
def needs_refinement(npts, area, density, pmax, min_area, dens_thresh):
    if npts > pmax and area > min_area:
        return True
    # if density > dens_thresh:
    #    return True
    return False

def build_AMR(x, y, m, 
              x0, x1, y0, y1, 
              max_particles=200, 
              min_area=0.01,
              dens_thresh=0.0):

    # stack of pending cells: (x0,x1,y0,y1,pts)
    stack = [(x0, x1, y0, y1, np.arange(x.size))]

    cells = []

    while stack:
        x0, x1, y0, y1, pts = stack.pop()

        sizex = x1 - x0
        sizey = y1 - y0
        area  = sizex * sizey
        mass  = compute_mass(pts, m)
        density = mass / area
        if needs_refinement(len(pts), area, density,
                            max_particles, min_area, dens_thresh):

            c0,c1,c2,c3 = split_indices(x, y, pts, x0, x1, y0, y1)
            for i, child in enumerate((c0, c1, c2, c3)):
                #if len(child) > 0 and len(child) < len(pts):
                if len(child) > 0 and (x1 - x0)*(y1 - y0) > min_area:
                    xm = 0.5*(x0+x1)
                    ym = 0.5*(y0+y1)
            
                    if i == 0:      # SW
                        stack.append((x0, xm, y0, ym, np.array(child)))
                    elif i == 1:    # SE
                        stack.append((xm, x1, y0, ym, np.array(child)))
                    elif i == 2:    # NW
                        stack.append((x0, xm, ym, y1, np.array(child)))
                    else:           # NE
                        stack.append((xm, x1, ym, y1, np.array(child)))
        else:
            # store cell
            # ignore particle ID -> keep it to verify uniqueness of binning
            cells.append([x0, x1, y0, y1, pts, mass, density])
            #cells.append([x0, x1, y0, y1, mass, density])

    return cells    
    
def validate_no_duplicates(cells, N,verbose=False):
     seen = np.zeros(N, dtype=np.int32)
     for (_,_,_,_,pts,_,_) in cells:
         seen[pts] += 1
 
     if np.all(seen==1):
         if verbose:
             print("OK: No duplicates, each particle appears once.")
     else:
         raise AssertionError("Some particles appear multiple times or not at all.")

# inspired from Translator/particle_galaxy.py
def clip_parts(x,y,m,clip_sigma=6,clip_thresh=.1):
    xm = np.sum(x*m)/np.sum(m)
    ym = np.sum(y*m)/np.sum(m)

    # this recentering needs to be re-added afterwards
    # I lost a day of work for this...
    x -= xm 
    y -= ym 

    i=0
    clip_frac_final  = 1
    while i<100:
        # clip coordinates outliers
        dists = np.sum(np.array([x,y])**2,axis=0)
        mask  = np.invert(sigma_clip(dists,sigma=clip_sigma).mask)
        
        clip_frac_final = 1-len(m[mask])/len(m)
        if clip_frac_final>clip_thresh:
            clip_sigma *= 1.1
            i+=1
        else:
            break
    if clip_frac_final>clip_thresh:
        raise RuntimeError(f"Even raising the sigma to {clip_sigma} we end up discarding {np.round(clip_frac_final*100,1)}% of the points")
        
    return x[mask]+xm,y[mask]+ym,m[mask]



def AMR_density_PLL(x, y, m, max_particles=300, min_area=None,
                    dens_thresh=None,Sigma_crit=None,dens_thresh_scale=0.1,
                   clip=False,clip_thresh=0.1,clip_sigma=6,
                   *args,**kwargs):
    """
        x,y,m : arrays w. coordinates and mass of particles
        max_particles: int, max n* of particle for 1 cell
        min_area: float, min area for 1 cell
        dens_thresh: float (optional), density threshold for 1 cell
        Sigma_crit: float (optional), critical density (if prev. meas.)
        dens_thresh_scale: float (optional), scaling of Sigma_crit to obtain dens_thresh
        clip: bool (def:False), if True ignore particles further than clip_sigma*sigma from the centre of mass
        clip_sigma: float (def:6), scale of sigma from CM from where we ignore the particles
        clip_thresh: float (def:.1), max. fraction of particle that we can discard with clipping
    """
    # numba strips units; if present, store them and add them a posteriori
    units = False
    try:
        units = True
        space_unit = x.unit
        x = x.value
        y = y.value
        min_area = min_area.value
    except:
        space_unit = 1
    try:
        units = True
        mass_unit = m.unit
        m = m.value
    except:
        mass_unit = 1 
    if clip:
        x,y,m = clip_parts(x,y,m,clip_sigma=clip_sigma,clip_thresh=clip_thresh)
    
    # Domain
    eps   = 1e-6
    x0, x1 = np.min(x)-eps, np.max(x)+eps
    y0, y1 = np.min(y)-eps, np.max(y)+eps
    if min_area is None:
        domain = (x1-x0)*(y1-y0)
        min_area = float(domain / 300)
        
    if dens_thresh is None:
        if Sigma_crit is None:
            raise RuntimeError("Provide either density threshold dens_thresh or critical density Sigma_crit")
        dens_thresh = dens_thresh_scale*Sigma_crit
        dens_thresh = float(np.squeeze(dens_thresh))

    cells = build_AMR(x, y, m,
                      x0,x1,y0,y1,
                      max_particles=int(max_particles),
                      min_area=min_area,dens_thresh=dens_thresh)
    validate_no_duplicates(cells,len(m))
    # if all particle are accounted for, we don't need particle ID
    cells = [c[:4] + c[5:] for c in cells]
    # cells: x0,x1,y0,y1,m,density
    if units:
        ucells = []
        for c in cells:
            x0, x1, y0, y1, mass, dns = c
            ucells.append([
                x0 * space_unit, x1 * space_unit,
                y0 * space_unit, y1 * space_unit,
                mass * mass_unit,
                dns * mass_unit / (space_unit**2)
            ])
        cells = ucells
    return cells

    
def get_MDfromAMRcells_PLL(AMR_cells,top_fraction=0.05):
    # for parallelised version
    try:
        dns_unit = AMR_cells[0][-1].unit
        density = np.array([c[-1].value for c in AMR_cells])
    except:
        dns_unit = 1
        density = np.array([c[-1] for c in AMR_cells])
    """c_MD      = AMR_cells[np.argmax(density)]
    MD_coords = (c_MD[0]+c_MD[1])/2.,(c_MD[2]+c_MD[3])/2.
        
    try:
        MD_coords = np.array([mdc.value for mdc in MD_coords])*MD_coords[0].unit
    except:
        pass
    MD_value  = np.max(density)
    assert MD_value == c_MD[-1]
    """
    
    # centroid of top 5% densest cells, weighted by density
    n_top    = max(1, int(len(density) * top_fraction))
    top_idx  = np.argpartition(density, -n_top)[-n_top:]

    top_dns  = density[top_idx]
    w        = top_dns / top_dns.sum()

    x_md = sum(w[k] * (AMR_cells[i][0] + AMR_cells[i][1]) / 2 
               for k, i in enumerate(top_idx))
    y_md = sum(w[k] * (AMR_cells[i][2] + AMR_cells[i][3]) / 2 
               for k, i in enumerate(top_idx))
    MD_value = sum(w[k] * density[i] for k, i in enumerate(top_idx))

    try:
        coord_unit = AMR_cells[0][0].unit
        MD_coords  = np.array([x_md.value, y_md.value])
    except:
        coord_unit = 1
        MD_coords  = np.array([x_md, y_md])
    return MD_coords*coord_unit, MD_value*dns_unit
    
def cell_in_extents(c,ext):
     x0,x1,y0,y1,mass,dns = c
     x = (x0+x1)/2
     y = (y0+y1)/2
     
     if ext[0]<x<ext[1] and ext[2]<y<ext[3]:
        return True
     return False

def plot_AMR_cells(kw_2Ddens,kw_extents=None,log10=True):
    fig, ax = plt.subplots(figsize=(8,8))    
    xc,yc = kw_2Ddens["MD_coords"] #kpc
    cells = kw_2Ddens["AMR_cells"]

    # to speed up the code I need to vectorise it -
    # but then I need to ingore the units
    try:
        x0_unit,x1_unit,y0_unit,y1_unit,mass_unit,dns_unit  = [c.unit for c in cells[0]]
        # verify that the units are consistent
        assert x0_unit==x1_unit
        assert x0_unit==y0_unit
        assert x0_unit==y1_unit
        assert x0_unit==xc.unit
        assert x0_unit==yc.unit
        length_unit = x0_unit
        # def. not the best implementation, but should work anyway
        cells = [[cc.value for cc in c] for c in cells]
    except AttributeError:
        warnings.warn("Units missing, assuming [x]=kpc, [m]=Msun")
        length_unit = u.kpc
        mass_unit   = u.Msun
        dns_unit    = mass_unit/(length_unit*length_unit)
        cells = cells
    
    

    if kw_extents:    
        if "arc" in str(length_unit):
            ext = kw_extents["extent_arcsec"]
        elif "kpc" in str(length_unit):
            ext = kw_extents["extent_kpc"]
        # Correct for the MD center
        try:
            xckpc = xc.to("kpc").value
            yckpc = yc.to("kpc").value
        except:
            xckpc = xc
            yckpc = yc
        ext   = np.array(ext) + np.array([xckpc,xckpc,yckpc,yckpc])
            
        cells = [c for c in cells if cell_in_extents(c,ext)]

    if len(cells)==0:
        warnings.warn(f"Cells are empty - skipping")
        return fig,ax
    elif len(cells)<10:
        warnings.warn(f"Very few cells: {len(cells)} - continuing but carefully")

    try:
        x0,x1,y0,y1,mass,dns  = np.array([[cc for cc in c] for c in cells]).T
    except ValueError as e:
        raise e

    if log10:
        dns = np.log10(dns)
    vmax,vmin = np.max(dns),np.min(dns)
    cmap = plt.get_cmap("hot")
    norm = Normalize(vmin=vmin, vmax=vmax)

    patches_list = [
        patches.Rectangle(
            (x0[i], y0[i]),
            x1[i] - x0[i],
            y1[i] - y0[i])
        for i in range(len(cells))
    ]

    colors = cmap(norm([d for d in dns]))

    pc = PatchCollection(
        patches_list,
        facecolor=colors,
        edgecolor="none", 
        linewidth=0.5)

    ax.add_collection(pc)
    ax.set_xlim(np.min(x0),np.max(x1))
    ax.set_ylim(np.min(y0),np.max(y1))

    ax.set_aspect("equal")
    ax.set_xlabel("x ["+str(length_unit)+"]")
    ax.set_ylabel("y ["+str(length_unit)+"]")
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm, ax=ax)
    if log10:
        lbl_dns = r"log$_{10}$(Density ["+str(dns_unit)+"])"
    else:
        lbl_dns = f"Density [{dns_unit}]"
    cbar.set_label(lbl_dns)
    return fig,ax
    
def _test_AMR():
    N = int(1e6)
    M = int(N/10)
    t0 = time()
    
    x,y = np.random.normal(0,2,(2,N))
    x0,y0 = np.random.normal(0,.4,(2,M))
    x0 +=3
    y0 +=-2
    x = np.hstack([x,x0])
    y = np.hstack([y,y0])
    m   = np.abs(np.random.normal(100,3,(len(x))))
    
    x0,x1 = np.min(x),np.max(x)
    y0,y1 = np.min(y),np.max(y)
    
    cells = AMR_density_PLL(x, y, m,dens_thresh=0) #, x0, x1, y0, y1)
    print(len(cells))
    t1 = time()
    print(t1-t0)
    fig,ax=plot_AMR_cells(cells)
    nm_test = "tmp/AMR_pll_test.pdf"
    plt.savefig(nm_test)
    print(f"Saving {nm_test}")
    