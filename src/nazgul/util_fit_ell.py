import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.isophote import EllipseGeometry,EllipseSample

# Fitting ellipses - fast and easy 
# to have a first parameter estimates

def fit_ellipse_moments(image, threshold=None):
    img = np.asarray(image, dtype=float)

    # Optional: restrict to significant region
    if threshold is not None:
        mask = img > threshold
    else:
        mask = np.ones_like(img, dtype=bool)

    y, x = np.indices(img.shape)

    w = img * mask
    w_sum = w.sum()

    # --- centroid ---
    x0 = (x * w).sum() / w_sum
    y0 = (y * w).sum() / w_sum

    # --- centered coordinates ---
    dx = x - x0
    dy = y - y0

    # --- second moments ---
    Ixx = (w * dx * dx).sum() / w_sum
    Iyy = (w * dy * dy).sum() / w_sum
    Ixy = (w * dx * dy).sum() / w_sum

    # --- covariance matrix ---
    cov = np.array([[Ixx, Ixy],
                    [Ixy, Iyy]])

    # --- eigen decomposition ---
    eigvals, eigvecs = np.linalg.eigh(cov)

    # sort largest → smallest
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # --- ellipse parameters ---
    a = np.sqrt(eigvals[0])   # semi-major
    b = np.sqrt(eigvals[1])   # semi-minor

    # angle (radians)
    pa = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    return {
        "x0": x0,
        "y0": y0,
        "a": a,
        "b": b,
        "pa": pa
    }


def find_valid_sma0(image, x0, y0, eps, pa, sma, growth=1.3,
                     max_sma=None, max_tries=30):
    """Grow sma0 until the local gradient at that isophote is finite and nonzero."""
    sma_init = sma
    for _ in range(max_tries):
        geom = EllipseGeometry(x0, y0, sma, eps, pa)
        sample = EllipseSample(image, sma, geometry=geom)
        sample.update(fixed_parameters=geom.fix)
        grad = sample.gradient
        if grad is not None and np.isfinite(grad) and grad != 0.0:
            return sma
        sma *= growth
        if max_sma is not None and sma > max_sma:
            break
    raise RuntimeError(f"No valid sma0 found starting from {sma_init}")


def get_initial_kwfit(image,threshold=None):
    image = np.clip(image, 0, None)  # moments need non-negative weights

    # Optional: restrict to significant region
    if threshold is not None:
        mask = image > threshold
    else:
        mask = np.ones_like(image, dtype=bool)
    image *= mask
    y, x = np.indices(image.shape)
    total = image.sum()
    xc = (image * x).sum() / total
    yc = (image * y).sum() / total
    xx = (image * (x - xc)**2).sum() / total
    yy = (image * (y - yc)**2).sum() / total
    xy = (image * (x - xc) * (y - yc)).sum() / total

    theta = 0.5 * np.arctan2(2 * xy, xx - yy)
    lam1 = 0.5*(xx + yy) + np.sqrt(((xx - yy)/2)**2 + xy**2)
    lam2 = 0.5*(xx + yy) - np.sqrt(((xx - yy)/2)**2 + xy**2)
    eps = 1 - np.sqrt(lam2 / lam1)
    sma0 = np.sqrt(lam1)  # rough scale; try a small multiple too, e.g. 0.5x-2x
    kw_initfit = {"x0":xc, "y0":yc, "eps":eps, "pa":theta, "sma":sma0}

    sma0_valid =  find_valid_sma0(image=image,**kw_initfit)
    kw_initfit["sma"] = sma0_valid
    return kw_initfit
    
    
def fit_ellipse_isocontour(image, level):
    mask = image >= level
    return fit_ellipse_moments(image * mask)
    
    

def plot_ellipse(ax, params, color="r"):
    t = np.linspace(0, 2*np.pi, 200)

    a, b = params["a"], params["b"]
    pa = params["pa"]
    x0, y0 = params["x0"], params["y0"]

    x = a * np.cos(t)
    y = b * np.sin(t)

    # rotate
    xr = x*np.cos(pa) - y*np.sin(pa)
    yr = x*np.sin(pa) + y*np.cos(pa)

    ax.plot(xr + x0, yr + y0, color=color, lw=1,ls="--")

def plot_ellipse_isocontours(map,ellipses,nm="tmp/fit_ellipses.png",label="Map",ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    im0 = ax.imshow(map, origin="lower")
    
    for ell in ellipses:
        plot_ellipse(ax, ell)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=label)
    plt.savefig(nm)
    print(f"Saved {nm}")
    return ax
    
# show params
    
def _get_prm(ellipses,_get_prm):
    prm = np.array([_get_prm(ell) for ell in ellipses])
    return prm

def get_eps(ellipses):
    def _get_eps(params):
        return 1-(params["b"]/params["a"])
    return _get_prm(ellipses,_get_eps)
    
def get_pa(ellipses):
    def _get_pa(ell):
        return ell["pa"]
    return _get_prm(ellipses,_get_pa)

def get_x0(ellipses):
    def _get_x(ell):
        return ell["x0"]
    return _get_prm(ellipses,_get_x)

def get_y0(ellipses):
    def _get_y(ell):
        return ell["y0"]
    return _get_prm(ellipses,_get_y)
def get_rad(ellipses):
    def _get_rad(ell):
        return np.sqrt(ell["a"]**2 + ell["b"]**2)
    return _get_prm(ellipses, _get_rad)
def get_sma(ellipses):
    def _get_a(ell):
        return ell["a"]
    return _get_prm(ellipses, _get_a)

prm_nms = "eps","x0","y0","pa","sma"

def get_prm(ellipses,prm_nm):
    if prm_nm=="eps":
        return get_eps(ellipses)
    elif prm_nm=="pa":
        return get_pa(ellipses)
    elif prm_nm=="x0":
            return get_x0(ellipses)
    elif prm_nm=="y0":
            return get_y0(ellipses)
    elif prm_nm=="rad":
            return get_rad(ellipses)
    elif prm_nm=="sma":
            return get_sma(ellipses)
    else:
        raise RuntimeError(f"Prm {prm_nm} not implemented")

def get_kw_prms(ellipses):
    kw_prms= {}
    for prm in prm_nms:
        kw_prms[prm] = get_prm(ellipses,prm)
    return kw_prms

# useful for isodensity fit:
"""
def _get_initial_kwfit(kw_prms,sma_in=5):
    sma = kw_prms["sma"]
    # define accurate initial parameters
    d_sma = np.abs(sma-sma_in)
    indx = np.where(d_sma<1)

    kw_init = {}
    for prm_nm in prm_nms:
        kw_init[prm_nm] = np.nanmean(kw_prms[prm_nm][indx])
    kw_init["sma"] = sma_in
    return kw_init

def get_initial_kwfit(map,sma_in=5,levels=None,nlevels=60,plot=False):
    if levels is None:
        levels=np.linspace(map.min(),map.max(),nlevels)
    ellipses = [fit_ellipse_moments(map, l) for l in levels]
    if plot:
        plot_ellipse_isocontours(map,ellipses)
    kw_prms = get_kw_prms(ellipses)
    kw_init = _get_initial_kwfit(kw_prms,sma_in=sma_in)
    return kw_init
"""
if __name__=="__main__":
    raise NotImplementedError("Example of use - read, do not run")
    
    levels=np.linspace(map.min(),map.max(),nlevels)
    
    ellipses = [fit_ellipse_moments(map, l) for l in levels]
    plot_ellipse_isocontours(map,ellipses)

    prms = []
    for p in prm_nms:
        prms.append(get_prm(ellipses,p))

    for i,prm in enumerate(prms):
        axes[i].scatter(np.log10(sma),prm)
        axes[i].set_ylabel(prm_nms[i])
        axes[i].set_xlabel("log10(sma)")
    
    nm ="tmp/param_ellipses.png"
    plt.savefig(nm)
    print(f"Saved {nm}")
    
