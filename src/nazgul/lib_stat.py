# basic statiscs and data analysis equations
import numpy as np
from scipy.stats import chi2, norm

def tau(val_i,val_j,sig_i,sig_j): 
    # from dirty_TDC-WST
    tau_val = np.abs(val_i-val_j)/np.sqrt(sig_i**2+sig_j**2) #~ Z val
    return tau_val
    
def compute_tension(out,truth):
    if np.shape(out)[0]==2:
        out_val,out_sig = out
    else:
        raise RuntimeError("Not implemented")
        out_val = out
        out_sig = None
    if np.shape(truth)==():
        truth = truth*np.ones_like(out_val)
    tension = tau(out_val,truth,out_sig,np.zeros_like(truth))
    return tension


def Mahalanobis_distance(distr, truth):
    # see https://en.wikipedia.org/wiki/Mahalanobis_distance
    # distr: array N,M
    # truth: array N
    # generalisation of the tension in multiple dimension
    distr = np.asarray(distr)
    distr_mean = distr.mean(axis=1)
    distr_cov  = np.cov(distr)
    dx = distr_mean - np.asarray(truth)
    d2 = dx@ np.linalg.solve(distr_cov,dx)
    return np.sqrt(d2)

def tension_multidim(distr,truth):
    # the mahalanobis distance doesn't represent the "n_sigma"
    # of distance beween the truth and the distribution
    # For that we have to compute explicitely the p value
    d2 = Mahalanobis_distance(distr,truth)**2
    p = chi2.sf(d2, df=np.size(truth))
    n_sigma = norm.isf(p/2)
    return n_sigma