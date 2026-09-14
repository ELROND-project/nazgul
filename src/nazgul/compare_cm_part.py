# TODO:
# Idea: we can take the centre of mass of baryonic particles and of DM parts and compare them, how much do they differ
# -> is it expected to have a large shift?
import dill
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from python_tools.tools import short_SciNot

from nazgul.Translator import std_simsuite
from nazgul.Translator.translator import get_rnd_PG,get_CM

Dcms = []
Gals_nms = []
for i in range(300):
    Gal    = get_rnd_PG(std_simsuite)
    if Gal.Name in Gals_nms:
        continue
    Gals_nms.append(Gal.Name)
    Gal.baryons = {}
    Gal.baryons["coords"] = np.vstack([Gal.stars["coords"],Gal.gas["coords"],Gal.bh["coords"]])
    Gal.baryons["mass"]   = np.hstack([Gal.stars["mass"],Gal.gas["mass"],Gal.bh["mass"]])
    
    mb = Gal.baryons["mass"]
    xb,yb,zb = Gal.baryons["coords"].T
    
    xcmb,ycmb,zcmb = get_CM(mb,xb,yb,zb)
    cm_b = np.array([xcmb,ycmb,zcmb])*u.Mpc
    mdm = Gal.dm["mass"]
    xdm,ydm,zdm = Gal.dm["coords"].T
    
    xcmdm,ycmdm,zcmdm = get_CM(mdm,xdm,ydm,zdm)
    
    cm_dm = np.array([xcmdm,ycmdm,zcmdm])*u.Mpc
    
    
    dcm = cm_b - cm_dm
    Dcm = np.linalg.norm(dcm) #np.sqrt(dcm[0]**2 + dcm[1]**2  + dcm[2]**2 )#.tolist())
    print(r"|$\Delta$CM|:",short_SciNot(Dcm.to("kpc")))

    Dcms.append(Dcm.to("kpc").value)

nm_file = "tmp/dcms.dll"
with open(nm_file,"wb") as f:
    dill.dump(Dcms,f)
print(f"Stored res in {nm_file}")

hist, bins = np.histogram(Dcms, bins=10)
#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
logbins = np.logspace(1e-10,np.log10(bins[-1]),len(bins))

plt.hist(Dcms,bins=logbins)
plt.ylabel(r"N (N$_{tot}$="+str(len(Dcms))+")")
plt.title("Distance between Bayrionic and DM Centre of Mass")
plt.xlabel(r"|$\Delta$CM| [kpc]")
plt.xscale("log")
nm = "tmp/Dcms_hist2.png"
print(f"Saving {nm}")
plt.savefig(nm)
