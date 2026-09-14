# Check v_disp w aperture radius
import dill
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from python_tools.get_res import load_whatever

from nazgul.Translator.EAGLE.sql_connect import exec_query
from nazgul.Translator.EAGLE.get_gal_indexes import get_query
from nazgul.Translator.EAGLE.fnct import get_snap
from nazgul.stat_lenses import get_all_gallens
from nazgul.test_vdisp import _get_tE_SIS



def get_kw_aperture(kwargs_query,
                    out_dll = "tmp/tE_vdisp_aperture.dll",reload=True):
    AP_sizes = [1, 3, 5, 10, 20, 30, 40, 50, 70 ,100]    
    if reload:
        try:
            kw_aperture = load_whatever(out_dll)
            print(f"Loaded prev. res. {out_dll}")
        except:
            print(f"Failed to load {out_dll}")
            reload = False
    if reload == False:
        kw_aperture = {"AP_sizes":AP_sizes,
              "Data":[]}
        for AP_size in AP_sizes:
            myQuery = get_query(AP_size=AP_size,**kwargs_query)
            myData = exec_query(myQuery)
            kw_aperture["Data"].append(myData)
    
        with open(out_dll,"wb") as f:
            dill.dump(kw_aperture,f)
    return kw_aperture

def get_lenses(kw_get_gallens):
    lenses  = get_all_gallens(**kw_get_gallens)
    for lensgal in lenses:
        lensgal.unpack()
        cosmo = lensgal.cosmo
        Ds    = cosmo.angular_diameter_distance(lensgal.z_source)
        Dls   = cosmo.angular_diameter_distance_z1z2(lensgal.z_lens,lensgal.z_source)
        lensgal.Ds = Ds
        lensgal.Dls = Dls
    return lenses

def plot_vdisp_vs_Ap(lenses,kw_aperture,nm = "tmp/vdisp_vs_Aperture.png"):

    fig,axes = plt.subplots(2)
    axis,axis_ratio_v =axes
    AP_sizes = kw_aperture["AP_sizes"]
    for i in range(len(AP_sizes)):
        ratios_tE = []
        query_out = kw_aperture["Data"][i]
        vdisp_stars_list = query_out["SVD"]*u.km/u.s
        list_Gn   = query_out["Gn"]
        list_SGn  = query_out["SGn"]
        list_z    = query_out["z"]
        list_snap = np.array([int(get_snap(zi)) for zi in list_z])

        for lensgal in lenses:
            #np.any(list_Gn==lensgal.Gal.Gn) , np.any(list_SGn==lensgal.Gal.SGn) ,np.any(list_snap==int(lensgal.Gal.snap)) )
            vdisp_stars = vdisp_stars_list[(list_Gn==lensgal.Gal.Gn) & (list_SGn==lensgal.Gal.SGn) & (list_snap==int(lensgal.Gal.snap))]
            if len(vdisp_stars)==0:
                print("Gal not present")
                continue
            assert len(vdisp_stars)==1
            theta_E_sis = _get_tE_SIS(vdisp_stars,Dls=lensgal.Dls,Ds=lensgal.Ds)
            ratio_tE    = lensgal.thetaE.value/theta_E_sis
            ratios_tE.append(ratio_tE)

            """
            vdisp_A10_ordered.append(svd_AP10[(list_Gn==lensgal.Gal.Gn) & (list_SGn==lensgal.Gal.SGn) & (list_snap==int(lensgal.Gal.snap))])
            for j in range(len(list_Gn)):
                gn=list_Gn[j]
                sgn = list_SGn[j]
                snp = list_snap[j]
                if 
            """
            
        y,y_sig = np.mean(ratios_tE),np.std(ratios_tE)
        axis.errorbar(AP_sizes[i],y,yerr=y_sig,ecolor="k",fmt="ko",
                      elinewidth=.8,markersize=4)

        axis_ratio_v.hist(vdisp_stars_list.value,bins=30,alpha=.5,
                          label=r"R$_{\rm{Aperture}}$="+str(AP_sizes[i]),
                          density=True)

    axis.axhline(1,c="k",label="ratio=1")
    axis.set_title(r"Ratio $\theta_E$/$\theta_{E,SIS}$(v$_{disp}$) averaged over all lenses, wrt to ap. size")
    axis.set_ylabel(r"$\theta_E$/ $\theta_{E,SIS}$(v$_{disp}$)")
    axis.set_xlabel(r"Aperture [pkpc]")
    axis.legend()

    axis_ratio_v.set_title(r"$\sigma_v(R)$ wrt to ap. size")
    axis_ratio_v.set_xlabel(r"$\sigma_v(R)$")
    axis_ratio_v.legend()
    plt.tight_layout()
    plt.savefig(nm)
    print(f"Saving {nm}")

if __name__=="__main__":
    reload = True
    kwargs_query   = {"sim":"RefL0050N0752",
                      "min_hmr":1,                                   
                      "min_z":0.1,#0.49,
                      "max_z":0.3,#0.51,
                      "min_vel_disp":120,
                      "min_mass_stars":1.76e10*.6777}

    kw_aperture = get_kw_aperture(kwargs_query)

    kw_get_gallens = {"snaps":[25,26,27],
                 "sim":"RefL0050N0752"}
    lenses = get_lenses(kw_get_gallens)
    plot_vdisp_vs_Ap(lenses,kw_aperture,nm = "tmp/vdisp_vs_Aperture.png")