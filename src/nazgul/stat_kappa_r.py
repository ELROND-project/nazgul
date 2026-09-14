"""
Study the kappa profiles from the AMR - note the limits in resolution in the center!
"""
# refactored via Claude, my version is in the ./old/ subdir w. the same name
import os
import sys
import argparse
import warnings
import dill
import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler

from python_tools.get_res import load_whatever

from nazgul.mount_doom.generate_gal_lens import get_kw_galpart
from nazgul.plot_AMRxpart import get_kw_1D_density
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.Translator.translator import get_all_PG,get_z_snap

default_kw_criteria = {
        "min_vel_disp": 120,
        "min_hmr": 1,
        "min_mass_stars": 1.76e10 * .6777,
    }

not_enclosed_str = "_not_encl"

def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Study statistically core and power law index from the 1D mass "
                     "distribution - directly from the galaxies, irrespective if they "
                     "are lenses or not",
    )
    parser.add_argument('-snap', '--snap', nargs="+", type=str, dest="snaps",
                         default=[], help="List of snaps to consider - default is all")
    parser.add_argument('-sim', '--sim', type=str, dest="sim",
                         default=std_sim, help="Simulation name")
    parser.add_argument('-ss', '--simsuite', type=str, dest="simsuite",
                         default=std_simsuite, help="Simulation suite name")
    parser.add_argument('-ssim', '--subsim', type=str, dest="subsim",
                         default=std_subsim, help="Sub-Simulation name")
    parser.add_argument('-no_enc', '--not_enclosed', dest="enclosed",
                         default=True, action="store_false",
                         help="Do not take the enclosed kappa(<r) but kappa(r) (more noisy)")
    parser.add_argument('-nr', '--no_reload', dest="reload",
                         default=True, action="store_false",
                         help="Do not try to reload prev. res.")
    return parser.parse_args()


def get_redshift_range(snaps, kw_galpart):
    if snaps != []:
        zs = [get_z_snap(snap=snap, **kw_galpart)[0] for snap in snaps]
        min_z = min(zs)
        max_z = max(zs)
        if min_z == max_z:
            min_z -= 1e-4
            max_z += 1e-4
    else:
        min_z = 0
        max_z = 10
        warnings.warn(f"Taking large range of redshift: {min_z}<=z<={max_z}")
    return min_z, max_z


def get_kappa_profiles(kw_galpart, kw_criteria,
                       enclosed = True,
                       reload=True,
                       kw_res_path="tmp/kw_kappa_r.dll"):
    """
    Try to load a cached kw_res from kw_res_path. If it exists and was computed
    with matching kw_galpart / kw_criteria, return it as-is. Otherwise recompute
    the 1D kappa profiles via get_kw_1D_density and cache the result.
    """
    if reload and os.path.exists(kw_res_path):
        kw_res = load_whatever(kw_res_path)
        if kw_res.get("kw_galpart") == kw_galpart and kw_res.get("kw_criteria") == kw_criteria:
            if getattr(kw_res,"enclosed",True)==enclosed:
                print(f"Loaded existing {kw_res_path}")
                return kw_res
        print(f"Cached {kw_res_path} found but kw_galpart/kw_criteria differ - recomputing")

    kw_galpart_full = dict(kw_galpart)
    kw_galpart_full["kw_criteria"] = kw_criteria
    kw_galpart_processed = get_kw_galpart(kw_galpart_full)

    profiler = Profiler()
    profiler.start()
    all_Gal = get_all_PG(**kw_galpart_processed)
    profiler.stop()
    print(profiler.output_text(color=True, show_all=False))

    kw_res = {
        "rs": [],
        "name": [],
        "kappas": [],
        "arcXkpcs": [],
        "REs": [],
        "kw_galpart": kw_galpart,
        "kw_criteria": kw_criteria,
        "enclosed":enclosed # if the returned kappa is kappa(r) or kappa(<r)!
    }

    for i,gal in enumerate(all_Gal):
        gal.unpack()
        for prji in range(3):
            kw_1d = get_kw_1D_density(gal, proj_index=prji, reload=reload,enclosed=enclosed)
            r = kw_1d["r_all"]
            if enclosed:
                kappa = kw_1d["Sigma_encl_all"] / kw_1d["Sigma_crit"]
            else:
                kappa = kw_1d["Sigma_all"] / kw_1d["Sigma_crit"]
                
            tE = kw_1d["tE"]  # arcsec
            RE = tE / kw_1d["arcXkpc"]
            arcXkpc =  kw_1d["arcXkpc"]
            

            kw_res["rs"].append(r.value)
            kw_res["REs"].append(RE.value)
            kw_res["kappas"].append(kappa.value)
            kw_res["arcXkpcs"].append(arcXkpc.value)
            kw_res["name"].append(gal.name + "_prj" + str(prji))
            if i==0 and prji==0:
                unit_rs = r.unit
                unit_REs = RE.unit
                unit_kappas = kappa.unit
                unit_arcXkpcs = arcXkpc.unit
            else:
                assert r.unit == unit_rs
                assert RE.unit == unit_REs
                assert kappa.unit == unit_kappas
                assert arcXkpc.unit == unit_arcXkpcs
    kw_res["unit_rs"] = unit_rs
    kw_res["unit_REs"] = unit_REs
    kw_res["unit_kappas"] = unit_kappas
    kw_res["unit_arcXkpcs"] = unit_arcXkpcs

    os.makedirs(os.path.dirname(kw_res_path) or ".", exist_ok=True)
    
    if not enclosed and not_enclosed_str not in kw_res_path:
        kw_res_path.replace(".dll",f"{not_enclosed_str}.dll")

    with open(kw_res_path, "wb") as f:
        dill.dump(kw_res, f)
    print(f"Saved {kw_res_path}")

    return kw_res


def plot_kappa_overlap(kw_res, enclosed=True, out_path="tmp/1D_overlap.png"):
    """Plot the overlap of the (scaled) 1D kappa profiles stored in kw_res."""
    fig, ax = plt.subplots()
    r_lbl = r"log$_{10}$ r/R$_{\rm{E}}$ [kpc]"
    if enclosed:    
        kappa_lbl = r"log$_{10} \kappa(<r)$ []"
    else:
        kappa_lbl = r"log$_{10} \kappa(r)$ []"
    ax.set_xlabel(r_lbl)
    ax.set_ylabel(kappa_lbl)
    if enclosed:
        ax.set_title("Overlap of enclosed and scaled 1D kappa profiles")
    else:
        ax.set_title("Overlap of scaled 1D kappa profiles")

    for r, kappa, RE in zip(kw_res["rs"], kw_res["kappas"], kw_res["REs"]):
        ax.plot(np.log10(r / RE), np.log10(kappa), alpha=.5, color="grey")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    if not enclosed and not_enclosed_str not in out_path:
        out_path.replace(".png",f"{not_enclosed_str}.png")
    
    fig.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    snaps = args.snaps
    sim = args.sim
    subsim = args.subsim
    simsuite = args.simsuite
    reload = args.reload
    enclosed = args.enclosed

    enclosed_str = ""
    if not enclosed:
        enclosed_str = not_enclosed_str
    kw_criteria = default_kw_criteria
    
    kw_galpart = {"sim": sim, "subsim": subsim, "simsuite": simsuite}

    min_z, max_z = get_redshift_range(snaps, kw_galpart)
    kw_galpart["min_z"] = min_z
    kw_galpart["max_z"] = max_z

    res_dir = "./tmp/"
    kw_res = get_kappa_profiles(
        kw_galpart, kw_criteria, reload=reload,
        kw_res_path=f"{res_dir}/kw_kappa_r{enclosed_str}.dll",
        enclosed=enclosed
    )
    plot_kappa_overlap(kw_res,enclosed=enclosed, out_path=f"{res_dir}/1D_overlap{enclosed_str}.png")


if __name__ == "__main__":
    main()
