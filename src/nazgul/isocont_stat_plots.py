# From isocont_stat results,
# obtain some nice plots

import glob
import sys
import argparse
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from python_tools.get_res import load_whatever

from nazgul.stat_lenses import get_all_gallens_paths

from nazgul.pathfinder import std_data_dir,tmp_dir
from nazgul.Translator import std_sim,std_simsuite,std_subsim

def select_path_galaxies(gal_lenses_path,number_gal=None,kwargs_selection=None):
    if number_gal is None:
        warnings.warn("Loading all results - might take a while")
    if kwargs_selection:
        raise RuntimeError("Pragma: TODO. implement an actual selection that makes sense")
    gal_path_sel = []
    warnings.warn("For now very crude selection, just checks if the kappa isofit is present for the given galaxy")
    for i,gal_lens_path in enumerate(gal_lenses_path):   
        mn_dr = str(gal_lens_path.parent)
        file_to_check = f"{mn_dr}/kw_res_isodens_prj*" 
        # check that all fit for all projection were successful
        if len(glob.glob(file_to_check))==3:
            if mn_dr not in gal_path_sel:
                gal_path_sel.append(mn_dr)
        if number_gal is not None:
            if len(gal_path_sel)==number_gal:
                return gal_path_sel
    return gal_path_sel
    
def _extract_needed_info(kw_res_path_list,output_list):
    for kw_path in kw_res_path_list:
        kw_i = load_whatever(kw_path)
        isolist = kw_i["isofit"]["isolist"]
        # we skip the very first isophote
        isolist.__dict__["_list"] = isolist.__dict__["_list"][1:]
        
        # get lens:
        pth = Path(kw_path)
        prj = pth.name.split("prj")[1][0]
        lens_name = glob.glob(str(pth.parent/f"Sub_*_Prj{prj}_*.pkl"))
        if len(lens_name)==1:
            lens_name = lens_name[0]
        else:
            raise RuntimeError("pragma no cover")
        lens = load_whatever(lens_name)
        lens.unpack(verbose=False)

        # the following should be the case if I coded correctly, 
        # but atm it's poisoned
        #cutoff_rad = kw_i["cutoff_rad"]
        cutoff_rad = lens.radius/lens.arcXkpc # kpc
        kpcXPix    = 2*cutoff_rad.value/lens.pixel_num
        sma_kpc    = isolist.sma*kpcXPix
        rE = lens.thetaE/lens.arcXkpc
        isolist.sma_kpc = sma_kpc/rE.value
        
        theta  = sma_kpc*lens.arcXkpc.value
        theta_nrm = theta/lens.thetaE.value
        i_thetaE = np.argmin(np.abs(theta_nrm-1))
        isolist.dpa = isolist.pa - isolist.pa[i_thetaE]

        geom = kw_i["isofit"]["geom"]
        dx = (isolist.x0-isolist.x0[0]) #pix
        dx_arc = dx*lens.deltaPix.value/lens.thetaE.value
        dy = (isolist.y0-isolist.y0[0]) #pix
        dy_arc = dy*lens.deltaPix.value/lens.thetaE.value
        
        output_list.append({"map":kw_i["isofit"]["map"],
                           "model":kw_i["isofit"]["model"],
                           "isolist":isolist,
                            "dx":dx_arc,
                            "dy":dy_arc,
                            "lens_name":lens.name.replace("Sub_Lens_","")
                           })
    return output_list



def plot_isophote_results(map_type,kwargs_list, pdf_path="tmp/isophote_results.pdf", rows_per_page=5):
    #breakpoint()
    if map_type not in kw_name_map:
        raise ValueError(f"Map type {map_type} not known, only {kw_name_map.keys()}")
    map_type_str = kw_name_map[map_type]
    n = len(kwargs_list)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(n)]

    profile_params = [
        ("dx",   "(x(theta) - x(0))/theta_E"),
        ("dy",   "(y(theta) - y(0))/theta_E"),
        ("eps",  "Ellipticity"),
        ("dpa",  "Pointing Angle - P.A.(theta_E)"),
        ("a3",   "A_3 multipole"),
        ("a4",   "A_4 multipole"),
        ("b3",   "B_3 multipole"),
        ("b4",   "B_4 multipole"),
    ]

    with PdfPages(pdf_path) as pdf:
        # --- Page 1: 8 profile plots vs log10(sma_kpc) ---
        fig, axes = plt.subplots(4, 2, figsize=(12, 16))
        plt.suptitle(map_type_str)
        for ax, (key, title) in zip(axes.ravel(), profile_params):
            for i, kw in enumerate(kwargs_list):
                iso = kw["isolist"]
                sma = np.asarray(iso.sma_kpc)
                if key in ("dx","dy"):
                    y = kw[key]
                    sma_fit = sma
                    ax.set_xlabel(r"sma$/\rm{r}_{\rm{E}})$")
            
                else:
                    ax.set_xlabel(r"$\log_{10}(\mathrm{sma}/\rm{r}_{\rm{E}}\ [\mathrm{kpc}])$")
                    sma_fit = np.log10(sma)
                    y = np.asarray(getattr(iso, key))
                ax.plot(sma_fit, y, color=colors[i], label=str(i))#+": "+kw["lens_name"])    

                # crop min-max for a34-b34
                if map_type=="kappa":
                    if key in ["a3","a4","b3","b4"]:
                        """min_y,max_y = ax.get_ylim()
                        if np.any(np.max(y)>0.3):
                            max_y = np.min([0.3,max_y])
                        if np.any(np.min(y)<-0.3):
                            min_y = np.max([-0.3,min_y])
                        """
                        min_y,max_y = -0.3,0.3
                        ax.set_ylim(min_y,max_y)

                # def min x to be -1
                min_x = -1
                if np.min(sma_fit)<min_x:
                    _,max_x = ax.get_xlim()
                    ax.set_xlim(min_x,max_x)
                #ax.plot(sma, y, color=colors[i], label=str(i))#+": "+kw["lens_name"])
            ax.set_title(title)
            
            #ax.set_xlabel(r"sma/r$_{\rm{E}}\ [\mathrm{kpc}])$")
            ax.set_ylabel(title)
            ax.legend(fontsize=6, ncol=2, title="idx")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: (dx, dy) trajectories ---
        fig, ax = plt.subplots(figsize=(7, 7))
        for i, kw in enumerate(kwargs_list):
            ax.plot(kw["dx"], kw["dy"], color=colors[i], label=str(i))#+": "+kw["lens_name"])
        ax.set_xlabel(r"$x(\theta) - x(0)$")
        ax.set_ylabel(r"$y(\theta) - y(0)$")
        ax.set_title("Center offset trajectory")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=6, ncol=2, title="idx")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: complex ellipticity eps * exp(i * dpa) ---
        fig, ax = plt.subplots(figsize=(7, 7))
        for i, kw in enumerate(kwargs_list):
            iso = kw["isolist"]
            e_complex = np.asarray(iso.eps) * np.exp(1j *2*(np.asarray(iso.dpa)))
            ax.plot(e_complex.real, e_complex.imag, color=colors[i], label=str(i))#+": "+kw["lens_name"])
        ax.set_xlabel(r"$\mathrm{Re}(\epsilon\, \rm{e}^{i\, 2\Delta \rm{PA}})$")
        ax.set_ylabel(r"$\mathrm{Im}(\epsilon\, \rm{e}^{i\, 2\Delta \rm{PA}})$")
        ax.set_title("Complex ellipticity")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=6, ncol=2, title="idx")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Image pages: map | model | residual, one row per instance ---
        if map_type == "psi":
            lbl_map = r"-$\psi$ +min($\psi$)"
        elif map_type=="kappa":
            lbl_map = r"log$_{10}$(clip(smooth($\kappa$)))$"
        for start in range(0, n, rows_per_page):
            chunk = kwargs_list[start:start + rows_per_page]
            fig, axes = plt.subplots(len(chunk), 3, figsize=(12, 4 * len(chunk)), squeeze=False)
            for row, kw in enumerate(chunk):
                idx = start + row
                m, mod = np.asarray(kw["map"]), np.asarray(kw["model"])
                if map_type=="psi":
                    mod[mod==0]=np.nan
                resid = m - mod
                vmax_r = np.nanmax(np.abs(resid))

                for col, (img, title, cm, kwimg) in enumerate([
                    (m,    f"[{idx}] {map_type_str}",        "hot", {}),
                    (mod,  f"[{idx}] model {map_type_str}",       "hot", {}),
                    (resid, f"[{idx}] {map_type_str} - model", "bwr", dict(vmin=-vmax_r, vmax=vmax_r))]):
                    ax = axes[row, col]
                    im = ax.imshow(img, cmap=cm, origin="lower", **kwimg)
                    ax.set_title(title)
                    
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,label=lbl_map)

                axes[row, 0].set_ylabel(f"Lens {idx} {kw['lens_name']}", fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path    

kw_name_map = {"psi":r"$\psi$",
               "kappa": r"$\kappa$"}

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Study statistic of isocontours (kappa and psi) of lenses")
    parser.add_argument('-snap','--snap',nargs="+",dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-n','--number_gal',type=int,dest="number_gal",default=5,help=f"Number of galaxies to plot (set negative if you want all of them)")
    args      = parser.parse_args()
    snaps     = args.snaps #[25,26,27]
    sim       = args.sim
    subsim    = args.subsim
    simsuite  = args.simsuite
    #reload    = args.reload
    number_gal = args.number_gal
    if number_gal<0:
        number_gal = None

    res_dir = tmp_dir #tmp res dir, to improve

    gal_lenses_path = get_all_gallens_paths(snaps=snaps,
                                            simsuite=simsuite,
                                            sim=sim,
                                            subsim=subsim,
                                            data_dir=std_data_dir)
    gal_path_sample = select_path_galaxies(gal_lenses_path=gal_lenses_path,number_gal=number_gal)
    kw_isopsi = []
    kw_isokappa = []

    for gal_path in gal_path_sample:
        kw_isopsi_res_list = glob.glob(gal_path+"/kw_res_isopsi_prj*")
        kw_isokappa_res_list = glob.glob(gal_path+"/kw_res_isodens_prj*")
        kw_isopsi = _extract_needed_info(kw_isopsi_res_list,output_list=kw_isopsi)
        kw_isokappa = _extract_needed_info(kw_isokappa_res_list,output_list=kw_isokappa)
    if number_gal is not None:
        pdf_path_k = f"{res_dir}/isophote_results_kappa.pdf"
        plot_isophote_results(map_type="kappa",kwargs_list=kw_isokappa,pdf_path=pdf_path_k)
        pdf_path_p = f"{res_dir}/isophote_results_psi.pdf"
        plot_isophote_results(map_type="psi",kwargs_list=kw_isopsi,pdf_path=pdf_path_p)
    else:
        _N_gals = 14
        chunks_kw_isopsi   = [kw_isopsi[x:x+_N_gals] for x in range(0, len(kw_isopsi), _N_gals)]
        chunks_kw_isokappa = [kw_isokappa[x:x+_N_gals] for x in range(0, len(kw_isokappa), _N_gals)]
        
        for i,kw_isokappa_i in enumerate(chunks_kw_isokappa):
            pdf_path_k = f"{res_dir}/isophote_results_kappa_{i}.pdf"
            plot_isophote_results(map_type="kappa",kwargs_list=kw_isokappa_i,pdf_path=pdf_path_k)
        for i,kw_isopsi_i in enumerate(chunks_kw_isopsi):
            pdf_path_p = f"{res_dir}/isophote_results_psi_{i}.pdf"
            plot_isophote_results(map_type="psi",kwargs_list=kw_isopsi_i,pdf_path=pdf_path_p)
    print("Success!")