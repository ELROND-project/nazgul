"""
compare_two_models.py
---------------------
For each lens present in both model1 and model2, plot a row containing:
  [Sim Image | Model1 | Resid1 (chi2) | P(thetaE) m1 | Model2 | Resid2 (chi2) | P(thetaE) m2]
One row per lens, multi-page PDF with lines_per_page rows per page.
"""
import gc
import sys
import argparse
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from chainconsumer import Chain
from chainconsumer.plotting import plot_dist

from python_tools.tools import short_SciNot
from python_tools.get_res import load_whatever

from nazgul.mount_doom.cracks_of_doom import LoadLens
from nazgul.Translator import std_sim, std_simsuite, std_subsim
from nazgul.Modelling.lib_models import get_red_chi2, get_model_plot

# reuse helpers from combined_modelling_results
from nazgul.combined_modelling_results import (
    get_full_chain, get_res_dir, name_models,
    get_all_lens_model_paths, get_model_title,
    _convert_shear2LOS, _convert_polarshear2LOS,
)

matplotlib.use('Agg')
plt.rcParams.update({'font.size': 10})
scale_fig = 7

# ── per-lens key: GnXSGnY_PrjZ extracted from model_res_dir.name ─────────────

def _lens_key(model_res_dir):
    """
    Extract the lens identity (GnXSGnY_PrjZ) from the model result dir name.
    Assumes dir name contains 'Gn' somewhere, e.g. 'snap_027_Gn31SGn1_Prj1_allLOS'.
    Adjust the split logic if your naming convention differs.
    """
    name = Path(model_res_dir).name
    # keep everything up to and including PrjN
    parts = name.split("_")
    key_parts = []
    for p in parts:
        key_parts.append(p)
        if p.startswith("Prj") and len(p) > 3:
            break
    return "_".join(key_parts)


def _load_lens_and_data(model_res_dir, model):
    """Load lens + kw_data for one model result dir. Returns (lens, kw_data)."""
    lens = LoadLens(model_res_dir / "link_gallens.pkl")
    lens.unpack()
    lens.model_res_dir = model_res_dir

    modelPlot  = get_model_plot(res_dir=model_res_dir)
    full_chain = get_full_chain(lens, model)
    kw_data    = dict(full_chain=full_chain, modelPlot=modelPlot)
    return lens, kw_data


def _free(lens, kw_data):
    del kw_data["full_chain"], kw_data["modelPlot"]
    try:
        lens.slim_down()
    except:
        pass
    del lens
    gc.collect()


# ── plot one comparison row ───────────────────────────────────────────────────

def plot_comparison_row(axes, i_row, nrows,
                        lens1, kw_data1, model1,
                        lens2, kw_data2, model2,
                        columns_ttl, _rnd=3):
    """
    Fill row i_row of axes with:
      col 0: sim image (shared)
      col 1: model1 image
      col 2: model1 residual + chi2
      col 3: P(thetaE) model1
      col 4: model2 image
      col 5: model2 residual + chi2
      col 6: P(thetaE) model2
    """
    fig        = axes.flatten()[0].get_figure()
    lens_name  = lens1.name.replace("Sub_", "").replace("LS_", "")

    def _band(kw_data):
        return kw_data["modelPlot"]._band_plot_list[0]

    band1 = _band(kw_data1)
    band2 = _band(kw_data2)

    kw_img1 = dict(vmin=band1._v_min_default, vmax=band1._v_max_default,
                   extent=band1._image_extent, origin="lower", cmap=band1._cmap)
    kw_img2 = dict(vmin=band2._v_min_default, vmax=band2._v_max_default,
                   extent=band2._image_extent, origin="lower", cmap=band2._cmap)
    kw_res  = dict(vmin=-3, vmax=3, origin="lower", cmap="bwr")

    def _add_cbar(ax, im, label):
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label=label)

    def _chi2_text(ax, band, kw_data, kw_img):
        chi2 = get_red_chi2(kw_data["modelPlot"], verbose=False)
        ext  = kw_img["extent"]
        x_txt = (ext[1] - ext[0]) * 3.5 / 5
        y_txt = (ext[3] - ext[2]) * 4.0 / 5
        ax.text(x_txt, y_txt,
                r"$\chi^2_{\rm red.}$=" + str(np.round(chi2, 2)),
                color="k", backgroundcolor="w", fontsize=8)
        return chi2

    # ── col 0: sim image (use model1's data image as reference) ──────────────
    ax = axes[i_row][0]
    ax.set_ylabel(lens_name, fontsize=8)
    if i_row == 0:
        ax.set_title(columns_ttl[0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])
    im = ax.imshow(np.log10(band1._data), **kw_img1)
    _add_cbar(ax, im, r"flux$_{\rm data}$")

    # ── col 1: model1 ─────────────────────────────────────────────────────────
    ax = axes[i_row][1]
    if i_row == 0:
        ax.set_title(columns_ttl[1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(np.log10(band1._model), **kw_img1)
    _add_cbar(ax, im, r"flux$_{\rm model}$")

    # ── col 2: model1 residual ────────────────────────────────────────────────
    ax = axes[i_row][2]
    if i_row == 0:
        ax.set_title(columns_ttl[2])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(band1._norm_residuals, **{**kw_res, "extent": kw_img1["extent"]})
    _add_cbar(ax, im, r"$(f_{\rm mod}-f_{\rm dat})/\sigma$")
    _chi2_text(ax, band1, kw_data1, kw_img1)

    # ── col 3: P(thetaE) model1 ───────────────────────────────────────────────
    ax = axes[i_row][3]
    if i_row == 0:
        ax.set_title(columns_ttl[3])
    if i_row == nrows - 1:
        ax.set_xlabel(r"$\theta_E$")
    fc1  = kw_data1["full_chain"]
    tE1  = fc1["theta_E_lens0"].to_numpy()
    plot_dist(ax, Chain(samples=fc1, name=model1, shade=True, color='#2c7fb8',
                        smooth=20, bins=10, shade_gradient=0.4, linewidth=2.0),
              px="theta_E_lens0")
    ax.axvline(np.median(tE1), ls="-.", color="#2c7fb8",
               label=rf"$\tilde\theta_E$={np.median(tE1):.{_rnd}f}±{np.std(tE1):.{_rnd}f}")
    ax.axvline(lens1.thetaE.value, ls="-", color="r",
               label=r"True=" + short_SciNot(lens1.thetaE.value) + '"')
    ax.legend(fontsize=6, loc="upper right")

    # ── col 4: model2 ─────────────────────────────────────────────────────────
    ax = axes[i_row][4]
    if i_row == 0:
        ax.set_title(columns_ttl[4])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(np.log10(band2._model), **kw_img2)
    _add_cbar(ax, im, r"flux$_{\rm model}$")

    # ── col 5: model2 residual ────────────────────────────────────────────────
    ax = axes[i_row][5]
    if i_row == 0:
        ax.set_title(columns_ttl[5])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(band2._norm_residuals, **{**kw_res, "extent": kw_img2["extent"]})
    _add_cbar(ax, im, r"$(f_{\rm mod}-f_{\rm dat})/\sigma$")
    _chi2_text(ax, band2, kw_data2, kw_img2)

    # ── col 6: P(thetaE) model2 ───────────────────────────────────────────────
    ax = axes[i_row][6]
    if i_row == 0:
        ax.set_title(columns_ttl[6])
    if i_row == nrows - 1:
        ax.set_xlabel(r"$\theta_E$")
    fc2  = kw_data2["full_chain"]
    tE2  = fc2["theta_E_lens0"].to_numpy()
    plot_dist(ax, Chain(samples=fc2, name=model2, shade=True, color='#d7301f',
                        smooth=20, bins=10, shade_gradient=0.4, linewidth=2.0),
              px="theta_E_lens0")
    ax.axvline(np.median(tE2), ls="-.", color="#d7301f",
               label=rf"$\tilde\theta_E$={np.median(tE2):.{_rnd}f}±{np.std(tE2):.{_rnd}f}")
    ax.axvline(lens2.thetaE.value, ls="-", color="r",
               label=r"True=" + short_SciNot(lens2.thetaE.value) + '"')
    ax.legend(fontsize=6, loc="upper right")

    gc.collect()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Compare two lens models side by side for each shared lens")
    parser.add_argument('-m1', '--model1', type=str, dest="model1",
                        help=f"Model 1 name. Accepted: {name_models}")
    parser.add_argument('-m2', '--model2', type=str, dest="model2",
                        help=f"Model 2 name. Accepted: {name_models}")
    parser.add_argument('-sim',  type=str, dest="sim",      default=std_sim)
    parser.add_argument('-ss',   type=str, dest="simsuite", default=std_simsuite)
    parser.add_argument('-ssim', type=str, dest="subsim",   default=std_subsim)
    parser.add_argument('-lpp', '--lines_per_page', dest="lines_per_page",
                        default=5, type=int)
    parser.add_argument('-nciwoi', '--not_check_if_workin_on_it', dest="check_if_workin_on_it",
                        default=True,action="store_false")
    args = parser.parse_args()

    model1       = args.model1
    model2       = args.model2
    sim          = args.sim
    subsim       = args.subsim
    simsuite     = args.simsuite
    check_if_workin_on_it = args.check_if_workin_on_it
    if not check_if_workin_on_it:
        warnings.warn("Not checking if workin on it")
    lines_per_page = args.lines_per_page
    _rnd = 3

    res_dir1 = get_res_dir(model1, simsuite=simsuite, sim=sim, subsim=subsim)
    res_dir2 = get_res_dir(model2, simsuite=simsuite, sim=sim, subsim=subsim)

    paths1 = get_all_lens_model_paths(res_dir1,\
                                      check_if_workin_on_it=check_if_workin_on_it)
    paths2 = get_all_lens_model_paths(res_dir2,\
                                      check_if_workin_on_it=check_if_workin_on_it)

    # ── match lenses by identity key ─────────────────────────────────────────
    key2path1 = {_lens_key(p): p for p in paths1}
    key2path2 = {_lens_key(p): p for p in paths2}
    shared_keys = sorted(set(key2path1) & set(key2path2))

    if not shared_keys:
        raise RuntimeError(
            f"No shared lenses found between {res_dir1} and {res_dir2}.\n"
            f"Keys model1: {sorted(key2path1)[:5]}...\n"
            f"Keys model2: {sorted(key2path2)[:5]}..."
        )
    print(f"Found {len(shared_keys)} shared lenses.")

    columns_ttl = [
        "Sim Image",
        f"{model1} model",
        f"{model1} residual",
        f"P($\\theta_E$) {model1}",
        f"{model2} model",
        f"{model2} residual",
        f"P($\\theta_E$) {model2}",
    ]
    ncols = len(columns_ttl)
    n_lenses = len(shared_keys)

    nm_out = (res_dir1.parent.parent /
              f"compare_{model1}_vs_{model2}.pdf")

    warnings.filterwarnings("ignore")

    with PdfPages(nm_out) as pdf:
        for page_start in range(0, n_lenses, lines_per_page):
            batch      = shared_keys[page_start: page_start + lines_per_page]
            nrows_page = len(batch)

            fig, axes = plt.subplots(
                nrows_page, ncols,
                figsize=(scale_fig * ncols, scale_fig * nrows_page),
                squeeze=False)

            for i_row, key in enumerate(batch):
                print(f"  Processing {key} ...")
                mrd1 = key2path1[key]
                mrd2 = key2path2[key]
                try:
                    lens1, kw_data1 = _load_lens_and_data(mrd1, model1)
                    lens2, kw_data2 = _load_lens_and_data(mrd2, model2)

                    plot_comparison_row(
                        axes, i_row, nrows_page,
                        lens1, kw_data1, model1,
                        lens2, kw_data2, model2,
                        columns_ttl, _rnd=_rnd)

                    _free(lens1, kw_data1)
                    _free(lens2, kw_data2)

                except Exception as e:
                    print(f"  Failed {key}: {e}")
                    for col in range(ncols):
                        axes[i_row][col].set_axis_off()
                        axes[i_row][col].text(
                            0.5, 0.5, f"FAILED\n{e}",
                            ha="center", va="center",
                            transform=axes[i_row][col].transAxes,
                            fontsize=7, color="red")

            fig.suptitle(
                f"{get_model_title(model1)}  vs  {get_model_title(model2)}",
                fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            del fig
            gc.collect()
            print(f"Saved page {page_start // lines_per_page + 1}")

    print(f"Done → {nm_out}")
