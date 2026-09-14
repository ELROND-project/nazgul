# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository — the
**Nazgul** package (`ELROND-project/nazgul`, formerly `EAGLE_Lensing`).
Author: Giacomo Queirolo, ELROND project.

Nazgul takes hydrodynamical cosmological simulation output (EAGLE, COLIBRE, or
an analytic test suite), finds galaxies that act as strong lenses, computes
their lensing from the **individual particles** on lenstronomy, simulates
observations, and fits parametric models to the simulated images.

## Environment

The environment recipe is `nazgul_env.yaml` plus an editable install:

```bash
mamba env create -f nazgul_env.yaml   # conda works too
mamba activate nazgul_env
python -m pip install -e .
```

`pyproject.toml` declares `requires-python >= 3.11` and **no dependencies**: the
dependency set lives in `nazgul_env.yaml` alone, so installing from
`pyproject.toml` by itself yields an unusable package. Nazgul is also consumed
as an editable path dependency by other projects; in that case the consuming
project's lockfile supplies the dependencies and the conda env is not involved.

Heavy non-PyPI dependencies:

- `python_tools` (GiacomoQueirolo) — `load_whatever`, `LoadClass`, `mkdir`,
  `to_dimless`, `ensure_unit`, and the `tools_WOI` cross-machine lock;
- the **ELROND fork of lenstronomy** — supplies the `EPL_BOXYDISKY_ELL`,
  `EPL_MULTIPOLE_M1M3M4_ELL` and `LOS_MINIMAL` profiles the modelling scripts
  fit; the upstream release does not have them;
- `swiftgalaxy` / `swiftsimio` — COLIBRE and SOAP reading;
- `photutils` (isophote fitting), `chainconsumer`/`emcee`/`corner` (chains),
  `numba` (the AMR kernel).

## Git

Branches: `main` (shared with collaborators) and `sam` (personal working
branch). `.gitignore` covers `RingBearer/` (simulation data), the
`__pycache__/` dirs and `Translator/EAGLE/.eagle_account.dll` — but **not**
`src/nazgul/results/` or `src/nazgul/tmp/`, which importing the package
creates. They stay invisible only while empty; as soon as a run writes into
them they show up as untracked.

## Running scripts, and testing

Nearly every top-level and `Modelling/` script is an argparse CLI; run them as
modules so the `nazgul.` imports resolve:

```bash
python -m nazgul.stat_kappa_r -snap 27 -ss COLIBRE
python -m nazgul.Modelling.model_simNoShear -rt 1 -nl 5   # -rt 1 = short test run
```

Common flags across scripts: `-snap/--snap` (list), `-sim`, `-ss/--simsuite`,
`-ssim/--subsim`, `-rt/--run_type` (0 = full PSO/MCMC, 1 = 3-iteration smoke
test, 2 = intermediate), `-nl/--n_lenses`, `-mtE/--min_thetaE`.

**There is no pytest suite**, despite what earlier revisions of this file said.
`src/nazgul/test/` is pre-restructuring legacy — its files import `remade_gal`,
`Gen_PM_PLL_AMR`, `project_gal_AMR`, none of which exist any more. The
top-level `test_*.py` files are ad-hoc study scripts (e.g. `test_vdisp.py` is
the σ_v vs θ_E study), not unit tests. The only `def test_` in the package is
`test_split_indices` in `AMR2D_PLL.py`.

So the cheap check after editing is an import:

```bash
python -c "import nazgul.project_gal, nazgul.mount_doom.lens_system"
```

Side effect to know about: importing `nazgul.pathfinder` (i.e. importing almost
anything) **creates `src/nazgul/results/` and `src/nazgul/tmp/`** via `mkdir()`
at module level.

## Architecture

### Configuration

All runtime parameters live in `configurations.py`: `nazgul_path`,
`nazgul_path_origin`, `forecast_telescope` (`'Euclid'`), `std_simsuite`,
`min_z`, `max_z`, `min_mass`, `min_thetaE`, `pixel_num`, `verbose`,
`z_source_max`, `scale_tE`.

- `std_simsuite = SimSuiteNames[0]` → **EAGLE** is the current default (the
  allowed list is `["EAGLE", "COLIBRE", "ANL_TEST"]`; index 1 is COLIBRE).
  Most scripts also accept `-ss/--simsuite` so you rarely need to edit this.
- Set `nazgul_path_origin` when loading pickles produced on another machine
  (e.g. cluster → local); `_resolve_gal_path()` in
  `mount_doom/cracks_of_doom.py` remaps stored paths, handling relative,
  same-machine and cross-machine absolute paths by re-anchoring on
  `RingBearer`.

### Data flow

1. **Translator** (`Translator/`) — dispatch layer over simulation-specific
   readers. `Translator/__init__.py` discovers suites by scanning its own
   subdirectories, and `translator.py`'s `get_sim_func(simsuite, name)` pulls
   the named object out of `nazgul.Translator.{simsuite}.particle_galaxy`. The
   uniform façade is `PartGal`, which proxies attribute access to the
   suite-specific `SimPartGal`. Available simulations:

   | suite | `sim` | `subsim` | galaxy key / dir name |
   |---|---|---|---|
   | EAGLE | `RefL0025N0752`, `RefL0012N0188`, `RefTuto` | — | `{"Gn":…,"SGn":…}` → `Gn{Gn}SGn{SGn}` |
   | COLIBRE | `L0050N0752` | `{"L0025N0752":["THERMAL_AGN_m5"], "L0050N0752":["THERMAL_AGN_m6"], "L0050N1504":["THERMAL_AGN_m5"]}` | `{"soap_index":…}` → `Gn{soap_index}` |
   | ANL_TEST | `SIS`, `SIE` | — | `TEST_GAL_{prof}_tE…_nS…` |

   Note the `subsim` dict still carries keys for boxes not listed in `sim`.
   `Translator/__init__.py` also exports `std_sim`, `std_subsim`, `test_sim`,
   `tutorial_sim` (`RefTuto`), and `get_simsuite_code`/`get_simsuite_from_code`
   (EAGLE→`EGL`, COLIBRE→`CLB`, ANL_TEST→`ANLT`).

   **COLIBRE specifics** (`Translator/COLIBRE/get_Gal.py`): reads SOAP through
   `swiftgalaxy`/`swiftsimio` from a **hardcoded**
   `colibre_base_path = /cosma8/data/dp004/colibre/Runs/`, catalogue
   `SOAP-HBT/halo_properties_{snap}.hdf5`, membership
   `SOAP-HBT/colibre_with_SOAP_membership_{snap}.hdf5`. Galaxy *selection*
   (`min_mass`, `min_mass_stars`, `min_vel_disp`, `min_hmr`) runs on SOAP's
   `bound_subhalo` group, with σ taken from
   `stellar_velocity_dispersion_matrix` as √(tr/3) (Vandenbroucke+24 eq. 17).
   `get_vdisp()` in `Translator/COLIBRE/particle_galaxy.py` instead
   **recomputes** σ from star-particle velocities (eqs. 15 & 17), mass-weighted
   about the stellar centre-of-mass velocity, returned in km/s. EAGLE's
   `get_vdisp()` reads `StellarVelDisp` from the SQL catalogue instead. So
   "the σ nazgul uses" is ambiguous: the selection cut and the per-galaxy value
   come from different definitions — check which one a given result depends on.

2. **Galaxy projection** (`project_gal.py`) — `ProjGal` wraps `PartGal`
   (projection index 0/1/2 = x/y/z), builds 2-D density maps with AMR
   (`AMR2D_PLL.py`, numba-parallel), locates the densest coordinate, and tests
   supercriticality for a source up to `z_source_max`. Cached per projection as
   `projection_{index}.pkl`. Useful helpers: `cells2SigEnclRad` (enclosed Σ(<r))
   vs `cells2SigRad` (local Σ(r)) vs `cells2MRad`, `get_rough_thetaE`,
   `theta_E_from_AMR_densitymap`.

3. **Particle lenses** (`particle_lenses.py`) — turns particle positions and
   masses into lenstronomy lens parameters. Two profiles, both defined here:
   `default_kwlens_part_AS = {"type":"AS","theta_cAS":5e-3}` (arcsinh, default)
   and `default_kwlens_part_PM = {"type":"PM"}` (point mass). Hierarchy:
   `PartLens_basis` (abstract) → `PartLens` (used by `GalLens`) and
   `PartLensExpanded` (adds `kw_add_lenses` for LOS perturbers).

4. **Defaults & observation config** (`mount_doom/cracks_of_doom.py`) —
   `kwargs_source_default` is an **absolute**-magnitude Sérsic ellipse
   (`{'abs_magnitude': -21, 'R_sersic': .1, 'n_sersic': 3, …}`, converted to
   apparent magnitude then to amplitude in `get_kwargs_sourceSim`),
   `kwargs_band_sim` (idealised HST-like: no read noise, 5400 s, gain 2.5,
   zero-point 30, `psf_type: NONE`), `kw_prior_z_source_stnd`. Also the base
   classes `BasicLensPart` (on `basic_gal.BasicGal`), `LoadLens(path)` and
   `_resolve_gal_path()`.

5. **GalLens** (`mount_doom/generate_gal_lens.py`) — the per-galaxy lens
   computation: runs the projection, sets up `PartLens`, and lazily computes
   and caches `alpha_map`, `kappa_map`, `hessian`, `psi_map`. Pass
   `ignore_OoBErr=True` to swallow lenstronomy out-of-bound errors. Each
   instance carries an `_identity()` tuple (galaxy, particle-lens kwargs,
   `pixel_num`, `z_source_max`, `scale_tE`) hashed into the filename, so
   changing any of them produces a new pickle rather than silently reusing an
   old one.

6. **LensSystem** (`mount_doom/lens_system.py`) — restructured from the old
   `generate_particle_lens_dom.py`. Wraps a `GalLens`, adds optional extra
   lenses (`kwargs_add_lenses`) through lenstronomy's `LensModel`, simulates
   the observation via `SimAPI`, samples a source position inside the tangential
   caustic (with a per-lens deterministic `_rnd_seed` derived from the pickle
   path) and produces the lensed image by interpolating the deflection map.

7. **Modelling** (`Modelling/`) — fits parametric models to the simulated
   images with lenstronomy's inference engine. `Modelling/lib_models.py` is the
   shared library: `setup_lens`, `setup_sim_obs`, masks
   (`get_lens_mask`, SEAGLE-style masking in `masking.py`), priors
   (`add_gaussian_tE_prior`), lens selection (`get_lenses2model`, which caches
   its selection in `cat_lens2model.dll` and recomputes when the selection
   kwargs change), result-dir management, and PSO/MCMC defaults
   (`n_it_std=1000`, `n_part_std=300`, `n_burn_std=700`, `n_run_std=7000`).

   Current entry points write under `results/models/` (`model_res_base`):

   | script | `lens_model_list` | results subdir |
   |---|---|---|
   | `model_simNoShear.py` | `EPL`, `LOS_MINIMAL` | `simNoShear/` |
   | `model_simNoShear_gausstE.py` | `EPL`, `LOS_MINIMAL` | `simNoShear_gausstE/` |
   | `model_SNS_multipole[_gausstE].py` | `EPL_BOXYDISKY_ELL`, `LOS_MINIMAL` | `SNS_multipole[_gausstE]/` |
   | `model_SNS_m134_gausstE.py` | `EPL_MULTIPOLE_M1M3M4_ELL`, `LOS_MINIMAL` | `SNS_m134_gausstE/` |

   "SNS" = simNoShear: the lens is *simulated* without shear but *modelled*
   with LOS, to study internal shear à la Etherington.
   `model_simNoShear_cored.py` raises on import — the cored elliptical profile
   is not implemented.

   The older exploratory scripts (`model_sim_lens.py`, `model_ext_shear*.py`,
   `model_fitLOS*.py`, `model_allLOS.py`, `modelling_sis_sie.py`,
   `modelling_2sis.py`, `modelling_wLOS.py`, `modelling_severals_wLOS.py`)
   write under **`tmp/models/`** instead, and `modelling_updated.py` raises
   `NotImplementedError`.

   Per-lens output lands in
   `{res_dir_base}/{SimCode}_{Sim}[_{subsim}]/snap_{NNN}_{lensname}/`
   (`get_res_dir` + `get_model_res_dir`); `run_type != 0` adds a `test/` level.

   **Cross-machine lock:** modelling scripts call `set_workin_on_it()` and
   readers call `is_someone_workin_on_it()` (`python_tools.tools_WOI`) on the
   result dir, so two hosts sharing a synced tree do not model the same lens
   twice. Analysis scripts take `-nciwoi/--not_check_if_workin_on_it` to ignore it.

   Chain analysis: `combined_modelling_results.py` (all lenses, chainconsumer,
   LOS shear statistics) and `combined_modelling_results_one_lens.py`;
   plus `plot_model_corner.py`, `compare_models_tE.py`, `image_model_vs_sim.py`,
   `los_res_study.py`, `check_chi2_res.py`.

8. **LensPop** (`LensPop/`) — pre-computed lens population catalogues for DES,
   Euclid and LSST, downloaded on demand from `tcollett/LensPop` by
   `likelihood_z_source.py`; the active telescope is `forecast_telescope` in
   `configurations.py`. Used as the z_source prior.

9. **Isocontour / ellipticity analysis** — renamed since the last revision of
   this file: `fit_iso_ell.py` (was `isodens.py`) fits elliptical isophotes to
   κ or ψ maps with `photutils.isophote`, plus `skimage.find_contours` contour
   fitting and a `rescale_pot`/`rescale_kappa` pre-processing step;
   `utils_fit_ell.py` (was `fit_ellipses.py`) supplies fast moment-based
   initial guesses; `isocont_stat.py` (was `isodens_stat.py`) is the batch
   driver that produces `kw_res_isodens_prj*` per galaxy; `isocont_stat_plots.py`
   turns those into population plots; `kappa_isocontVSlensmodel.py` compares
   isocontour ellipticity against the fitted lens model.

10. **Profile / dispersion statistics** — population statistics over the
    computed lenses:
    `stat_gamma_1Ddens.py` (cored-power-law γ and core size from the 1-D mass
    profile, cross-checked against model chains), `stat_kappa_r.py` (κ(r) and
    enclosed κ(<r) profiles from AMR, `-no_enc/--not_enclosed` toggling between
    them; note the low central resolution of the AMR profiles),
    `plot_AMRxpart.py` and `reproduce_fig5SEAGLE.py` (SEAGLE-I Fig. 5-style
    surface-density profiles by particle type), `test_vdisp.py` (θ_E vs
    θ_E(σ_v), including the cored-profile correction
    θ_E/θ_E(σ_v) = √(1 − 1/κ_max)), `vdisp_vs_Aperture.py` (σ_v aperture
    dependence, EAGLE SQL), `multipole_tension.py` (do multipoles reduce
    tension with γ_LOS), `compare_cm_part.py` (baryon vs DM centre of mass),
    `stat_lenses.py` (lens catalogue helpers: `get_all_gallens`,
    `get_all_gallens_paths`, deduplication of re-computed lenses).

11. **`output_analysis/`** — two standalone notebooks that do **not** import
    nazgul (analytic modelling of what the pipeline produced):
    `radial profiles/modelling_COLIBRE_cores.ipynb` (disk-averaged κ̄(θ), cored
    profiles fitted to COLIBRE lenses) and
    `drifting elliptical potential/cored_drifting_elliptical_power_law.ipynb`
    (cored elliptical power-law potential, ellipticity drift with radius).

### Data storage tree

Per-galaxy cached objects go under `src/nazgul/RingBearer/` (gitignored),
model fits under `src/nazgul/results/`, scratch under `src/nazgul/tmp/`
(see `pathfinder.py`: `std_data_dir`, `results_dir`, `tmp_dir`). None of them
are in the repo: they are created on first run, and on HPC `ParticleData/` is
normally a symlink into the simulation's own storage rather than a copy.

```
RingBearer/
  {SimSuite}/                     EAGLE/ | COLIBRE/ | ANL_TEST/
    {Sim}/                        e.g. RefL0025N0752/, L0050N0752/
      [{subsim}/]                 COLIBRE only, e.g. THERMAL_AGN_m6/
        CatGal/                   galaxy catalogue pickles
        CatLens/                  lens catalogue pickles
        snap_{NNN}/
          ParticleData/           HDF5 snapshot files (symlinks on HPC)
          {galname}/              Gn{Gn}SGn{SGn} (EAGLE) | Gn{soap_index} (COLIBRE)
            Gal/                  PartGal pickle
            Projection/           projection_{0,1,2}.pkl  (ProjGal)
            Sub/                  Sub_Lens_{galname}_Prj{i}_{hash}.pkl  (GalLens)
            Dom/                  high-level lens-model computation
            LensSystem/           LS_Lens_{galname}_Prj{i}_{hash}.pkl  (LensSystem)

results/models/{model}/{SimCode}_{Sim}[_{subsim}]/snap_{NNN}_{lensname}/
tmp/                              scratch, intermediate plots, older model runs
```

The `_{hash}` suffix is base64 of the object's `_identity()`; a stale pickle
with different settings will not be picked up, it will simply be recomputed.

### Key classes and the `reload` flag

`PartGal` proxies to `SimPartGal`; `ProjGal` proxies to `PartGal`; `GalLens`
inherits `BasicLensPart`/`BasicGal`; `LensSystem` wraps a `GalLens`. All
persistent classes serialise with `dill`.

`reload=True` loads a cached `.pkl`; `reload=False` recomputes from scratch
(slow — re-reads HDF5 particle data and re-runs AMR). Large attributes
(`Gal`, `PartLens`, `cosmo`, `kwargs_lens`, `lens_prof`) are stripped before
serialisation and rebuilt by `unpack()` / `LoadLens()`. Several classes carry
`MONKEY_PATCH` branches that reconstruct attributes missing from older
pickles — expect to see them print during loads.

### Entry points

```python
# Batch: forge a lens from every galaxy matching the filters.
# NOTE the name: the older wrapper_get_all_lens no longer exists.
from nazgul.mount_doom.generate_gal_lens import wrapper_forge_all_lenses, wrapper_get_rnd_lens
all_lenses = wrapper_forge_all_lenses(
    kw_galpart={"sim":"RefL0050N0752","min_z":0.1,"max_z":0.3,
                "min_vel_disp":120,"min_hmr":1,"min_mass_stars":1.76e10*.6777},
    kw_lenspart={"ignore_OoBErr":True},
    reload=True)

# Random single lens (useful for testing)
lens = wrapper_get_rnd_lens(reload=True)

# Single galaxy — EAGLE keys on Gn/SGn, COLIBRE on soap_index
from nazgul.Translator.translator import PartGal
from nazgul.mount_doom.generate_gal_lens import GalLens
Gal  = PartGal({"Gn":23,"SGn":0}, simsuite="EAGLE", sim="RefL0025N0752", snap="27", reload=True)
lens = GalLens(Gal, projection_index=0, reload=True)
lens.run()
lens.kappa_map   # 2-D numpy array

# Load previously computed lenses for a snapshot
from nazgul.stat_lenses import get_all_gallens
lenses = get_all_gallens(snaps=[27], sim="RefL0025N0752")
```

## Stale imports: what is fixed, what still is not

The package was restructured repeatedly (`Translator/`, `mount_doom/`,
`Modelling/`, and the isodens→fit_iso_ell rename) and a number of scripts were
left pointing at the old names. Most were repaired on 2026-09-14; the audit that
finds them is a static walk of every intra-package import, checking that the
target module exists and exports the name (`ast`-based, no side effects — worth
re-running after any rename).

**Still broken — these need a rewrite, not an import fix:**

| Module | Why |
|---|---|
| `test_lens_part.py`, `test_lens_part_LOS_sim.py`, `test_masking.py` | want `LensPart` from `mount_doom/generate_particle_lens_dom.py`. Its successor `LensSystem` takes `kwargs_gallens` (a dict of `GalLens` kwargs) rather than a `PartGal`, has `setup()` rather than `run()`, and no `.image_sim` attribute — the simulated image now comes from `lens.get_lensed_image(...)`, as `Modelling/lib_models.setup_lens` does it. They also call `PartGal(5,0,…)` with the pre-Translator positional signature. |
| `test_AnlVsNumLOS.py` | `LensPartLOS` exists nowhere in the package any more (`lens_part_LOS.py` is down to `get_kw_los`), and `mount_doom/generate_particle_lens.py` is gone — `get_extents` and `kw_prior_z_source_minimal` moved to `mount_doom/cracks_of_doom.py`. |
| `lens_part.py` | WIP `SinglePlaneLensPart` superstructure built on `LensPart`, `LensPartLOS` and a `bounds_error` decorator, none of which survive. Its imports are still bare pre-package ones (`from project_gal import …`). |
| `test/` (whole directory) | pre-restructuring legacy: imports `remade_gal`, `Gen_PM_PLL*`, `project_gal_AMR`, `proj_part`, `fnct`, `get_gal_indexes` as top-level modules. The first four no longer exist in any form. |
| `Tutorial/Tutorial.ipynb` | out of date end to end — `nazgul.fnct`, `nazgul.particle_galaxy`, `nazgul.isodens`, `mount_doom.generate_particle_lens`, plus removed helpers (`projection_main_AMR`, `getDensAtRad`, `plot_amr_cells`). Read it for the intended flow, not as runnable code. |

Deliberate import-time raises (not bugs): `plot_one_gal.py` (WIP — go through
the Translator instead of reaching for `Gal.dm` directly),
`Modelling/model_simNoShear_cored.py`, `Modelling/modelling_updated.py`, and the
`__main__` block of `fit_iso_ell.py` (`RuntimeError("Outdated")`).

**Fixed, for the record** (so old scripts and commits still read):
`fit_iso_ell` (`util_fit_ell`→`utils_fit_ell`), `isocont_stat` (`isodens`→
`fit_iso_ell`), `kappa_isocontVSlensmodel` (`modelling_severals`→
`Modelling.modelling_severals_wLOS`), `create_all_lenses`
(`wrapper_get_all_lens`→`wrapper_forge_all_lenses`), `Modelling/model_sim_lens`
(`wrapper_get_rnd_lens` moved to `generate_gal_lens`), `Translator/translator`
(EAGLE's `get_rnd_kw_gal` is `get_rnd_gal_indexes`), `plotting_all` and
`test_vdisp` (bare local imports), `test_tEsis_parts_pos` / `vdisp_vs_Aperture`
(`get_tEsis`/`get_tE_sis` → `get_tE_SIS` / `_get_tE_SIS`, picked by signature),
`compare_cm_part` (`particle_galaxy`→`Translator.translator`, and `get_rnd_PG`
now needs a simsuite), `Tutorial/get_gal_indexes_tutorial` (`std_gal_dir` is
gone from `pathfinder`; the tutorial now derives its own from
`Tutorial/data_Tuto/{sim}/Gals`), and the `nazgul.lens_part_los` →
`nazgul.lens_part_LOS` case mismatches, which only worked because macOS is
case-insensitive and would have failed on a cluster.

`plot_model_corner.py` and `combined_modelling_results_one_lens.py` both import
`get_all_lens_models`, which had been parked inside the triple-quoted block at
the end of `combined_modelling_results.py`. That one function is now live code
again (next to `get_all_lens_model_paths`); the old `__main__` block parked
alongside it was left commented. `plot_model_corner.py`'s four
`from nazgul.model_* import res_dir_base` branches were replaced by
`get_res_dir(model)`, which maps a model name to its `Modelling/` module *and*
appends the `{SimCode}_{Sim}[_{subsim}]` level where the `snap_*` dirs actually
live — the old code globbed one level too high.

Rename map, for reading old commits and old scripts:

```
isodens.py                  -> fit_iso_ell.py
fit_ellipses.py             -> utils_fit_ell.py
isodens_stat.py             -> isocont_stat.py
particle_galaxy.py, fnct.py -> Translator/{suite}/…, Translator/translator.py
generate_particle_lens*.py  -> mount_doom/generate_gal_lens.py (GalLens),
                               mount_doom/lens_system.py (LensSystem)
model*.py, modelling*.py    -> Modelling/
reproduce_fig5SEAGLE.py     -> plot_AMRxpart.py  (a new reproduce_fig5SEAGLE.py has since been added)
```

## Adding a new simulation

Create `Translator/{NewSim}/` with:

- `__init__.py` defining `simsuite_name`, `simsuite_short_name`, `sim` (list),
  optionally `subsim` (dict), `part_type_list` and `check_part_type()`;
- `particle_galaxy.py` exposing the names `get_sim_func` looks up:
  `SimPartGal`, `get_kw_SimPartGal`, `get_rnd_SPG`, `get_all_SPG`, `Gal2MXYZ`,
  `Gal2MXYZ_part`, `gal_path2kwGal`, `get_z_snap`, `get_vdisp`
  (re-exporting them from a sibling module is fine — EAGLE imports `get_z_snap`
  from `fnct.py`, COLIBRE from `get_Gal.py`);
- `pathfinder.py` with `get_galname(...)`.

Then add the suite to `SimSuiteNames` in `configurations.py` and a branch to
`translate_galname()` in `Translator/pathfinder.py`. The suite is picked up
automatically by the directory scan in `Translator/__init__.py`.

## EAGLE data setup

`python -m nazgul.Translator.EAGLE.setup_eagle_data` interactively
downloads EAGLE particle snapshots — **run it with `src/nazgul/` as the working
directory**, it shells out to the relative path
`Translator/EAGLE/download_eagle_data.sh`. Credentials are dilled into
`Translator/EAGLE/.eagle_account.dll` (gitignored). Galaxy selection goes
through the EAGLE SQL database (`sql_connect.py`, `get_gal_indexes.py`), which
is where `min_vel_disp` maps to `gal.StellarVelDisp`.

## Tutorial

`src/nazgul/Tutorial/Tutorial.ipynb` is an end-to-end walkthrough on
pre-packaged data under `Tutorial/data_Tuto/` (EAGLE `RefTuto`, snap 20). It
predates the restructuring and its imports no longer resolve — read it for the
intended flow, not as runnable code.
