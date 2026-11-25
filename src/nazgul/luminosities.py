"""Rest-frame dust-free stellar luminosities of SWIFT star particles.

SWIFT computes per-star-particle AB luminosities at runtime from BC03
(GALAXEV) and writes them to the snapshot as the "Luminosities" named
columns, one per GAMA band, already divided by 3631 Jy — i.e. the stored
values are rest-frame maggies (SWIFT ``src/stars/EAGLE/stars_io.h``). No
dust attenuation is applied on the simulation side.

This module is deliberately stdlib-only, so that light extraction
environments (e.g. the COSMA stage-2 venv) can share this single
implementation with ``nazgul.Translator.COLIBRE`` without pulling in the
full nazgul dependency tree.
"""
import warnings

GAMA_LUM_BANDS = ("GAMA_u", "GAMA_g", "GAMA_r", "GAMA_i", "GAMA_z",
                  "GAMA_Y", "GAMA_J", "GAMA_H", "GAMA_K")


def read_star_luminosities(stars, bands=None, name=None):
    """Read per-star-particle luminosities from a swiftsimio ``stars`` dataset.

    Returns ``{band: swiftsimio array}`` for the requested bands the snapshot
    carries, aligned with ``stars``; missing columns are skipped with a
    warning. ``bands`` defaults to the nine GAMA bands. ``name`` only labels
    the warnings.
    """
    label = f"{name}: " if name else ""
    lums = getattr(stars, "luminosities", None)
    if lums is None:
        warnings.warn(f"{label}snapshot has no stars.luminosities; skipped")
        return {}
    out = {}
    for band in (bands or GAMA_LUM_BANDS):
        try:
            out[band] = getattr(lums, band)
        except AttributeError:
            warnings.warn(f"{label}no stars.luminosities column {band!r}; skipped")
    return out
