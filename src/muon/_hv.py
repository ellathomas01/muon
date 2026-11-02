"""Experimental, opt-in HoloViews backend for :mod:`muon.pl`.

``holoviews`` and ``hv-anndata`` are imported lazily so that they stay optional
dependencies: nothing here is imported unless a plotting function is called with
``backend="holoviews"``.

HoloViews does not understand :class:`~mudata.MuData` (``hv-anndata`` only
registers an interface for :class:`~anndata.AnnData`), so the plotting functions
in :mod:`muon.pl` first gather what they need into a temporary ``AnnData`` and
hand that to the functions in this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import holoviews as hv
    from anndata import AnnData

_INSTALL_HINT = (
    "The 'holoviews' plotting backend needs holoviews, hv-anndata and bokeh. "
    "Install them with `pip install holoviews hv-anndata bokeh` "
    "(hv-anndata currently needs pre-release versions: add `--pre`)."
)

_registered = False


def _import_holoviews() -> tuple[Any, Any]:
    """Import holoviews and hv-anndata lazily; register the AnnData interface once."""
    global _registered
    try:
        import holoviews as hv
        from hv_anndata import A, register
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e

    if not _registered:
        register()
        _registered = True
    # Make sure the bokeh backend is loaded so `.opts(tools=...)` etc. are valid.
    if "bokeh" not in getattr(hv.Store, "renderers", {}):
        hv.extension("bokeh")
    return hv, A


def _as_list(color: str | Sequence[str] | None) -> list[str | None]:
    """Turn the ``color`` argument into a list with one entry per plot (``[None]`` for no colour)."""
    if color is None:
        return [None]
    if isinstance(color, str):
        return [color]
    return list(color)


def _check_in_obs(ad: AnnData, keys: Sequence[str | None], verb: str) -> None:
    """Raise if any of ``keys`` is not a column of ``ad.obs``."""
    missing = [k for k in keys if k is not None and k not in ad.obs.columns]
    if missing:
        raise NotImplementedError(
            f"{missing} not found in .obs. The holoviews backend can only {verb} .obs columns "
            "for a plain AnnData; for a MuData, features from any modality are supported."
        )


def _scatter(hv: Any, A: Any, ad: AnnData, x: Any, y: Any, color: str | None, user_opts: dict[str, Any]) -> hv.Scatter:
    """One scatter plot of the ``x`` and ``y`` accessors, coloured by an obs column or not coloured."""
    if color is None:
        plot = hv.Scatter(ad, x, [y])
        opts: dict[str, Any] = {"aspect": "square", "tools": ["hover"]}
    else:
        plot = hv.Scatter(ad, x, [y, A.obs[color]])
        opts = {"aspect": "square", "tools": ["hover"], "title": color, "color": A.obs[color]}
        values = ad.obs[color]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            # continuous annotation
            opts.update(colorbar=True, cmap="viridis")
        else:
            # categorical annotation
            opts["legend_position"] = "right"
    opts.update(user_opts)
    return plot.opts(backend="bokeh", **opts)


def _combine(hv: Any, plots: list[hv.Scatter]) -> hv.Scatter | hv.Layout:
    """Return a single plot as is, and several plots as a layout with at most three columns."""
    if len(plots) == 1:
        return plots[0]
    return hv.Layout(plots).cols(min(len(plots), 3))


def hv_embedding(
    ad: AnnData, basis: str, color: str | Sequence[str] | None = None, **kwargs: Any
) -> hv.Scatter | hv.Layout:
    """Interactive embedding scatter plot of the cells in ``ad``.

    Args:
        ad: AnnData object holding the embedding in ``.obsm`` and everything that is used for colouring in ``.obs``.
        basis: Key in ``ad.obsm`` (``"X_"`` is prepended if needed). The first two columns are plotted.
        color: Column(s) of ``ad.obs``. One plot is made per column.
        **kwargs: Forwarded to ``.opts()`` of each HoloViews element, overriding the defaults.
            These are *bokeh* options, not scanpy/matplotlib ones.

    Returns:
        A single :class:`holoviews.Scatter` or, for several colours, a :class:`holoviews.Layout`.
    """
    hv, A = _import_holoviews()

    if basis not in ad.obsm and f"X_{basis}" in ad.obsm:
        basis = f"X_{basis}"
    if basis not in ad.obsm:
        available = ", ".join(ad.obsm) or "none"
        raise KeyError(f"Basis {basis!r} is not present in .obsm (available: {available}).")
    if np.shape(ad.obsm[basis])[1] < 2:
        raise ValueError(f"Basis {basis!r} needs at least two dimensions to be plotted.")

    colors = _as_list(color)
    _check_in_obs(ad, colors, "colour by")

    x, y = A.obsm[basis][0], A.obsm[basis][1]
    return _combine(hv, [_scatter(hv, A, ad, x, y, c, kwargs) for c in colors])


def hv_scatter(
    ad: AnnData, x: str, y: str, color: str | Sequence[str] | None = None, **kwargs: Any
) -> hv.Scatter | hv.Layout:
    """Interactive scatter plot of two ``.obs`` columns of ``ad``.

    Args:
        ad: AnnData object holding ``x``, ``y`` and everything that is used for colouring in ``.obs``.
        x: Column of ``ad.obs`` for the x axis.
        y: Column of ``ad.obs`` for the y axis.
        color: Column(s) of ``ad.obs``. One plot is made per column.
        **kwargs: Forwarded to ``.opts()`` of each HoloViews element, overriding the defaults.
            These are *bokeh* options, not scanpy/matplotlib ones.

    Returns:
        A single :class:`holoviews.Scatter` or, for several colours, a :class:`holoviews.Layout`.
    """
    hv, A = _import_holoviews()

    colors = _as_list(color)
    _check_in_obs(ad, [x, y, *colors], "plot")

    return _combine(hv, [_scatter(hv, A, ad, A.obs[x], A.obs[y], c, kwargs) for c in colors])
