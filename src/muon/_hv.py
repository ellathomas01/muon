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


def _scatter(hv: Any, A: Any, ad: AnnData, basis: str, color: str | None, user_opts: dict[str, Any]) -> hv.Scatter:
    x, y = A.obsm[basis][0], A.obsm[basis][1]
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

    colors: list[str | None]
    if color is None:
        colors = [None]
    elif isinstance(color, str):
        colors = [color]
    else:
        colors = list(color)

    missing = [c for c in colors if c is not None and c not in ad.obs.columns]
    if missing:
        raise NotImplementedError(
            f"{missing} not found in .obs. The holoviews backend can only colour by .obs columns "
            "for a plain AnnData; for a MuData, features from any modality are supported."
        )

    plots = [_scatter(hv, A, ad, basis, c, kwargs) for c in colors]
    if len(plots) == 1:
        return plots[0]
    return hv.Layout(plots).cols(min(len(plots), 3))
