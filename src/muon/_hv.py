"""Plotting under scanpy's ``ScanpyV2Preview`` preset, where scanpy's plotting functions return HoloViews elements.

Muon registers no HoloViews interface itself. ``sc.pl.hv_init`` registers the one from ``hv-anndata``, which reads a
:class:`~mudata.MuData` as well as an :class:`~anndata.AnnData` since version 0.1.3 (``MuData`` understands the same
``anndata.acc`` references, e.g. ``mdata[A.obs["x"]]``). ``scanpy`` and ``holoviews`` are imported lazily, so nothing
here is imported unless the preset is active.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    import holoviews as hv
    from anndata import AnnData
    from anndata.acc import AdAcc, AdRef
    from mudata import MuData


def _scanpy_preset_is_v2() -> bool:
    """Whether the installed scanpy has presets and the active one is ``ScanpyV2Preview``."""
    import scanpy as sc

    preset_enum = getattr(sc, "Preset", None)  # scanpy versions without presets have no `Preset`
    v2 = getattr(preset_enum, "ScanpyV2Preview", None)
    return v2 is not None and getattr(sc.settings, "preset", None) is v2


def _names_only(obj: object) -> bool:
    """Whether ``obj`` is a name or a collection of names, and not an ``anndata.acc`` reference or a list of them.

    A collection is a list, tuple, set, array, index and so on. Iterators count as names, as they cannot be inspected
    without being consumed.
    """
    if isinstance(obj, str):
        return True
    if isinstance(obj, Collection):
        return all(isinstance(k, str) for k in obj)
    return isinstance(obj, Iterator)


def _has_names(obj: object) -> bool:
    """Whether ``obj`` is a name or a collection with a name in it, i.e. is not made of references only."""
    if isinstance(obj, str):
        return True
    if isinstance(obj, Collection):
        return any(isinstance(k, str) for k in obj)
    return isinstance(obj, Iterator)


def check_accessors(*args: object) -> bool:
    """Whether any of ``args`` uses accessor references; raise if they are used without ``ScanpyV2Preview``."""
    used = any(a is not None and not _names_only(a) for a in args)
    if used and not _scanpy_preset_is_v2():
        msg = "Accessor references are only supported with scanpy's ScanpyV2Preview preset."
        raise ValueError(msg)
    return used


def require_only_accessors(**args: object) -> None:
    """Raise if any of ``args`` is or contains a name (for ``MuData`` input, where names are not resolved)."""
    names = [k for k, v in args.items() if v is not None and _has_names(v)]
    if names:
        msg = (
            f"With accessor references, {', '.join(f'`{k}`' for k in names)} must be given as accessor references "
            "too: names cannot be mixed with them for MuData input."
        )
        raise ValueError(msg)


def reject_unused_options(args: Sequence[object], **options: object) -> None:
    """Raise if options that resolve names are set but every argument in ``args`` is an accessor reference.

    ``layers`` and ``gene_symbols`` apply to the arguments given as names and are ignored for references. If there are
    only references, there is nothing for them to apply to, which is most likely a mistake.
    """
    given = [a for a in args if a is not None]
    set_options = [k for k, v in options.items() if v not in (None, False)]
    if set_options and given and not any(_has_names(a) for a in given):
        msg = (
            f"Options that only apply to names are set ({', '.join(f'`{k}`' for k in set_options)}), but every "
            "argument is an accessor reference, which already says where its data comes from."
        )
        raise ValueError(msg)


def _layer_triple(layers: str | tuple[str, str, str] | None) -> tuple[str | None, str | None, str | None]:
    """Expand ``layers`` as ``scanpy.pl.scatter`` does: a single name stands for x, y and the colour."""
    if layers is None or isinstance(layers, str):
        return (layers, layers, layers)
    if len(layers) != 3 or not all(isinstance(layer, str) for layer in layers):
        msg = "`layers` must be a layer name or a tuple of three layer names for AnnData input."
        raise ValueError(msg)
    return layers


def _v2_ref(ad: AnnData | MuData, A: AdAcc, key: str | AdRef, layer: str | None, gene_symbols: str | None) -> AdRef:
    """Turn a name into an ``hv-anndata`` reference: an ``.obs`` column, or a variable read from ``.X`` or a layer.

    As in ``scanpy``, ``gene_symbols`` is the ``.var`` column that variable names are looked up in. A name that is
    both an ``.obs`` column and a variable is ambiguous and raises an error. A reference is returned as it is.
    """
    if not isinstance(key, str):  # already an accessor reference
        return key

    if gene_symbols is None:
        var_names = [key] if key in ad.var_names else []
    elif gene_symbols in ad.var.columns:
        var_names = list(ad.var_names[(ad.var[gene_symbols] == key).to_numpy()])
    else:
        msg = f"`gene_symbols` column {gene_symbols!r} not found in .var."
        raise KeyError(msg)

    if key in ad.obs.columns:
        if var_names:
            msg = f"{key!r} is both an .obs column and a variable name; rename one of them."
            raise ValueError(msg)
        return A.obs[key]
    if not var_names:
        where = "variable name" if gene_symbols is None else f"value of .var[{gene_symbols!r}]"
        msg = f"{key!r} is neither an .obs column nor a {where}."
        raise KeyError(msg)
    if len(var_names) > 1:
        msg = f"{key!r} matches several variables in .var[{gene_symbols!r}]: {', '.join(map(str, var_names))}."
        raise ValueError(msg)

    if layer is None:
        return A.X[:, var_names[0]]
    if layer not in ad.layers:
        msg = f"Layer {layer!r} not found in .layers."
        raise KeyError(msg)
    return A.layers[layer][:, var_names[0]]


def _v2_refs(
    ad: AnnData | MuData,
    A: AdAcc,
    keys: str | Sequence[str] | AdRef | Sequence[AdRef] | None,
    layer: str | None = None,
    gene_symbols: str | None = None,
) -> AdRef | list[AdRef] | None:
    """Turn names into ``hv-anndata`` references (``None`` stays ``None``; one name gives one reference).

    References are returned as they are, and a list can mix names and references.
    """
    if keys is None:
        return None
    if isinstance(keys, str):
        items: list[str | AdRef] = [keys]
    elif _names_only(keys) or isinstance(keys, list | tuple):
        items = list(cast("Sequence[str | AdRef]", keys))
    else:  # a single accessor reference
        return cast("AdRef", keys)
    refs = [_v2_ref(ad, A, k, layer, gene_symbols) for k in items]
    return refs[0] if len(refs) == 1 else refs


def _v2_options(plot: hv.Scatter | hv.Layout, options: dict[str, object]) -> hv.Scatter | hv.Layout:
    """Add a hover tool (Bokeh) and apply extra keyword arguments as options of every scatter plot.

    The options are applied to the plots and not to a surrounding ``Layout``, which HoloViews does not support.
    Keyword arguments override the defaults, e.g. ``tools=["pan"]`` removes the hover tool.
    """
    import holoviews as hv

    if "bokeh" in getattr(hv.Store, "renderers", {}):
        plot = plot.opts(hv.opts.Scatter(tools=["hover"], backend="bokeh"))
    if options:
        plot = plot.opts(hv.opts.Scatter(**options))
    return plot


def v2_scatter(
    ad: AnnData | MuData,
    x: str | AdRef | None,
    y: str | AdRef | None,
    color: str | Sequence[str] | AdRef | Sequence[AdRef] | None = None,
    layers: str | tuple[str, str, str] | None = None,
    gene_symbols: str | None = None,
    **kwargs: object,
) -> hv.Scatter | hv.Layout:
    """Counterpart of ``scanpy.pl.scatter(ad, x=x, y=y, color=color)`` under the ``ScanpyV2Preview`` preset.

    ``x``, ``y`` and ``color`` are ``.obs`` columns or variable names, or accessor references such as
    ``A.X[:, "gene"]``. ``layers`` are the layers variables are read from for x, y and the colour (``.X`` by default)
    and ``gene_symbols`` is the ``.var`` column variable names are looked up in. Both apply to the names only, as a
    reference says where its data comes from, and it is an error to set them if there are only references. Extra
    keyword arguments are HoloViews options, not matplotlib ones.
    """
    import scanpy as sc

    if x is None or y is None:
        msg = "`x` and `y` are required with scanpy's ScanpyV2Preview preset."
        raise ValueError(msg)
    reject_unused_options((x, y, color), layers=layers, gene_symbols=gene_symbols)
    layer_x, layer_y, layer_color = _layer_triple(layers)
    A = sc.pl.hv_init(None)  # registers the AnnData interface; does not load a plotting backend
    kdims = [_v2_ref(ad, A, x, layer_x, gene_symbols), _v2_ref(ad, A, y, layer_y, gene_symbols)]
    color_refs = _v2_refs(ad, A, color, layer_color, gene_symbols)
    return _v2_options(sc.pl.scatter(ad, kdims, color=color_refs), kwargs)


def v2_embedding(
    ad: AnnData | MuData,
    basis: str | AdRef,
    color: str | Sequence[str] | AdRef | Sequence[AdRef] | None = None,
    layer: str | None = None,
    gene_symbols: str | None = None,
    **kwargs: object,
) -> hv.Scatter | hv.Layout:
    """Counterpart of ``scanpy.pl.embedding(ad, basis=basis, color=color)`` under the ``ScanpyV2Preview`` preset.

    The first two components of ``ad.obsm[basis]`` are plotted; ``basis`` can also be an accessor reference to the two
    coordinates, e.g. ``A.obsm["X_umap"][:, [0, 1]]``. ``color`` are ``.obs`` columns or variable names, or accessor
    references; variables are read from ``layer`` (``.X`` by default) and looked up in the ``.var`` column
    ``gene_symbols``. Both apply to the names only, and it is an error to set them if ``color`` has only references.
    Extra keyword arguments are HoloViews options, not matplotlib ones.
    """
    import scanpy as sc

    reject_unused_options((color,), layer=layer, gene_symbols=gene_symbols)
    A = sc.pl.hv_init(None)
    if isinstance(basis, str):
        if basis not in ad.obsm:
            available = ", ".join(ad.obsm.keys()) or "none"
            msg = f"Basis {basis!r} is not present in .obsm (available: {available})."
            raise KeyError(msg)
        kdims = A.obsm[basis][:, [0, 1]]
    else:  # already an accessor reference
        kdims = basis
    color_refs = _v2_refs(ad, A, color, layer, gene_symbols)
    return _v2_options(sc.pl.scatter(ad, kdims, color=color_refs), kwargs)


def reject_v2_options(**options: object) -> None:
    """Raise if a matplotlib/scanpy-v1 only option was set while the ``ScanpyV2Preview`` preset is active."""
    set_options = [k for k, v in options.items() if v not in (None, False)]
    if set_options:
        msg = f"{set_options} are not supported with scanpy's ScanpyV2Preview preset for AnnData input."
        raise NotImplementedError(msg)
