import matplotlib
import numpy as np
import pytest
from anndata import AnnData
from mudata import MuData

import muon as mu

matplotlib.use("Agg")


@pytest.fixture
def mdata() -> MuData:
    mdata = MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )
    mdata.var_names_make_unique()
    return mdata


def test_pl_scatter(mdata: MuData, rng: np.random.Generator) -> None:
    mdata.obs["condition"] = rng.choice(["a", "b"], mdata.n_obs)
    mu.pl.scatter(mdata, x="mod1:0", y="mod2:0", color="condition")


@pytest.fixture
def mdata_umap(mdata: MuData, rng: np.random.Generator) -> MuData:
    mdata.obsm["X_umap"] = rng.random((mdata.n_obs, 2))
    mdata.obs["condition"] = rng.choice(["a", "b"], mdata.n_obs)
    mdata.obs["score"] = rng.random(mdata.n_obs)
    return mdata


def test_pl_embedding_invalid_backend(mdata_umap: MuData) -> None:
    with pytest.raises(ValueError, match="backend"):
        mu.pl.embedding(mdata_umap, basis="umap", color="condition", backend="plotly")  # type: ignore[arg-type]


@pytest.mark.parametrize("color", [None, "condition", "score"])
def test_pl_embedding_holoviews(mdata_umap: MuData, color: str | None) -> None:
    pytest.importorskip("hv_anndata")
    hv = pytest.importorskip("holoviews")
    plot = mu.pl.embedding(mdata_umap, basis="umap", color=color, backend="holoviews")
    assert isinstance(plot, hv.Scatter)


def test_pl_embedding_holoviews_multiple_colors(mdata_umap: MuData) -> None:
    pytest.importorskip("hv_anndata")
    hv = pytest.importorskip("holoviews")
    plot = mu.pl.embedding(mdata_umap, basis="umap", color=["condition", "score"], backend="holoviews")
    assert isinstance(plot, hv.Layout)
    assert len(plot) == 2


def test_pl_embedding_holoviews_anndata_missing_obs_column(mdata_umap: MuData) -> None:
    pytest.importorskip("hv_anndata")
    pytest.importorskip("holoviews")
    adata = AnnData(obs=mdata_umap.obs, obsm=mdata_umap.obsm)
    with pytest.raises(NotImplementedError, match="only colour by .obs"):
        mu.pl.embedding(adata, basis="umap", color="not_a_column", backend="holoviews")
