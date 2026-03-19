# Maintainer: Walid Ghariani (wagh@dhigroup.com)
# Description: Geospatial utilities.

from typing import Union

import numpy as np
import torch
import xarray as xr
import rioxarray
from pyproj import Transformer


def build_latlons(da: xr.DataArray, target_crs: str = "EPSG:4326") -> torch.Tensor:
    """Compute lat/lon tensor for all pixels in a raster.

    Args:
      da: DataArray with CRS in `da.rio.crs` and spatial coords (x, y).
      target_crs: Target CRS (default EPSG:4326).

    Returns:
      Tensor of shape [n_pixels, 2] with (lat, lon).
    """
    if getattr(da, "rio", None) is None or da.rio.crs is None:
        raise ValueError("DataArray must have a valid CRS in `da.rio.crs`.")
    xs, ys = da.x.values, da.y.values
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    transformer = Transformer.from_crs(da.rio.crs, target_crs, always_xy=True)
    lons, lats = transformer.transform(xx.ravel(), yy.ravel())
    return torch.from_numpy(np.stack([lats, lons], axis=1)).float()


def set_geo_metadata(
    target: Union[xr.DataArray, xr.Dataset],
    source: Union[xr.DataArray, xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Copy geospatial metadata (CRS and affine transform) from a source
    xr.DataArray or xr.Dataset to a target xr.DataArray or xr.Dataset.

    Args:
      target: xr.DataArray or xr.Dataset to which CRS and transform
        metadata will be written.
      source: xr.DataArray or xr.Dataset providing CRS and transform
        metadata.

    Returns:
      xr.DataArray or xr.Dataset with CRS and transform copied from
      the source (if available).
    """
    try:
        if hasattr(source, "rio") and hasattr(target, "rio"):
            if source.rio.crs is not None:
                target = target.rio.write_crs(source.rio.crs)
            if source.rio.transform() is not None:
                target = target.rio.write_transform(source.rio.transform())
    except ImportError:
        pass
    return target


def features_to_xarray(
    features: torch.Tensor,
    y: xr.DataArray,
    x: xr.DataArray,
    embed_dim_name: str = "embed",
) -> xr.DataArray:
    """Convert [n_pixels, feature_dim] to DataArray [y, x, embed].

    Args:
      features: Tensor [n_pixels, feature_dim].
      y: DataArray of y coordinates.
      x: DataArray of x coordinates.
      embed_dim_name: Name of embedding dimension.

    Returns:
      xr.DataArray with dims (y, x, embed).
    """
    h, w = len(y), len(x)
    feat = features.numpy().reshape(h, w, -1)
    feat = feat.transpose(2, 0, 1)

    return xr.DataArray(
        feat,
        dims=(embed_dim_name, y.dims[0], x.dims[0]),
        coords={
            embed_dim_name: np.arange(1, feat.shape[0] + 1),
            y.dims[0]: y,
            x.dims[0]: x,
        },
        name="embeds",
    )


def cosine_similarity_map(
    embeds: xr.DataArray,
    ref_vector: np.ndarray,
    embed_dim: str = "embed",
    eps: float = 1e-8,
) -> xr.DataArray:
    """
    Compute cosine similarity between each pixel embedding and a reference vector.

    Args:
        embeds (xr.DataArray): Embeddings with dimensions (embed, y, x).
        ref_vector (np.ndarray): Reference embedding vector of shape (embed,).
        embed_dim (str): Name of the embedding dimension.
        eps (float): Small constant for numerical stability.

    Returns:
        xr.DataArray: Cosine similarity map with dimensions (y, x).
    """
    if embed_dim not in embeds.dims:
        raise ValueError(f"'{embed_dim}' must be a dimension of embeds")

    if embeds.sizes[embed_dim] != ref_vector.shape[0]:
        raise ValueError("Reference vector length does not match embedding dimension")

    ref = ref_vector / (np.linalg.norm(ref_vector) + eps)
    embeds_yxe = embeds.transpose("y", "x", embed_dim)
    y, x, c = embeds_yxe.shape
    E = embeds_yxe.values.reshape(y * x, c)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + eps)
    sim = E_norm @ ref  # (y*x,)
    sim_img = sim.reshape(y, x)

    sim_xr = xr.DataArray(
        sim_img,
        dims=("y", "x"),
        coords={"y": embeds.coords["y"], "x": embeds.coords["x"]},
        name="cosine_similarity",
    )
    return set_geo_metadata(sim_xr, embeds)
