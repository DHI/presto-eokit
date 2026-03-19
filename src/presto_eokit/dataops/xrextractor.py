# Maintainer: Walid Ghariani (wagh@dhigroup.com)
# Description: Performs dataarray/feature extraction (embeddings) and wrap it in an xarray.

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
import rioxarray 
from pyproj import Transformer
from tqdm import tqdm

from .pipelines.s1_s2_era5_srtm import (
    BANDS,
    ERA5_BANDS,
    NORMED_BANDS,
    REMOVED_BANDS,
    S1_BANDS,
    S1_S2_ERA5_SRTM,
    S2_BANDS,
    SRTM_BANDS,
)
from .geo import (
    build_latlons,
    set_geo_metadata,
    features_to_xarray,
)

def _expected_bands(modality: str) -> List[str]:
    """Return the canonical band list for a modality."""
    if modality == "s2":
        return S2_BANDS
    if modality == "s1":
        return S1_BANDS
    if modality == "era5":
        return ERA5_BANDS
    if modality == "srtm":
        return SRTM_BANDS
    raise ValueError(f"Unknown modality: {modality}")


def _plan_indexers(
    da_band_names: List[str],
    band_configs: Dict[str, List[str]],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], bool]:
    """Precompute (data_idx, out_idx) per modality and NDVI availability.

    Args:
      da_band_names: Names in the DataArray band coordinate.
      band_configs: Dict modality -> requested band names.

    Returns:
      indexers: modality -> (data_idx, out_idx) arrays.
      has_ndvi_sources: True if S2 includes both B4 and B8.
    """
    name_to_pos = {b: i for i, b in enumerate(da_band_names)}
    indexers = {}
    has_ndvi = False

    for modality, req in band_configs.items():
        expected = [b for b in _expected_bands(modality) if b not in REMOVED_BANDS]
        missing = [b for b in req if b not in da_band_names]
        if missing:
            raise ValueError(f"Requested {modality} bands not in DataArray: {missing}")
        bad = [b for b in req if b not in expected]
        if bad:
            raise ValueError(f"Requested {modality} bands not valid for Presto: {bad}")

        data_idx = np.array([name_to_pos[b] for b in req], dtype=np.int64)
        out_idx = np.array([BANDS.index(b) for b in req], dtype=np.int64)
        indexers[modality] = (data_idx, out_idx)

        if modality == "s2" and ("B4" in req) and ("B8" in req):
            has_ndvi = True

    return indexers, has_ndvi


def datarray_extractor(
    da: xr.DataArray,
    band_configs: Dict[str, List[str]],
    timestep_dim: int = 1,
    month: int = 6,
    target_crs: str = "EPSG:4326",
    spatial_dims: Tuple[str, str] = ("y", "x"),
    band_dim: str = "band",
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized extractor: build Presto inputs for all pixels without per-pixel loops.

    Args:
      da: Input DataArray with a CRS and band names in `band_dim`.
      band_configs: Dict modality -> list of band names to use.
      timestep_dim: Number of timesteps to repeat per pixel.
      month: Month (1–12) assigned to all pixels (converted to 0–11).
      target_crs: Target CRS for output lat/lon.
      spatial_dims: Tuple of (y_dim, x_dim).
      band_dim: Name of the band dimension.
      chunk_size: If set, process pixels in chunks to limit RAM.

    Returns:
      x: [n_pixels, timestep_dim, len(NORMED_BANDS)] features (B9 removed, NDVI added).
      mask: [n_pixels, timestep_dim, len(NORMED_BANDS)].
      dw: [n_pixels, timestep_dim] (long).
      latlons: [n_pixels, 2] (lat, lon).
      months: [n_pixels] (long, 0–11).
    """
    y_dim, x_dim = spatial_dims
    if getattr(da, "rio", None) is None or da.rio.crs is None:
        raise ValueError("DataArray must have a valid CRS in `da.rio.crs`.")
    da = da.transpose(y_dim, x_dim, band_dim)

    stacked = da.stack(z=(y_dim, x_dim))  # [band, n_pixels]
    n_pixels = stacked.z.size
    band_names = stacked[band_dim].values.tolist()

    indexers, has_ndvi = _plan_indexers(band_names, band_configs)
    latlons = build_latlons(da, target_crs=target_crs)

    keep_indices = [i for i, b in enumerate(BANDS) if b != "B9"]
    months = torch.full((n_pixels,), month - 1, dtype=torch.long)

    if not chunk_size:
        x, mask, dw = _build_presto_tensors_chunk(
            stacked.values.astype(np.float32),  # [band, n_pixels]
            indexers,
            timestep_dim,
            has_ndvi,
            keep_indices,
        )
        return x, mask, dw, latlons, months

    arrays, masks, dws = [], [], []
    for start in tqdm(range(0, n_pixels, chunk_size), desc="Building Presto tensors"):
        end = min(start + chunk_size, n_pixels)
        sub = stacked.values[:, start:end].astype(np.float32)
        x_c, mask_c, dw_c = _build_presto_tensors_chunk(
            sub, indexers, timestep_dim, has_ndvi, keep_indices
        )
        arrays.append(x_c)
        masks.append(mask_c)
        dws.append(dw_c)

    x = torch.cat(arrays, dim=0)
    mask = torch.cat(masks, dim=0)
    dw = torch.cat(dws, dim=0)
    return x, mask, dw, latlons, months


def _build_presto_tensors_chunk(
    sub_band_np: np.ndarray,
    indexers: Dict[str, Tuple[np.ndarray, np.ndarray]],
    timestep_dim: int,
    has_ndvi: bool,
    keep_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build x/mask/dw for a pixel chunk in a single vectorized pass.

    Args:
      sub_band_np: Array [band, n_pixels] for a chunk.
      indexers: modality -> (data_idx, out_idx).
      timestep_dim: Number of timesteps to repeat per pixel.
      has_ndvi: True if NDVI can be computed.
      keep_indices: Indices for removing B9.

    Returns:
      x: [n_chunk_pixels, timestep_dim, len(NORMED_BANDS)].
      mask: [n_chunk_pixels, timestep_dim, len(NORMED_BANDS)].
      dw: [n_chunk_pixels, timestep_dim] (long).
    """
    n_pixels = sub_band_np.shape[1]
    T = timestep_dim
    D = len(BANDS)

    x_full = torch.zeros((n_pixels, T, D), dtype=torch.float32)
    mask_full = torch.ones((n_pixels, T, D), dtype=torch.float32)

    for modality, (data_idx, out_idx) in indexers.items():
        vals = torch.from_numpy(sub_band_np[data_idx, :].T).unsqueeze(1).repeat(1, T, 1)  # [N,T,K]
        idx = torch.from_numpy(out_idx).long().view(1, 1, -1).expand(n_pixels, T, -1)
        x_full.scatter_(2, idx, vals)
        mask_full.scatter_(2, idx, torch.zeros_like(vals))

    x_full = S1_S2_ERA5_SRTM.normalize(x_full)  # removes B9 and writes NDVI
    mask = mask_full[:, :, keep_indices]
    if has_ndvi:
        mask[:, :, NORMED_BANDS.index("NDVI")] = 0

    dw = torch.full((n_pixels, T), 9, dtype=torch.long)
    return x_full, mask, dw


def encode_presto(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    dw: torch.Tensor,
    latlons: torch.Tensor,
    months: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 1024,
) -> torch.Tensor:
    """Run Presto encoder in batches.

    Args:
      model: Presto model with `.encoder`.
      x: [n_pixels, timestep, features].
      mask: [n_pixels, timestep, features].
      dw: [n_pixels, timestep] (long).
      latlons: [n_pixels, 2] (lat, lon).
      months: [n_pixels] (long, 0–11).
      device: Device string.
      batch_size: Pixels per batch.

    Returns:
      Tensor [n_pixels, feature_dim].
    """
    n = x.shape[0]
    model = model.to(device)
    model.eval()

    feats = []
    with torch.no_grad():
        for s in tqdm(range(0, n, batch_size), desc="Encoding"):
            e = min(s + batch_size, n)
            inputs = {
                "x": x[s:e].to(device),
                "mask": mask[s:e].to(device),
                "dynamic_world": dw[s:e].to(device).long(),
                "latlons": latlons[s:e].to(device),
                "month": months[s:e].to(device).long(),
            }
            feats.append(model.encoder(**inputs).cpu())
    return torch.cat(feats, dim=0)


def generate_embeddings(
    da: xr.DataArray,
    model: torch.nn.Module,
    band_configs: Dict[str, List[str]],
    month: int = 6,
    timestep_dim: int = 1,
    target_crs: str = "EPSG:4326",
    device: str = "cpu",
    batch_size: int = 1024,
    spatial_dims: Tuple[str, str] = ("y", "x"),
    band_dim: str = "band",
    embed_dim_name: str = "embed",
    chunk_size: Optional[int] = None,
) -> xr.DataArray:
    """Full pipeline: DataArray → vectorized Presto tensors → embeddings DataArray.

    Args:
      da: Input DataArray with CRS and named bands in `band_dim`.
      model: Presto model with `.encoder`.
      band_configs: Dict modality -> band names present in `da`.
      month: Month (1–12) assigned to all pixels.
      timestep_dim: Number of timesteps per pixel.
      target_crs: CRS for output lat/lon.
      device: Device for encoding.
      batch_size: Pixels per encoding batch.
      spatial_dims: Tuple of (y_dim, x_dim).
      band_dim: Name of the band dimension.
      embed_dim_name: Name of embedding dimension.
      chunk_size: Pixel chunk size for vectorized building.

    Returns:
      xr.DataArray [embed, y, x].
    """
    x, mask, dw, latlons, months = datarray_extractor(
        da=da,
        band_configs=band_configs,
        timestep_dim=timestep_dim,
        month=month,
        target_crs=target_crs,
        spatial_dims=spatial_dims,
        band_dim=band_dim,
        chunk_size=chunk_size,
    )
    feats = encode_presto(
        model=model,
        x=x,
        mask=mask,
        dw=dw,
        latlons=latlons,
        months=months,
        device=device,
        batch_size=batch_size,
    )
    embeds = features_to_xarray(
        feats,
        da[spatial_dims[0]],
        da[spatial_dims[1]],
        embed_dim_name
    )

    return set_geo_metadata(embeds, da)
