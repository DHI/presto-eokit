# Maintainer: Walid Ghariani (wagh@dhigroup.com)
# Description: Performs data/feature extraction based on specified user configurations.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .utils import construct_single_presto_input


class MonthSource(Enum):
    COLUMN = "column"
    DATETIME = "datetime"


@dataclass
class ExtractorConfig:
    band_configs: Dict[str, List[str]]
    timestep_dim: int
    locations: Tuple[str, str]
    target_column: str
    month_source: Optional[MonthSource] = None
    month_column: Optional[str] = None
    date_column: Optional[str] = None

    def __post_init__(self):
        if self.month_column and self.date_column:
            raise ValueError(
                "Only one of `month_column` or `date_column` may be specified."
            )
        if not (self.month_column or self.date_column):
            raise ValueError(
                "One of `month_column` or `date_column` must be specified."
            )
        if self.month_source and self.month_source not in MonthSource:
            raise ValueError(f"Invalid month source: {self.month_source}")


class Extractor:
    def __init__(self, config: ExtractorConfig):
        self.config = config

    def __call__(self, row: dict) -> tuple:
        return (
            self._extract_bands(row),
            *self._extract_location(row),
            self._extract_label(row),
            self._extract_month(row),
        )

    def _extract_bands(self, row: dict) -> dict:
        return {
            source: self._to_tensor(row, bands)
            for source, bands in self.config.band_configs.items()
        } | {
            f"{source}_bands": bands
            for source, bands in self.config.band_configs.items()
        }

    def _to_tensor(self, row: dict, bands: List[str]) -> torch.Tensor:
        missing = [band for band in bands if band not in row]
        if missing:
            raise ValueError(f"Missing bands: {missing}")
        values = [float(row[band]) for band in bands]
        return torch.tensor([values] * self.config.timestep_dim, dtype=torch.float32)

    def _extract_location(self, row: dict) -> Tuple[float, float]:
        lat, lon = self.config.locations
        return (
            float(self._get_or_error(row, lat, "latitude")),
            float(self._get_or_error(row, lon, "longitude")),
        )

    def _extract_label(self, row: dict) -> float:
        return float(self._get_or_error(row, self.config.target_column, "target"))

    def _extract_month(self, row: dict) -> int:
        if self.config.month_column:
            return int(self._get_or_error(row, self.config.month_column, "month"))
        elif self.config.date_column:
            try:
                return pd.to_datetime(row[self.config.date_column]).month
            except Exception as e:
                raise ValueError(f"Date parsing error: {e}")
        raise ValueError("No method available to extract month.")

    def _get_or_error(self, row: dict, key: str, label: str) -> float:
        if key not in row:
            raise ValueError(f"Missing {label} column: '{key}'")
        return row[key]


def create_extractor(
    band_configs: Dict[str, List[str]],
    timestep_dim: int,
    locations: Tuple[str, str],
    target_column: str,
    month_source: Optional[MonthSource] = None,
    month_column: Optional[str] = None,
    date_column: Optional[str] = None,
) -> Callable[[dict], tuple]:
    config = ExtractorConfig(
        band_configs=band_configs,
        timestep_dim=timestep_dim,
        locations=locations,
        target_column=target_column,
        month_source=month_source,
        month_column=month_column,
        date_column=date_column,
    )
    return Extractor(config)


def data_extractor(
    df: pd.DataFrame,
    data_extractor_fn: Callable[
        [pd.Series], Tuple[Dict[str, Any], float, float, float, int]
    ],
) -> Tuple[torch.Tensor, ...]:
    """
    General-purpose encoding function that works with extractors created via `create_extractor`.

    Args:
        df (pd.DataFrame): Input dataframe.
        data_extractor_fn (Callable): Extractor function returning (input_dict, lat, lon, label, month).

    Returns:
        Tuple of tensors: (x, mask, dynamic_world, latlons, labels, month_names)
    """
    arrays, masks, dynamic_worlds = [], [], []
    latlons, labels, month_names = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_dict, lat, lon, label, month = data_extractor_fn(row)

        # Construct tensors from input_dict
        x, mask, dynamic_world = construct_single_presto_input(**input_dict)

        arrays.append(x)
        masks.append(mask)
        dynamic_worlds.append(dynamic_world)
        latlons.append(torch.tensor([lat, lon], dtype=torch.float32))
        labels.append(label)
        month_names.append(month)

    return (
        torch.stack(arrays),
        torch.stack(masks),
        torch.stack(dynamic_worlds),
        torch.stack(latlons),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(month_names, dtype=torch.int64),
    )


def features_extractor(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract features from a dataloader using the model's encoder.

    Args:
        model (torch.nn.Module): PyTorch model with an encoder component.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the data batches.
        device (Union[str, torch.device], optional): Device to run the model on, by default uses CUDA if available.
        verbose (bool, optional): Whether to show progress bar, by default True.

    Returns
        np.ndarray
            Array of encoded features with shape [n_samples, feature_dim].

    """
    model = model.to(device)
    model.eval()

    features_list = []
    iterator = tqdm(dataloader) if verbose else dataloader

    with torch.no_grad():
        for batch in iterator:
            x, mask, dw, latlons, y, month = batch

            inputs = {
                "x": x.to(device),
                "mask": mask.to(device),
                "dynamic_world": dw.to(device),
                "latlons": latlons.to(device),
                "month": month.to(device),
            }
            encodings = model.encoder(**inputs)
            features_list.append(encodings.detach().cpu().numpy())

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.concatenate(features_list)
