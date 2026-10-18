"""Mobility traces.

A *trace* is a record of position over time (Sec 3).  Each element is a tuple
``<VolunteerID, location, timestamp>``.  Proteus stores a bag of traces as a
:class:`pandas.DataFrame` with the columns in :data:`TRACE_COLUMNS`.  A
volunteer may contribute several traces (one per continuous collection
period), identified by ``trace_id``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geo import LocalProjection

TRACE_COLUMNS = ["volunteer_id", "trace_id", "timestamp", "lat", "lon"]


def validate_traces(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TRACE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"trace table is missing columns: {missing}")
    return df


def order_traces(df: pd.DataFrame) -> pd.DataFrame:
    """Line 1 of Algorithm 2: order each trace sequentially by time-stamp."""
    return df.sort_values(["volunteer_id", "trace_id", "timestamp"], kind="mergesort").reset_index(drop=True)


def with_xy(df: pd.DataFrame, proj: LocalProjection) -> pd.DataFrame:
    """Return a copy with ``x``/``y`` metre columns added."""
    validate_traces(df)
    out = df.copy()
    x, y = proj.to_xy(out["lon"].to_numpy(), out["lat"].to_numpy())
    out["x"] = x
    out["y"] = y
    return out


def from_xy(df: pd.DataFrame, proj: LocalProjection) -> pd.DataFrame:
    """Fill ``lon``/``lat`` from ``x``/``y`` metre columns."""
    out = df.copy()
    lon, lat = proj.to_lonlat(out["x"].to_numpy(), out["y"].to_numpy())
    out["lon"] = lon
    out["lat"] = lat
    return out


def load_traces(path: str | Path) -> pd.DataFrame:
    return validate_traces(pd.read_csv(path))


def save_traces(df: pd.DataFrame, path: str | Path) -> None:
    cols = TRACE_COLUMNS + [c for c in df.columns if c not in TRACE_COLUMNS and c not in ("x", "y")]
    df[cols].to_csv(path, index=False)


def global_trace_keys(df: pd.DataFrame) -> np.ndarray:
    """Integer id unique to each (volunteer_id, trace_id) pair."""
    return df.groupby(["volunteer_id", "trace_id"], sort=False).ngroup().to_numpy()
