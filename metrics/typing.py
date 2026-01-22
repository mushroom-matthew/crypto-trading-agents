"""Type definitions for metrics service."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Protocol

import pandas as pd

from .base import MetricResult


class MetricFunction(Protocol):
    """Protocol describing a metric implementation."""

    def __call__(self, df: pd.DataFrame, **kwargs) -> MetricResult:
        ...


MetricRegistry = Dict[str, MetricFunction]
MetricParams = Mapping[str, Mapping[str, object]]
MetricResultList = List[MetricResult]
