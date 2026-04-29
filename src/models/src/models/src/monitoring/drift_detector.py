"""
PSI-Based Drift Detector
Triggers automated Kubeflow retraining when PSI > 0.2
Author: Ravi Teja Chittaluri
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
PSI_THRESHOLD = 0.2  # Industry standard threshold for significant drift


@dataclass
class DriftReport:
    feature_name: str
    psi_score: float
    drifted: bool
    severity: str  # low / medium / high


class PSIDriftDetector:
    """
    Population Stability Index (PSI) drift detection.
    PSI < 0.1  → No significant change
    PSI 0.1–0.2 → Moderate change, monitor
    PSI > 0.2  → Significant change, trigger retraining
    """

    def __init__(
        self,
        psi_threshold: float = PSI_THRESHOLD,
        n_bins: int = 10,
        alert_callback=None,
    ):
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins
        self.alert_callback = alert_callback

    def compute_psi(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """Compute PSI between reference and current distributions."""
        # Create bins from reference distribution
        breakpoints = np.percentile(reference, np.linspace(0, 100, self.n_bins + 1))
        breakpoints = np.unique(breakpoints)

        def get_bucket_counts(data: np.ndarray) -> np.ndarray:
            counts = np.histogram(data, bins=breakpoints)[0]
            counts = np.maximum(counts, 0.0001)  # Avoid log(0)
            return counts / counts.sum()

        ref_pct = get_bucket_counts(reference)
        cur_pct = get_bucket_counts(current)
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def check_features(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> list[DriftReport]:
        """Check all features for drift and trigger alert if needed."""
        reports = []
        drifted_features = []

        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            psi = self.compute_psi(
                reference_df[col].dropna().values,
                current_df[col].dropna().values,
            )
            drifted = psi > self.psi_threshold
            severity = (
                "high" if psi > 0.25
                else "medium" if psi > 0.15
                else "low"
            )
            report = DriftReport(
                feature_name=col,
                psi_score=round(psi, 4),
                drifted=drifted,
                severity=severity,
            )
            reports.append(report)
            if drifted:
                drifted_features.append(col)
                logger.warning(f"DRIFT DETECTED: {col} PSI={psi:.4f} (threshold={self.psi_threshold})")

        # Trigger retraining if any feature drifted
        if drifted_features and self.alert_callback:
            logger.info(f"Triggering retraining — {len(drifted_features)} features drifted")
            self.alert_callback(drifted_features=drifted_features, reports=reports)

        return reports

    def summary(self, reports: list[DriftReport]) -> dict:
        return {
            "total_features": len(reports),
            "drifted_features": sum(1 for r in reports if r.drifted),
            "high_severity": sum(1 for r in reports if r.severity == "high"),
            "max_psi": max((r.psi_score for r in reports), default=0),
            "retraining_triggered": any(r.drifted for r in reports),
        }
