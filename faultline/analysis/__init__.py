"""Aggregation, confidence intervals, and leaderboard data generation."""

from faultline.analysis.aggregate import aggregate_runs, load_runs
from faultline.analysis.report import generate_report
from faultline.analysis.stats import wilson_interval

__all__ = ["aggregate_runs", "generate_report", "load_runs", "wilson_interval"]
