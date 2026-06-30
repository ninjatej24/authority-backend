"""Backward-compatible re-exports."""

from services.scoring_engine import compute_authority_score_legacy as compute_authority_score

__all__ = ["compute_authority_score"]
