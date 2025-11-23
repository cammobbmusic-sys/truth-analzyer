"""
Verification module for the Multi-AI System.
Handles cross-validation, consensus scoring, and conflict resolution.
"""

from .cross_validate import CrossValidator
from .conflict_resolver import ConflictResolver

__all__ = ["CrossValidator", "ConflictResolver"]
