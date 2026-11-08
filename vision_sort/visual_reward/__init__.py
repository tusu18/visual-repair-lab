"""Internal reward implementation for the standalone Batch 4 distribution.

Historical Gym IDs are intentionally not registered by this new distribution.
"""
from .env import VisualRewardEnv

__all__ = ['VisualRewardEnv']
