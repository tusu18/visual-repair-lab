"""Internal task implementation for the standalone Batch 4 distribution.

The supported Gym ID is registered by visual_repair_lab only.
"""
from .env import VisualCausalEnv

__all__ = ['VisualCausalEnv']
