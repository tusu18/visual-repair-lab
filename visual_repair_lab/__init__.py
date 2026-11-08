"""Trusted evaluator factories; actors use the separate HTTP client only."""
from gymnasium.envs.registration import register, registry
from vision_sort.visual_training.env import VisualTrainingEnv
__version__ = "7.0.0"
if "VisualTrainingRepair-v0" not in registry:
    register(id="VisualTrainingRepair-v0",entry_point="vision_sort.visual_training.env:VisualTrainingEnv")
def make_env(spec=None, **kwargs):
    if spec is not None:
        from .v7.env import make_env as adaptive_factory
        return adaptive_factory(spec, **kwargs)
    return VisualTrainingEnv(**kwargs)
__all__ = ["VisualTrainingEnv", "__version__", "make_env"]

if "VisualAdaptiveRepair-v0" not in registry:
    register(id="VisualAdaptiveRepair-v0", entry_point="visual_repair_lab.adaptive.env:AdaptiveRepairEnv")

if "VisualRepairDelivery-v7" not in registry:
    register(id="VisualRepairDelivery-v7", entry_point="visual_repair_lab.v7.env:RepairEnv")
