"""Internal bounded-runtime support for the standalone Batch 4 distribution."""

from .env import VisualDiscoveryEnv



def make_env(family="canvas_layout", max_steps=48, **kwargs):
    return VisualDiscoveryEnv(family=family, max_steps=max_steps, **kwargs)


__all__ = ["VisualDiscoveryEnv", "make_env"]
