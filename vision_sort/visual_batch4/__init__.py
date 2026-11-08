"""Batch 4 standalone task revision, with lazy public environment export."""


def __getattr__(name):
    if name == 'VisualBatch4Env':
        from .env import VisualBatch4Env
        return VisualBatch4Env
    raise AttributeError(name)


__all__ = ['VisualBatch4Env']
