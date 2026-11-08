"""Trusted dispatcher. The train generator never imports this private authority."""
def make_env(spec,**kwargs):
    if spec['family']=='detection_overlay':
        from .private_cv import DetectionRepairEnv
        return DetectionRepairEnv(spec,**kwargs)
    from .env import AdaptiveRepairEnv
    return AdaptiveRepairEnv(spec,**kwargs)
