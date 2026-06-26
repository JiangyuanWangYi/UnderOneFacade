"""
Model registry and engine dispatch.
Each model file registers an engine via @register_engine.
Engine provides: build_model, build_dataset, get_optimizer, get_scheduler,
                 collate_fn, train_step, eval_step.
"""

ENGINE_REGISTRY = {}


def register_engine(name):
    def decorator(cls):
        ENGINE_REGISTRY[name] = cls
        return cls
    return decorator


def get_engine(name):
    if name not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(ENGINE_REGISTRY.keys())}"
        )
    return ENGINE_REGISTRY[name]


def list_models():
    return list(ENGINE_REGISTRY.keys())


# Make pointops importable (CUDA extension in models/pointops_lib/pointops/)
import os as _os, sys as _sys
_POINTOPS_PARENT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pointops_lib")
if _os.path.isdir(_POINTOPS_PARENT) and _POINTOPS_PARENT not in _sys.path:
    _sys.path.insert(0, _POINTOPS_PARENT)

# Import all engine modules to trigger registration
from . import pointnet2 
from . import dgcnn      
try:
    from . import ptv1
except Exception:
    pass
try:
    from . import ptv3
except Exception:
    pass
from . import octformer  
from . import kpconv    
from . import kpconv_full
from . import pointnet   
