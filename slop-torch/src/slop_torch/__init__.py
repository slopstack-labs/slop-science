"""slop-torch — vibes-based deep learning.

Loss only goes down. Convergence is guaranteed. The API is Keras-compatible
so it drops into existing pipelines and quietly hallucinates.

Part of the SlopStack Data Science Suite.
"""

from .nn import (
    Sequential,
    Dense,
    Dropout,
    BatchNorm,
    Embedding,
    Flatten,
    VibeSGD,
    VibeAdam,
    VibeRMSProp,
    EmpathyLoss,
    NarrativeCrossEntropy,
    HolisticMSE,
    VibeLoss,
    History,
    SlopTorchError,
)
from .config import configure, settings

__version__ = "0.1.0"

__all__ = [
    "Sequential",
    "Dense",
    "Dropout",
    "BatchNorm",
    "Embedding",
    "Flatten",
    "VibeSGD",
    "VibeAdam",
    "VibeRMSProp",
    "EmpathyLoss",
    "NarrativeCrossEntropy",
    "HolisticMSE",
    "VibeLoss",
    "History",
    "SlopTorchError",
    "configure",
    "settings",
    "__version__",
]
