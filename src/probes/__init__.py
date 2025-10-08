"""
Cognitive Action Probe System for Gemma 3 4B

This package provides tools for:
- Capturing activations from Gemma 3 4B using nnsight
- Training linear/logistic probes for cognitive action detection
- Real-time inference during text generation
- Integration with liminal_backrooms for live visualization
"""

from .dataset_utils import load_cognitive_actions_dataset, create_splits
from .probe_models import LinearProbe, MultiHeadProbe
from .probe_inference import ProbeInferenceEngine

__version__ = "0.1.0"
__all__ = [
    "load_cognitive_actions_dataset",
    "create_splits",
    "LinearProbe",
    "MultiHeadProbe",
    "ProbeInferenceEngine",
]
