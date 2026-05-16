"""Hierarchical reaction-diffusion toolkit.

A ``HierarchicalRDStack`` is a cascade of two-field RD stages
(:class:`GrayScottStage`, :class:`FitzHughNagumoStage`) where the
product of stage *k* feeds (as a spatial modulation of feed/source)
into stage *k+1*. Two drivers explore the parameter space:

* :class:`PreferenceDriver` — interactive breeder.
* :class:`PerceptualDriver` — Gabor-bank perceptual distance + (1, lambda) ES.
"""

from .core import (
    RDStage,
    GrayScottStage,
    FitzHughNagumoStage,
    laplacian,
)
from .stack import HierarchicalRDStack, normalize01
from .perceptual import (
    gabor_bank,
    perceptual_features,
    perceptual_distance,
)
from .drivers import (
    PreferenceDriver,
    PerceptualDriver,
    get_stack_params,
    set_stack_params,
    stack_param_layout,
)

__all__ = [
    "RDStage",
    "GrayScottStage",
    "FitzHughNagumoStage",
    "laplacian",
    "HierarchicalRDStack",
    "normalize01",
    "gabor_bank",
    "perceptual_features",
    "perceptual_distance",
    "PreferenceDriver",
    "PerceptualDriver",
    "get_stack_params",
    "set_stack_params",
    "stack_param_layout",
]
