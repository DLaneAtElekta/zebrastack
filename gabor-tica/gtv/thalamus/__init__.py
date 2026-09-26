"""Thalamic modulation (Phases 6-8).

Modules:
    gain.py         attention field from generative templates; applied as a gain
                    on stage energies before normalization (Phase 6)
    expectation.py  predictive subtraction / explaining away (Phase 7)
    routing.py      pulvinar-style routing: priority map -> object-centered window (Phase 8)
"""

from .expectation import expectation
from .gain import feature_gain, feature_similarity_field, pass_through_gain
from .routing import (learned_priority, route_window, saliency_map, select_location, select_topk,
                      template_match_map)

__all__ = ["expectation", "feature_gain", "feature_similarity_field", "pass_through_gain", "learned_priority",
           "route_window", "saliency_map", "select_location", "select_topk", "template_match_map"]
