"""Thalamic modulation (Phases 6-8).

Modules:
    gain.py         attention field from generative templates; applied as a gain
                    on stage energies before normalization (Phase 6)
    expectation.py  predictive subtraction / explaining away (Phase 7)
    routing.py      context neighborhoods, pulvinar routing (Phase 8, planned)
"""

from .expectation import expectation
from .gain import feature_gain, feature_similarity_field, pass_through_gain

__all__ = ["expectation", "feature_gain", "feature_similarity_field", "pass_through_gain"]
