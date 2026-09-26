"""Thalamic modulation (Phases 6-8).

Modules:
    gain.py         attention field from generative templates; applied as a gain
                    on stage energies before normalization (Phase 6)
    expectation.py  predictive subtraction / explaining away (Phase 7, planned)
    routing.py      context neighborhoods, pulvinar routing (Phase 8, planned)
"""

from .gain import feature_gain, feature_similarity_field, pass_through_gain

__all__ = ["feature_gain", "feature_similarity_field", "pass_through_gain"]
