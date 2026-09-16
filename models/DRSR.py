"""Public name for the validated DRSR implementation.

The inherited module names are retained for checkpoint and provenance
compatibility. Experiment scripts construct the model with the frozen values in
the released design JSON files.
"""

from models.RWHEDNPseudoConditional import RWHEDNPseudoConditional


class DRSR(RWHEDNPseudoConditional):
    """Dual-role source routing model."""


__all__ = ["DRSR"]
