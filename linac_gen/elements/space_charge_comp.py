"""Space charge compensation element."""
from linac_gen.elements.base import PassiveElement


class SpaceChargeComp(PassiveElement):
    """Space charge compensation element.

    A passive, zero-length element that stores a neutralisation factor.
    The tracker reads ``self.factor`` to scale space-charge kicks; this
    element itself performs no beam modification.

    Parameters
    ----------
    factor : float
        Neutralisation fraction, 0 (no compensation) to 1 (full compensation).
    """

    def __init__(self, name: str, factor: float = 0.0):
        super().__init__(name=name)
        self.factor = factor

    def apply(self, beam) -> None:
        """No-op: beam is unchanged.  The tracker uses self.factor."""
        pass
