"""ML surrogate elements for HELIX.

Per-element-type mixins that combine a HELIX :class:`Element` subclass
with a :class:`torch.nn.Module` so the trained ML surrogate is a
drop-in replacement at the tracker's isinstance-based dispatch seam.

M1 ships :class:`SurrogateFieldMap` — predicts the linearised 6x6
transfer matrix from an MLP given ``ref`` kinematics + element params,
for envelope-mode use.  Multiparticle tracking and matcher integration
are M7 (see ``docs/plans/surrogates.md``).
"""
from linac_gen.surrogates.base import (
    MlpHead,
    OutOfScopeError,
    Scope,
    SurrogateFieldMap,
    SurrogateMetadata,
)
from linac_gen.surrogates import registry

__all__ = [
    "MlpHead",
    "OutOfScopeError",
    "Scope",
    "SurrogateFieldMap",
    "SurrogateMetadata",
    "registry",
]
