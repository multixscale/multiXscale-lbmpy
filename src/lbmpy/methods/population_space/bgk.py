from typing import List

import pystencils as ps

from lbmpy import LBStencil
from lbmpy.equilibrium import GenericDiscreteEquilibrium
from lbmpy.methods.population_space import PopulationSpaceSRT


class PopulationSpaceBGK(PopulationSpaceSRT):
    def __init__(
        self,
        stencil: LBStencil,
        feq: GenericDiscreteEquilibrium,
        isothermal: bool = None,
    ):
        super().__init__(stencil=stencil, method="bgk", feq=feq, isothermal=isothermal)

    def get_neq_assignments(self) -> List[ps.Assignment]:
        neq_assignments = [
            ps.Assignment(fneq, fneq_expr)
            for fneq, fneq_expr in zip(
                self.fneq_symbols, self.bgk_nonequilibrium_populations()
            )
        ]
        return neq_assignments
