import abc
import warnings

import sympy as sp
import pystencils as ps

from lbmpy import LBStencil
from lbmpy.methods import AbstractLbMethod, LbmCollisionRule
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.equilibrium import GenericDiscreteEquilibrium

from typing import List


class PopulationSpaceSRT(AbstractLbMethod):

    def __init__(
        self,
        stencil: LBStencil,
        method: str,
        feq: GenericDiscreteEquilibrium,
        isothermal: bool,
        entropic_collision: bool = False,
        entropic_collision_type: str = None,
    ):

        super().__init__(stencil)
        self._method_type = method
        self._equilibrium = feq
        if isothermal is None:
            warnings.warn(
                "'isothermal' boolean argument not specified. Assuming an isothermal simulation."
            )
        self._isothermal = isothermal if isothermal is not None else True

        self._entropic_collision = entropic_collision
        if self.entropic_collision:
            warnings.warn(
                "Entropic Collision is not yet implemented. Setting entropic_collision to False."
            )
            self._entropic_collision = False

        self._entropic_collision_type = entropic_collision_type
        if self.entropic_collision and self.entropic_collision_type is None:
            warnings.warn(
                "Entropic Collision type not specified. Using Third Order Essentially Entropic Scheme"
            )
            self._entropic_collision_type = "third_order_eelbm"

        self._alpha = sp.Symbol("alpha") if self.entropic_collision else sp.S("2")
        self._beta = sp.Symbol("beta")

        self._omega = self.alpha * self.beta

        if self.isothermal:
            try:
                self._isothermal_temperature = self.equilibrium.theta
            except AttributeError:
                self._isothermal_temperature = self.stencil.theta0

        self._cqc = DensityVelocityComputation(stencil, True, zero_centered=False)

        self._f_eq_symbols = sp.symbols(f"f_eq_:{self.stencil.Q}")
        self._f_neq_symbols = sp.symbols(f"f_neq_:{self.stencil.Q}")

        # Ignore: For API compatibility with PSM methods
        self.fraction_field = None

    @property
    def method_type(self):
        return self._method_type

    @property
    def equilibrium(self):
        return self._equilibrium

    @property
    def isothermal(self):
        return self._isothermal

    @property
    def entropic_collision(self):
        return self._entropic_collision

    @property
    def entropic_collision_type(self):
        return self._entropic_collision_type

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def isothermal_temperature(self):
        if self._isothermal_temperature is None:
            print("Simulation is not isothermal")
        return self._isothermal_temperature

    @property
    def weights(self):
        print(
            "PopulationSpaceSRT methods takes the equilibrium as an input and do not have weights"
        )
        return None

    @property
    def conserved_quantity_computation(self):
        return self._cqc

    @property
    def relaxation_rates(self):
        return (self._omega,) * self.stencil.Q

    @property
    def feq_symbols(self):
        return self._f_eq_symbols

    @property
    def feq_symbolic_matrix(self):
        return sp.Matrix(self._f_eq_symbols)

    @property
    def fneq_symbols(self):
        return self._f_neq_symbols

    @property
    def fneq_symbolic_matrix(self):
        return sp.Matrix(self._f_neq_symbols)

    def get_equilibrium(self, conserved_quantity_equations=None):
        ac = self._derive_collision(1, conserved_quantity_equations)
        return ac.new_without_subexpressions()

    def get_collision_rule(self, **kwargs):
        cq_eqs = self._cqc.equilibrium_input_equations_from_pdfs(
            self.pre_collision_pdf_symbols
        )

        return self._derive_collision(self._omega, cq_eqs=cq_eqs)

    def _derive_collision(
        self, rrate: sp.Expr, cq_eqs: ps.AssignmentCollection | None = None
    ) -> ps.AssignmentCollection:

        eq_assignments = list()

        try:
            eq_assignments += self._equilibrium.get_subexpression_assignments()
        except AttributeError:
            pass

        eq_assignments += [
            ps.Assignment(f_i, eq_i)
            for f_i, eq_i in zip(
                self.feq_symbols, self._equilibrium.discrete_populations
            )
        ]

        neq_assignments = self.get_neq_assignments()

        # relaxation in regularized form
        relaxation = self.feq_symbolic_matrix + (1 - rrate) * self.fneq_symbolic_matrix
        if self.method_type == "onsager":
            if self.onsager_correction is not None:
                relaxation += (1 - rrate) * sp.Matrix(
                    self.oreg_populations_correction_symbols
                )

        subexprs = []
        if cq_eqs:
            subexprs += cq_eqs.all_assignments
        subexprs += eq_assignments
        subexprs += neq_assignments
        mains = [
            ps.Assignment(f_i, r_i)
            for f_i, r_i in zip(self.post_collision_pdf_symbols, relaxation)
        ]

        ac = LbmCollisionRule(self, mains, subexpressions=subexprs)
        return ac

    @abc.abstractmethod
    def get_neq_assignments(self) -> List[ps.Assignment]:
        """Return the non-equilibrium assignments list"""

    def bgk_nonequilibrium_populations(self):
        return sp.Matrix(self.pre_collision_pdf_symbols) - self.feq_symbolic_matrix

    def compute_alpha(self):
        if self.entropic_collision:
            raise NotImplementedError("Entropic Collision is not yet implemented.")
        else:
            alpha = 2.0
        return alpha

    def compute_beta_from_reynolds_number(self, Re, N, u):
        if not self.isothermal:
            raise NotImplementedError(
                "Relaxation rate computation is currently supported only for isothermal simulations"
            )

        nu = self.compute_lattice_viscosity_from_reynolds_number(Re, N, u)
        # need to understand how the value of theta will be supplied in case of thermal situations
        theta = (
            float(self.isothermal_temperature)
            if self.isothermal
            else sp.symbols("theta")
        )
        tau = nu / theta
        return 1 / (2 * tau + 1)

    def compute_lattice_viscosity_from_reynolds_number(self, Re, N, u):
        return (u * N) / Re
