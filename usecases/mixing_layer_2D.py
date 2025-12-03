from lbmpy.methods.population_space import PopulationSpaceBGK

from lbmpy import LBStencil, LBMOptimisation, create_lb_update_rule
from lbmpy.equilibrium import DiscreteHydrodynamicMaxwellian
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter

import numpy
import sympy
import pystencils as ps

print('''
=====================
2D Mixing Layer Test
=====================
Ref: Jonnalagadda et al. PRE (2021)

LB Configuration:
-----------------
    a) D2Q9 lattice with O(u^3) polynomial equilibrium
    b) BGK Collision Model
    b) Isothermal temperature: lattice reference temperature (theta = 1/3)

Case Parameters:
----------------
    1. Re = 30000    # Reynolds
    2. kappa = 80    # shear layer width
    3. u0 = 0.04     # velocity scale
    4. delta = 0.05  # perturbation parameter

Grid Parameters:
----------------
    1. N = Nx = Ny = 256

Run time:
---------
    nsteps = N/u0 = 6400

========================
'''
)

stencil = LBStencil("D2Q9")

# number of ghost layers
ngl = max([element for velocity in stencil.stencil_entries for element in velocity])

method = "bgk"
equilibrium_type = "polynomial"
base_temperature = stencil.theta0

equilibrium = DiscreteHydrodynamicMaxwellian(
    stencil=stencil,
    compressible=True,
    deviation_only=False,
    c_s_sq=base_temperature,
    order = 3)

srt_method = PopulationSpaceBGK(
    stencil=stencil,
    feq=equilibrium,
    isothermal=True,
)

# Domain and Arrays

# grid points
N = Ny = Nx = 256
# data handler (NxN periodic grid laid out in Structure of Arrays format)
dh = ps.create_data_handling(
        domain_size=(Ny, Nx),
        periodicity=(True, True),
        default_ghost_layers=ngl,
        default_target=ps.Target.CPU,
        default_layout="fzyx")

# create population arrays
f = dh.add_array(name="f", values_per_cell=stencil.Q)
f_tmp = dh.add_array(name="f_tmp", values_per_cell=stencil.Q)

# create field arrays
rho = dh.add_array(name="rho", values_per_cell=1)
u = dh.add_array(name="u", values_per_cell=stencil.D)


# Stream-Collide Kernel

lbm_opt = LBMOptimisation(
    symbolic_field=f,
    symbolic_temporary_field=f_tmp
)

output_fields = dict({"density": rho, "velocity": u})

update_rule = create_lb_update_rule(
    lb_method = srt_method,
    lbm_optimisation=lbm_opt,
    output = output_fields
)

# update_rule

# create stream-collide kernel
ker_stream_collide = ps.create_kernel(update_rule).compile()

ac_init = macroscopic_values_setter(
    lb_method=srt_method,
    density=rho.center,       # rho_(0,0)
    velocity=u.center_vector, # \vec{u}_(0,0)
    pdfs=f,
    set_pre_collision_pdfs=True
)

ker_init = ps.create_kernel(ac_init).compile()

## Initial state
# \begin{align}
#     \rho &= 1 \\
#     u_x
#     &=
#     \begin{cases}
#         u_0 \tanh \left[ \kappa \left( \frac{y}{N} - \frac{1}{4} \right)\right], \, y \leq \frac{N}{2} \\
#         u_0 \tanh \left[ \kappa \left( \frac{3}{4} - \frac{y}{N} \right)\right], \, y > \frac{N}{2}
#     \end{cases}\\
#     u_y
#     &=
#     \delta u_0 \sin \left[2\pi\left( \frac{x}{N} + \frac{1}{4} \right)\right]
# \end{align}

# Parameters
kappa = 80    # shear layer width
u0 = 0.04     # velocity scale
delta = 0.05  # perturbation parameter

# Initialize the density and velocity fields to unity and zero
dh.fill(rho.name, 1.0)
dh.fill(u.name, 0.0)

# initialize the velocity field
_x = numpy.linspace(0, 1, Nx)
_y = numpy.linspace(0, 1, Ny)

for y in range(Ny):
    dh.cpu_arrays[u.name][ngl:-1*ngl, ngl+y, 1] = delta * numpy.sin(2*numpy.pi*(_x + 0.25))

tmp = numpy.zeros_like(_y)
tmp[:Ny//2] = numpy.tanh(kappa * (_y[:Ny//2] - 0.25))
tmp[Ny//2:] = numpy.tanh(kappa * (0.75 - _y[Ny//2:]))

for x in range(Nx):
    dh.cpu_arrays[u.name][ngl+x, ngl:-1*ngl, 0] = tmp

dh.cpu_arrays[u.name] *= u0

# run the initalization kernel
dh.run_kernel(ker_init)

# Integration Loop

# global synchronization function
gl_sync = dh.synchronization_function(f.name)

def step():
    dh.run_kernel(
        kernel_function=ker_stream_collide,
        alpha=srt_method.compute_alpha(),
        beta=srt_method.compute_beta_from_reynolds_number(Re=30000, u=u0, N=N)
    )

    dh.swap(f.name, f_tmp.name)
    gl_sync()

def loop(n):
    for iteration in range(n):
        step()


# Simulation Run

# characteristic time
t0 = int(N/u0)

# run
print(f"Running {t0} iterations ... ", end=" ")
loop(t0)
print("done")

print("Computing normalized averaged kinetic energy ... ", end=" ")
ufinal = dh.gather_array(u.name)
print("done")
keNorm = numpy.mean(ufinal[:,:,0]**2 + ufinal[:,:,1]**2)
keNorm /= (u0)**2

print(f"\tNormalized Average Kinetic Energy =  {keNorm:.4f}")
