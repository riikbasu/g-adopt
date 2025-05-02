from gadopt import *
from gadopt.inverse import *


# define the helmholtz solver
def helmholtz(V, source):
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(v), grad(u)) * dx + 100.0*v*u*dx - v*source*dx

    solve(F == 0, u)
    return u


# ## Define the forward problem and the objective functional

# +
# define a mesh
mesh = UnitIntervalMesh(10)
num_processes = mesh.comm.size
mesh_checkpoint = f"mesh_helmholtz_np{num_processes}.h5"
# create a checkpointable mesh by writing to disk and restoring
with CheckpointFile(mesh_checkpoint, "w") as f:
    f.save_mesh(mesh)
with CheckpointFile(mesh_checkpoint, "r") as f:
    mesh = f.load_mesh("firedrake_default")

# define the space and sources
V = FunctionSpace(mesh, "CG", 1)
source_ref = Function(V)
x = SpatialCoordinate(mesh)
source_ref.interpolate(cos(pi * x**2))

# compute reference solution
with stop_annotating():
    u_ref = helmholtz(V, source_ref)

# tape the forward solution
source = Function(V)
c = Control(source)
u = helmholtz(V, source)

# define the reduced objective functional
J = assemble(1e6 * (u - u_ref)**2 * dx)
rf = ReducedFunctional(J, c)

# define the boundary conditions
T_lb = Function(V, name="Lower bound")
T_ub = Function(V, name="Upper bound")
T_lb.assign(-1.0)
T_ub.assign(1.0)


# -

# ## Using Scipy minimize - This part fails--------------------------------------------------------------------------

# Setting up the problem using minimize that uses Scipy
sol = minimize(rf, bounds=(T_lb, T_ub), tol=1e-12)

#--------------------------------------------------------------------------------------------------------------------

# ## Using Lin-more optimiser - ROL optimiser works with the same framework -----------------------------------------

# +
# define the optimiser run
def run(optimiser, rf, rank, filename):
    if rank == 0:
        with open(filename, "w") as f:
            rf.eval_cb_post = lambda val, *args: f.write(f"{val}\n")
            optimiser.run()
            rf.eval_cb_pots = lambda *args: None
    else:
        optimiser.run()

# set up the minimisation problem
minimisation_problem = MinimizationProblem(rf, bounds=(T_lb, T_ub))
minimisation_parameters["Status Test"]["Iteration Limit"] = 10

# run full optimisation, checkpointing every iteration
checkpoint_dir = f"optimisation_checkpoint_np{num_processes}"
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir=checkpoint_dir,
)
run(optimiser, rf, mesh.comm.rank, f"full_optimisation_np{num_processes}.dat")
# -
#--------------------------------------------------------------------------------------------------------------------


