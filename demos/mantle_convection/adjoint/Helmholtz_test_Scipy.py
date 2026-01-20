# ## Define the forward problem and the objective functional

# +
from gadopt import *
from gadopt.inverse import *
import inspect
from gadopt.utility import log

# define the helmholtz solver
def helmholtz(V, source):
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(v), grad(u)) * dx + 100.0*v*u*dx - v*source*dx
    solve(F == 0, u)
    return u

# define a mesh
mesh = UnitIntervalMesh(300)
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
reduced_functional = ReducedFunctional(J, c)

# define the boundary conditions
T_lb = Function(V, name="Lower bound")
T_ub = Function(V, name="Upper bound")
T_lb.assign(-1.0)
T_ub.assign(1.0)

def callback(intermediate_result):
    sf = inspect.stack()[3].frame.f_locals["sf"]
    log(f"{intermediate_result.fun=}")
    log(f"{sf.nfev=}, {sf.ngev=}")

sol = minimize(reduced_functional, method = 'L-BFGS-B', bounds=(T_lb, T_ub), tol=1e-12, callback=callback, options={"disp": True, "maxiter": 2})


# -


