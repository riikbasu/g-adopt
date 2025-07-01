# +
# from gadopt import *
# from gadopt.inverse import *

# +
# # define the helmholtz solver
# def helmholtz(V, source):
#     u = Function(V)
#     v = TestFunction(V)
#     F = inner(grad(v), grad(u)) * dx + 100.0*v*u*dx - v*source*dx

#     solve(F == 0, u)
#     return u
# -

# ## Define the forward problem and the objective functional

# +
# # define a mesh
# mesh = UnitIntervalMesh(10)
# num_processes = mesh.comm.size
# mesh_checkpoint = f"mesh_helmholtz_np{num_processes}.h5"
# # create a checkpointable mesh by writing to disk and restoring
# with CheckpointFile(mesh_checkpoint, "w") as f:
#     f.save_mesh(mesh)
# with CheckpointFile(mesh_checkpoint, "r") as f:
#     mesh = f.load_mesh("firedrake_default")

# # define the space and sources
# V = FunctionSpace(mesh, "CG", 1)
# source_ref = Function(V)
# x = SpatialCoordinate(mesh)
# source_ref.interpolate(cos(pi * x**2))

# # compute reference solution
# with stop_annotating():
#     u_ref = helmholtz(V, source_ref)

# # tape the forward solution
# source = Function(V)
# c = Control(source)
# u = helmholtz(V, source)

# # define the reduced objective functional
# J = assemble(1e6 * (u - u_ref)**2 * dx)
# rf = ReducedFunctional(J, c)

# # define the boundary conditions
# T_lb = Function(V, name="Lower bound")
# T_ub = Function(V, name="Upper bound")
# T_lb.assign(-1.0)
# T_ub.assign(1.0)


# -

# ## Using Scipy minimize - This part fails--------------------------------------------------------------------------

# +
# # Setting up the problem using minimize that uses Scipy
# sol = minimize(rf, bounds=(T_lb, T_ub), tol=1e-12)
# -

# --------------------------------------------------------------------------------------------------------------------

# ## Using Lin-more optimiser - ROL optimiser works with the same framework -----------------------------------------

# +
# # define the optimiser run
# def run(optimiser, rf, rank, filename):
#     if rank == 0:
#         with open(filename, "w") as f:
#             rf.eval_cb_post = lambda val, *args: f.write(f"{val}\n")
#             optimiser.run()
#             rf.eval_cb_pots = lambda *args: None
#     else:
#         optimiser.run()

# # set up the minimisation problem
# minimisation_problem = MinimizationProblem(rf, bounds=(T_lb, T_ub))
# minimisation_parameters["Status Test"]["Iteration Limit"] = 10

# # run full optimisation, checkpointing every iteration
# checkpoint_dir = f"optimisation_checkpoint_np{num_processes}"
# optimiser = LinMoreOptimiser(
#     minimisation_problem,
#     minimisation_parameters,
#     checkpoint_dir=checkpoint_dir,
# )
# run(optimiser, rf, mesh.comm.rank, f"full_optimisation_np{num_processes}.dat")
# -
# --------------------------------------------------------------------------------------------------------------------

# +
# from firedrake import *
# from firedrake.adjoint import *
from gadopt import *
from gadopt.inverse import *
import inspect
import datetime
import time

continue_annotation()

# define the space and sources
mesh = UnitIntervalMesh(3000000)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
uref = Function(V)
v = TestFunction(V)

# compute target solution
with stop_annotating():
  x = SpatialCoordinate(mesh)
  Fref = inner(grad(v), grad(uref)) * dx + 100.0*v*uref*dx - v*cos(pi * x[0]**2)*dx
  solve(Fref == 0, uref)

# create forward problem
source = Function(V)
F = inner(grad(v), grad(u)) * dx + 100.0*v*u*dx - v*source*dx
solve(F == 0, u)

# define the reduced objective functional
J = assemble((u - uref)**2 * dx)
rf = ReducedFunctional(J, Control(source))

# define the boundary conditions
T_lb = Function(V, name="Lower bound")
T_ub = Function(V, name="Upper bound")
T_lb.assign(-1.0)
T_ub.assign(1.0)

functional_values = []
initial_misfit_values = []
final_misfit_values = []
counter_hess = 0
counter_func = 0
counter_grad = 0
start_time_hess = 0
start_time_func = 0
start_time_grad = 0
elapsed_time_hess = 0
elapsed_time_func = 0
elapsed_time_grad = 0
iteration = 1

# Profiling
def record_pre_hess(*args):
    global counter_hess
    global start_time_hess
    counter_hess = counter_hess + 1
    start_time_hess = datetime.datetime.now()
    start_time_hess_disp = start_time_hess.strftime("%a, %b %d, %Y %I:%M:%S %p")
    log(f"Hessian calculation started with count: {counter_hess} at time: {start_time_hess_disp}")

def record_post_hess(*args):
    global counter_hess
    global start_time_hess
    global elapsed_time_hess
    end_time_hess = datetime.datetime.now()
    elapsed_time = end_time_hess-start_time_hess
    elapsed_time_hess = elapsed_time_hess + elapsed_time.total_seconds()
    end_time_hess_disp = end_time_hess.strftime("%a, %b %d, %Y %I:%M:%S %p")
    total_seconds = int(elapsed_time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    log(f"Hessian calculation finished with count: {counter_hess} at time: {end_time_hess_disp} and completed in: {hours:02}:{minutes:02}:{seconds:02}")

def record_pre_func(*args):
    global counter_func
    global start_time_func
    counter_func = counter_func + 1
    start_time_func = datetime.datetime.now()
    start_time_func_disp = start_time_func.strftime("%a, %b %d, %Y %I:%M:%S %p")
    log(f"Functional calculation started with count: {counter_func} at time: {start_time_func_disp}")

def record_post_func(func_value, *args):
    global start_time_func
    global elapsed_time_func
    end_time_func = datetime.datetime.now()
    elapsed_time = end_time_func-start_time_func
    elapsed_time_func = elapsed_time_func + elapsed_time.total_seconds()
    end_time_func_disp = end_time_func.strftime("%a, %b %d, %Y %I:%M:%S %p")
    total_seconds = int(elapsed_time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    log(f"Functional calculation finished with count: {counter_func} at time: {end_time_func_disp} and completed in: {hours:02}:{minutes:02}:{seconds:02}")
    functional_values.append(func_value)

def record_pre_grad(controls, *args):
    global start_time_grad
    global counter_grad
    counter_grad = counter_grad + 1
    start_time_grad = datetime.datetime.now()
    start_time_grad_disp = start_time_grad.strftime("%a, %b %d, %Y %I:%M:%S %p")
    log(f"Gradient calculation started with count: {counter_grad} at time: {start_time_grad_disp}")
    return controls

def record_post_grad(checkpoint, derivatives, values, *args):
    global start_time_grad
    global elapsed_time_grad
    end_time_grad = datetime.datetime.now()
    elapsed_time = end_time_grad-start_time_grad
    elapsed_time_grad = elapsed_time_grad + elapsed_time.total_seconds()
    end_time_grad_disp = end_time_grad.strftime("%a, %b %d, %Y %I:%M:%S %p")
    total_seconds = int(elapsed_time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    log(f"Gradient calculation finished with count: {counter_grad} at time: {end_time_grad_disp} and completed in: {hours:02}:{minutes:02}:{seconds:02}")
    return derivatives


def callback(intermediate_result):
    global iteration
    sf = inspect.stack()[3].frame.f_locals["sf"]
    # k = inspect.stack()[3].frame.f_locals["k"]
    print(f"Iteration: {iteration} completed")
    print(f"{intermediate_result.fun=}")
    print(f"{sf.nfev=}, {sf.ngev=}, {sf.nhev=}")
    
    # Print output for ease of tracking simulation progress:
    if counter_hess == 0 and counter_func == 0 and counter_grad == 0:
        log(f"No Hessians, functionals and gradients calculated \n")
    elif counter_hess == 0 and counter_grad == 0:
        log(f"Total Hessians: {counter_hess}, Hessian time avg: 0.0 ; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: 0.0\n")
    elif counter_hess == 0:
        log(f"Total Hessians: {counter_hess}, Hessian time avg: 0.0; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: {elapsed_time_grad/counter_grad}\n")
    else:
        log(f"Total Hessians: {counter_hess}, Hessian time avg: {elapsed_time_hess/counter_hess}; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: {elapsed_time_grad/counter_grad}\n") 

    # Write functional and misfit values to a file (appending to avoid overwriting)
    if MPI.COMM_WORLD.Get_rank() == 0:        
        with open("minimal_failing_example.txt", "a") as f:
            f.write(f"Iteration: {iteration} \n")
            if counter_hess == 0 and counter_func == 0 and counter_grad == 0:
                f.write(f"No Hessians, functionals and gradients calculated \n")
            elif counter_hess == 0 and counter_grad == 0:
                f.write(f"Total Hessians: {counter_hess}, Hessian time avg: 0.0; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: 0.0\n")
            elif counter_hess == 0:
                f.write(f"Total Hessians: {counter_hess}, Hessian time avg: 0.0; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: {elapsed_time_grad/counter_grad}\n")
            else:
                f.write(f"Total Hessians: {counter_hess}, Hessian time avg: {elapsed_time_hess/counter_hess}; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: {elapsed_time_grad/counter_grad}\n")

    iteration = iteration + 1

rf.eval_cb_pre = record_pre_func
rf.eval_cb_post = record_post_func
rf.derivative_cb_pre = record_pre_grad
rf.derivative_cb_post = record_post_grad
rf.hessian_cb_pre = record_pre_hess
rf.hessian_cb_post = record_post_hess

# Setting up the problem using minimize that uses Scipy
print(f"Iteration: 0 completed")
sol = minimize(rf, method='L-BFGS-B', bounds=(T_lb, T_ub), tol=1e-12, options={"disp": True, "maxiter": 5}, 
derivative_options={"riesz_representation": "l2"})

# +
from firedrake import *
from firedrake.adjoint import *

continue_annotation()

# define the space and sources
mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
uref = Function(V)
v = TestFunction(V)

# compute target solution
with stop_annotating():
  x = SpatialCoordinate(mesh)
  Fref = inner(grad(v), grad(uref)) * dx + 100.0*v*uref*dx - v*cos(pi * x[0]**2)*dx
  solve(Fref == 0, uref)

# create forward problem
source = Function(V)
F = inner(grad(v), grad(u)) * dx + 100.0*v*u*dx - v*source*dx
solve(F == 0, u)

# define the reduced objective functional
J = assemble((u - uref)**2 * dx)
rf = ReducedFunctional(J, Control(source))

# define the boundary conditions
T_lb = Function(V, name="Lower bound")
T_ub = Function(V, name="Upper bound")
T_lb.assign(-1.0)
T_ub.assign(1.0)

# Setting up the problem using minimize that uses Scipy
sol = minimize(rf, method='Newton-CG', bounds=(T_lb, T_ub), tol=1e-12, options={"disp": True, "maxiter": 5}, 
derivative_options={"riesz_representation": "l2"})
# +
import firedrake as fd
from firedrake import *
from firedrake.adjoint import *
from petsc4py import PETSc
import numpy as np

continue_annotation()

# === Mesh and Function Space ===
mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="State")
uref = Function(V, name="Reference")
v = TestFunction(V)
source = Function(V, name="Control")

# === Reference solution uref ===
with stop_annotating():
    x = SpatialCoordinate(mesh)
    Fref = inner(grad(v), grad(uref)) * dx + 100.0 * v * uref * dx - v * cos(pi * x[0]**2) * dx
    solve(Fref == 0, uref)

# === Hessian operator class ===
class AdjointHessianOperator:
    def __init__(self, V_space):
        self.V = V_space
        self.current_rf = None
        self.v_fd = Function(self.V)

    def set_reduced_functional(self, rf_instance):
        self.current_rf = rf_instance

    def mult(self, mat_shell, in_vec_petsc, out_vec_petsc):
        PETSc.Sys.Print("Inside Hessian mult", comm=PETSc.COMM_WORLD)
        try:
            if self.current_rf is None:
                raise RuntimeError("ReducedFunctional not set in Hessian operator.")

            v_arr = in_vec_petsc.getArray()
            if len(v_arr) != self.v_fd.dat.data.shape[0]:
                raise ValueError(f"Input vector length {len(v_arr)} != {self.v_fd.dat.data.shape[0]}")

            self.v_fd.dat.data[:] = v_arr
            Hv_fd = self.current_rf.hessian(self.v_fd)
            Hv_arr = Hv_fd.dat.data_ro[:]

            out_vec_petsc.array[:] = Hv_arr

        except Exception as e:
            PETSc.Sys.Print(f"Exception in Hessian mult: {e}", comm=PETSc.COMM_WORLD)
            out_vec_petsc.array[:] = 0.0  # Avoid PETSc error by zeroing output

# Instantiate Hessian operator
hessian_op = AdjointHessianOperator(V)

# === Initial solve and create ReducedFunctional once ===
F = inner(grad(v), grad(u)) * dx + 100.0 * v * u * dx - v * source * dx
solve(F == 0, u)
J = assemble((u - uref)**2 * dx)
rf = ReducedFunctional(J, Control(source))
hessian_op.set_reduced_functional(rf)

# === Objective and Gradient callback ===
def tao_objgrad(tao, x_petsc, g_petsc):
    PETSc.Sys.Print("--- Inside tao_objgrad ---", comm=PETSc.COMM_WORLD)

    s_array = x_petsc.getArray(readonly=True)
    source.dat.data[:] = s_array

    # Solve state PDE with updated control
    F = inner(grad(v), grad(u)) * dx + 100.0 * v * u * dx - v * source * dx
    solve(F == 0, u)

    # Evaluate objective and gradient using existing ReducedFunctional
    J_val = rf(source)            # Evaluate functional at current control
    dJ = rf.derivative()          # Correct: no argument here!

    g_petsc.array[:] = dJ.dat.data_ro[:]

    PETSc.Sys.Print(f"Objective J: {J_val:.6e}", comm=PETSc.COMM_WORLD)
    PETSc.Sys.Print(f"Gradient norm: {np.linalg.norm(dJ.dat.data_ro[:]):.6e}", comm=PETSc.COMM_WORLD)
    PETSc.Sys.Print("--- Exiting tao_objgrad ---", comm=PETSc.COMM_WORLD)

    return J_val

# === Setup PETSc vectors and bounds ===
x_np = source.dat.data_ro[:].copy()
N = len(x_np)
x = PETSc.Vec().createWithArray(x_np.copy(), comm=PETSc.COMM_WORLD)
xl = PETSc.Vec().createWithArray(-np.ones_like(x_np), comm=PETSc.COMM_WORLD)
xu = PETSc.Vec().createWithArray(np.ones_like(x_np), comm=PETSc.COMM_WORLD)

# === Setup TAO solver ===
tao = PETSc.TAO().create(comm=PETSc.COMM_WORLD)
tao.setType("tron")

H_mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
H_mat.setSizes((N, N))  # global size (rows, cols)
H_mat.setType('python')
H_mat.setPythonContext(hessian_op)
H_mat.setUp()
H_mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)

tao.setObjectiveGradient(tao_objgrad)
tao.setHessian(H_mat, H_mat)
tao.setSolution(x)
tao.setVariableBounds(xl, xu)

# === Initial call to setup Hessian operator properly ===
PETSc.Sys.Print("\nPerforming initial tao_objgrad call to set up Hessian operator...", comm=PETSc.COMM_WORLD)
dummy_grad = PETSc.Vec().createWithArray(np.zeros_like(x_np), comm=PETSc.COMM_WORLD)
_ = tao_objgrad(tao, x, dummy_grad)
PETSc.Sys.Print("Initial tao_objgrad call complete.\n", comm=PETSc.COMM_WORLD)

# === Optional PETSc TAO options for debug ===
PETSc.Options()["tao_monitor"] = None
PETSc.Options()["tao_view"] = None
PETSc.Options()["tao_max_it"] = 20
PETSc.Options()["tao_converged_reason"] = None

tao.setFromOptions()

# === Run optimization ===
print("\nStarting optimization...")
try:
    tao.solve()
    print("Optimization finished.")
except Exception as e:
    print(f"\nTAO solver failed: {e}")
    import traceback
    print(traceback.format_exc())

# === Write output ===
source.dat.data[:] = tao.getSolution().getArray(readonly=True)
fd.output.VTKFile("optimized_source.pvd").write(source)
print("✅ Optimization complete. Solution written to 'optimized_source.pvd'")

# -


