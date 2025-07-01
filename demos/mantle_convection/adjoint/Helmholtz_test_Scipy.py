# +
# import petsc4py
# petsc4py.init(['-log_view', 'solver_log_helmholtz.txt', '-ksp_monitor'])
# -

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
mesh = UnitIntervalMesh(3000000)
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


# -

# ## Using Line search with Netwon-Krylov method

# +
import datetime
import time

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
minimisation_parameters["Status Test"]["Iteration Limit"] = 5
minimisation_parameters["Step"]["Line Search"] = {
  "Descent Method": {"Type": "Newton-Krylov"}
}
# minimisation_parameters["General"]["Secant"]["Type"] = "Limited-Memory BFGS"
# try:
#     rol_secant = ROL.lBFGS(parameters["General"]["Secant"]["Maximum Storage"])
# except KeyError:
#     # Use the default storage value
#     rol_secant = ROL.lBFGS()
rol_solver = ROLSolver(minimisation_problem, minimisation_parameters, inner_product="L2")
rol_params = ROL.ParameterList(minimisation_parameters, "Parameters")
rol_algorithm = ROL.LineSearchAlgorithm(rol_params)

solutions_vtk = VTKFile("solutions_Helmholtz.pvd")
solution_source = Function(source.function_space(), name="Source")
solution_u = Function(u.function_space(), name="Solution")    
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
iteration = 0

# Profiling
def record_pre_hess(*args):
    global counter_hess
    global start_time_hess
    counter_hess = counter_hess + 1
    start_time_hess = datetime.datetime.now()
    start_time_hess_disp = start_time_hess.strftime("%a, %b %d, %Y %I:%M:%S %p")
    log(f"Hessian calculation started with count: {counter_hess} at: {start_time_hess_disp}")

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
    log(f"Functional calculation started at: {start_time_func_disp}")

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
    log(f"Functional calculation finished at time: {end_time_func_disp} and completed in: {hours:02}:{minutes:02}:{seconds:02}")
    functional_values.append(func_value)

def record_pre_grad(controls, *args):
    global start_time_grad
    global counter_grad
    counter_grad = counter_grad + 1
    start_time_grad = datetime.datetime.now()
    start_time_grad_disp = start_time_grad.strftime("%a, %b %d, %Y %I:%M:%S %p")
    log(f"Gradient calculation started at: {start_time_grad_disp}")
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
    log(f"Gradient calculation finished at time: {end_time_grad_disp} and completed in: {hours:02}:{minutes:02}:{seconds:02}")
    return derivatives

# Log values of initial and final misfit:
def record_misfit_values(init_misfit, final_misfit):
    initial_misfit_values.append(init_misfit)
    final_misfit_values.append(final_misfit)



class StatusTest(ROL.StatusTest):
    def check(self, status):
        # callback stuff goes here
        initial_misfit = assemble(
            (source.block_variable.checkpoint - source_ref) ** 2 * dx
        )
        final_misfit = assemble(
            (u.block_variable.checkpoint - u_ref) ** 2 * dx
        )

        reduced_functional.eval_cb_pre = record_pre_func
        reduced_functional.eval_cb_post = record_post_func
        reduced_functional.derivative_cb_pre = record_pre_grad
        reduced_functional.derivative_cb_post = record_post_grad
        reduced_functional.hessian_cb_pre = record_pre_hess
        reduced_functional.hessian_cb_post = record_post_hess
        record_misfit_values(initial_misfit, final_misfit)
        
        # Print output for ease of tracking simulation progress:
        if counter_hess == 0 and counter_func == 0 and counter_grad == 0:
            log(f"No Hessians, functionals and gradients calculated \n")
        elif counter_hess == 0 and counter_grad == 0:
            log(f"Total Hessians: {counter_hess}, Hessian time avg: 0.0 ; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: 0.0\n")
        elif counter_hess == 0:
            log(f"Total Hessians: {counter_hess}, Hessian time avg: 0.0; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: {elapsed_time_grad/counter_grad}\n")
        else:
            log(f"Total Hessians: {counter_hess}, Hessian time avg: {elapsed_time_hess/counter_hess}; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: {elapsed_time_grad/counter_grad}\n")  
        if functional_values:
            log(f"Functional value: {functional_values[-1]};  Misfit (Source): {initial_misfit};  Misfit (Final): {final_misfit}")
        else:
            log(f"Functional value not recorded; Misfit (Source): {initial_misfit}; Misfit (Final): {final_misfit}")

        # Write functional and misfit values to a file (appending to avoid overwriting)
        if MPI.COMM_WORLD.Get_rank() == 0:        
            with open("functional_helmholtz_test.txt", "a") as f:
                f.write(f"Iteration: {iteration} \n")
                if counter_hess == 0 and counter_func == 0 and counter_grad == 0:
                    f.write(f"No Hessians, functionals and gradients calculated \n")
                elif counter_hess == 0 and counter_grad == 0:
                    f.write(f"Total Hessians: {counter_hess}, Hessian time avg: 0.0; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: 0.0\n")
                elif counter_hess == 0:
                    f.write(f"Total Hessians: {counter_hess}, Hessian time avg: 0.0; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: {elapsed_time_grad/counter_grad}\n")
                else:
                    f.write(f"Total Hessians: {counter_hess}, Hessian time avg: {elapsed_time_hess/counter_hess}; Total functionals: {counter_func}, Functional time avg: {elapsed_time_func/counter_func}; Total Gradients: {counter_grad}, Gradient time avg: {elapsed_time_grad/counter_grad}\n")
                if functional_values:            
                    f.write(f"Functional value: {functional_values[-1]}, Misfit (Source): {initial_misfit}, Misfit (Final): {final_misfit}\n")
                else:
                    f.write(f"Functional value: 0.0, Misfit (Source): {initial_misfit}, Misfit (Final): {final_misfit}\n")

        # Write VTK output:
        solution_u.assign(u.block_variable.checkpoint)
        solution_source.assign(source.block_variable.checkpoint)        
        solutions_vtk.write(solution_u, solution_source)

        return super().check(status)


rol_algorithm.setStatusTest(StatusTest(rol_params), False)
rol_algorithm.run(rol_solver.rolvector, rol_solver.rolobjective)
# -
# --------------------------------------------------------------------------------------------------------------------


