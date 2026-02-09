# Adjoint inverse reconstruction
# ==============================
#
# Introduction
# ------------
# In this tutorial, we will demonstrate how to perform an inversion to recover the initial temperature field of an
# idealised mantle convection simulation using G-ADOPT. This tutorial is published as the first synthetic experiment in
# *Ghelichkhan et al. (2024)*. The full inversion showcased in the publication involves a total number of 80 timesteps.
# For the tutorial here we start with only 5 timesteps to go through the basics.
#
# The tutorial involves a *twin experiment*, where we assess the performance of the inversion scheme by inverting the
# initial state of a synthetic reference simulation, known as the "*Reference Twin*". To create this reference twin, we
# run a forward mantle convection simulation and record all relevant fields (velocity and temperature) at each time step.
#
# We have pre-run this simulation by running [the forward case](../adjoint_forward), and stored model output as a
# checkpoint file on our servers.  These fields serve as benchmarks for evaluating our inverse problem's performance. To
# download the reference benchmark checkpoint file if it doesn't already exist, execute the following command:



# + tags=["active-ipynb"]
# ![ ! -f adjoint-demo-checkpoint-state.h5 ] && wget https://data.gadopt.org/demos/adjoint-demo-checkpoint-state.h5
# -

# In this file, fields from the reference simulation are stored under the names "Temperature" and "Velocity".
# After importing g-adopt and the associated inverse module (gadopt.inverse - discussed further below), we can
# retrieve timestepping information from the pre-computed forward run as follows

# +
from gadopt import *
from gadopt.inverse import *
from pyadjoint import TAOSolver
# Open the checkpoint file and subsequently load the mesh:
checkpoint_filename = "adjoint-demo-checkpoint-state-highres.h5"
checkpoint_file = CheckpointFile(checkpoint_filename, mode="r")
mesh = checkpoint_file.load_mesh("firedrake_default_extruded")
mesh.cartesian = True

# Specify boundary markers, noting that for extruded meshes the upper and lower boundaries are tagged as
# "top" and "bottom" respectively.
boundary = get_boundary_ids(mesh)

# Retrieve the timestepping information for the Velocity and Temperature functions from checkpoint file:
temperature_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Temperature")
velocity_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Velocity")
# -

# We can check the information for each:

# + tags=["active-ipynb"]
# print("Timestepping info for Temperature", temperature_timestepping_info)
# print("Timestepping info for Velocity", velocity_timestepping_info)
# -

# The timestepping information reveals that there are 80 time-steps (from 0 to 79) in the reference simulation,
# with the temperature field stored only at the initial (index=0) and final (index=79) timesteps, while the
# velocity field is stored at all timesteps. We can visualise the benchmark fields using Firedrake's built-in VTK
# functionality. For example, initial and final temperature fields can be loaded:

# Load the final state, analagous to the present-day "observed" state:
Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][-1]))
Tobs.rename("Observed Temperature")
# Load the reference initial state - i.e. the state that we wish to recover:
Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][0]))
Tic_ref.rename("Reference Initial Temperature")
checkpoint_file.close()

# These fields can be visualised using standard VTK software, such as Paraview or pyvista.

# + tags=["active-ipynb"]
# import pyvista as pv
# VTKFile("./visualisation_vtk.pvd").write(Tobs, Tic_ref)
# # dataset = pv.read('./visualisation_vtk.pvd')
# # # Create a plotter object
# # plotter = pv.Plotter()
# # # Add the dataset to the plotter
# # plotter.add_mesh(dataset, scalars='Observed Temperature', cmap='coolwarm')
# # # Adjust the camera position
# # plotter.camera_position = [(0.5, 0.5, 2.5), (0.5, 0.5, 0), (0, 1, 0)]
# # # Show the plot
# # plotter.show(jupyter_backend="static")
# -

# The Inverse Code
# ----------------
#
# The novelty of using the overloading approach provided by pyadjoint is that it requires
# minimal changes to our script to enable the inverse capabalities of G-ADOPT.
# To turn on the adjoint, one simply imports the inverse module (already done above) to
# enable all taping functionality from pyadjoint.
#
# Doing so will turn Firedrake's objects to overloaded types, in a way
# that any UFL operation will be annotated and added to the tape, unless
# otherwise specified.
#
# We first ensure that the tape is cleared of any previous operations, using the following code:

tape = get_working_tape()
tape.clear_tape()

# + tags=["active-ipynb"]
# # To verify the tape is empty, we can print all blocks:
# print(tape.get_blocks())
# -

# From here on, all user operations are specified with minimal differences relative to
# to our forward code. Under the hood, however, the tape will be populated
# by *blocks* that record their dependencies. Knowing the mesh was loaded above, we continue
# in a manner that is consistent with our most basic forward modelling tutorials.

# +
# Set up function spaces:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "DQ", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space

# Specify test functions and functions to hold solutions:
z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
T = Function(Q, name="Temperature")

# Specify important constants for the problem, alongside the approximation:
Ra = Constant(1e6)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# Define time-stepping parameters:
delta_t = Constant(4e-6)  # Constant time step
timesteps = int(temperature_timestepping_info["index"][-1]) + 1  # number of timesteps from forward

# Nullspaces for the problem are next defined:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Followed by boundary conditions, noting that all boundaries are free slip, whilst the domain is
# heated from below (T = 1) and cooled from above (T = 0).
stokes_bcs = {
    boundary.bottom: {"uy": 0},
    boundary.top: {"uy": 0},
    boundary.left: {"ux": 0},
    boundary.right: {"ux": 0},
}
temp_bcs = {
    boundary.bottom: {"T": 1.0},
    boundary.top: {"T": 0.0},
}

# Setup Energy and Stokes solver
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace, constant_jacobian=True)
# -

# Specify Problem Length
# ------------------------
#
# For the purpose of this demo, we only invert for a total of 10 time-steps. This makes it
# tractable to run this within a tutorial session.
#
# To run for the simulation's full duration, change the initial_timestep to `0` below, rather than
# `timesteps - 10`.

# initial_timestep = timesteps - 10
initial_timestep = 0

# Define the Control Space
# ------------------------
#
# In this section, we define the control space, which can be restricted to reduce the risk of encountering an
# undetermined problem. Here, we select the Q1 function space for the initial condition $T_{ic}$. We also provide an
# initial guess for the control value, which in this synthetic test is the temperature field of the reference
# simulation at the final time-step (`timesteps - 1`). In other words, our guess for the initial temperature
# is the final model state.

# +
# Define control function space:
Q1 = FunctionSpace(mesh, "CG", 1)

# Create a function for the unknown initial temperature condition, which we will be inverting for. Our initial
# guess is set to the 1-D average of the forward model. We first load that, at the relevant timestep.
# Note that this layer average will later be used for the smoothing term in our objective functional.
with CheckpointFile(checkpoint_filename, mode="r") as checkpoint_file:
    Taverage = checkpoint_file.load_function(mesh, "Average_Temperature", idx=initial_timestep)
Tic = Function(Q1, name="Initial_Condition_Temperature").assign(Taverage)

# Given that Tic will be updated during the optimisation, we also create a function to store our initial guess,
# which we will later use for smoothing. Note that since smoothing is executed in the control space, we must
# specify boundary conditions on this term in that same Q1 space.
T0_bcs = [DirichletBC(Q1, 0., boundary.top), DirichletBC(Q1, 1., boundary.bottom)]
T0 = Function(Q1, name="Initial_Guess_Temperature").project(Tic, bcs=T0_bcs)

# We next make pyadjoint aware of our control problem:
control = Control(Tic)

# Take our initial guess and project to T, simultaneously applying boundary conditions in the Q2 space:
T.project(Tic, bcs=energy_solver.strong_bcs)

# We continue by integrating the solutions at each time-step.
# Notice that we cumulatively compute the misfit term with respect to the
# surface velocity observable.

# +
u_misfit = 0.0

# Next populate the tape by running the forward simulation.
for time_idx in range(initial_timestep, timesteps):
    stokes_solver.solve()
    energy_solver.solve()
    # Update the accumulated surface velocity misfit using the observed value.
    with CheckpointFile(checkpoint_filename, mode="r") as checkpoint_file:
        uobs = checkpoint_file.load_function(mesh, name="Velocity", idx=time_idx)
    u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)
# -

# Define the Objective Functional
# -------------------------------
#
# Now that all calculations are in place, we must define *the objective functional*.
# The objective functional is our way of expressing our goal for this optimisation.
# It is composed of several terms, each representing a different aspect of the model's
# performance and regularisation.
#
# Regularisation involves imposing constraints on solutions to prevent overfitting, ensuring that the model
# generalises well to new data. In this context, we use the one-dimensional (1-D) temperature profile derived from
# the reference simulation as our regularisation constraint. This profile, referred to below as `Taverage`, helps
# stabilise the inversion process by providing a benchmark that guides the solution towards physically plausible states.
#
# We use `Taverage` as a part of the damping and smoothing terms in our regularisation.
# Consequently, the complete objective functional is defined mathematically as follows:
#
# Reiterating that:
# - $T_{ic}$ is the initial temperature condition.
# - $T_{\text{average}}$ is the average temperature profile representing mantle's geotherm.
# - $T_{F}$ is the the temperature field at final time-step.
# - $T_{\text{obs}}$ is the observed temperature field at the final time-step.
# - $u_{\text{obs}}$ is the observed velocity field at *each time-step*.
# - $\alpha_u$, $\alpha_d$, $\alpha_s$ are the three different
#   weighting terms for the velocity, damping and smoothing terms.
#
# We define the objective functional as
# $$ \text{Objective Functional}= \int_{\Omega}(T - T_{\text{obs}}) ^ 2 \, dx \\
#                  +\alpha_u\, \frac{D_{T_{obs}}}{N\times D_{u_{obs}}}\sum_{i}\int_{\partial \Omega_{\text{top}}}(u - u_{\text{obs}}) \cdot(u - u_{\text{obs}}) \, ds \\
#                  +\alpha_s\, \frac{D_{T_{obs}}}{D_{\text{smoothing}}}\int_{\Omega} \nabla(T_{ic} - T_{\text{average}}) \cdot \nabla(T_{ic} - T_{\text{average}}) \, dx \\
#                  +\alpha_d\, \frac{D_{T_{obs}}}{D_{\text{damping}}}\int_{\Omega}(T_{ic} - T_{\text{average}}) ^ 2 \, dx $$

# With the three *normlisation terms* of:
# + $D_{\text{damping}} = \int_{\Omega} T_{\text{average}}^2 \, dx$,
# + $D_{\text{smoothing}} = \int_{\Omega} \nabla T_{\text{obs}} \cdot \nabla T_{\text{obs}} \, dx$,
# + $D_{T_{obs}} = \int_{\Omega} T_{\text{obs}} ^ 2 \, dx$, and
# + $D_{\text{damping}} = \int_{\partial \Omega_{\text{top}}} u_{\text{obs}} \cdot u_{\text{obs}} \, ds$
#
# which we specify through the `objective` below:

# +
# Define component terms of overall objective functional and their normalisation terms:
damping = assemble((T0 - Taverage) ** 2 * dx)
norm_damping = assemble(Taverage**2 * dx)
smoothing = assemble(dot(grad(T0 - Taverage), grad(T0 - Taverage)) * dx)
norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * dx)
norm_obs = assemble(Tobs**2 * dx)
norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

# Define temperature misfit between final state solution and observation:
t_misfit = assemble((T - Tobs) ** 2 * dx)

# Weighting terms
alpha_u = 1e-2
alpha_d = 1e-4
alpha_s = 1e-3

# Define overall objective functional:
objective = (
    t_misfit +
    alpha_u * (norm_obs * u_misfit / timesteps / norm_u_surface) +
    alpha_d * (norm_obs * damping / norm_damping) +
    alpha_s * (norm_obs * smoothing / norm_smoothing)
)
# -

print(type(t_misfit))

# Define the Reduced Functional
# -----------------------------
#
# In optimisation terminology, a reduced functional is a functional that takes a given value for the control and outputs
# the value of the objective functional defined for it. It does this without explicitly depending on all intermediary
# state variables, hence the name "reduced".
#
# To define the reduced functional, we provide the class with an objective (which is an overloaded UFL object) and the control.

reduced_functional = ReducedFunctional(objective, control)

# At this point, we have completed annotating the tape with the necessary information from running the forward simulation.
# To prevent further annotations during subsequent operations, we stop the annotation process. This ensures that no additional
# solves are unnecessarily recorded, keeping the tape focused only on the essential steps.

pause_annotation()

# We can print the contents of the tape at this stage to verify that it is not empty.

# + tags=["active-ipynb"]
# # print(tape.get_blocks())
# -

# Verification of Gradients: Taylor Remainder Convergence Test
# ------------------------------------------------------------
#
# A fundamental tool for verifying gradients is the Taylor remainder convergence test. This test helps ensure that
# the gradients computed by our optimisation algorithm are accurate. For the reduced functional, $J(T_{ic})$, and its derivative,
# $\frac{\mathrm{d} J}{\mathrm{d} T_{ic}}$, the Taylor remainder convergence test can be expressed as:
#
# $$ \left| J(T_{ic} + h \,\delta T_{ic}) - J(T_{ic}) - h\,\frac{\mathrm{d} J}{\mathrm{d} T_{ic}} \cdot \delta T_{ic} \right| \longrightarrow 0 \text{ at } O(h^2). $$
#
# The expression on the left-hand side is termed the second-order Taylor remainder. This term's convergence rate of $O(h^2)$ is a robust indicator for
# verifying the computational implementation of the gradient calculation. Essentially, if you halve the value of $h$, the magnitude
# of the second-order Taylor remainder should decrease by a factor of 4.
#
# We employ these so-called *Taylor tests* to confirm the accuracy of the determined gradients. The theoretical convergence rate is
# $O(2.0)$, and achieving this rate indicates that the gradient information is accurate down to floating-point precision.
#
# ### Performing Taylor Tests
#
# In our implementation, we perform a second-order Taylor remainder test for each term of the objective functional. The test involves
# computing the functional and the associated gradient when randomly perturbing the initial temperature field, $T_{ic}$, and subsequently
# halving the perturbations at each level.
#
# Here is how you can perform a Taylor test in the code:

# + tags=["active-ipynb"]
# # # Define the perturbation in the initial temperature field
# # import numpy as np
# # Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
# # Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
#
# # # Perform the Taylor test to verify the gradients
# # minconv = taylor_test(reduced_functional, Tic, Delta_temp)
# -

# The `taylor_test` function computes the Taylor remainder and verifies that the convergence rate is close to the theoretical value of $O(2.0)$. This ensures
# that our gradients are accurate and reliable for optimisation.

# gradJ = reduced_functional.derivative(options={"riesz_representation": "L2"})

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tripcolor(gradJ, axes=axes, cmap='viridis')
# fig.colorbar(collection);

# Running the inversion
# ---------------------
# In the final section of this tutorial, we run the optimisation method. First, we define lower and upper bounds for the optimisation problem to guide
# the optimisation method towards a more constrained solution.
#
# For this simple problem, we perform a bounded nonlinear optimisation where the temperature is only permitted to lie in the range [0, 1]. This means that the
# optimisation problem should not search for solutions beyond these values.

# +
# Define lower and upper bounds for the temperature
T_lb = Function(Tic.function_space(), name="Lower Bound Temperature")
T_ub = Function(Tic.function_space(), name="Upper Bound Temperature")

# # Assign the bounds
# T_lb.assign(0.0)
# T_ub.assign(1.0)

# Define the minimisation problem, with the goal to minimise the reduced functional
# Note: in some scenarios, the goal might be to maximise (rather than minimise) the functional.
# minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
minimisation_problem = MinimizationProblem(reduced_functional)

# +
# from scipy.optimize import minimize
# import inspect

# def callback(intermediate_result):
#     sf = inspect.stack()[3].frame.f_locals["sf"]
#     print(f"{intermediate_result.fun=}")
#     print(f"{sf.nfev=}, {sf.ngev=}")

# # Define the minimisation problem, with the goal to minimise the reduced functional
# # Note: in some scenarios, the goal might be to maximise (rather than minimise) the functional.
# options = { 'maxcor': 10, # The maximum number of variable metric corrections (memory)
#             'ftol': 1e7* np.finfo(float).eps,# 1e14 for low accuracy, 1e7 for mid-accuracy, 10 for high accuracy
#             'gtol': 0,#1e-5, # Iteration stops if projection of gradient is smaller than this
#             'eps': 1e-8, # Only used when approx grad is True
#             'maxfun': 15000, # Maximum number of evaluations
#             'maxiter': 5, # Maximum number of iterations
#             'maxls': 20, # Maximum number of line-searth steps
#         }

# # Setting up the problem using minimize that uses Scipy
# sol = minimize(J, m_global, bounds=bounds, method="L-BFGS-B", tol=1e-12, callback=callback, options = options)
# -

# ## Using Line search with Quasi-Newton method

# +
# Define callbacks

functional_file = "functional_TAO_LS_LMVM_orig_20_secant.txt"
solutions_vtk = VTKFile("solutions_TAO_LS_LMVM_orig_20_secant.pvd")

import datetime

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
    print(f"Hessian calculation started with count: {counter_hess} at time: {start_time_hess_disp}")

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
    print(f"Hessian calculation finished with count: {counter_hess} at time: {end_time_hess_disp} and completed in: {hours:02}:{minutes:02}:{seconds:02}")

def record_pre_func(*args):
    global counter_func
    global start_time_func
    counter_func = counter_func + 1
    start_time_func = datetime.datetime.now()
    start_time_func_disp = start_time_func.strftime("%a, %b %d, %Y %I:%M:%S %p")
    print(f"Functional calculation started with count: {counter_func} at time: {start_time_func_disp}")

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
    print(f"Functional calculation finished with count: {counter_func} at time: {end_time_func_disp} and completed in: {hours:02}:{minutes:02}:{seconds:02}")
    # functional_values.append(func_value)

def record_pre_grad(controls, *args):
    global start_time_grad
    global counter_grad
    counter_grad = counter_grad + 1
    start_time_grad = datetime.datetime.now()
    start_time_grad_disp = start_time_grad.strftime("%a, %b %d, %Y %I:%M:%S %p")
    print(f"Gradient calculation started with count: {counter_grad} at time: {start_time_grad_disp}")
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
    print(f"Gradient calculation finished with count: {counter_grad} at time: {end_time_grad_disp} and completed in: {hours:02}:{minutes:02}:{seconds:02}")
    return derivatives


reduced_functional.eval_cb_pre = record_pre_func
reduced_functional.eval_cb_post = record_post_func
reduced_functional.derivative_cb_pre = record_pre_grad
reduced_functional.derivative_cb_post = record_post_grad
reduced_functional.hessian_cb_pre = record_pre_hess
reduced_functional.hessian_cb_post = record_post_hess

# import datetime
# func_count = grad_count = hess_count = 0
# func_time = grad_time = hess_time = 0.0

# def timed_objective(tao, x):
#     global func_count, func_time
#     func_count += 1
#     start = datetime.datetime.now()

#     # The actual evaluation of the functional
#     val = reduced_functional(x)

#     func_time += (datetime.datetime.now() - start).total_seconds()
#     return val

# # Signature: timed_gradient(tao, x, g)
# def timed_gradient(tao, x, g):
#     global grad_count, grad_time
#     grad_count += 1
#     start = datetime.datetime.now()

#     # Calculate the gradient and fill the PETSc Vec 'g'
#     g[:] = reduced_functional.derivative(x)

#     grad_time += (datetime.datetime.now() - start).total_seconds()


# def timed_hessian(tao, x, H, Hpre):
#     global hess_count, hess_time
#     hess_count += 1
#     start = datetime.datetime.now()

#     reduced_functional.hessian(x, H)          # assemble PETSc Mat

#     hess_time += (datetime.datetime.now() - start).total_seconds()


# ==========================================
# TRUST REGION + L-BFGS (BOUNDED) CONFIG
# ==========================================
# solver = TAOSolver(minimisation_problem, {
#     "tao_type": "ntr",                   # Newton Trust Region; default: nls (line search Newton)
#     "tao_ntr_ksp_type": "stcg",
    
#     # --- Monitoring and Stopping ---
#     "tao_monitor": True,                 # Print iteration log; default: False
#     "tao_converged_reason": True,        # Print convergence reason; default: False
#     "tao_gatol": 1e-8,                 # Gradient absolute tolerance; default: 1e-8
#     "tao_grtol": 1e-8,                    # Relative gradient tolerance; default: 1e-8
#     "tao_gttol": 1e-12,                    # Step tolerance; default: 1e-12
#     "tao_max_it": 5,                   # Max iterations; default: 200

#     # --- Trust-Region Parameters ---
#     "tao_trust0": 1.0,                   # Initial trust-region radius; default: 1.0
#     "tao_trust_max": 100.0,              # Maximum trust-region radius; default: 1e8
#     "tao_eta": 0.1,                     # Step acceptance threshold; default: 0.1
#     "tao_alpha1": 0.25,                  # Shrink factor (mild rejection); default: 0.25
#     "tao_alpha2": 0.5,                   # Shrink factor (strong rejection); default: 0.5
#     "tao_gamma1": 1.5,                   # Expand factor (mild); default: 1.5
#     "tao_gamma2": 2.0,                   # Expand factor (moderate); default: 2.0
#     "tao_gamma3": 4.0,                   # Expand factor (aggressive); default: 4.0

#     # --- Hessian and Preconditioner (L-BFGS) ---
#     "tao_ntr_pc_type": "lmvm",           # Use LMVM (L-BFGS) as preconditioner; default: none
#     "tao_lmm_vectors": 10,               # Memory length for secant pairs; default: 5
#     "tao_lmm_scale_type": "scalar",      # Initial Hessian scaling; default: scalar
#     "tao_bfgs_rescale": True,            # Rescale secant pairs if curvature fails; default: True

#     # --- Linear Solver (Inner Iterations) ---
#     "tao_ntr_ksp_rtol": 1e-4,            # Relative tolerance for KSP; default: 1e-4
#     "tao_ntr_ksp_max_it": 50,           # Max KSP iterations; default: 50
#     "tao_ntr_pc_type": "lmvm",           # Preconditioner for KSP; default: none

#     # --- Bounds and Active Set Handling ---
#     "tao_bounded_type": "clip",          # Clip variables to bounds; default: 'clip'
#     "tao_bmvm_eps": 1e-8,                # Feasibility tolerance; default: 1e-8
#     "tao_blmvm_as_type": "grad",         # Active-set strategy; default: 'grad'

#     # --- Optional diagnostics ---
#     "tao_view": True,                   # Print solver setup; default: False
#     "tao_view_solution": True,          # Print solution vector; default: False
# })

# ==========================================
# BOUNDED L-BFGS (LINE SEARCH) SOLVER CONFIG
# ==========================================

# ================================================================
# Step 1: Set PETSc TAO options to enable internal text output
# ================================================================
# import petsc4py.PETSc as PETSc
from firedrake.petsc import PETSc
# PETSc.Options().setValue("tao_monitor", "")        # show f, ||g||, step
# PETSc.Options().setValue("tao_monitor_globalization", "")        # show f, ||g||, step
# PETSc.Options().setValue("tao_converged_reason", "")   # print convergence reason
# PETSc.Options().setValue("tao_view", "")               # print final TAO summary
# PETSc.Options().setValue("log_view", "")               # print final TAO summary

solver = TAOSolver(minimisation_problem, {
    # -------------------------------------------------------
    # Core Algorithm Selection
    # -------------------------------------------------------
    "tao_type": "lmvm",                         # Use LMVM (unconstrained L-BFGS); default: nls


    # -------------------------------------------------------
    # 1. Limited-Memory Quasi-Newton Matrix (LMVM)
    # -------------------------------------------------------
    "tao_lmm_vectors": 20,                        # Number of (s,y) correction pairs; default: 5
    "tao_lmm_scale_type": "scalar",              # Initial Hessian scaling ('none'|'scalar'|'diagonal'); default: 'scalar'
    "tao_lmm_limit_type": "none",                # Memory limiting strategy ('none'|'average'); default: 'none'
    "tao_lmm_bfgs": True,                        # Use BFGS updates (True) or DFP (False); default: True
    "tao_lmm_alpha": 1.0,                        # Damping factor for L-BFGS update; default: 1.0
    "tao_bfgs_rescale": True,                    # Rescale if curvature condition violated; default: True


    # -------------------------------------------------------
    # 2. Globalization via Line Search
    # -------------------------------------------------------
    "tao_ls_type": "more-thuente",               # Line search ('more-thuente'|'armijo'|'gpcg'); default: 'more-thuente'
    "tao_ls_ftol": 1e-4,                         # Armijo sufficient decrease; default: 1e-4
    "tao_ls_gtol": 0.9,                          # Curvature (Wolfe) condition; default: 0.9
    "tao_ls_stepmin": 1e-20,                     # Minimum step; default: 1e-20
    "tao_ls_stepmax": 1e10,                      # Maximum step; default: 1e10
    "tao_ls_rtol": 1e-10,                        # Relative line-search tolerance; default: 1e-10
    # "tao_ls_monitor": "",                      # Print line search progress; default: off


    # -------------------------------------------------------
    # 3. Stopping Criteria
    # -------------------------------------------------------
    "tao_gatol": 1e-8,                           # Absolute gradient tolerance; default: 1e-8
    "tao_grtol": 1e-8,                           # Relative gradient tolerance; default: 1e-8
    "tao_gttol": 1e-12,                          # Step-size tolerance; default: 1e-12


    # -------------------------------------------------------
    # 4. Iteration and Evaluation Limits
    # -------------------------------------------------------
    "tao_max_it": 200,                           # Maximum TAO iterations; default: 200
    "tao_max_funcs": 10000,                      # Maximum function evaluations; default: 10000


    # -------------------------------------------------------
    # 5. Initialization Options
    # -------------------------------------------------------
    # "tao_init_type": "given",                  # Start point type ('given'|'constant'|'random'); default: 'given'
    # "tao_zero_guess": True,                    # Force initial guess = 0; default: off


    # -------------------------------------------------------
    # 6. Monitoring, Output & Debugging
    # -------------------------------------------------------
    # "tao_monitor": "",                         # Print iteration summary; default: off
    # "tao_monitor_all": "",                    # Print all internal data each iteration; default: off
    # "tao_view": "",                            # Print solver options; default: off
    # "tao_view_solution": "",                   # Print final solution; default: off
    # "tao_view_gradient": "",                   # Print gradient at solution; default: off
    # "tao_view_hessian": "",                    # Print L-BFGS stats; default: off
    # "tao_converged_reason": "",                # Print convergence reason; default: off
    # "tao_history": "",                         # Record objective/grad history; default: off
    # "tao_draw_solution": "",                   # Draw solution each iteration; default: off
    # "tao_view_kkt": "",                        # KKT diagnostics; default: off
})


# def my_iteration_monitor(tao, its, f, gnorm, gnormTrue, step):
#     """
#     Called by TAO after each optimization iteration.

#     Args:
#         tao (PETSc.TAO): The TAO solver object.
#         its (int): The current iteration number.
#         f (float): The current objective function value.
#         gnorm (float): The 2-norm of the gradient (projected gradient for bounded).
#         gnormTrue (float): The true (unprojected) gradient 2-norm (often the same as gnorm).
#         step (float): The size of the last step taken.
#     """
    
    # --- Standard TAO Information ---
    # This prints the basic information TAO provides automatically via "tao_monitor"
    # log(f"TAO Iteration: {its:02d} | Functional: {f:.6e} | Gradient Norm: {gnorm:.3e} | Step Size: {step:.3e}")

    # # --- Custom Information (e.g., your profiling data) ---
    # # You can access your global counters here, just as you did in your PyROL StatusTest.
    # # Ensure no division by zero!
    # avg_func = elapsed_time_func / counter_func if counter_func > 0 else 0.0
    
    # # Only print on rank 0 to avoid redundant output
    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     log(f"  > Total Func Evals: {counter_func} (Avg Time: {avg_func:.4f}s)")
        
    #     # You could also add your misfit calculations here, as detailed in the previous response.
    #     # initial_misfit = assemble((Tic - Tic_ref) ** 2 * dx)
    #     # log(f"  > Misfit (IC): {initial_misfit:.6e}")
        
    # log("-" * 50)

solution_IC = Function(Tic.function_space(), name="Initial_Temperature")
solution_final = Function(T.function_space(), name="Final_Temperature")    
functional_values = []
initial_misfit_values = []
final_misfit_values = []

def monitor(tao):
    global counter_hess
    global counter_func
    global counter_grad
    initial_misfit = assemble(
        (Tic.block_variable.checkpoint - Tic_ref) ** 2 * dx
    )
    final_misfit = assemble(
        (T.block_variable.checkpoint - Tobs) ** 2 * dx
    )
    # Get the complete solution status tuple: 
    # (its, f, res, cnorm, step)
    try:
        status = tao.getSolutionStatus()
        its, f, res, cnorm, step, reason = status
        print(f"TAO Iteration: {its} | Functional: {f} | Function evaluations: {counter_func} | Gradient evaluations: {counter_grad} | Hessian evaluations: {counter_hess} | Gradient Norm: {res} | C-Norm: {cnorm} | Step Size: {step} | Initial Misfit: {initial_misfit} | Final Misfit: {final_misfit}")
        # Write functional and misfit values to a file (appending to avoid overwriting)
        if MPI.COMM_WORLD.Get_rank() == 0:        
            with open(functional_file, "a") as file:
                file.write(f"TAO Iteration: {its} | Functional: {f} | Function evaluations: {counter_func} | Gradient evaluations: {counter_grad} | Hessian evaluations: {counter_hess} | Gradient Norm: {res} | C-Norm: {cnorm} | Step Size: {step} | Initial Misfit: {initial_misfit} | Final Misfit: {final_misfit} \n")
        # Write VTK output:
        solution_IC.assign(Tic.block_variable.checkpoint)
        solution_final.assign(T.block_variable.checkpoint)        
        solutions_vtk.write(solution_IC, solution_final)

    except AttributeError:
        print("Attribute Error")
        # Fallback for older/different petsc4py versions 
        # where the monitor still might only receive (tao,)
        # its = tao.getIterationNumber()
        # f = tao.getObjectiveValue()
        # reason = tao.getConvergedReason
        # # The other values are not reliably accessible without 
        # # getSolutionStatus(), which is the key method for this.
        # res, cnorm, step = float('nan'), float('nan'), float('nan')

tao = solver.tao
tao.setMonitor(monitor)
# Explicitly view the TAO object properties
tao.view() # This should print the solver configuration and type!
T_opt = solver.solve()

# Get convergence
def tao_reason_to_text(reason_code):
    reason_map = {
        PETSc.TAO.ConvergedReason.CONVERGED_GATOL: "Converged: ||g|| ≤ gatol",
        PETSc.TAO.ConvergedReason.CONVERGED_GRTOL: "Converged: ||g||/f ≤ grtol",
        PETSc.TAO.ConvergedReason.CONVERGED_GTTOL: "Converged: trust region too small",
        PETSc.TAO.ConvergedReason.CONVERGED_STEPTOL: "Converged: step size small",
        PETSc.TAO.ConvergedReason.CONVERGED_MINF: "Converged: f ≤ f_min",
        PETSc.TAO.ConvergedReason.DIVERGED_MAXITS: "Diverged: maximum iterations reached",
        PETSc.TAO.ConvergedReason.DIVERGED_NAN: "Diverged: NaN encountered",
        PETSc.TAO.ConvergedReason.DIVERGED_MAXFCN: "Diverged: max function evals reached",
        PETSc.TAO.ConvergedReason.DIVERGED_LS_FAILURE: "Diverged: line search failure",
        PETSc.TAO.ConvergedReason.DIVERGED_TR_REDUCTION: "Diverged: trust region reduction",
        PETSc.TAO.ConvergedReason.DIVERGED_USER: "Diverged: user defined",
        PETSc.TAO.ConvergedReason.CONTINUE_ITERATING: "Still iterating",
    }
    return reason_map.get(reason_code, f"Unknown reason code {reason_code}")

reason_code = tao.getConvergedReason()
print(f"Converged Reason Code: {reason_code}, Converged Reason Text: {tao_reason_to_text(reason_code)}")
if MPI.COMM_WORLD.Get_rank() == 0:
    with open(functional_file, "a") as file:
        file.write(f"Converged Reason Code: {reason_code}, Converged Reason Text: {tao_reason_to_text(reason_code)}")

# Explicitly view the TAO object properties
tao.view() # This should print the solver configuration and type!
# -


