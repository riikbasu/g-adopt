#!/usr/bin/env python
# coding: utf-8
# %%
from gadopt import *
from gadopt.inverse import *
import csv
import sys
import subprocess
import os


# %%
def functional(functional_file, iteration_limit, initial_radius, radius_growing_rate, radius_shrinking_rate_negative_rho, radius_shrinking_rate_positive_rho, radius_shrinking_threshold, radius_growing_threshold):

    # Check if file exists
    if not os.path.isfile("adjoint-demo-checkpoint-state.h5"):
        # Download the file
        subprocess.run(["wget", "https://data.gadopt.org/demos/adjoint-demo-checkpoint-state.h5"], check=True)
    # Open the checkpoint file and subsequently load the mesh:
    checkpoint_filename = "adjoint-demo-checkpoint-state.h5"
    checkpoint_file = CheckpointFile(checkpoint_filename, mode="r")
    mesh = checkpoint_file.load_mesh("firedrake_default_extruded")
    mesh.cartesian = True
    
    # Specify boundary markers, noting that for extruded meshes the upper and lower boundaries are tagged as
    # "top" and "bottom" respectively.
    boundary = get_boundary_ids(mesh)
    
    # Retrieve the timestepping information for the Velocity and Temperature functions from checkpoint file:
    temperature_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Temperature")
    velocity_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Velocity")
    
    # Load the final state, analagous to the present-day "observed" state:
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][-1]))
    Tobs.rename("Observed Temperature")
    # Load the reference initial state - i.e. the state that we wish to recover:
    Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][0]))
    Tic_ref.rename("Reference Initial Temperature")
    checkpoint_file.close()
    
    # Write vtk for the observed and reference temperatures
    VTKFile("./visualisation_vtk.pvd").write(Tobs, Tic_ref)
    
    # Clear tape
    tape = get_working_tape()
    tape.clear_tape()
    
    # Set up function spaces:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
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
    
    # Set problem length
    initial_timestep = timesteps - 50
    
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
    
    # Initialise misfit
    u_misfit = 0.0
    
    # Next populate the tape by running the forward simulation.
    for time_idx in range(initial_timestep, timesteps):
        stokes_solver.solve()
        energy_solver.solve()
        # Update the accumulated surface velocity misfit using the observed value.
        with CheckpointFile(checkpoint_filename, mode="r") as checkpoint_file:
            uobs = checkpoint_file.load_function(mesh, name="Velocity", idx=time_idx)
        u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)
    
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
    alpha_u = 1e-1
    alpha_d = 1e-3
    alpha_s = 1e-3
    
    # Define overall objective functional:
    objective = (
        t_misfit +
        alpha_u * (norm_obs * u_misfit / timesteps / norm_u_surface) +
        alpha_d * (norm_obs * damping / norm_damping) +
        alpha_s * (norm_obs * smoothing / norm_smoothing)
    )
    
    # Define the reduced functional
    reduced_functional = ReducedFunctional(objective, control)
    
    # Pause writing to tape
    pause_annotation()
    # End of forward model
    
    #--------------------------------------------------------------------------------------------------------------------------------
    
    # Inversion
    
    # Define lower and upper bounds for the temperature
    T_lb = Function(Tic.function_space(), name="Lower Bound Temperature")
    T_ub = Function(Tic.function_space(), name="Upper Bound Temperature")
    
    # Assign the bounds
    T_lb.assign(0.0)
    T_ub.assign(1.0)
    
    # Define the minimisation problem, with the goal to minimise the reduced functional
    # Note: in some scenarios, the goal might be to maximise (rather than minimise) the functional.
    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
    
    # Minimisation Parameters            
    functional_file = functional_file
    minimisation_parameters["Status Test"]["Iteration Limit"] = int(iteration_limit)
    minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = float(initial_radius)
    minimisation_parameters["Step"]["Trust Region"]["Radius Growing Rate"] = float(radius_growing_rate)
    minimisation_parameters["Step"]["Trust Region"]["Radius Shrinking Rate (Negative rho)"] = float(radius_shrinking_rate_negative_rho)
    minimisation_parameters["Step"]["Trust Region"]["Radius Shrinking Rate (Positive rho)"] = float(radius_shrinking_rate_positive_rho)
    minimisation_parameters["Step"]["Trust Region"]["Radius Shrinking Threshold"] = float(radius_shrinking_threshold)
    minimisation_parameters["Step"]["Trust Region"]["Radius Growing Threshold"] = float(radius_growing_threshold)
    # Define the LinMore Optimiser class with checkpointing capability:
    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint",
    )
    functional_values = []
    def record_value(value, *args):
        functional_values.append(value)
    
    reduced_functional.eval_cb_post = record_value
    
    # If it existed, we could restore the optimisation from last checkpoint:
    # optimiser.restore()
    
    # Run the optimisation
    optimiser.run()
    
    # Write the functional values to a file
    with open(functional_file, "w") as f:
        f.write("\n".join(str(x) for x in functional_values))
    
    # # Write the final solution
    # VTKFile("./solution_Set_2_"+str(parameter_set)+".pvd").write(optimiser.rol_solver.rolvector.dat[0])
    
    # # Store solutions after each iteration
    # solutions_vtk = VTKFile("solutions.pvd")
    # solution_container = Function(Tic.function_space(), name="Solutions")
    # functional_values = []
    
    # def callback():
    #     solution_container.assign(Tic.block_variable.checkpoint)
    #     solutions_vtk.write(solution_container)
    #     final_temperature_misfit = assemble(
    #         (T.block_variable.checkpoint - Tobs) ** 2 * dx
    #     )
    #     log(f"Terminal Temperature Misfit: {final_temperature_misfit}")
    
    # optimiser.add_callback(callback)


# %%
def main():
    functional_file = sys.argv[1]
    iteration_limit = sys.argv[2]
    initial_radius = sys.argv[3]
    radius_growing_rate = sys.argv[4]
    radius_shrinking_rate_negative_rho = sys.argv[5]
    radius_shrinking_rate_positive_rho = sys.argv[6]
    radius_shrinking_threshold = sys.argv[7]
    radius_growing_threshold = sys.argv[8]
    functional(functional_file, iteration_limit, initial_radius, radius_growing_rate, radius_shrinking_rate_negative_rho, radius_shrinking_rate_positive_rho, radius_shrinking_threshold, radius_growing_threshold)


# %%
if __name__ == '__main__':
    main()


# %%




