# Computing dynamic topography from a pygplates simulation
from gadopt import *
from pathlib import Path

def __main__():
    # Getting all the filenames
    all_filenames = get_all_files()

    # Construct a CubedSphere mesh and then extrude into a sphere (or load from checkpoint):
    with CheckpointFile(str(all_filenames[0]), mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False

    # Boundary ids
    bottom_id, top_id = "bottom", "top"

    # Set up function spaces - currently using the bilinear Q2Q1 element pair for Stokes and DQ2 T:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "DQ", 2)  # Temperature function space (DG scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    z = Function(Z, name="Stokes")
    T = Function(Q, name="Temperature")
    surface_force = Function(W, name="DyanamicTopography")

    # Test functions and functions to hold solutions:
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

    # We next specify the important constants for this problem, including compressibility parameters,
    # and set up the TALA approximation.
    Ra = Constant(1.24e8)  # Rayleigh number
    Di = Constant(0.9492824165791792)  # Dissipation number
    H_int = Constant(9.93)  # Internal heating
    kappa = Constant(3.0)  # Thermal conductivity = yields a diffusivity of 7.5e-7 at surface.

    # getting TALA parameters for the approximation
    tala_fields = get_TALA_fields(Q)

    # We next prepare our viscosity, starting with a radial profile.
    mu_rad = Function(Q, name="Viscosity_Radial")  # Depth dependent component of viscosity
    interpolate_1d_profile(function=mu_rad, one_d_filename="initial_condition_mat_prop/mu2_radial.txt")

    # Full temperature including the adiabatic effects
    FullT = Function(Q, name="FullTemperature")

    # Temperature average field
    T_avg = Function(Q, name='Layer_Averaged_Temp')
    T_dev = Function(Q, name='Deviatoric_Temperature')

    mu_field = Function(Q, name="Viscosity")

    # Now that we have the average T profile, we add lateral viscosity variation due to temperature variations:
    # For T_dev[-0.5,0.5], this leads to a viscosity range of [7.07106781e+01,1.41421356e-02]; 1000[30,0.3]
    delta_mu_T = Constant(10000.)
    mu = mu_rad * exp(-ln(delta_mu_T) * T_dev)

    # These fields are used to set up our Truncated Anelastic Liquid Approximation.
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra,
        Di,
        rho=tala_fields["rhobar"],
        Tbar=tala_fields["Tbar"],
        alpha=tala_fields["alphabar"],
        cp=tala_fields["cpbar"],
        g=tala_fields["gbar"],
        H=H_int,
        mu=mu,
        kappa=kappa
    )

    # Nullspaces and near-nullspace objects are next set up,
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

    # Followed by boundary conditions for velocity and temperature.
    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'un': 0},
    }

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                                 constant_jacobian=False,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)
    stokes_solver.solver_parameters['snes_rtol'] = 1e-2
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

    averager = LayerAveraging(mesh, quad_degree=6)

    vtk_output = VTKFile("cao_etal.pvd")

    for fname in all_filenames:
        # TODO: Load temperature field
        with CheckpointFile(str(fname), mode="r") as f:
            T.interpolate(f.load_function(mesh, name="Temperature"))
            z.interpolate(f.load_function(mesh, name="Stokes"))

        # Solve Stokes sytem:
        stokes_solver.solve()

        # Calculate Full T and update gradient fields:
        FullT.assign(T + tala_fields["Tbar"])

        # Compute deviation from layer average
        averager.extrapolate_layer_average(T_avg, averager.get_layer_average(FullT))
        T_dev.assign(FullT - T_avg)

        # update mu field
        mu_field.interpolate(mu)

        # Compute diagnostics:
        surface_force.interpolate(stokes_solver.force_on_boundary(subdomain_id=top_id))

        vtk_output.write(u, p, T_dev, surface_force)


def get_TALA_fields(Q):
    # Truncated Anelastic liquid approximation fields
    rhobar = Function(Q, name="CompRefDensity")
    interpolate_1d_profile(function=rhobar, one_d_filename="initial_condition_mat_prop/rhobar.txt")
    rhobar.assign(rhobar / 3200.)

    Tbar = Function(Q, name="CompRefTemperature")
    interpolate_1d_profile(function=Tbar, one_d_filename="initial_condition_mat_prop/Tbar.txt")
    Tbar.assign((Tbar - 1600.) / 3700.)

    alphabar = Function(Q, name="IsobaricThermalExpansivity")
    interpolate_1d_profile(function=alphabar, one_d_filename="initial_condition_mat_prop/alphabar.txt")
    alphabar.assign(alphabar / 4.1773e-05)

    cpbar = Function(Q, name="IsobaricSpecificHeatCapacity")
    interpolate_1d_profile(function=cpbar, one_d_filename="initial_condition_mat_prop/CpSIbar.txt")
    cpbar.assign(cpbar / 1249.7)

    gbar = Function(Q, name="GravitationalAcceleration")
    interpolate_1d_profile(function=gbar, one_d_filename="initial_condition_mat_prop/gbar.txt")
    gbar.assign(gbar / 9.8267)

    return {
        "rhobar": rhobar,
        "Tbar": Tbar,
        "alphabar": alphabar,
        "cpbar": cpbar,
        "gbar": gbar,
    }


def get_all_files():
    main_path = Path("/g/data/xd2/rad552/FIREDRAKE_Simulations/DG_GPlates_Late_2024/GPlates_2e8/Cao/")
    all_files = sorted(main_path.glob("*/Final_State.h5"), key=lambda x: int(x.parent.name.replace("C", "")))
    return all_files 
