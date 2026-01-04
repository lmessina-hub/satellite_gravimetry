# Load standard modules
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from orbit_simulator import OrbitalElements
from plotter import Plotter
from noise_generator import NoiseGenerator

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime

###################################################################
#
#                   PROPAGATOR & ENVIRONMENT SETUP
#
###################################################################

spice.load_standard_kernels()

# Create default body settings for "Earth"
bodies_to_create = ["Earth"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = dynamics.environment_setup.get_default_body_settings(
   bodies_to_create, global_frame_origin, global_frame_orientation)

body_settings.add_empty_settings("Grace-A")
body_settings.add_empty_settings("Grace-B")

# Create system of bodies
bodies = dynamics.environment_setup.create_system_of_bodies(body_settings)

# Define bodies that are propagated
bodies_to_propagate = ["Grace-A", "Grace-B"]

# Define central bodies of propagation
central_bodies = ["Earth", "Earth"]

# Define accelerations acting on the GRACE satellites
acceleration_settings_grace = dict(
   Earth=[dynamics.propagation_setup.acceleration.point_mass_gravity()]
)

acceleration_settings = {"Grace-A": acceleration_settings_grace, "Grace-B": acceleration_settings_grace}

# Create acceleration models
acceleration_models = dynamics.propagation_setup.create_acceleration_models(
   bodies, acceleration_settings, bodies_to_propagate, central_bodies)

simulation_start_epoch = DateTime(2003, 1, 1, 0, 0, 0).to_epoch()
simulation_end_epoch = DateTime(2003, 1, 1, 2, 0, 0).to_epoch()

# Create numerical integrator settings
integrator_settings = dynamics.propagation_setup.integrator.runge_kutta_fixed_step(
   time_step=100.0, coefficient_set=dynamics.propagation_setup.integrator.rk_4
)

propagator_type = dynamics.propagation_setup.propagator.cowell

# Create termination settings
termination_settings = dynamics.propagation_setup.propagator.time_termination(simulation_end_epoch)

earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter

###################################################################
#
#                       GRACE ORBIT SIMULATION
#
###################################################################

# GRACE B

grace_b_initial_orbital_elements = OrbitalElements(
                              a_km=6878.0,
                              e=0.01,
                              i_deg=89.0,
                              raan_deg=0.0,
                              argp_deg=0.0,
                              M_deg=0.0,
                           )

grace_b_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
   gravitational_parameter=earth_gravitational_parameter,
   semi_major_axis=grace_b_initial_orbital_elements.a_km * 1e3,
   eccentricity=grace_b_initial_orbital_elements.e,
   inclination=np.radians(grace_b_initial_orbital_elements.i_deg),
   longitude_of_ascending_node=np.radians(grace_b_initial_orbital_elements.raan_deg),
   argument_of_periapsis=np.radians(grace_b_initial_orbital_elements.argp_deg),
   true_anomaly=element_conversion.mean_to_true_anomaly(
                  mean_anomaly=np.radians(grace_b_initial_orbital_elements.M_deg),
                  eccentricity=grace_b_initial_orbital_elements.e,
               ),
)


# GRACE A

grace_a_initial_orbital_elements = grace_b_initial_orbital_elements.get_along_track_shift(
    separation_km=220.0
)
grace_a_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
   gravitational_parameter=earth_gravitational_parameter,
   semi_major_axis=grace_a_initial_orbital_elements.a_km * 1e3,
   eccentricity=grace_a_initial_orbital_elements.e,
   inclination=np.radians(grace_a_initial_orbital_elements.i_deg),
   longitude_of_ascending_node=np.radians(grace_a_initial_orbital_elements.raan_deg),
   argument_of_periapsis=np.radians(grace_a_initial_orbital_elements.argp_deg),
   true_anomaly=element_conversion.mean_to_true_anomaly(
                  mean_anomaly=np.radians(grace_a_initial_orbital_elements.M_deg),
                  eccentricity=grace_a_initial_orbital_elements.e,
              ),
   )


initial_states = np.hstack((grace_a_initial_state, grace_b_initial_state))

# Create propagation settings
propagator_settings = dynamics.propagation_setup.propagator.translational(
   central_bodies,
   acceleration_models,
   bodies_to_propagate,
   initial_states,
   simulation_start_epoch,
   integrator_settings,
   termination_settings,
   propagator=propagator_type,
)

# Create simulation object and propagate the dynamics
dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
   bodies, propagator_settings
)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)


# =====================================
# PLOTTING GRACE RELATED DATA
# =====================================

plotter = Plotter(output_path=Path("./GRACE/plots"))


time_data = states_array[:, 0]
grace_position_data = [
    states_array[:, 1:4],   # GRACE-A position
    states_array[:, 7:10],  # GRACE-B position
]
grace_velocity_data = [
    states_array[:, 4:7],    # GRACE-A velocity
    states_array[:, 10:13],  # GRACE-B velocity
]

plotter.plot_orbits(
    no_satellites=2,
    position_data=grace_position_data,
    title="GRACE-A and GRACE-B orbits",
    sat_labels=["GRACE-A", "GRACE-B"],
    file_name="grace_nominal_orbits.png"
)

plotter.plot_relative_position(
   time_data=time_data,
   position_data=grace_position_data,
   velocity_data=grace_velocity_data,
   title="RTN relative position components - GRACE",
   file_name="grace_rtn_relative_position.png"
)

# =====================================
# GPS POSITION MEASUREMENT SIMULATION 
# =====================================

sigma_gps_position_rtn = np.array([0.01, 0.02, 0.02])  # [R, T, N] in meters

eci_gps_position_noise_grace_a, rtn_gps_position_noise_grace_a = NoiseGenerator.generate_gps_position_noise(
    state_vector=states_array[:, 1:7],  # GRACE-A state
    sigma_rtn=sigma_gps_position_rtn,
    seed=40
)

eci_gps_position_noise_grace_b, rtn_gps_position_noise_grace_b = NoiseGenerator.generate_gps_position_noise(
    state_vector=states_array[:, 7:13],  # GRACE-B state
    sigma_rtn=sigma_gps_position_rtn,
    seed=41
)

Plotter.plot_rtn_error_projections(
    plotter,
    samples_rtn=rtn_gps_position_noise_grace_a[0, :, :],  
    sigma_rtn=sigma_gps_position_rtn,
    epoch_idx=10,
    file_name="grace_a_gps_noise_rtn_projections_epoch_10.png"
)

Plotter.plot_rtn_error_projections(
    plotter,
    samples_rtn=rtn_gps_position_noise_grace_b[0, :, :],  
    sigma_rtn=sigma_gps_position_rtn,
    epoch_idx=10,
    file_name="grace_b_gps_noise_rtn_projections_epoch_10.png"
)


# =====================================
# KBR RANGE MEASUREMENT SIMULATION 
# =====================================

sigma_rho = 1e-6  # KBR range noise standard deviation in meters
num_epochs = states_array.shape[0]

kbr_range_noise = NoiseGenerator.generate_kbr_range_noise(
    num_epochs=num_epochs,
    sigma_rho=sigma_rho,
    seed=42)

Plotter.plot_kbr_range_noise_histogram_and_distribution(
    plotter,
    range_error_samples=kbr_range_noise[10, :],
    sigma_rho=sigma_rho,
    epoch_idx=10,
    file_name="kbr_range_noise_epoch_10.png",
)