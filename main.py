# NOTE: Load warning module and filter out specific deprecation warnings
# to avoid cluttering log outpuyt with irrelevant messages.
import warnings

from tudatpy.dynamics import propagation_setup
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

# Load standard modules
from pathlib import Path
import numpy as np
from orbit_simulator import OrbitalElements
from plotter import Plotter
from noise_generator import NoiseGenerator
from environment_customizer import EnvironmentCustomizer

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.astro import element_conversion
from tudatpy.astro import gravitation
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics.propagation_setup import dependent_variable


simulation_start_epoch = DateTime(2005, 3, 1, 0, 0, 0).to_epoch()
simulation_end_epoch = DateTime(2005, 3, 2, 0, 0, 0).to_epoch() 
time_step = 5.0  # seconds
number_epochs = int(np.floor((simulation_end_epoch - simulation_start_epoch) / time_step)) + 1

# =====================================
# ASD NOISE GENERATION 
# =====================================

plotter = Plotter(output_path=Path("./GRACE-FO/plots"))

pitch_history_json_path=Path("data/pitch_angles_asd_data.json")
yaw_history_json_path=Path("data/yaw_angles_asd_data.json")
roll_history_json_path=Path("data/roll_angles_asd_data.json")


Plotter.plot_pointing_angles_asd(
   plotter,
   file_name="pointing_angles_asd.png",
   pitch_history_json_path=pitch_history_json_path,
   yaw_history_json_path=yaw_history_json_path,
   roll_history_json_path=roll_history_json_path,
)

error_free_pointing_angles_time_series = dict()
noisy_attitude_time_series = dict()

white_noise_asd_values = {
    "roll": 20e-6,
    "pitch": 20e-6,
    "yaw": 20e-6,
}

bias_noise_values = {
    "roll": 2e-3,
    "pitch": 2e-3,
    "yaw": 2e-3,
}

for satellite, seed in zip(["Grace-FO_A", "Grace-FO_B"], [42, 43]):
    
   # Generate pointing angles noise time series for each satellite

    error_free_pointing_angles_time_series[satellite], noisy_attitude_time_series[satellite] = NoiseGenerator.generate_pointing_angles_noise(
        plotter,
        pitch_history_json_path,
        yaw_history_json_path,
        roll_history_json_path,
        number_epochs,
        satellite_label=satellite,
        seed=seed,
        white_noise_asd_values=white_noise_asd_values,
        bias_noise_values=bias_noise_values
    )

rotation_model_context: dict[str, object] = {"bodies": None}
attitude_noise_sample_times = simulation_start_epoch + np.arange(number_epochs, dtype=float) * time_step

grace_fo_a_custom_rotation_matrix_callable = EnvironmentCustomizer.create_custom_spacecraft_rotation_function(
    spacecraft_name="Grace-FO A",
    counterpart_name="Grace-FO B",
    attitude_noise_history=noisy_attitude_time_series["Grace-FO_A"],
    sample_times=attitude_noise_sample_times,
    rotation_model_context=rotation_model_context,
)

grace_fo_b_custom_rotation_matrix_callable = EnvironmentCustomizer.create_custom_spacecraft_rotation_function(
    spacecraft_name="Grace-FO B",
    counterpart_name="Grace-FO A",
    attitude_noise_history=noisy_attitude_time_series["Grace-FO_B"],
    sample_times=attitude_noise_sample_times,
    rotation_model_context=rotation_model_context,
)


###################################################################
#
#                   PROPAGATOR & ENVIRONMENT SETUP
#
###################################################################

spice.load_standard_kernels()
spice.load_kernel("./kernels/nep105.bsp")
spice.load_kernel("./kernels/plu060.bsp")
spice.load_kernel("./kernels/ura182.bsp")
spice.load_kernel("./kernels/gm_de440.pck")

# Vehicle configuration parameters
# TODO: Improve the drag and solar radiation pressure modelling to match GRACE-FO specifications.
# Ref: https://isdc-data.gfz.de/grace-fo/DOCUMENTS/Level-1/GRACE-FO_L1_Data_Product_User_Handbook_20190911.pdf
# Ref: https://agupubs-onlinelibrary-wiley-com.tudelft.idm.oclc.org/doi/epdf/10.1029/2020JB021297
drag_coefficient = 2.4
# Ref: https://essd.copernicus.org/articles/9/833/2017/
grace_fo_mass = 655.0   # [kg]

grace_fo_a_per_source_occulting_bodies = {
   "Sun": ["Earth", "Moon"],
   }

grace_fo_b_per_source_occulting_bodies = {
   "Sun": ["Earth", "Moon"],
   }

# Gravitational field parameters for the Sun
# Ref: https://iopscience.iop.org/article/10.3847/1538-4357/aca8a4/pdf
J2_sun = 2.07e-7
unormalized_cosine_coefficients_sun = np.zeros((3, 3))
unormalized_sine_coefficients_sun = np.zeros((3, 3))
unormalized_cosine_coefficients_sun[0, 0] = 1.0
unormalized_cosine_coefficients_sun[2, 0] = -J2_sun
# Normalize the Sun's gravitational field coefficients
normalized_cosine_coefficients_sun, normalized_sine_coefficients_sun = gravitation.normalize_spherical_harmonic_coefficients(
   unormalized_cosine_coefficients_sun,
   unormalized_sine_coefficients_sun,
)
sun_gravitational_parameter = spice.get_body_gravitational_parameter("Sun")
sun_astrometric_mean_radius = 695508000  # [m], astrometric (mean) radius of the Sun

# Create body settings 
bodies_to_create = [
               "Earth",
               "Sun",
               "Mercury",
               "Venus",
               "Mars",
               "Moon",
               "Jupiter",
               "Io",
               "Europa",
               "Ganymede",
               "Callisto",
               "Amalthea",
               "Saturn",
               "Titan",
               "Rhea",
               "Iapetus",
               "Dione",
               "Tethys",
               "Enceladus",
               "Hyperion",
               "Mimas",
               "Uranus",
               "Neptune",
               "Pluto",
               "Ceres",
               "Vesta",
               "Phobos",
               "Deimos",
               ]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = dynamics.environment_setup.get_default_body_settings(
   bodies_to_create, global_frame_origin, global_frame_orientation)

body_settings.get("Sun").gravity_field_settings = dynamics.environment_setup.gravity_field.spherical_harmonic(
      gravitational_parameter = sun_gravitational_parameter,
      reference_radius = sun_astrometric_mean_radius,
      normalized_cosine_coefficients = normalized_cosine_coefficients_sun,
      normalized_sine_coefficients = normalized_sine_coefficients_sun,
      associated_reference_frame = "IAU_Sun"
)

body_settings.add_empty_settings("Grace-FO A")
body_settings.add_empty_settings("Grace-FO B")

body_settings.get("Grace-FO A").rotation_model_settings = dynamics.environment_setup.rotation_model.custom_rotation_model(
    base_frame="J2000",
    target_frame="Grace-FO_A_SF",
    custom_rotation_matrix_function=grace_fo_a_custom_rotation_matrix_callable,
    finite_difference_time_step=time_step,
)

body_settings.get("Grace-FO B").rotation_model_settings = dynamics.environment_setup.rotation_model.custom_rotation_model(
    base_frame="J2000",
    target_frame="Grace-FO_B_SF",
    custom_rotation_matrix_function=grace_fo_b_custom_rotation_matrix_callable,
    finite_difference_time_step=time_step,
)

# Loading the macromodel
grace_fo_material_properties = {
    "SiOx_Kapton": dynamics.environment_setup.vehicle_systems.material_properties(
        specular_reflectivity=0.40,
        diffuse_reflectivity=0.26,
    ),
    "Si_Glass": dynamics.environment_setup.vehicle_systems.material_properties(
        specular_reflectivity=0.03,
        diffuse_reflectivity=0.07,
    ),
    "Teflon": dynamics.environment_setup.vehicle_systems.material_properties(
        specular_reflectivity=0.68,
        diffuse_reflectivity=0.20,
    ),
    }   

grace_fo_reradiation_settings = {
    "SiOx_Kapton": True,
    "Si_Glass": True,
    "Teflon": True,
}

grace_fo_frame_origin = np.array([0.0, 0.0, 0.0])  # [m], origin of the spacecraft bus frame in the SF frame

grace_fo_bus_panels = dynamics.environment_setup.vehicle_systems.body_panel_settings_list_from_dae(
    file_path=str(Path("./data/grace_fo_low_fidelity.dae")),
    frame_origin=grace_fo_frame_origin,
    material_properties=grace_fo_material_properties,
    reradiation_settings=grace_fo_reradiation_settings,
)

grace_fo_full_panelled_body = dynamics.environment_setup.vehicle_systems.full_panelled_body_settings(
    grace_fo_bus_panels,
)

body_settings.get("Grace-FO A").vehicle_shape_settings = grace_fo_full_panelled_body
body_settings.get("Grace-FO B").vehicle_shape_settings = grace_fo_full_panelled_body

grace_fo_a_target_settings = dynamics.environment_setup.radiation_pressure.panelled_radiation_target(
    grace_fo_a_per_source_occulting_bodies
)

grace_fo_b_target_settings = dynamics.environment_setup.radiation_pressure.panelled_radiation_target(
    grace_fo_b_per_source_occulting_bodies
)

body_settings.get("Grace-FO A").radiation_pressure_target_settings = (
    grace_fo_a_target_settings
)

body_settings.get("Grace-FO B").radiation_pressure_target_settings = (
    grace_fo_b_target_settings
)

aero_coefficients_settings = dynamics.environment_setup.aerodynamic_coefficients.constant_variable_cross_section(
    [drag_coefficient, 0.0, 0.0])

# Add the aerodynamic interface to the body settings
body_settings.get("Grace-FO A").aerodynamic_coefficient_settings = aero_coefficients_settings
body_settings.get("Grace-FO B").aerodynamic_coefficient_settings = aero_coefficients_settings

# create atmosphere settings and add to body settings of body "Earth"
body_settings.get( "Earth" ).atmosphere_settings = dynamics.environment_setup.atmosphere.nrlmsise00()

# Group all Solar System bodies into a single effective barycentric mass
# and model their combined gravitational effect as one equivalent source.

# Martian System
mars_gravitational_parameter = spice.get_body_gravitational_parameter("Mars")
phobos_gravitational_parameter = spice.get_body_gravitational_parameter("Phobos")
deimos_gravitational_parameter = spice.get_body_gravitational_parameter("Deimos")
mars_system_gravitational_parameter = mars_gravitational_parameter + phobos_gravitational_parameter + deimos_gravitational_parameter
body_settings.get("Mars").gravity_field_settings = dynamics.environment_setup.gravity_field.central(
   gravitational_parameter = mars_system_gravitational_parameter
)
body_settings.get( "Mars" ).ephemeris_settings = dynamics.environment_setup.ephemeris.direct_spice(
            global_frame_origin, global_frame_orientation, "Mars Barycenter" )

# Jovian System
jupiter_gravitational_parameter = spice.get_body_gravitational_parameter("Jupiter")
io_gravitational_parameter = spice.get_body_gravitational_parameter("Io")
europa_gravitational_parameter = spice.get_body_gravitational_parameter("Europa")
ganymede_gravitational_parameter = spice.get_body_gravitational_parameter("Ganymede")
callisto_gravitational_parameter = spice.get_body_gravitational_parameter("Callisto")
amalthea_gravitational_parameter = spice.get_body_gravitational_parameter("Amalthea")
jupiter_system_gravitational_parameter = (jupiter_gravitational_parameter + io_gravitational_parameter +\
                                           europa_gravitational_parameter + ganymede_gravitational_parameter +\
                                               callisto_gravitational_parameter + amalthea_gravitational_parameter)
body_settings.get("Jupiter").gravity_field_settings = dynamics.environment_setup.gravity_field.central(
   gravitational_parameter = jupiter_system_gravitational_parameter
)
body_settings.get( "Jupiter" ).ephemeris_settings = dynamics.environment_setup.ephemeris.direct_spice(
            global_frame_origin, global_frame_orientation, "Jupiter Barycenter" )

# Saturnian System
saturn_gravitational_parameter = spice.get_body_gravitational_parameter("Saturn")
titan_gravitational_parameter = spice.get_body_gravitational_parameter("Titan")
rhea_gravitational_parameter = spice.get_body_gravitational_parameter("Rhea")
iapetus_gravitational_parameter = spice.get_body_gravitational_parameter("Iapetus")
dione_gravitational_parameter = spice.get_body_gravitational_parameter("Dione")
tethys_gravitational_parameter = spice.get_body_gravitational_parameter("Tethys")
enceladus_gravitational_parameter = spice.get_body_gravitational_parameter("Enceladus")
hyperion_gravitational_parameter = spice.get_body_gravitational_parameter("Hyperion")
mimas_gravitational_parameter = spice.get_body_gravitational_parameter("Mimas")
saturn_system_gravitational_parameter = (saturn_gravitational_parameter + titan_gravitational_parameter +\
                                          rhea_gravitational_parameter + iapetus_gravitational_parameter +\
                                               dione_gravitational_parameter + tethys_gravitational_parameter +\
                                                  enceladus_gravitational_parameter + hyperion_gravitational_parameter +\
                                                     mimas_gravitational_parameter)

body_settings.get("Saturn").gravity_field_settings = dynamics.environment_setup.gravity_field.central(
   gravitational_parameter = saturn_system_gravitational_parameter
)
body_settings.get( "Saturn" ).ephemeris_settings = dynamics.environment_setup.ephemeris.direct_spice(
            global_frame_origin, global_frame_orientation, "Saturn Barycenter" )

# Create system of bodies
bodies = dynamics.environment_setup.create_system_of_bodies(body_settings)
rotation_model_context["bodies"] = bodies

bodies.get("Grace-FO A").mass = grace_fo_mass
bodies.get("Grace-FO B").mass = grace_fo_mass

# Define bodies that are propagated
bodies_to_propagate = ["Grace-FO A", "Grace-FO B"]

# Define central bodies of propagation
central_bodies = ["Earth", "Earth"]

# Define relativistic correction settings for the Earth
# TODO: Activate Lense-Thirring acceleration.
# Add Angular momentum vector per unit mass (in global frame)
# that is to be used for the calculation of the Lense-Thirring acceleration.
# Select terms to be used
use_schwarzschild = True
use_lense_thirring = False
use_de_sitter = True

# Define accelerations acting on the GRACE-FO satellites
acceleration_settings_grace_fo = dict(
   Earth=[dynamics.propagation_setup.acceleration.relativistic_correction(
               use_schwarzschild,
               use_lense_thirring,
               use_de_sitter,
               de_sitter_central_body="Sun",
          ),
          dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(20, 20),  # Default Max Degree: 200, Max Order: 200
          dynamics.propagation_setup.acceleration.aerodynamic(),
          ],
   Sun=[dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
         dynamics.propagation_setup.acceleration.radiation_pressure(),
        ],
   Moon=[dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)],    # Default Max Degree: 200, Max Order: 200
   Mars=[dynamics.propagation_setup.acceleration.point_mass_gravity()],                # Default Max Degree: 120, Max Order: 120
   Venus=[dynamics.propagation_setup.acceleration.point_mass_gravity()],               # Default Max Degree: 180, Max Order: 180   
   Mercury=[dynamics.propagation_setup.acceleration.point_mass_gravity()],             # Default Max Degree: 160, Max Order: 160
   Jupiter=[dynamics.propagation_setup.acceleration.point_mass_gravity()],             # Zonal coefficients up to degree 8
   Saturn=[dynamics.propagation_setup.acceleration.point_mass_gravity()],                           
   Uranus=[dynamics.propagation_setup.acceleration.point_mass_gravity()],
   Neptune=[dynamics.propagation_setup.acceleration.point_mass_gravity()],
   Ceres=[dynamics.propagation_setup.acceleration.point_mass_gravity()],
   Vesta=[dynamics.propagation_setup.acceleration.point_mass_gravity()],
)

acceleration_settings = {"Grace-FO A": acceleration_settings_grace_fo, "Grace-FO B": acceleration_settings_grace_fo}

# Create acceleration models
acceleration_models = dynamics.propagation_setup.create_acceleration_models(
   bodies, acceleration_settings, bodies_to_propagate, central_bodies)

# Create numerical integrator settings
integrator_settings = dynamics.propagation_setup.integrator.runge_kutta_fixed_step(
   time_step=5.0, coefficient_set=dynamics.propagation_setup.integrator.rkf_1412
)

propagator_type = dynamics.propagation_setup.propagator.cowell

# Create termination settings
termination_settings = dynamics.propagation_setup.propagator.time_termination(simulation_end_epoch)

earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter

###################################################################
#
#                       GRACE-FO ORBIT SIMULATION
#
###################################################################

# GRACE-FO B

grace_fo_b_initial_altitude_km = 477.7
earth_radius_km = bodies.get("Earth").shape_model.average_radius / 1e3
grace_fo_b_initial_orbit_semi_major_axis_km = earth_radius_km + grace_fo_b_initial_altitude_km

grace_fo_b_initial_orbital_elements = OrbitalElements(
                              a_km=grace_fo_b_initial_orbit_semi_major_axis_km,
                              e=0.0019,
                              i_deg=89.0081,
                              raan_deg=0.0,
                              argp_deg=0.0,
                              M_deg=0.0,
                           )

grace_fo_b_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
   gravitational_parameter=earth_gravitational_parameter,
   semi_major_axis=grace_fo_b_initial_orbital_elements.a_km * 1e3,
   eccentricity=grace_fo_b_initial_orbital_elements.e,
   inclination=np.radians(grace_fo_b_initial_orbital_elements.i_deg),
   longitude_of_ascending_node=np.radians(grace_fo_b_initial_orbital_elements.raan_deg),
   argument_of_periapsis=np.radians(grace_fo_b_initial_orbital_elements.argp_deg),
   true_anomaly=element_conversion.mean_to_true_anomaly(
                  mean_anomaly=np.radians(grace_fo_b_initial_orbital_elements.M_deg),
                  eccentricity=grace_fo_b_initial_orbital_elements.e,
               ),
)


# GRACE-FO A

grace_fo_a_initial_orbital_elements = grace_fo_b_initial_orbital_elements.get_along_track_shift(
    separation_km=238.0
)
grace_fo_a_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
   gravitational_parameter=earth_gravitational_parameter,
   semi_major_axis=grace_fo_a_initial_orbital_elements.a_km * 1e3,
   eccentricity=grace_fo_a_initial_orbital_elements.e,
   inclination=np.radians(grace_fo_a_initial_orbital_elements.i_deg),
   longitude_of_ascending_node=np.radians(grace_fo_a_initial_orbital_elements.raan_deg),
   argument_of_periapsis=np.radians(grace_fo_a_initial_orbital_elements.argp_deg),
   true_anomaly=element_conversion.mean_to_true_anomaly(
                  mean_anomaly=np.radians(grace_fo_a_initial_orbital_elements.M_deg),
                  eccentricity=grace_fo_a_initial_orbital_elements.e,
              ),
   )


initial_states = np.hstack((grace_fo_a_initial_state, grace_fo_b_initial_state))

dependent_variables_to_save = [
    dependent_variable.single_acceleration_norm(
        dynamics.propagation_setup.acceleration.radiation_pressure_type,  "Grace-FO A", "Sun",
    ),
    dependent_variable.single_acceleration_norm(
        dynamics.propagation_setup.acceleration.radiation_pressure_type,  "Grace-FO B", "Sun",
    ),
    dependent_variable.single_acceleration(
        dynamics.propagation_setup.acceleration.radiation_pressure_type,  "Grace-FO A", "Sun",
    ),
    dependent_variable.single_acceleration(
        dynamics.propagation_setup.acceleration.radiation_pressure_type,  "Grace-FO B", "Sun",
    ),
    dependent_variable.single_acceleration_norm(
        dynamics.propagation_setup.acceleration.aerodynamic_type, "Grace-FO A", "Earth"
    ),
    dependent_variable.single_acceleration_norm(
        dynamics.propagation_setup.acceleration.aerodynamic_type, "Grace-FO B", "Earth"
    ),
    dependent_variable.single_acceleration(
        dynamics.propagation_setup.acceleration.aerodynamic_type, "Grace-FO A", "Earth"
    ),
    dependent_variable.single_acceleration(
        dynamics.propagation_setup.acceleration.aerodynamic_type, "Grace-FO B", "Earth"
    ),
    ]

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
   output_variables=dependent_variables_to_save,
)

# Create simulation object and propagate the dynamics
dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
   bodies, propagator_settings
)

cpu_time_history = dynamics_simulator.cumulative_computation_time_history
total_cpu_time = list(cpu_time_history.values())[-1]
function_evaluation = dynamics_simulator.cumulative_number_of_function_evaluations
total_function_evaluations = list(function_evaluation.values())[-1]
print("\n=================================")
print(f"Propagation CPU time : ", total_cpu_time)
print(f"Number of function evaluations : ", total_function_evaluations)
print("=================================\n")


# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)
dependent_variables_history = dynamics_simulator.propagation_results.dependent_variable_history
dependent_variables_array = result2array(dependent_variables_history)


# =====================================
# PLOTTING GRACE-FO RELATED DATA
# =====================================

time_data = states_array[:, 0]
grace_fo_position_data = [
    states_array[:, 1:4],   # GRACE-FO A position
    states_array[:, 7:10],  # GRACE-FO B position
]
grace_fo_velocity_data = [
    states_array[:, 4:7],    # GRACE-FO A velocity
    states_array[:, 10:13],  # GRACE-FO B velocity
]

plotter.plot_orbits(
    no_satellites=2,
    position_data=grace_fo_position_data,
    title="GRACE-FO A and GRACE-FO B orbits",
    sat_labels=["GRACE-FO A", "GRACE-FO B"],
    file_name="grace_fo_nominal_orbits.png"
)

plotter.plot_relative_position(
   time_data=time_data,
   position_data=grace_fo_position_data,
   velocity_data=grace_fo_velocity_data,
   title="RTN relative position components - GRACE-FO",
   file_name="grace_fo_rtn_relative_position.png"
)

plotter.plot_srp_acceleration_time_series(
    dependent_variables_array=dependent_variables_array,
)

plotter.plot_aerodynamic_acceleration_time_series(
    dependent_variables_array=dependent_variables_array,
)

# =====================================
# GPS POSITION MEASUREMENT SIMULATION 
# =====================================

# sigma_gps_position_rtn = np.array([0.02, 0.02, 0.02])  # [R, T, N] in meters

# eci_gps_position_noise_grace_fo_a, rtn_gps_position_noise_grace_fo_a = NoiseGenerator.generate_gps_position_noise(
#     state_vector=states_array[:, 1:7],  # GRACE-FO A state
#     sigma_rtn=sigma_gps_position_rtn,
#     seed=40
# )

# eci_gps_position_noise_grace_fo_b, rtn_gps_position_noise_grace_fo_b = NoiseGenerator.generate_gps_position_noise(
#     state_vector=states_array[:, 7:13],  # GRACE-FO B state
#     sigma_rtn=sigma_gps_position_rtn,
#     seed=41
# )

# Plotter.plot_rtn_error_projections(
#     plotter,
#     samples_rtn=rtn_gps_position_noise_grace_fo_a[0, :, :],  
#     sigma_rtn=sigma_gps_position_rtn,
#     epoch_idx=10,
#     file_name="grace_fo_a_gps_noise_rtn_projections_epoch_10.png"
# )

# Plotter.plot_rtn_error_projections(
#     plotter,
#     samples_rtn=rtn_gps_position_noise_grace_fo_b[0, :, :],  
#     sigma_rtn=sigma_gps_position_rtn,
#     epoch_idx=20,
#     file_name="grace_fo_b_gps_noise_rtn_projections_epoch_20.png"
# )

# =====================================
# KBR RANGE MEASUREMENT SIMULATION 
# =====================================

# Generate KBR system and oscillator noise time series for each satellite
kbr_system_and_oscillator_noise_timeseries = NoiseGenerator.generate_kbr_system_and_oscillator_noise(
   plotter,
   time_data.shape[0],
   seed=42
)

antenna_phase_center_offset_vector_sf = {       
      "Grace-FO_A": np.array([1.4444, -170.0e-6, 448e-6]),  # [m]
      "Grace-FO_B": np.array([1.4445, 54e-6, 230e-6]),  # [m]
}

bias_value = 2e-2  # KBR range bias in meters

kbr_range_noise = NoiseGenerator.generate_kbr_range_noise(
    angles_noise_timeseries=error_free_pointing_angles_time_series,
    kbr_system_and_oscillator_noise_timeseries=kbr_system_and_oscillator_noise_timeseries,
    position_data=grace_fo_position_data,
    antenna_phase_center_offset_vector_sf=antenna_phase_center_offset_vector_sf,
    bias_value=bias_value,
    plotter=plotter,
    )
