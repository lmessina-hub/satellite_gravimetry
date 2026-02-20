""" Noise Generator Module """

from typing import List
from scipy.stats import qmc, norm
from helpers import rtn_basis
from pathlib import Path
from plotter import Plotter

import json
import numpy as np
from tudatpy import math 
from pycbc import types, noise, psd
from scipy.spatial.transform import Rotation

class NoiseGenerator:
    """Class to generate different types of noise for satellite measurements. """

    @staticmethod
    def generate_gps_position_noise(state_vector: np.ndarray, sigma_rtn: np.ndarray, seed: int) -> np.ndarray:
        """ Generate GPS position measurement noise in RTN frame using Sobol sequences and Gaussian distribution."""

        num_epochs = state_vector.shape[0]
        exponent_num_samples = 10

        eci_position_errors = np.empty((num_epochs, 2**exponent_num_samples, 3))
        rtn_position_errors = np.empty((num_epochs, 2**exponent_num_samples, 3))

        eps = np.finfo(np.float64).eps

        for k in range(num_epochs):
            r = state_vector[k, 0:3]
            v = state_vector[k, 3:6]
            C_rtn_eci = rtn_basis(r, v)

            # Generate Sobol samples
            sampler = qmc.Sobol(d=3, scramble=True, rng=seed)
            sobol_samples = sampler.random_base2(m=exponent_num_samples)
            sobol_samples = np.clip(sobol_samples, eps, 1.0 - eps)

            # Transform to normal distribution
            normal_distribution_samples = norm.ppf(sobol_samples)

            # Scale by standard deviation
            position_errors_rtn = normal_distribution_samples[:, 0:3] * sigma_rtn

            # Transform errors from RTN to ECI frame
            position_errors_eci = position_errors_rtn @ C_rtn_eci.T

            eci_position_errors[k, :, :] = position_errors_eci
            rtn_position_errors[k, :, :] = position_errors_rtn

        return eci_position_errors, rtn_position_errors
    
    def generate_pointing_angles_noise(
        plotter: Plotter,
        pitch_history_json_path: Path,
        yaw_history_json_path: Path,
        roll_history_json_path: Path,
        num_epochs: int,
        satellite_label: str,
        seed: int,
        white_noise_asd_values: dict[str, float],
        bias_noise_values: dict[str, float],
        ) -> dict[str, types.TimeSeries]:
        """ Generate pointing angles noise based on ASD data from JSON files. """

        json_paths =[pitch_history_json_path, yaw_history_json_path, roll_history_json_path]
        file_prefixes = ['pitch', 'yaw', 'roll']
        error_free_pointing_angles_time_series = {}
        noisy_attitude_time_series = {}

        # Load the ASD data from the uploaded JSON file
        for path, file_prefix in zip(json_paths, file_prefixes):
            with open(path, 'r') as file:
                asd_data = json.load(file)

            frequencies = np.array([float(entry['x']) for entry in asd_data])
            asd_values = np.array([float(entry['y']) for entry in asd_data])

            # Create a dictionary for the interpolator
            data_to_interpolate = dict(zip(frequencies, asd_values))

            # Set the interpolator settings (linear interpolation)
            interpolator_settings = math.interpolators.linear_interpolation()
            interpolator = math.interpolators.create_one_dimensional_scalar_interpolator(data_to_interpolate, interpolator_settings)

            # Create a regular frequency span and interpolate ASD values
            time_step = 5.0  # seconds
            num_samples = num_epochs 
            delta_f = 1.0 / (num_samples * time_step)
            frequencies_uniform_span = np.arange(frequencies.min(), frequencies.max(), delta_f)
            asd_interpolated = np.array([interpolator.interpolate(freq) for freq in frequencies_uniform_span])

            # Plot original vs interpolated data
            plotter.plot_linear_interpolation_comparison(
                frequencies,
                asd_values,
                frequencies_uniform_span,
                asd_interpolated,
                file_name=f"{satellite_label}_{file_prefix}_asd_interpolation_comparison.png"
            )

            # Convert ASD to PSD
            psd_interpolated = types.frequencyseries.FrequencySeries(asd_interpolated**2, delta_f)

            # Generate noise using the PSD, sample rate of 5 seconds for a time span of 31 days
            noise_time_series = noise.gaussian.noise_from_psd(num_samples, time_step, psd_interpolated, seed)

            error_free_pointing_angles_time_series[file_prefix] = noise_time_series

            # Print noise time series and basic stats
            plotter.plot_angle_noise_time_series(
                noise_time_series,
                file_name=f"{satellite_label}_{file_prefix}_pointing_angle_noise_time_series.png",
            )

            # Estimate PSD of time series via Welch

            segment_len = int(num_samples / 31)

            # 50% overlap
            seg_stride = segment_len // 2

            estimated_psd = psd.welch(noise_time_series, seg_len=segment_len, seg_stride=seg_stride)

            # Extract frequency and PSD values from estimated PSD
            estimated_frequencies = estimated_psd.sample_frequencies.numpy()
            estimated_psd_values = estimated_psd.numpy()

            input_frequencies = psd_interpolated.sample_frequencies.numpy()
            input_psd_values = psd_interpolated.numpy()

            plotter.plot_welch_estimated_psd_comparison(
                estimated_frequencies,
                estimated_psd_values,
                input_frequencies,
                input_psd_values,
                file_name=f"{satellite_label}_{file_prefix}_pointing_angle_welch_estimated_psd_comparison.png",
            )

            # Create a flat ASD over frequency span
            white_noise_asd = np.ones_like(frequencies_uniform_span) * white_noise_asd_values[file_prefix]

            # Convert to PSD
            white_noise_psd = types.frequencyseries.FrequencySeries(white_noise_asd**2, delta_f)

            white_noise_time_series = noise.gaussian.noise_from_psd(
                num_samples,
                time_step,
                white_noise_psd,
                seed + 10  # shift seed to avoid correlation
            )

            # Add bias noise
            bias_time_series = types.timeseries.TimeSeries(
                    np.full(num_samples, bias_noise_values[file_prefix], dtype=float),
                    delta_t=time_step
                )
            
            
            
            plotter.plot_angle_noise_time_series(
                noise_time_series + bias_time_series + white_noise_time_series,
                file_name=f"{satellite_label}_{file_prefix}_total_pointing_angle_noise_time_series.png",
            )
            
            noisy_attitude_time_series[file_prefix] = noise_time_series + bias_time_series + white_noise_time_series
            
        return error_free_pointing_angles_time_series, noisy_attitude_time_series
    
    def generate_kbr_system_and_oscillator_noise(
        plotter: Plotter,
        num_epochs: int,
        seed: int,
        )-> types.TimeSeries:
        """ Generate system and oscillator noise time series. """

        # Create a regular frequency span
        time_step = 5.0  # seconds
        delta_f = 1.0 / (num_epochs * time_step)
        frequency_interval = [delta_f, 1e-1 + delta_f]  # Hz
        frequencies_uniform_span = np.arange(frequency_interval[0], frequency_interval[1], delta_f)
        analytical_asd = 1e-6 * np.sqrt(1 + (0.0018 / frequencies_uniform_span)**4)  # [m Hz^-1/2]
        analytical_psd = (1e-6 * np.sqrt(1 + (0.0018 / frequencies_uniform_span)**4))**2  # [m^2 Hz^-1]


        # Plot original vs interpolated data
        plotter.plot_kbr_system_and_oscillator_asd(
            frequencies_uniform_span,
            analytical_asd,
            file_name=f"kbr_system_and_oscillator_asd.png"
        )

        # Convert ASD to PSD
        analytical_psd = types.frequencyseries.FrequencySeries(analytical_psd, delta_f)

        # Generate noise using the PSD, sample rate of 5 seconds for a time span of 31 days
        num_samples = num_epochs 
        noise_time_series = noise.gaussian.noise_from_psd(num_samples, time_step, analytical_psd, seed)

        # Print noise time series and basic stats
        plotter.plot_kbr_system_and_oscillator_noise_time_series(
            noise_time_series,
            file_name=f"kbr_system_and_oscillator_noise_time_series.png",
        )

        # Estimate PSD of time series via Welch
        segment_len = int(num_samples // 3)

        # 50% overlap
        seg_stride = segment_len // 2

        estimated_psd = psd.welch(noise_time_series, seg_len=segment_len, seg_stride=seg_stride)

        # Extract frequency and PSD values from estimated PSD
        estimated_frequencies = estimated_psd.sample_frequencies.numpy()
        estimated_psd_values = estimated_psd.numpy()

        input_frequencies = analytical_psd.sample_frequencies.numpy()
        input_psd_values = analytical_psd.numpy()

        plotter.plot_welch_estimated_psd_comparison(
            estimated_frequencies,
            estimated_psd_values,
            input_frequencies,
            input_psd_values,
            file_name=f"kbr_system_and_oscillator_welch_estimated_psd_comparison.png",
            ordinate_label=r"PSD [m$^2$ Hz$^{-1}$]",
            title="KBR System and Oscillator PSD"
        )
        
        return noise_time_series
    
    def generate_kbr_range_noise(
            angles_noise_timeseries: dict[str, types.TimeSeries],
            kbr_system_and_oscillator_noise_timeseries: types.TimeSeries,
            position_data: List[np.ndarray],
            antenna_phase_center_offset_vector_sf: dict[str, np.ndarray],
            bias_value: float,
            plotter: Plotter,
        ) -> types.TimeSeries:
        """ 
        Generate KBR range measurement noise using pointing angles noise time series,
        satellite position data, and systema and oscillator noise timeseries.
        """

        # ==================================================================
        # ANTENNA PHASE CENTRE POINTING JITTER COUPLING NOISE GENERATION 
        # ==================================================================

        # Compute rotation matrices from SF to LOSF and from LOSF to J2000 for both satellites
        rot_matrices_sf_to_losf = dict()
        rot_matrices_losf_to_j2000 = dict()
        # Compute rotation matrices from SF to J2000 for both satellites
        rot_matrices_sf_to_j2000 = dict()
        los_vectors_j2000 = dict()

        # Antenna phase center pointing jitter coupling noise
        apc_pointing_jitter_coupling_noise = dict()

        position_data = {
            "Grace-FO_A": position_data[0],
            "Grace-FO_B": position_data[1],
        }

        num_epochs = len(kbr_system_and_oscillator_noise_timeseries)

        for satellite in ["Grace-FO_A", "Grace-FO_B"]:

            roll_noise  = np.asarray(angles_noise_timeseries[satellite]["roll"], dtype=float)
            pitch_noise = np.asarray(angles_noise_timeseries[satellite]["pitch"], dtype=float)
            yaw_noise   = np.asarray(angles_noise_timeseries[satellite]["yaw"], dtype=float)

            pointing_angles = np.column_stack([yaw_noise, pitch_noise, roll_noise]) 
            rot_losf_to_sf = Rotation.from_euler('ZYX', pointing_angles, degrees=False)
            rot_sf_to_losf = rot_losf_to_sf.inv()
            rot_matrices_sf_to_losf[satellite] = rot_sf_to_losf.as_matrix()

            primary_position = position_data[satellite]
            secondary_position = position_data["Grace-FO_B" if satellite == "Grace-FO_A" else "Grace-FO_A"]

            # Compute LOSF to J2000 matrices for both satellites
            x_losf = (secondary_position - primary_position) / np.linalg.norm(secondary_position \
                                                                              - primary_position, axis=-1, keepdims=True)
            
            los_vectors_j2000[satellite] = x_losf
            
            y_losf = np.cross(x_losf, primary_position) / np.linalg.norm(np.cross(x_losf, primary_position), axis=-1, keepdims=True)
            
            z_losf = np.cross(x_losf, y_losf) / np.linalg.norm(np.cross(x_losf, y_losf), axis=-1, keepdims=True)
            rot_matrices_losf_to_j2000[satellite] = np.stack([x_losf, y_losf, z_losf], axis=-1)

            rot_matrices_sf_to_j2000[satellite] = rot_matrices_losf_to_j2000[satellite] @ rot_matrices_sf_to_losf[satellite]

            antenna_phase_center_offset_vector_j2000 = np.einsum("nij,j->ni", rot_matrices_sf_to_j2000[satellite],\
                                                                  antenna_phase_center_offset_vector_sf[satellite])
            
            apc_pointing_jitter_coupling_noise[satellite] = - np.einsum("ni,ni->n", los_vectors_j2000[satellite],\
                                                                        antenna_phase_center_offset_vector_j2000)
            
            plotter.plot_apc_pointing_jitter_coupling_time_series_demeaned(
                apc_pointing_jitter_coupling_noise=apc_pointing_jitter_coupling_noise,
                time_seconds=np.asarray(kbr_system_and_oscillator_noise_timeseries.sample_times, dtype=float),
                satellite_label=satellite,
                file_name=f"{satellite}_apc_pointing_jitter_coupling_noise_demeaned.png",
            )

        # =========================
        # TOTAL KBR RANGE NOISE 
        # =========================   

        bias = 0.0 * kbr_system_and_oscillator_noise_timeseries + bias_value

        # Final length checks
        if len(apc_pointing_jitter_coupling_noise["Grace-FO_A"]) != num_epochs\
              or len(apc_pointing_jitter_coupling_noise["Grace-FO_B"]) != num_epochs\
                or len(bias) != num_epochs:
            raise ValueError(
                f"TimeSeries length mismatch."
            )

        total_kbr_range_noise = (apc_pointing_jitter_coupling_noise["Grace-FO_A"] + \
                                apc_pointing_jitter_coupling_noise["Grace-FO_B"] + \
                                bias + \
                                np.asarray(kbr_system_and_oscillator_noise_timeseries, dtype=float))
        
        return total_kbr_range_noise
