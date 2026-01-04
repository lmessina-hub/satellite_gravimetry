""" Noise Generator Module """

from scipy.stats import qmc, norm
from helpers import rtn_basis
import numpy as np
from matplotlib import pyplot as plt


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
    

    def generate_kbr_range_noise(
        num_epochs: int,
        sigma_rho: float,        
        seed: int,
    ) -> tuple[np.ndarray]:
        """ Generate KBR range measurement noise using Sobol sequences and Gaussian distribution. """

        exponent_num_samples = 10
        eps = np.finfo(np.float64).eps
        range_error  = np.empty((num_epochs, 2**exponent_num_samples))                  

        for k in range(num_epochs):

            sampler = qmc.Sobol(d=1, scramble=True, rng=seed)
            sobol_samples = sampler.random_base2(m=exponent_num_samples)               
            sobol_samples = np.clip(sobol_samples, eps, 1.0 - eps)
            normal_distribution_samples = norm.ppf(sobol_samples).reshape(-1)               

            range_error_samples = normal_distribution_samples * sigma_rho                           
            range_error[k, :]  = range_error_samples

        return range_error

        

