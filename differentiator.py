import numpy as np
from findiff import Diff

def compute_acceleration(
        position: np.ndarray,
        time: np.ndarray,
        accuracy: int
        ) -> tuple[np.ndarray, dict]:
    """
    Compute the acceleration by double differentiating the position time history.

    :param position: Array of position vectors at different time steps (shape: (N, 3)).
    :param time_: Array of time values corresponding to the position measurements (shape: (N,)).
    :param accuracy: Order of accuracy for the finite difference approximation (2, 4, 6 ...).
    :return: Array of acceleration vectors (shape: (N, 3)).
    """

    num_epochs = position.shape[0]
    acceleration = np.zeros_like(position)
    metadata = {}
    
    # second derivative operator along time axis
    second_derivative_operator = Diff(0, time, acc=accuracy)**2

    # Compute acceleration for each spatial component
    for i in range(3):  
        acceleration[:, i] = second_derivative_operator(position[:, i])

    # Capture stencils used at each k (1D shape only)
    stencil_points = second_derivative_operator.stencil((num_epochs,))

    metadata['stencil_points'] = stencil_points

    return acceleration, metadata