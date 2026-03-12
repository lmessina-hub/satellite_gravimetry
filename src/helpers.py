import numpy as np

def wrap_deg(x: float) -> float:
    """Wrap angle to [0, 360)."""
    return x % 360.0

def wrap_rad(x: float) -> float:
    """Wrap angle to [0, 2pi)."""
    return x % (2.0 * np.pi)

def rtn_basis(r, v):
    Rhat = r / np.linalg.norm(r)
    h = np.cross(r, v); Nhat = h / np.linalg.norm(h)
    That = np.cross(Nhat, Rhat)
    return np.column_stack((Rhat, That, Nhat))

def transform_vector_history_inertial_to_satellite_frame(
    vectors_inertial: np.ndarray,
    rot_inertial_to_sf_flat: np.ndarray,
) -> np.ndarray:
    rot_inertial_to_sf = np.asarray(rot_inertial_to_sf_flat, dtype=float).reshape(-1, 3, 3)
    vectors_inertial = np.asarray(vectors_inertial, dtype=float).reshape(-1, 3)
    return np.einsum("nij,nj->ni", rot_inertial_to_sf, vectors_inertial)
