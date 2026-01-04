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