from dataclasses import dataclass
import math
from helpers import wrap_deg

@dataclass
class OrbitalElements:
    """Class to hold the classical orbital elements of a satellite orbit."""
    a_km: float                 # semi-major axis            [km]
    e: float                    # eccentricity               [-]
    i_deg: float                # inclination                [deg]
    raan_deg: float             # RAAN Ω                     [deg]
    argp_deg: float             # argument of periapsis ω    [deg]
    M0_deg: float               # mean anomaly at epoch      [deg]

    def get_along_track_shift(
        cls, 
        separation_km: float
        ) -> 'OrbitalElements':
        """Get the orbital elements of the targeter given a separation from the chaser.

        Args:
            separation_km (float): Along-track separation between chaser and targeter [km].
        """
        if cls.a_km <= 0:
            raise ValueError("Semi-major axis must be positive.")
        if cls.e > 1e-2:
            # Not forbidden, but warn via exception message to force explicit user intent.
            raise ValueError(
                f"e={cls.e} is not ~0. This workflow assumes circular orbit. "
            "For non-circular orbits, 'km along-track separation' is not a constant anomaly offset."
        )


        delta_M_rad = separation_km / cls.a_km
        delta_M_deg = math.degrees(delta_M_rad)

        return OrbitalElements(
                a_km=cls.a_km,
                e=cls.e,
                i_deg=cls.i_deg,
                raan_deg=cls.raan_deg,
                argp_deg=cls.argp_deg,
                M0_deg=wrap_deg(cls.M0_deg + delta_M_deg),
                )


