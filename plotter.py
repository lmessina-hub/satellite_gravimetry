from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from pathlib import Path

class Plotter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)

    def plot_orbits(self, no_satellites, position_data, title, sat_labels, file_name):
        """Plot the 3D orbits of the satellites."""

        fig = plt.figure(figsize=(8, 7), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)

        cmap = plt.get_cmap("tab10")
        linestyles = ["-", "--", "-.", ":"]

        for i in range(no_satellites):
            reference_color = cmap(i+3)
            starting_point_color = self._mix_with_white(reference_color, amount=0.35)  # lighter
            ending_point_color   = self._mix_with_black(reference_color, amount=0.25)  # darker
            ax.plot(position_data[i][:, 0], position_data[i][:, 1], position_data[i][:, 2], label=sat_labels[i],
                     linewidth=1.8, color=reference_color, linestyle=linestyles[i % len(linestyles)])
            ax.scatter(
                position_data[i][0, 0], position_data[i][0, 1], position_data[i][0, 2],
                marker="o", s=35, color=starting_point_color, edgecolor="k", linewidth=0.6,
                label=f"{sat_labels[i]} start"
            )
            ax.scatter(
                position_data[i][-1, 0], position_data[i][-1, 1], position_data[i][-1, 2],
                marker="x", s=45, color=ending_point_color, linewidth=1.4,
                label=f"{sat_labels[i]} end"
            )
        R_earth = 6378.1363e3  # [m] WGS-84 equatorial radius (used only for plotting)
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xs = R_earth * np.outer(np.cos(u), np.sin(v))
        ys = R_earth * np.outer(np.sin(u), np.sin(v))
        zs = R_earth * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.30, linewidth=0, antialiased=True)

        data = np.vstack([position_data[i] for i in range(no_satellites)])
        self._set_equal_3d_axes(ax, data)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_path / file_name)

    @staticmethod
    def _mix_with_white(color, amount=0.5):
        """If the amount=0 it yields original color, whilst it the amount=1 
        the method returns white."""
        rgb = np.array(mcolors.to_rgb(color))
        return tuple((1 - amount) * rgb + amount * np.ones(3))

    @staticmethod
    def _mix_with_black(color, amount=0.3):
        """If the amount=0 it yields original color, whilst it the amount=1 
        the method returns black."""
        rgb = np.array(mcolors.to_rgb(color))
        return tuple((1 - amount) * rgb)
    
    @staticmethod
    def _set_equal_3d_axes(ax, data: np.ndarray) -> None:
        """Set 3D plot axes to equal scale."""
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        max_range = np.array([x.max() - x.min(),
                            y.max() - y.min(),
                            z.max() - z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) / 2.0
        mid_y = (y.max() + y.min()) / 2.0
        mid_z = (z.max() + z.min()) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_box_aspect([1, 1, 1])

    def plot_relative_position(self, time_data, position_data, velocity_data, title, file_name):
        """Plot the relative position in the RTN frame between the satellites pair over time."""

        t_days = (time_data - time_data[0]) / 86400.0
        # Along-track distance computation (RTN of chaser satellite)
        # Along-track unit vector 
        v_chaser_norm = np.linalg.norm(velocity_data[1], axis=1)
  
        r_hat = position_data[1] / np.linalg.norm(position_data[1], axis=1)[:, None]  
        h_vector = np.cross(position_data[1], velocity_data[1])
        h_hat = h_vector / np.linalg.norm(h_vector, axis=1)[:, None]
        t_hat = np.cross(h_hat, r_hat)/ np.linalg.norm(np.cross(h_hat, r_hat), axis=1)[:, None]
        

        # Relative position (target relative to chaser)
        relative_position = position_data[0] - position_data[1]

        # Along-track distance (signed)
        along_track_distance = np.einsum("ij,ij->i", relative_position, t_hat) / 1e3 # [km]
        radial_distance = np.einsum("ij,ij->i", relative_position, r_hat) / 1e3      # [km]
        cross_track_distance = np.einsum("ij,ij->i", relative_position, h_hat) / 1e3 # [km]
        relative_position_norm = np.linalg.norm(relative_position, axis=1) / 1e3     # [km]

        plt.figure(figsize=(9, 4.5), dpi=140)
        plt.plot(t_days, along_track_distance, linewidth=1.8, label="Along-track distance", color="tab:blue")
        plt.plot(t_days, radial_distance, linewidth=1.8, linestyle="--", label="Radial distance", color="tab:orange")
        plt.plot(t_days, cross_track_distance, linewidth=1.8, linestyle=":", label="Cross-track distance", color="tab:green")
        plt.plot(t_days, relative_position_norm, linewidth=1.8, linestyle="-.", label="separation magnitude", color="tab:red")
        plt.axhline(0.0, linewidth=1.0, linestyle="--")

        plt.title(title)
        plt.xlabel("Propagation time [days]")
        plt.ylabel("Distance [km]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / file_name)