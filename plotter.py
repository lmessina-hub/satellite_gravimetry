from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from pathlib import Path

import matplotlib as mpl

def apply_publication_style():
    mpl.rcParams.update({
        # Fonts
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,

        # Axes
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.2,
        "axes.grid": True,

        # Grid
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,

        # Ticks
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,

        # Legend
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",

    })

apply_publication_style()

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
            if i == 0:
                reference_color =  "#073AA0"
            else:
                reference_color = cmap(i)
            starting_point_color = self._mix_with_white(reference_color, amount=0.35)  # lighter
            ending_point_color   = self._mix_with_black(reference_color, amount=0.25)  # darker
            ax.plot(position_data[i][:, 0], position_data[i][:, 1], position_data[i][:, 2], label=sat_labels[i],
                     linewidth=3, color=reference_color, linestyle=linestyles[i % len(linestyles)])
            ax.scatter(
                position_data[i][0, 0], position_data[i][0, 1], position_data[i][0, 2],
                marker="o", s=80, color=starting_point_color, edgecolor="k", linewidth=0.8,
                label=f"{sat_labels[i]} start"
            )
            ax.scatter(
                position_data[i][-1, 0], position_data[i][-1, 1], position_data[i][-1, 2],
                marker="x", s=110, color=ending_point_color, linewidth=1.6,
                label=f"{sat_labels[i]} end"
            )
        R_earth = 6378.1363e3  # [m] WGS-84 equatorial radius (used only for plotting)
        u = np.linspace(0, 2*np.pi, 120)
        v = np.linspace(0, np.pi, 120)
        xs = R_earth * np.outer(np.cos(u), np.sin(v))
        ys = R_earth * np.outer(np.sin(u), np.sin(v))
        zs = R_earth * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.25, zorder=0)

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

        plt.figure(figsize=(7, 4.5), dpi=140)
        plt.plot(t_days, along_track_distance, linewidth=2.5, label="Along-track (T)", color="tab:blue")
        plt.plot(t_days, radial_distance, linewidth=2.5, linestyle="--", label="Radial (R)", color="tab:orange")
        plt.plot(t_days, cross_track_distance, linewidth=2.8, linestyle=":", label="Cross-track (N)", color="tab:green")
        plt.plot(t_days, relative_position_norm, linewidth=2.5, linestyle="-.", label=r"Range ($\rho$)", color="tab:red")

        plt.title(title)
        plt.xlabel("Propagation time [days]")
        plt.ylabel("Distance [km]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / file_name)

    def plot_rtn_error_projections(self, samples_rtn: np.ndarray, sigma_rtn: np.ndarray, epoch_idx: int, file_name: str):
        """Plot RTN projections of noise samples with 1σ/2σ/3σ ellipses."""
        planes = [
            (0, 1, "R", "T"),  # R-T plane
            (0, 2, "R", "N"),  # R-N plane
            (1, 2, "T", "N"),  # T-N plane
        ]

        ellipse_angle = np.linspace(0.0, 2.0 * np.pi, 400)
        cmap = plt.get_cmap("tab10")


        fig = plt.figure(figsize=(14, 5))
        for i, (a, b, la, lb) in enumerate(planes, start=1):
            ax = fig.add_subplot(1, 3, i)

            # Scatter of samples projected into the plane
            ax.scatter(samples_rtn[:, a], samples_rtn[:, b], s=6, alpha=0.9, color="tab:blue")

            for ksig in (1, 2, 3):

                reference_color = cmap(ksig)
                xa = ksig * sigma_rtn[a] * np.cos(ellipse_angle)
                xb = ksig * sigma_rtn[b] * np.sin(ellipse_angle)
                ax.plot(xa, xb, linewidth=2, label=f"{ksig}σ", color=reference_color)

            ax.set_xlabel(f"{la} [m]")
            ax.set_ylabel(f"{lb} [m]")
            ax.set_title(f"Epoch {epoch_idx}: {la}-{lb} plane projection")
            ax.axis("equal")
            ax.grid(True, which="major", linewidth=0.8, alpha=0.25)
            ax.minorticks_on()
            ax.grid(False, which="minor")   
            ax.legend()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

        fig.suptitle("RTN noise samples and uncertainty ellipses")
        plt.tight_layout()
        plt.savefig(self.output_path / file_name)

    def plot_kbr_range_noise_histogram_and_distribution(
        self,
        range_error_samples: np.ndarray,
        sigma_rho: float,
        epoch_idx: int,
        file_name: str,
        bins: int = 40,
    ):
        """
        Plot KBR range noise samples for a given epoch as a histogram (density)
        with the target Gaussian PDF N(0, sigma_rho^2) overlaid.

        Parameters
        ----------
        range_error_samples : np.ndarray
            Array of sampled range noise for a single epoch.
        sigma_rho : float
            Standard deviation of range noise [m].
        epoch_idx : int
            Epoch index.
        file_name : str
            Output filename (saved under self.output_path).
        bins : int
            Histogram bins.
        """

        range_error_samples = np.asarray(range_error_samples).reshape(-1)

        x_lim = 4.0 * sigma_rho
        x = np.linspace(-x_lim, x_lim, 800)

        # Gaussian PDF computation
        pdf = (1.0 / (sigma_rho * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (x / sigma_rho) ** 2)

        fig = plt.figure(figsize=(7.0, 4.5), dpi=140)
        ax = fig.add_subplot(111)

        # Histogram as density
        ax.hist(
            range_error_samples,
            bins=bins,
            density=True,
            color="0.65",
            edgecolor="0.25",
            linewidth=0.8,
            alpha=0.75,
            label="Sampled KBR noise",
        )

        # Overlay Gaussian
        ax.plot(x, pdf, linewidth=2.5, color="#073AA0", label=r"Gaussian PDF")

        ax.set_title(f"KBR range noise distribution (Epoch {epoch_idx})")
        ax.set_xlabel("Range noise error [m]")
        ax.set_ylabel("Probability density [-]")

        # Publication-style cosmetics consistent with your other plots
        ax.grid(True, which="major", linewidth=0.8, alpha=0.25)
        ax.minorticks_on()
        ax.grid(False, which="minor")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)


        ax.legend(frameon=True, fontsize=11, loc="upper right")
        fig.tight_layout()
        fig.savefig(self.output_path / file_name, bbox_inches="tight")
        plt.close(fig)
