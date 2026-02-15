import numpy as np
import matplotlib.colors as mcolors
import json


from pathlib import Path

import matplotlib as mpl
mpl.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

        earth_texture_path = Path("data/earth_texture.jpg")

        self._add_earth(ax, texture_path=earth_texture_path)

        cmap = plt.get_cmap("tab10")
        linestyles = ["-", "--", "-.", ":"]
        linewidth = [4, 2]


        for i in range(no_satellites):
            if i == 0:
                reference_color =  "#073AA0"
            else:
                reference_color = cmap(i)
            point_color = self._mix_with_white(reference_color, amount=0.35)  # lighter
            ax.plot(position_data[i][:, 0], position_data[i][:, 1], position_data[i][:, 2], label=sat_labels[i],
                     linewidth=linewidth[i % len(linewidth)], color=reference_color, linestyle=linestyles[i % len(linestyles)])
            ax.scatter(
                position_data[i][0, 0], position_data[i][0, 1], position_data[i][0, 2],
                marker="o", s=80, color=point_color, edgecolor="k", linewidth=0.8,
                label=f"{sat_labels[i]} start"
            )
            ax.scatter(
                position_data[i][-1, 0], position_data[i][-1, 1], position_data[i][-1, 2],
                marker="x", s=110, color=point_color, linewidth=1.6,
                label=f"{sat_labels[i]} end"
            )

        data = np.vstack([position_data[i] for i in range(no_satellites)])
        self._set_equal_3d_axes(ax, data)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_path / file_name, dpi = 720)

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

    @staticmethod
    def _add_earth(ax, texture_path: Path | None) -> None:
        """
        Add an Earth sphere centered at the origin.
        """
        # Sphere parameterization
        n_lon, n_lat = 720, 360  # increase for smoother sphere, decrease for speed
        lon = np.linspace(-np.pi, np.pi, n_lon)
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon2, lat2 = np.meshgrid(lon, lat)
        radius_m = 6378137.0 - 1100000.0

        x = radius_m * np.cos(lat2) * np.cos(lon2)
        y = radius_m * np.cos(lat2) * np.sin(lon2)
        z = radius_m * np.sin(lat2)

        # Read texture (expects equirectangular: width=2*height typically)
        img = mpimg.imread(str(texture_path))
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0

        # Map lon/lat -> texture coordinates (u in [0,1], v in [0,1])
        u = (lon2 + np.pi) / (2 * np.pi)
        v = (lat2 + np.pi / 2) / np.pi

        # Convert to pixel indices
        h, w = img.shape[0], img.shape[1]
        px = np.clip((u * (w - 1)).astype(int), 0, w - 1)
        py = np.clip(((1 - v) * (h - 1)).astype(int), 0, h - 1)  # flip vertical for image coords

        facecolors = img[py, px]

        ax.plot_surface(
            x, y, z,
            facecolors=facecolors,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,   # IMPORTANT: keep False so texture colors are not altered
            alpha=1.0
        )


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


        fig = plt.figure(figsize=(14.9, 5))
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
            ax.spines["top"].set_linewidth(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

        fig.suptitle("RTN noise samples and uncertainty ellipses")
        plt.tight_layout()
        plt.savefig(self.output_path / file_name)


    def plot_acceleration_finite_difference_statistics(
        self,
        scenario: str,
        time: np.ndarray,
        results: dict[int, dict],
    ) -> None:
        """
        Create plots for finite-difference acceleration validation.

        Produces:
          - plots of the error components time evolution, overlaying all accuracy orders
          - plots of the error norm time evolution, overlaying all accuracy orders and including RMS in legend

        Parameters
        ----------
        satellite_name : str
            Identifier used in titles and filenames.
        time : np.ndarray, shape (N,)
            Time stamps [s].
        results : dict[int, dict]
            Output of validate_numerical_position_differentiation keyed by accuracy order.
            Must contain keys: 'error_vector', 'error_norm', 'error_rms'.
        file_prefix : str | None
            Optional prefix for filenames. If None, derived from satellite_name.
        """
        if time.ndim != 1:
            raise ValueError("Time must be a 1D array.")
        file_prefix = scenario.replace(" ", "_")
        propagation_time = time - time[0] # [s] time since start


        # -------------
        # Component plots
        # -------------
        comp_labels = [
            r"$|\epsilon_{a_x}(t)|$",
            r"$|\epsilon_{a_y}(t)|$",
            r"$|\epsilon_{a_z}(t)|$",
        ]

        file_suffix = ["x", "y", "z"]
        comp_titles = [
            r"Acceleration error component: $\epsilon_{a_x}(t)$",
            r"Acceleration error component: $\epsilon_{a_y}(t)$",
            r"Acceleration error component: $\epsilon_{a_z}(t)$",
        ]

        for j in range(3):
            fig = plt.figure(figsize=(8.5, 4.8), dpi=160)
            ax = fig.add_subplot(111)

            for accuracy in sorted(results.keys()):
                error_vector = np.asarray(results[accuracy]["error_vector"], dtype=float)
                
                ax.plot(
                    propagation_time,
                    np.abs(error_vector[:, j]),
                    "-o",
                    markersize=2.5,
                    linewidth=1.5,
                    label=rf"$p={accuracy}$",
                )

            ax.set_title(rf"{scenario} — {comp_titles[j]}", pad=14)
            ax.set_xlabel("Propagation Time [s]")
            ax.set_ylabel(rf"{comp_labels[j]} [m/s$^2$] ")
            ax.set_yscale("log")


            self._style_axes(ax)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),   # below x-axis
                ncol=min(5, len(results)),     # one row (up to 5 columns; adjust if needed)
                frameon=True,
                fontsize=10,
                handlelength=2.0,
                columnspacing=1.2,
            )

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.25)   # add room for legend below
            fig.savefig(self.output_path / f"{file_prefix}_acceleration_error_component_{file_suffix[j]}.png", bbox_inches="tight")
            plt.close(fig)

        # -------------
        # Norm plot with RMS in legend
        # -------------
        fig = plt.figure(figsize=(9.5, 5.3), dpi=160)
        ax = fig.add_subplot(111)

        for accuracy in sorted(results.keys()):
            error_norm = np.asarray(results[accuracy]["error_norm"], dtype=float).reshape(-1)

            rms = float(results[accuracy]["error_rms"])
            ax.plot(
                propagation_time,
                error_norm,
                "-o",
                markersize=2.2,
                linewidth=1.2,
                label=rf"$p={accuracy}\;(\mathrm{{RMS}}={rms:.3e}\,\mathrm{{m\,s^{{-2}}}})$",
            )

        ax.set_title(rf"{scenario} — Acceleration error norm $\|\epsilon(t)\|$", pad=10)
        ax.set_xlabel("Propagation Time [s]")
        ax.set_ylabel(r"$\|\epsilon(t)\|$ [m/s$^2$]")
        ax.set_yscale("log")

        self._style_axes(ax)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),   # below x-axis
            ncol=min(3, len(results)),     # one row (up to 5 columns; adjust if needed)
            frameon=True,
            fontsize=10,
            handlelength=2.0,
            columnspacing=1.2,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.10)   # add room for legend below
        out = self.output_path / f"{file_prefix}_acceleration_error_norm.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


    def plot_los_intersatellite_acceleration_finite_difference_statistics(
        self,
        scenario: str,
        time: np.ndarray,
        results: dict[int, dict],
    ) -> None:
        """
        Plot LOS-projected inter-satellite acceleration validation results.

        Produces:
        - Absolute Error with legend including RMS for each accuracy order (log y)

        """
        if time.ndim != 1:
            raise ValueError("Time must be a 1D array.")

        file_prefix = scenario.replace(" ", "_")
        propagation_time = time - time[0]

        fig = plt.figure(figsize=(9.5, 5.3), dpi=160)
        ax = fig.add_subplot(111)

        for accuracy in sorted(results.keys()):
            absolute_error = np.asarray(results[accuracy]["absolute_error"], dtype=float).reshape(-1)
            error_rms = float(results[accuracy]["error_rms"])

            ax.plot(
                propagation_time,
                absolute_error,
                "-o",
                markersize=2.2,
                linewidth=1.2,
                label=rf"$p={accuracy}\;(\mathrm{{RMS}}={error_rms:.3e}\,\mathrm{{m\,s^{{-2}}}})$",
            )

        ax.set_title(rf"{scenario} ", pad=10)
        ax.set_xlabel("Propagation Time [s]")
        ax.set_ylabel(r"$|\epsilon_{a_{\mathrm{LOS}}(t)}|$ [m/s$^2$]")
        ax.set_yscale("log")

        self._style_axes(ax)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(results)),
            frameon=True,
            fontsize=10,
            handlelength=2.0,
            columnspacing=1.2,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.10)
        fig.savefig(self.output_path / f"{file_prefix}.png", bbox_inches="tight")
        plt.close(fig)

    def _style_axes(self, ax: plt.Axes) -> None:
        ax.minorticks_on()
        ax.grid(False, which="minor")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["top"].set_linewidth(1.2)
        ax.spines["right"].set_linewidth(1.2)
        ax.set_axisbelow(True)

    def plot_pointing_angles_asd(
        self,
        file_name: str,
        pitch_history_json_path: Path,
        yaw_history_json_path: Path,
        roll_history_json_path: Path
    ) -> None:
        """Plot ASD of roll/pitch/yaw pointing angles from plot-digitizer JSON files."""

        json_paths = [roll_history_json_path, pitch_history_json_path, yaw_history_json_path]
        color = ["blue", "red", "green"]
        labels = ["Roll", "Pitch", "Yaw"]

        fig = plt.figure(figsize=(10, 6))

        for i, path in enumerate(json_paths):
            with open(path, "r") as f:
                data = json.load(f)

            x = np.array([float(d["x"]) for d in data], dtype=float)
            y = np.array([float(d["y"]) for d in data], dtype=float)
            # Sort by frequency
            idx = np.argsort(x)
            # Match the paper legend colors: Roll=blue, Pitch=red, Yaw=green
            plt.loglog(x[idx],  y[idx],  linewidth=2, color=color[i],  label=labels[i])

        plt.title("ASD of Pointing Angles (plot digitizer)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"ASD [rad Hz$^{-1/2}$]")
        plt.legend(loc="upper right", frameon=True)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_linear_interpolation_comparison(
        self,
        original_frequencies: np.ndarray,
        original_asd_values: np.ndarray,
        interpolated_frequencies: np.ndarray,
        interpolated_asd_values: np.ndarray,
        file_name: str
    ) -> None:
        """Plot comparison between original and linearly interpolated ASD data."""

        fig = plt.figure(figsize=(10, 6))

        plt.loglog(original_frequencies,  original_asd_values, color="#073AA0", label='Original ASD Data', linewidth=4)
        plt.loglog(interpolated_frequencies, interpolated_asd_values, color="#D8660E", label='Interpolated ASD Data', linewidth=2, linestyle='--')

        plt.title("ASD Data: Original vs Interpolated")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"ASD [rad Hz$^{-1/2}$]")
        plt.legend(loc="upper right", frameon=True)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_angle_noise_time_series(
        self,
        noise_time_series,
        file_name: str
    ) -> None:
        """Plot time series of pointing angle noise."""

        fig = plt.figure(figsize=(10, 6))

        plt.plot(noise_time_series.sample_times, noise_time_series, color="#073AA0", linewidth=1.8)

        plt.title("Pointing Angle Noise Time Series")
        plt.xlabel("Time [s]")
        plt.ylabel("Pointing Angle Noise [rad]")

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_welch_estimated_psd_comparison(
        self,
        estimated_frequencies: np.ndarray,
        estimated_psd_values: np.ndarray,
        input_frequencies: np.ndarray,
        input_psd_values: np.ndarray,
        file_name: str,
        ordinate_label: str = r"PSD [rad$^2$ Hz$^{-1}$]",
        title: str = "Pointing Angle PSD"
    ) -> None:
        """Plot comparison between Welch estimated PSD and input PSD."""

        fig = plt.figure(figsize=(10, 6))

        plt.loglog(estimated_frequencies, estimated_psd_values, color="#D8660E", label='Welch Estimated PSD', linewidth=1.8, linestyle='--')
        plt.loglog(input_frequencies,  input_psd_values, color="#073AA0", label='Input PSD', linewidth=3)

        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(ordinate_label)
        plt.legend(loc="upper right", frameon=True)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)


    def plot_kbr_system_and_oscillator_asd(
        self,
        frequencies: np.ndarray,
        asd_values: np.ndarray,
        file_name: str
    ) -> None:
        """Plot KBR system and oscillator ASD."""

        fig = plt.figure(figsize=(10, 6))

        plt.loglog(frequencies,  asd_values, color="#073AA0", label='Analytical ASD', linewidth=4)

        plt.title("KBR System and Oscillator Noise ASD")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"ASD [m Hz$^{-1/2}$]")
        plt.legend(loc="upper right", frameon=True)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_kbr_system_and_oscillator_noise_time_series(
        self,
        noise_time_series,
        file_name: str
    ) -> None:
        """Plot time series of KBR noise and oscillator noise."""

        fig = plt.figure(figsize=(10, 6))

        plt.plot(noise_time_series.sample_times, noise_time_series, color="#073AA0", linewidth=1.8)

        plt.title("KBR System and Oscillator Noise Time Series")
        plt.xlabel("Time [s]")
        plt.ylabel("KBR System and Oscillator Noise [m]")

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    
    def plot_apc_pointing_jitter_coupling_time_series_demeaned(
        self,
        apc_pointing_jitter_coupling_noise: dict[str, np.ndarray],
        time_seconds: np.ndarray,
        satellite_label: str,
        file_name: str,
    ) -> None:
        """
        Plot APC offset pointing jitter coupling noise (demeaned) for a single satellite.

        Parameters
        ----------
        apc_pointing_jitter_coupling_noise : dict[str, np.ndarray]
            Dict containing APC coupling noise arrays in meters, keyed by satellite name.
        time_seconds : np.ndarray
            Time array in seconds (shape (N,)).
        satellite_label : str
            Satellite key.
        file_name : str
            Output file name (saved under self.output_path).
        """

        if satellite_label not in apc_pointing_jitter_coupling_noise:
            raise KeyError(f"Satellite '{satellite_label}' not found in APC coupling noise dict.")

        value = np.asarray(apc_pointing_jitter_coupling_noise[satellite_label], dtype=float).reshape(-1)
        time = np.asarray(time_seconds, dtype=float).reshape(-1)

        if time.shape[0] != value.shape[0]:
            raise ValueError(f"Time and APC noise length mismatch: len(time)={time.shape[0]} vs len(value)={value.shape[0]}")
        value_demeaned = value - np.mean(value)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(time, value_demeaned, color="#073AA0", linewidth=1.8)

        plt.title(f"APC Pointing Jitter Coupling Noise (Demeaned) — {satellite_label}")
        plt.xlabel("Time [s]")
        plt.ylabel("APC coupling noise (demeaned) [m]")

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)
