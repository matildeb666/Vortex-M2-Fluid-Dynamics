"""
Vortex Dynamics Simulation Module.

This module provides a class-oriented framework for simulating the dynamics
of point vortices using the Biot-Savart law and Runge-Kutta 4 (RK4) integration.
It includes visualization tools, initialization utilities, and convergence analysis.

The core physics relies on the singular point vortex model in 2D inviscid flow.
It supports:
- Free space dynamics.
- Square domains via the Method of Images (first shell approximation).
- Calculation of conserved quantities (Hamiltonian) and stability metrics (MSD).
- Visualization of trajectories, phase space, and convergence rates.

Authors: Refactored by Gemini (Original: Ivan Delbende, Matilde Bureau, Gaston Ravanas)
Copyright: Sorbonne University 2021 - 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.ticker import FuncFormatter
from typing import Tuple, List, Optional, Union, Callable
from tqdm import tqdm
import warnings

# --- Matplotlib Configuration ---
# formatter = FuncFormatter(lambda x, _: f"{x:g}")

try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
except Exception as e:
    warnings.warn(f"Could not set LaTeX mode: {e}. Falling back to standard fonts.")


class PointVortexSimulation:
    """
    Handles the physics and numerical integration of a system of 2D point vortices.

    This class manages the state (positions, circulations) of a system of vortices
    and evolves them in time using the Biot-Savart law and a 4th-order Runge-Kutta
    integration scheme. It supports periodic-like boundary effects in a square domain
    using the method of images.

    Attributes:
        positions (np.ndarray): Array of shape (N, 2) storing the [x, y] coordinates
            of the *real* vortices.
        circulations (np.ndarray): Array of shape (N,) storing the circulation
            strength (Gamma) of each real vortex.
        delta (float): Regularization parameter (core radius) to prevent velocity
            singularities at r=0.
        box_size (Optional[float]): The side length L of the square domain [-L/2, L/2].
            If provided, image vortices are generated dynamically to enforce flow tangency.
        collision_threshold (float): The critical distance between any two vortices
            below which the simulation is halted to preserve physical validity.
        num_real (int): The number of physical vortices in the simulation.
    """

    def __init__(self, 
                 positions: Union[np.ndarray, List[List[float]]], 
                 circulations: Union[np.ndarray, List[float]], 
                 delta: float = 0.0,
                 box_size: Optional[float] = None,
                 collision_threshold: float = 1e-2):
        """
        Initialize the vortex system.

        Args:
            positions: Initial coordinates of REAL vortices as a list or array of shape (N, 2).
            circulations: Circulation strengths of REAL vortices as a list or array of shape (N,).
            delta: Regularization parameter (smoothing radius). Defaults to 0.0 (singular).
            box_size: If provided, enforces a square boundary condition [-L/2, L/2]
                using the method of images (first shell approximation). Defaults to None (free space).
            collision_threshold: Distance below which a collision is flagged and the simulation stops.
                Defaults to 1e-2.

        Raises:
            ValueError: If the number of positions does not match the number of circulations.
        """
        self.positions = np.array(positions, dtype=np.float64)
        self.circulations = np.array(circulations, dtype=np.float64)
        self.delta = delta
        self.box_size = box_size
        self.collision_threshold = collision_threshold
        self.num_real = len(self.circulations)
        
        if self.positions.shape[0] != self.num_real:
            raise ValueError("Number of positions must match number of circulations.")

    def _get_augmented_state(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the full set of vortices (Real + Images) for boundary handling.

        This method implements the Method of Images for a square domain. It generates
        8 image vortices for every real vortex (the first "shell" of neighbors):
        - 4 Wall reflections (inverted circulation).
        - 4 Corner reflections (preserved circulation).

        Args:
            positions: Current coordinates of the real vortices, shape (N, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Augmented positions array of shape (9N, 2).
                - Augmented circulations array of shape (9N,).
        """
        if self.box_size is None:
            return positions, self.circulations
            
        L = self.box_size
        
        # List of positions lists, starting with real ones
        pos_list = [positions]
        circ_list = [self.circulations]
        
        # --- Generate Images (First Shell) ---
        # 1. Wall Reflections (Circulation -> -Gamma)
        # Right (x -> L-x), Left (x -> -L-x), Top (y -> L-y), Bottom (y -> -L-y)
        
        p_right = positions.copy(); p_right[:, 0] = L - p_right[:, 0]
        p_left = positions.copy();  p_left[:, 0] = -L - p_left[:, 0]
        p_top = positions.copy();   p_top[:, 1] = L - p_top[:, 1]
        p_bot = positions.copy();   p_bot[:, 1] = -L - p_bot[:, 1]
        
        pos_list.extend([p_right, p_left, p_top, p_bot])
        circ_list.extend([-self.circulations] * 4)
        
        # 2. Corner Reflections (Circulation -> +Gamma)
        # TR, TL, BR, BL
        p_tr = positions.copy(); p_tr[:, 0] = L - p_tr[:, 0];   p_tr[:, 1] = L - p_tr[:, 1]
        p_tl = positions.copy(); p_tl[:, 0] = -L - p_tl[:, 0];  p_tl[:, 1] = L - p_tl[:, 1]
        p_br = positions.copy(); p_br[:, 0] = L - p_br[:, 0];   p_br[:, 1] = -L - p_br[:, 1]
        p_bl = positions.copy(); p_bl[:, 0] = -L - p_bl[:, 0];  p_bl[:, 1] = -L - p_bl[:, 1]
        
        pos_list.extend([p_tr, p_tl, p_br, p_bl])
        circ_list.extend([self.circulations] * 4)
        
        return np.vstack(pos_list), np.concatenate(circ_list)

    def _biot_savart_law(self, positions: np.ndarray) -> np.ndarray:
        """
        Computes the induced velocity field using the Biot-Savart law.

        Calculates the velocity of the REAL vortices induced by ALL vortices 
        (Real + Images if box_size is set).
        
        The velocity induced by vortex j on vortex i is given by:
        u_i = - (1/2pi) * sum_j [ Gamma_j * (y_i - y_j) / r_ij^2 ]
        v_i = + (1/2pi) * sum_j [ Gamma_j * (x_i - x_j) / r_ij^2 ]
        
        Args:
            positions: Current coordinates of REAL vortices, shape (N, 2).
            
        Returns:
            velocities: Velocity vectors [u, v] for REAL vortices, shape (N, 2).
        """
        # 1. Generate full set of vortices (Real + Virtual)
        all_pos, all_gammas = self._get_augmented_state(positions)
        
        # 2. Compute interactions
        # Target (Real)
        x_target = positions[:, 0]
        y_target = positions[:, 1]
        
        # Source (All)
        x_source = all_pos[:, 0]
        y_source = all_pos[:, 1]
        
        # Difference matrices: shape (N_real, N_total)
        dx = x_target[:, np.newaxis] - x_source[np.newaxis, :]
        dy = y_target[:, np.newaxis] - y_source[np.newaxis, :]
        
        r2 = dx**2 + dy**2 + self.delta**2
        
        # Handle self-interaction (diagonal of the top-left N*N block)
        if self.delta == 0:
            diag_indices = np.arange(self.num_real)
            r2[diag_indices, diag_indices] = np.inf
        
        inv_r2 = 1.0 / r2
        
        u_contributions = - (all_gammas[np.newaxis, :] * dy) * inv_r2
        v_contributions = + (all_gammas[np.newaxis, :] * dx) * inv_r2
        
        u = np.sum(u_contributions, axis=1) / (2 * np.pi)
        v = np.sum(v_contributions, axis=1) / (2 * np.pi)
        
        return np.column_stack((u, v))

    def step_rk4(self, dt: float) -> np.ndarray:
        """
        Advances the system by one time step `dt` using the Runge-Kutta 4 (RK4) method.

        RK4 provides 4th-order accuracy, with local error O(dt^5) and global error O(dt^4).
        
        Args:
            dt: Time step size.

        Returns:
            velocity: The velocity vector at the *start* of the time step (k1).
        """
        k1 = self._biot_savart_law(self.positions)
        k2 = self._biot_savart_law(self.positions + 0.5 * dt * k1)
        k3 = self._biot_savart_law(self.positions + 0.5 * dt * k2)
        k4 = self._biot_savart_law(self.positions + dt * k3)
        
        self.positions += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return k1

    def check_collision(self) -> bool:
        """
        Checks if any two real vortices have approached closer than `collision_threshold`.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        r2 = dx**2 + dy**2
        
        # Mask diagonal
        np.fill_diagonal(r2, np.inf)
        
        min_dist_sq = np.min(r2)
        return min_dist_sq < self.collision_threshold**2

    def run(self, t_max: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the full simulation from t=0 to t_max.

        Checks for collisions at every time step. If a collision is detected,
        the simulation terminates early and returns the truncated history.

        Args:
            t_max: Total simulation time.
            dt: Time step size.

        Returns:
            times: 1D array of time stamps.
            trajectories: Array of shape (Steps, N, 2) containing position history.
            velocities: Array of shape (Steps, N, 2) containing velocity history.
        """
        steps = int(t_max / dt)
        
        # Pre-allocate (can be truncated later)
        times = np.zeros(steps + 1)
        trajectories = np.zeros((steps + 1, self.num_real, 2))
        velocities = np.zeros((steps + 1, self.num_real, 2))
        
        # Initial State
        times[0] = 0.0
        trajectories[0] = self.positions.copy()
        velocities[0] = self._biot_savart_law(self.positions)
        
        final_step_idx = steps
        
        for i in range(steps):
            current_time = (i + 1) * dt
            
            # Check collision at start of step
            if self.check_collision():
                print(f"\n[Simulation Stopped] Collision detected at t = {times[i]:.4f}s")
                final_step_idx = i
                break
                
            # Integrate
            vel_at_step_start = self.step_rk4(dt)
            
            # Store
            times[i+1] = current_time
            velocities[i] = vel_at_step_start
            trajectories[i+1] = self.positions.copy()
            
            # Check collision of new state
            if self.check_collision():
                print(f"\n[Simulation Stopped] Collision detected at t = {current_time:.4f}s")
                final_step_idx = i + 1
                velocities[i+1] = self._biot_savart_law(self.positions)
                break
        else:
             # If no break, compute final velocity
             velocities[-1] = self._biot_savart_law(self.positions)

        # Truncate arrays to actual valid steps
        return (times[:final_step_idx+1], 
                trajectories[:final_step_idx+1], 
                velocities[:final_step_idx+1])


class VortexInitializer:
    """
    Utilities for generating initial configurations of vortex systems.
    """
    
    @staticmethod
    def create_circle(n_vortices: int, radius: float) -> np.ndarray:
        """
        Creates N positions distributed equally on a circle.

        Args:
            n_vortices: Number of vortices to generate.
            radius: Radius of the circle.

        Returns:
            np.ndarray: Array of shape (N, 2) containing positions.
        """
        angles = np.linspace(0, 2 * np.pi, n_vortices, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.column_stack((x, y))

    @staticmethod
    def apply_square_boundaries(positions: np.ndarray, 
                                circulations: np.ndarray, 
                                size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Statically augments a state with image vortices for a square domain.

        Note: This is primarily a helper function. The `PointVortexSimulation` class
        handles this dynamically during runtime if `box_size` is passed to it.

        Args:
            positions: Original vortex positions (N, 2).
            circulations: Original circulations (N,).
            size: Length of the square side L.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented positions (9N, 2) and circulations (9N,).
        """
        L = size
        
        new_pos_list = [positions]
        new_circ_list = [circulations]
        
        # 1. Wall Reflections (Circulation -> -Gamma)
        p_right = positions.copy(); p_right[:, 0] = L - p_right[:, 0]
        p_left = positions.copy();  p_left[:, 0] = -L - p_left[:, 0]
        p_top = positions.copy();   p_top[:, 1] = L - p_top[:, 1]
        p_bottom = positions.copy(); p_bottom[:, 1] = -L - p_bottom[:, 1]
        
        new_pos_list.extend([p_right, p_left, p_top, p_bottom])
        new_circ_list.extend([-circulations] * 4)
        
        # 2. Corner Reflections (Circulation -> +Gamma)
        p_tr = positions.copy(); p_tr[:, 0] = L - p_tr[:, 0]; p_tr[:, 1] = L - p_tr[:, 1]
        p_tl = positions.copy(); p_tl[:, 0] = -L - p_tl[:, 0]; p_tl[:, 1] = L - p_tl[:, 1]
        p_br = positions.copy(); p_br[:, 0] = L - p_br[:, 0]; p_br[:, 1] = -L - p_br[:, 1]
        p_bl = positions.copy(); p_bl[:, 0] = -L - p_bl[:, 0]; p_bl[:, 1] = -L - p_bl[:, 1]
        
        new_pos_list.extend([p_tr, p_tl, p_br, p_bl])
        new_circ_list.extend([circulations] * 4)
        
        return np.vstack(new_pos_list), np.concatenate(new_circ_list)


class VortexAnalysis:
    """
    Helper class for computing derived physical quantities from simulation data.
    """
    
    @staticmethod
    def compute_omega(times: np.ndarray, trajectories: np.ndarray) -> np.ndarray:
        """
        Computes the angular velocity of vortices relative to the system centroid.
        
        Calculates d(theta)/dt, where theta is the angle of the vortex position vector
        relative to the instantaneous centroid of the system.

        Args:
            times: 1D array of time stamps.
            trajectories: Array of shape (Time, N, 2) containing positions.

        Returns:
            np.ndarray: Array of shape (Time, N) containing angular velocities.
        """
        centroid = np.mean(trajectories, axis=1, keepdims=True)
        rel_pos = trajectories - centroid
        x_rel = rel_pos[:, :, 0]
        y_rel = rel_pos[:, :, 1]
        theta = np.arctan2(y_rel, x_rel)
        theta_unwrapped = np.unwrap(theta, axis=0)
        
        # Handle variable dt if collision stopped sim early
        omega = np.gradient(theta_unwrapped, times, axis=0)
        return omega
    
    @staticmethod
    def compute_msd(trajectories: np.ndarray, circulations: np.ndarray) -> np.ndarray:
        """
        Computes the Mean Square Displacement (MSD) or Average Radial Variance.
        
        The metric is defined as:
        R^2(t) = (1/N) * sum_i |z_i(t) - z_c(t)|^2
        where z_c is the center of vorticity. This serves as a stability metric.

        Args:
            trajectories: Array of shape (Time, N, 2).
            circulations: Array of shape (N,).

        Returns:
            np.ndarray: 1D array of MSD values over time.
        """
        gammas = circulations # Assumes only real circulations passed
        total_circulation = np.sum(gammas)
        
        if abs(total_circulation) < 1e-10:
             z_c = np.mean(trajectories, axis=1)
        else:
            weighted_pos = trajectories * gammas[np.newaxis, :, np.newaxis]
            z_c = np.sum(weighted_pos, axis=1) / total_circulation
            
        diff = trajectories - z_c[:, np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=2)
        msd = np.mean(dist_sq, axis=1)
        return msd

    @staticmethod
    def compute_hamiltonian(trajectories: np.ndarray, circulations: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Computes the Hamiltonian H(t) of the system.

        For a system of point vortices, the Hamiltonian is given by:
        H = - (1/4pi) * sum_{j != k} [ Gamma_j * Gamma_k * ln(r_jk) ]
        
        (Note: The implementation here computes a positive sum form which differs by a sign convention, 
        effectively H_implemented = -H_standard. The conservation property holds regardless).

        Args:
            trajectories: Array of shape (Time, N, 2).
            circulations: Array of shape (N,).

        Returns:
            Tuple[np.ndarray, bool]: 
                - 1D array of Hamiltonian values over time.
                - Boolean flag indicating if a "collision" (separation < 1e-3) occurred.
        """
        T, N, _ = trajectories.shape
        gammas = circulations
        
        H = np.zeros(T)
        min_dist_threshold = 1e-3
        collision_flag = False
        
        factor = 1.0 / (4 * np.pi)
        G_matrix = np.outer(gammas, gammas)
        
        for t in range(T):
            pos = trajectories[t]
            x = pos[:, 0]
            y = pos[:, 1]
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            r_sq = dx**2 + dy**2
            
            r_off_diag = r_sq.copy()
            np.fill_diagonal(r_off_diag, np.inf)
            if np.min(r_off_diag) < min_dist_threshold**2:
                collision_flag = True
            
            r_sq_safe = r_sq.copy()
            np.fill_diagonal(r_sq_safe, 1.0)
            ln_r = 0.5 * np.log(r_sq_safe)
            step_energy = np.sum(G_matrix * ln_r)
            H[t] = factor * step_energy
            
        return H, collision_flag

    @staticmethod
    def _compute_theoretical_dynamics(times, initial_pos, circulations, v_th, omega_th):
        """
        Calculates theoretical trajectories based on a simple rotation+translation model.
        
        Assumes the system translates with velocity `v_th` and rotates rigidly with
        angular velocity `omega_th` around the initial center of vorticity.

        Args:
            times: Time array.
            initial_pos: Initial positions (N, 2).
            circulations: Vortex strengths (N,).
            v_th: Theoretical translational velocity vector (2,).
            omega_th: Theoretical angular velocity scalar.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Theoretical trajectories (Time, N, 2).
                - Theoretical velocities (Time, N, 2).
        """
        N = len(initial_pos)
        total_gamma = np.sum(circulations)
        if abs(total_gamma) < 1e-10:
            z_c = np.mean(initial_pos, axis=0)
        else:
            z_c = np.sum(initial_pos * circulations[:, np.newaxis], axis=0) / total_gamma
            
        traj_th = np.zeros((len(times), N, 2))
        vel_th = np.zeros((len(times), N, 2))
        
        if v_th is None: v_th = np.zeros((N, 2))
        if omega_th is None: omega_th = np.zeros(N)
        
        if np.ndim(v_th) == 1 and len(v_th) == 2:
            v_th = np.tile(v_th, (N, 1))
        
        if np.ndim(omega_th) == 0:
            omega_th = np.full(N, omega_th)
        
        for i in range(N):
            rel_0 = initial_pos[i] - z_c
            theta = omega_th[i] * times
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rx = rel_0[0] * cos_t - rel_0[1] * sin_t
            ry = rel_0[0] * sin_t + rel_0[1] * cos_t
            tx = v_th[i, 0] * times
            ty = v_th[i, 1] * times
            traj_th[:, i, 0] = z_c[0] + tx + rx
            traj_th[:, i, 1] = z_c[1] + ty + ry
            vel_th[:, i, 0] = v_th[i, 0] - omega_th[i] * ry
            vel_th[:, i, 1] = v_th[i, 1] + omega_th[i] * rx
            
        return traj_th, vel_th


class VortexVisualizer:
    """
    Handles visualization of simulation results including trajectories, phase planes,
    and convergence metrics.
    """

    @staticmethod
    def plot_analysis(times: np.ndarray,
                      trajectories: np.ndarray,
                      velocities: np.ndarray,
                      circulations: np.ndarray,
                      theoretical_vel: Optional[Union[Callable[[float], np.ndarray], np.ndarray]] = None,
                      theoretical_omega: Optional[Union[float, Callable[[float], float], np.ndarray]] = None,
                      domain_size: Optional[float] = None,
                      alpha_range: Tuple[float, float] = (0.2, 1.0),
                      save_fig: Optional[str] = None,
                      time_cmap: Optional[str] = None):
        """
        Master plotting function producing a suite of analysis figures.

        Generates 4 figures:
        1. Phase Spaces: Spatial Trajectories (x, y) and Velocity Phase Plane (u, v).
        2. Angular Velocity: Omega vs Time.
        3. Hamiltonian: H vs Time (checks for conservation).
        4. Stability: Mean Square Displacement (log R^2) vs Time.

        Args:
            times: Simulation time array.
            trajectories: Position history (Time, N, 2).
            velocities: Velocity history (Time, N, 2).
            circulations: Vortex strengths (N,).
            theoretical_vel: Theoretical velocity function or static array for comparison.
            theoretical_omega: Theoretical angular velocity function or value.
            domain_size: Size of the square domain to draw boundary lines.
            alpha_range: (min_alpha, max_alpha) tuple for time-fading opacity in trajectory plots.
            save_fig: If provided, saves figures with this suffix (e.g., 'run1' -> 'phase_spaces_run1.pdf').
            time_cmap: If provided (e.g., 'viridis'), trajectories are colored using this colormap
                       based on time, instead of the default vortex-specific color with alpha gradient.
                       Start/End markers retain their vortex-specific colors.
        """
        n_real = trajectories.shape[1]
        real_gammas = circulations[:n_real]
        
        msd = VortexAnalysis.compute_msd(trajectories, circulations)
        H_vals, collision = VortexAnalysis.compute_hamiltonian(trajectories, circulations)
        
        traj_th, vel_th = None, None
        has_th_v = theoretical_vel is not None and not callable(theoretical_vel)
        has_th_w = theoretical_omega is not None and not callable(theoretical_omega)
        
        if has_th_v or has_th_w:
             traj_th, vel_th = VortexAnalysis._compute_theoretical_dynamics(
                 times, trajectories[0], real_gammas, theoretical_vel, theoretical_omega
             )
        
        # --- Figure 1: Phase Spaces (Trajectories & Velocities) ---
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        VortexVisualizer._plot_trajectories_on_ax(ax1, trajectories, real_gammas, domain_size, alpha_range, traj_th, time_cmap)
        VortexVisualizer._plot_velocities_on_ax(ax2, times, velocities, theoretical_vel, alpha_range, vel_th, time_cmap)
        
        # Add Time Colorbar
        if time_cmap:
            cmap_time = plt.get_cmap(time_cmap)
        else:
            n_bins = 256
            alphas = np.linspace(alpha_range[0], alpha_range[1], n_bins)
            zeros = np.zeros(n_bins)
            colors = np.stack([zeros, zeros, zeros, alphas], axis=1)
            cmap_time = ListedColormap(colors)
            
        norm = mcolors.Normalize(vmin=times[0], vmax=times[-1])
        sm = plt.cm.ScalarMappable(cmap=cmap_time, norm=norm)
        sm.set_array([])
        cbar = fig1.colorbar(sm, ax=ax2, orientation='vertical', pad=0.02)
        cbar.set_label('$t$')
        
        plt.tight_layout()
        if save_fig:
            fig1.savefig(f"phase_spaces_{save_fig}.pdf", bbox_inches='tight')
            fig1.savefig(f"phase_spaces_{save_fig}.svg", bbox_inches='tight')
        plt.show()

        # --- Figure 2: Angular Velocity Omega(t) ---
        fig2, ax3 = plt.subplots(figsize=(6, 4))
        VortexVisualizer._plot_omega_on_ax(ax3, times, omegas=VortexAnalysis.compute_omega(times, trajectories), 
                                           theoretical_val=theoretical_omega)
        plt.tight_layout()
        if save_fig:
            fig2.savefig(f"omega_{save_fig}.pdf", bbox_inches='tight')
            fig2.savefig(f"omega_{save_fig}.svg", bbox_inches='tight')
        plt.show()
        
        # --- Figure 3: Hamiltonian H(t) ---
        fig3, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(times, H_vals, 'k-', lw=1.5)
        ax4.set_xlabel(r'$t~[s]$')
        ax4.set_ylabel(r'$H~[m^5.s^{-2}]$')
        ax4.grid(True, linestyle=':', alpha=0.6)
        
        if collision:
            ax4.text(0.5, 0.9, "COLLISION WARNING!", transform=ax4.transAxes, 
                     color='red', fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        plt.tight_layout()
        if save_fig:
            fig3.savefig(f"hamiltonian_{save_fig}.pdf", bbox_inches='tight')
            fig3.savefig(f"hamiltonian_{save_fig}.svg", bbox_inches='tight')
        plt.show()
        
        # --- Figure 4: MSD R^2(t) ---
        fig4, ax5 = plt.subplots(figsize=(6, 4))
        ax5.semilogy(times, msd, 'k-', lw=1.5)
        ax5.set_xlabel(r'$t~[s]$')
        ax5.set_ylabel(r'$R^2~[m^2]$')
        ax5.set_yticks([0.01, 0.1, 1, 10])
        ax5.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax5.grid(True, which="both", ls=":", alpha=0.6)
        plt.tight_layout()
        if save_fig:
            fig4.savefig(f"msd_{save_fig}.pdf", bbox_inches='tight')
            fig4.savefig(f"msd_{save_fig}.svg", bbox_inches='tight')
        plt.show()

    @staticmethod
    def _plot_trajectories_on_ax(ax, trajectories, circulations, domain_size=None, alpha_range=(0.2, 1.0), traj_th=None, time_cmap=None):
        ax.set_xlabel(r'$x~[m]$')
        ax.set_ylabel(r'$y~[m]$')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if domain_size is not None:
            limit = domain_size / 2.0
            # Add 5% margin to limits so the dashed boundary is not on the edge
            margin_factor = 1.05
            ax.set_xlim(-limit * margin_factor, limit * margin_factor)
            ax.set_ylim(-limit * margin_factor, limit * margin_factor)
            
            rect = plt.Rectangle((-limit, -limit), domain_size, domain_size, 
                                 fill=False, linestyle='--', color='k', alpha=0.5)
            ax.add_patch(rect)
        else:
             all_x = trajectories[..., 0].flatten()
             all_y = trajectories[..., 1].flatten()
             if len(all_x) > 0:
                 margin = 0.1 * (all_x.max() - all_x.min() + 1e-6)
                 ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
                 ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        
        colors_list = []
        for i in range(trajectories.shape[1]):
            color = f"C{i%10}" 
            colors_list.append(color)
            label = rf'$\Gamma_{i}={circulations[i]:.2g}~[m^2.s^{'{-1}'}]$'
            
            points = trajectories[:, i, :].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_segs = len(segments)
            
            if time_cmap:
                cmap = plt.get_cmap(time_cmap)
                t_vals = np.linspace(0, 1, n_segs)
                colors = cmap(t_vals)
            else:
                alphas = np.linspace(alpha_range[0], alpha_range[1], n_segs)
                colors = np.zeros((n_segs, 4))
                colors[:] = to_rgba(color)
                colors[:, 3] = alphas
            
            lc = LineCollection(segments, colors=colors, linewidths=3)
            ax.add_collection(lc)
            ax.plot([], [], color=color, label=label) # Proxy legend

            if traj_th is not None:
                if i == trajectories.shape[1] - 1:
                    ax.plot(traj_th[:, i, 0], traj_th[:, i, 1], '--', color='k', alpha=1., lw=1.5, label='Theory')
                else:
                    ax.plot(traj_th[:, i, 0], traj_th[:, i, 1], '--', color='k', alpha=1., lw=1.5)

        ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1], 
                   c=colors_list, marker='x', zorder=1)
        ax.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], 
                   c=colors_list, marker='o', zorder=1)
        
        ax.scatter([], [], c='k', marker='x', label='Start')
        ax.scatter([], [], c='k', marker='o', label='End')
        ax.legend()

    @staticmethod
    def _plot_velocities_on_ax(ax, times, velocities, theoretical_func_or_arr=None, alpha_range=(0.2, 1.0), vel_th=None, time_cmap=None):
        ax.set_xlabel(r'$u~[m.s^{-1}]$')
        ax.set_ylabel(r'$v~[m.s^{-1}]$')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        all_u = velocities[..., 0].flatten()
        all_v = velocities[..., 1].flatten()
        if len(all_u) > 0:
            margin = 0.1 * (all_u.max() - all_u.min() + 1e-6)
            ax.set_xlim(all_u.min() - margin, all_u.max() + margin)
            margin_v = 0.1 * (all_v.max() - all_v.min() + 1e-6)
            ax.set_ylim(all_v.min() - margin_v, all_v.max() + margin_v)

        v_theo_plot = vel_th 
        
        if v_theo_plot is None and callable(theoretical_func_or_arr):
             v_list = [theoretical_func_or_arr(t) for t in times]
             v_theo_plot = np.array(v_list)
        
        for i in range(velocities.shape[1]):
            color = f"C{i%10}"
            
            u_sim = velocities[:, i, 0]
            v_sim = velocities[:, i, 1]
            points = np.column_stack([u_sim, v_sim]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_segs = len(segments)
            
            if time_cmap:
                cmap = plt.get_cmap(time_cmap)
                t_vals = np.linspace(0, 1, n_segs)
                colors = cmap(t_vals)
            else:
                alphas = np.linspace(alpha_range[0], alpha_range[1], n_segs)
                colors = np.zeros((n_segs, 4))
                colors[:] = to_rgba(color)
                colors[:, 3] = alphas

            lc = LineCollection(segments, colors=colors, linewidths=3)
            ax.add_collection(lc)
            
            ax.scatter(u_sim[0], v_sim[0], c=color, marker='x', zorder=1)
            ax.scatter(u_sim[-1], v_sim[-1], c=color, marker='o', zorder=1)
            
            if v_theo_plot is not None:
                if v_theo_plot.ndim == 3 and i < v_theo_plot.shape[1]:
                    ax.plot(v_theo_plot[:, i, 0], v_theo_plot[:, i, 1], '--', color='k', alpha=1., lw=1.5)
                elif v_theo_plot.ndim == 2 and i == 0:
                     ax.plot(v_theo_plot[:, 0], v_theo_plot[:, 1], '--', color='k', alpha=1., lw=1.5)

    @staticmethod
    def _plot_omega_on_ax(ax, times, omegas, theoretical_val=None):
        ax.set_xlabel(r'$t~[s]$')
        ax.set_ylabel(r'$\Omega~[s^{-1}]$')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        for i in range(omegas.shape[1]):
            color = f"C{i%10}"
            # ax.plot(times, omegas[:, i], color=color, label=f'Vortex {i}')
            ax.plot(times, omegas[:, i], color=color)
            
        if theoretical_val is not None:
            if callable(theoretical_val):
                vals = [theoretical_val(t) for t in times]
                ax.plot(times, vals, 'k--', lw=1.5, label='Theory')
            elif isinstance(theoretical_val, (np.ndarray, list)) and np.size(theoretical_val) > 1:
                 vals = np.array(theoretical_val)
                 for j in range(len(vals)):
                     color = f"C{j%10}"
                     ax.axhline(vals[j], color='k', linestyle='--', lw=1.5, alpha=1)
            else:
                ax.axhline(theoretical_val, color='k', linestyle='--', lw=1.5, label='Theory')
        
        # ax.legend()

    @staticmethod
    def plot_convergence(n_steps_list: List[int], 
                         errors: List[float], 
                         slope: float, 
                         intercept: float, 
                         fit_range_mask: np.ndarray,
                         save_fig: Optional[str] = None) -> None:
        """
        Plots the convergence error vs Number of Timesteps (N) and the fitted slope.

        Args:
            n_steps_list: List of N values tested.
            errors: List of computed error metrics.
            slope: Fitted convergence slope (expected -4 for RK4).
            intercept: Fitted intercept for the convergence line.
            fit_range_mask: Boolean array indicating which points were used for the fit.
            save_fig: Filename suffix for saving the plot.
        """
        N = np.array(n_steps_list)
        errors = np.array(errors)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot all data points
        ax.loglog(N, errors, 'o', label='Simulation Error', color='k', alpha=0.6)
        
        # Plot fitted line only within the fit range
        N_fit = N[fit_range_mask]
        if len(N_fit) > 0:
            fit_line = np.exp(intercept) * N_fit**slope
            ax.loglog(N_fit, fit_line, 'k--', label=f'Fit Slope: {slope:.3f}')

        ax.set_xlabel(r'$N_t~[-]$')
        ax.set_ylabel(r'$\mathcal{E}~[m]$')
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        
        if save_fig:
            fig.savefig(f"convergence_{save_fig}.pdf", bbox_inches='tight')
            fig.savefig(f"convergence_{save_fig}.svg", bbox_inches='tight')
            
        plt.show()

    @staticmethod
    def animate_trajectories(times: np.ndarray, 
                             trajectories: np.ndarray, 
                             filename: Optional[str] = None,
                             domain_size: Optional[float] = None) -> animation.FuncAnimation:
        """
        Creates an animation of the vortex motions.

        Args:
            times: Time array.
            trajectories: Position history.
            filename: Output filename (e.g., 'movie.mp4').
            domain_size: Size of the square domain for setting axis limits.

        Returns:
            animation.FuncAnimation: The animation object.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        if domain_size is not None:
            limit = domain_size / 2.0
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
        else:
            x_min, x_max = trajectories[..., 0].min(), trajectories[..., 0].max()
            y_min, y_max = trajectories[..., 1].min(), trajectories[..., 1].max()
            margin = 0.5
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            
        ax.set_xlabel(r'$x~[m]$')
        ax.set_ylabel(r'$y~[m]$')
        ax.grid(True)
        ax.set_aspect('equal')

        lines = [ax.plot([], [], '-', lw=1.5)[0] for _ in range(trajectories.shape[1])]
        points = [ax.plot([], [], 'o')[0] for _ in range(trajectories.shape[1])]
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            time_text.set_text('')
            return lines + points + [time_text]

        def update(frame):
            start_idx = max(0, frame - 50)
            for i, (line, point) in enumerate(zip(lines, points)):
                line.set_data(trajectories[start_idx:frame, i, 0], 
                              trajectories[start_idx:frame, i, 1])
                point.set_data([trajectories[frame, i, 0]], 
                               [trajectories[frame, i, 1]])
            time_text.set_text(f'Time = {times[frame]:.2f}')
            return lines + points + [time_text]

        anim = animation.FuncAnimation(fig, update, frames=len(times),
                                       init_func=init, blit=True, interval=30)
        
        if filename:
            anim.save(filename, writer='ffmpeg', fps=30)
            
        return anim


def run_convergence_study(
    initial_pos: np.ndarray,
    circulations: np.ndarray,
    t_end: float,
    n_steps_range: Tuple[float, float] = (1e1, 1e4),
    num_samples: int = 10,
    fit_range: Tuple[float, float] = (0, np.inf),
    exact_solution_func: Optional[callable] = None,
    box_size: Optional[float] = None,
    save_fig: Optional[str] = None
):
    """
    Runs a convergence study by varying the number of time steps N.

    Executes multiple simulations with increasing temporal resolution (decreasing dt)
    and computes the global error relative to a reference solution (either exact or finest).
    Performs a log-log regression to estimate the order of convergence.

    Args:
        initial_pos: Starting positions (N, 2).
        circulations: Vortex strengths (N,).
        t_end: Final simulation time.
        n_steps_range: Tuple (min_N, max_N) defining the range of steps to test.
        num_samples: Number of simulations to run within the range (log-spaced).
        fit_range: Tuple (min_N, max_N) specifying the range of points to include in the slope fit.
        exact_solution_func: Optional function f(t_end) returning exact positions.
            If None, the simulation with the highest N is used as the reference.
        box_size: Side length of the square domain (if applicable).
        save_fig: Filename suffix for saving the convergence plot.
    """
    # Generate N values logarithmically spaced
    min_n, max_n = n_steps_range
    n_values = np.logspace(np.log10(min_n), np.log10(max_n), num_samples).astype(int)
    n_values = np.unique(n_values)
    n_values = np.sort(n_values)
    dts_list = t_end / n_values
    
    # Progress Bar Variables (Weighted by number of steps)
    total_steps = np.sum(n_values)
    
    final_positions = []
    
    print(f"Running convergence study for N steps: {n_values}")
    
    with tqdm(total=total_steps, desc="Simulation Progress", unit="step") as pbar:
        for dt, n_step in zip(dts_list, n_values):
            sim = PointVortexSimulation(initial_pos, circulations, box_size=box_size)
            _, traj, _ = sim.run(t_end, dt)
            final_positions.append(traj[-1])
            pbar.update(n_step)
    
    errors = []
    
    # Determine reference
    if exact_solution_func:
        ref_pos = exact_solution_func(t_end)
        compare_list = final_positions
        n_plot = n_values
    else:
        # Use finest resolution (last element) as reference
        ref_pos = final_positions[-1]
        compare_list = final_positions[:-1]
        n_plot = n_values[:-1]

    # Calculate errors
    for i, pos in enumerate(compare_list):
        diff = pos - ref_pos
        err = np.sqrt(np.sum(diff**2))
        errors.append(err)
    
    # Linear Regression (Log-Log)
    n_plot = np.array(n_plot)
    errors = np.array(errors)
    
    # Filter valid range for fitting
    fit_mask = (n_plot >= fit_range[0]) & (n_plot <= fit_range[1]) & (errors > 0)
    
    if np.sum(fit_mask) >= 2:
        log_n = np.log(n_plot[fit_mask])
        log_e = np.log(errors[fit_mask])
        slope, intercept = np.polyfit(log_n, log_e, 1)
        print(f"Convergence Slope fitted in range {fit_range}: {slope:.4f}")
    else:
        slope, intercept = np.nan, np.nan
        print("Not enough points in fit range to calculate slope.")
        
    VortexVisualizer.plot_convergence(n_plot, errors, slope, intercept, fit_mask, save_fig=save_fig)


if __name__ == "__main__":
    # Initialization
    
    # Parameters
    B = 1
    N = 10
    TIME_SIM = 10
    DT = 0.001
    
    
    # # (a)
    # domain_size = None
    # positions = np.array([[-B/2, 0], [B/2, 0]])
    # gammas = np.array([1, -1])
    # theoretical_vel = np.array([[0, 1/(2*np.pi*B)], [0, 1/(2*np.pi*B)]])
    # theoretical_omega = np.array([0, 0])
    
    # # (b)
    # domain_size = None
    # positions = np.array([[-B, 0], [B, 0]])
    # gammas = np.array([0.5, 0.5])
    # theoretical_vel = np.array([[0, 0], [0, 0]])
    # theoretical_omega = np.array([1/(8*np.pi*(B**2)), 1/(8*np.pi*(B**2))])
    
    # # (c)
    # domain_size = None
    # positions = VortexInitializer.create_circle(n_vortices=N, radius=B)
    # gammas = np.ones(N)/N
    # theoretical_vel = np.zeros((N, 2))
    # theoretical_omega = np.ones(N)*((N-1)/(N*4*np.pi*(B**2)))
    
    # # (d)
    # domain_size = None
    # positions = VortexInitializer.create_circle(n_vortices=N, radius=B)
    # positions += np.random.normal(loc=0., scale=B/1e3, size=positions.shape)
    # gammas = np.ones(N)/N
    # theoretical_vel = None
    # theoretical_omega = None
    
    # (e)
    domain_size = None
    positions = np.array([[0, 0], [1, 0], [1, 1/np.sqrt(3)]])
    gammas = np.array([1, 1/2, -1/3])
    theoretical_vel = None
    theoretical_omega = None
    
    # # (f) Square Boundaries
    # domain_size = 4
    # positions = np.array([[0, 0], [1, 0], [1, 1/np.sqrt(3)]])
    # gammas = np.array([1, 1/2, -1/3])
    # theoretical_vel = None
    # theoretical_omega = None
    
    # Run Physics
    pvs = PointVortexSimulation(positions=positions, 
                                circulations=gammas, 
                                box_size=domain_size)
                                
    times, trajectories, velocities = pvs.run(t_max=TIME_SIM, dt=DT)
    
    # Analyze
    VortexVisualizer.plot_analysis(times=times, 
                                   trajectories=trajectories, 
                                   velocities=velocities, 
                                   circulations=gammas, 
                                   theoretical_vel=theoretical_vel, 
                                   theoretical_omega=theoretical_omega, 
                                   domain_size=domain_size,
                                   save_fig='2e',
                                   time_cmap='inferno',
                                   )
    
    # Convergence study
    print("\nStarting Convergence Analysis...")
    run_convergence_study(
        initial_pos=positions,
        circulations=gammas,
        t_end=10.0,
        n_steps_range=(1e1, 1e5),
        num_samples=10,
        fit_range=(1e1, 7e2),
        box_size=domain_size,
        save_fig='2e'
    )