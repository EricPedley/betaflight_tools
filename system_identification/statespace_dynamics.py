#!/usr/bin/env python3
"""
Linear state-space model fitting for Betaflight flight controller logs.

Model:
    x[k+1] = A @ x[k] + B @ u[k]
    z[k]   = C @ x[k]

Where:
    u: motor commands (4 dims)
    z: observations (12 dims): gyro[3], acc[3], vbat, current, eRPM[4]
    x: hidden state (tunable dimension, typically 8-32)
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Channel names for plotting
CHANNEL_NAMES = [
    'gyro_x', 'gyro_y', 'gyro_z',
    'acc_x', 'acc_y', 'acc_z',
    'vbat', 'current',
    'eRPM_0', 'eRPM_1', 'eRPM_2', 'eRPM_3',
]

INPUT_DIM = 4   # motor[0-3]
OUTPUT_DIM = 12  # gyro[3] + acc[3] + vbat + current + eRPM[4]


def load_flight_log(csv_path: str | Path) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load flight log CSV and extract state-space inputs/outputs.

    Args:
        csv_path: Path to Betaflight CSV log file.

    Returns:
        u: (T, 4) motor PWM commands
        z: (T, 12) observations [gyro(3), acc(3), vbat(1), current(1), rpm(4)]
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    print(f"Loading log file: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Filter pre-throttle data
    if 'rcCommand[3]' in df.columns:
        throttle_min = 1000
        first_active = (df['rcCommand[3]'] > throttle_min).idxmax()
        if df['rcCommand[3]'].iloc[first_active] > throttle_min:
            original_len = len(df)
            df = df.iloc[first_active:].reset_index(drop=True)
            print(f"Filtered: removed {original_len - len(df)} rows before throttle activation")

    # Input: motor commands
    u_cols = ['motor[0]', 'motor[1]', 'motor[2]', 'motor[3]']
    if not all(col in df.columns for col in u_cols):
        print(f"Error: Missing motor columns. Available: {list(df.columns)}")
        sys.exit(1)
    u = df[u_cols].values

    # Output: stack all observation channels
    z_cols = [
        'gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]',
        'accSmooth[0]', 'accSmooth[1]', 'accSmooth[2]',
        'vbatLatest (V)', 'amperageLatest (A)',
        'eRPM[0]', 'eRPM[1]', 'eRPM[2]', 'eRPM[3]',
    ]
    missing_cols = [col for col in z_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing observation columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    z = df[z_cols].values

    print(f"Input shape: {u.shape}, Output shape: {z.shape}")

    return jnp.array(u, dtype=jnp.float32), jnp.array(z, dtype=jnp.float32)


def normalize_data(
    data: jnp.ndarray,
    mean: jnp.ndarray | None = None,
    std: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Z-score normalize data per channel.

    Args:
        data: (T, D) array to normalize
        mean: Optional pre-computed mean (for test data)
        std: Optional pre-computed std (for test data)

    Returns:
        normalized: (T, D) normalized data
        mean: (D,) channel means
        std: (D,) channel stds
    """
    if mean is None:
        mean = jnp.mean(data, axis=0)
    if std is None:
        std = jnp.std(data, axis=0)
        std = jnp.where(std < 1e-8, 1.0, std)  # Avoid division by zero

    normalized = (data - mean) / std
    return normalized, mean, std


def detect_residual_cutoffs(residuals: np.ndarray) -> tuple[int, int]:
    """
    Detect cutoff indices by finding the longest contiguous block
    where residuals are within ±threshold of zero (centered at 0).

    Args:
        residuals: (T,) array of residuals

    Returns:
        start_idx: Start index of good data region
        end_idx: End index of good data region
    """
    std = np.std(residuals)
    threshold = 1.3 * std

    # Find where residuals are within ±threshold of zero
    within_threshold = np.abs(residuals) <= threshold

    if not np.any(within_threshold):
        # No data within threshold, return full range
        return 0, len(residuals) - 1

    # Find contiguous blocks using diff
    # Add padding to handle edges
    padded = np.pad(within_threshold, (1, 1), constant_values=False)
    diff = np.diff(padded.astype(int))

    # Start indices where diff == 1 (False -> True)
    starts = np.where(diff == 1)[0]
    # End indices where diff == -1 (True -> False)
    ends = np.where(diff == -1)[0]

    # Find longest block
    block_lengths = ends - starts
    longest_idx = np.argmax(block_lengths)

    start_idx = starts[longest_idx]
    end_idx = ends[longest_idx]

    return start_idx, end_idx


def stable_random_init(
    key: jax.Array,
    state_dim: int,
    input_dim: int = INPUT_DIM,
    output_dim: int = OUTPUT_DIM,
    spectral_radius: float = 0.95,
) -> dict[str, jnp.ndarray]:
    """
    Initialize state-space parameters with A having controlled spectral radius.

    Args:
        key: JAX random key
        state_dim: Hidden state dimension
        input_dim: Input dimension (4 for motor commands)
        output_dim: Output dimension (12 for observations)
        spectral_radius: Target spectral radius for A (< 1 for stability)

    Returns:
        params: Dict with 'A', 'B', 'C' matrices
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Initialize A with controlled spectral radius
    A = jax.random.normal(k1, (state_dim, state_dim)) / jnp.sqrt(state_dim)
    eigvals = jnp.linalg.eigvals(A)
    current_radius = jnp.max(jnp.abs(eigvals))
    A = A * (spectral_radius / (current_radius + 1e-8))

    # Initialize B and C with small values
    B = jax.random.normal(k2, (state_dim, input_dim)) * 0.1
    C = jax.random.normal(k3, (output_dim, state_dim)) * 0.1

    return {'A': A, 'B': B, 'C': C}


def simulate_statespace(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    u_seq: jnp.ndarray,
    x0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simulate state-space model using jax.lax.scan.

    Args:
        params: Dict with 'A', 'B', 'C' matrices
        u_seq: (T, input_dim) input sequence
        x0: (state_dim,) initial state, defaults to zeros

    Returns:
        z_seq: (T, output_dim) output sequence
    """


    def step(x, u):
        z = C @ x
        x_next = A @ x + B @ u
        return x_next, z

    _, z_seq = jax.lax.scan(step, x0, u_seq)
    return z_seq


# JIT-compiled version for efficiency
simulate_statespace_jit = jax.jit(simulate_statespace)


def trim_data_by_residuals(
    u_data: jnp.ndarray,
    z_data: jnp.ndarray,
    state_dim: int = 8,
    maxiter_initial: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, int, int]:
    """
    Fit initial model and trim data to most consistent region.

    Args:
        u_data: (T, 4) input data
        z_data: (T, 12) output data
        state_dim: State dimension for initial fit (smaller = faster)
        maxiter_initial: Max iterations for initial fit
        seed: Random seed
        verbose: Print progress

    Returns:
        u_trimmed: Trimmed input data
        z_trimmed: Trimmed output data
        start_idx: Start index of trimmed region
        end_idx: End index of trimmed region
    """
    if verbose:
        print("\n=== Trimming data by residual analysis ===")
        print(f"  Initial data length: {len(u_data)} samples")

    # Fit quick initial model on middle 50% of data
    mid_start = len(u_data) // 4
    mid_end = 3 * len(u_data) // 4

    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    init_params = stable_random_init(key, state_dim)
    n_x = init_params['A'].shape[0]
    n_u = init_params['B'].shape[1]
    n_z = init_params['C'].shape[0]
    init_params_flat = jnp.concatenate([
        init_params['A'].flatten(),
        init_params['B'].flatten(),
        init_params['C'].flatten(),
    ])

    # Fit on middle section
    def residual_fn(params: jnp.ndarray):
        A = params[:n_x*n_x].reshape((n_x, n_x))
        B = params[n_x*n_x: n_x*n_x+n_x*n_u].reshape((n_x, n_u))
        C = params[n_x*n_x+n_x*n_u:].reshape((n_z, n_x))
        z_pred = simulate_statespace(A, B, C, u_data[mid_start:mid_end], jnp.zeros(A.shape[0]))
        return (z_pred - z_data[mid_start:mid_end]).ravel()

    solver = jaxopt.LevenbergMarquardt(
        residual_fun=residual_fn,
        maxiter=maxiter_initial,
        tol=1e-6,
        verbose=False,
    )
    result = solver.run(init_params_flat)

    # Extract fitted params
    params_flat = result.params
    A = params_flat[:n_x*n_x].reshape((n_x, n_x))
    B = params_flat[n_x*n_x: n_x*n_x+n_x*n_u].reshape((n_x, n_u))
    C = params_flat[n_x*n_x+n_x*n_u:].reshape((n_z, n_x))

    # Compute residuals on full data
    z_pred_full = simulate_statespace(A, B, C, u_data, jnp.zeros(A.shape[0]))
    residuals_full = z_data - z_pred_full

    # Use mean squared residual across all channels as the metric
    residuals_mean = np.array(jnp.mean(residuals_full ** 2, axis=1))

    # Detect cutoffs
    start_idx, end_idx = detect_residual_cutoffs(residuals_mean)

    if verbose:
        print(f"  Detected cutoffs: start={start_idx}, end={end_idx}")
        print(f"  Trimmed data length: {end_idx - start_idx + 1} samples")
        print(f"  Kept {100 * (end_idx - start_idx + 1) / len(u_data):.1f}% of data")

    u_trimmed = u_data[start_idx:end_idx+1]
    z_trimmed = z_data[start_idx:end_idx+1]

    return u_trimmed, z_trimmed, start_idx, end_idx


def fit_statespace_model(
    u_data: jnp.ndarray,
    z_data: jnp.ndarray,
    state_dim: int = 16,
    maxiter: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[dict[str, jnp.ndarray], any]:
    """
    Fit linear state-space model using jaxopt.LevenbergMarquardt.

    Args:
        u_data: (T, 4) normalized motor commands
        z_data: (T, 12) normalized observations
        state_dim: Hidden state dimension
        maxiter: Maximum optimization iterations
        seed: Random seed for initialization
        verbose: Print progress

    Returns:
        params: Fitted parameters {'A', 'B', 'C'}
        opt_state: Optimization state from jaxopt
    """
    if verbose:
        print(f"\nFitting state-space model:")
        print(f"  State dimension: {state_dim}")
        print(f"  Data length: {len(u_data)} samples")
        print(f"  Max iterations: {maxiter}")

    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    init_params = stable_random_init(key, state_dim)
    n_x = init_params['A'].shape[0]
    n_u = init_params['B'].shape[1]
    n_z = init_params['C'].shape[0]
    init_params_flat = jnp.concatenate([
        init_params['A'].flatten(),
        init_params['B'].flatten(),
        init_params['C'].flatten(),
    ])

    if verbose:
        stability = check_stability(init_params['A'])
        print(f"  Initial spectral radius: {stability['spectral_radius']:.4f}")

    # Define residual function for LM solver
    def residual_fn(params: jnp.ndarray):
        A = params[:n_x*n_x].reshape((n_x, n_x))
        B = params[n_x*n_x: n_x*n_x+n_x*n_u].reshape((n_x, n_u))
        C = params[n_x*n_x+n_x*n_u:].reshape((n_z, n_x))
        z_pred = simulate_statespace(A,B,C, u_data, jnp.zeros(A.shape[0]))
        return (z_pred - z_data).ravel()

    # Create and run solver
    solver = jaxopt.LevenbergMarquardt(
        residual_fun=residual_fn,
        maxiter=maxiter,
        tol=1e-6,
        verbose=verbose,
    )

    result = solver.run(init_params_flat)

    if verbose:
        stability = check_stability(result.params[:n_x*n_x].reshape((n_x, n_x)))
        print(f"  Final spectral radius: {stability['spectral_radius']:.4f}")
        print(f"  Stable: {stability['is_stable']}")

    return result.params, result.state


def check_stability(A: jnp.ndarray) -> dict:
    """
    Analyze stability of state transition matrix.

    Args:
        A: (state_dim, state_dim) state transition matrix

    Returns:
        Dict with eigenvalues, spectral_radius, and is_stable flag
    """
    eigvals = jnp.linalg.eigvals(A)
    spectral_radius = float(jnp.max(jnp.abs(eigvals)))

    return {
        'eigenvalues': eigvals,
        'spectral_radius': spectral_radius,
        'is_stable': spectral_radius < 1.0,
    }


def compute_fit_metrics(
    z_true: jnp.ndarray,
    z_pred: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """
    Compute fit quality metrics.

    Args:
        z_true: (T, output_dim) true observations
        z_pred: (T, output_dim) predicted observations

    Returns:
        Dict with RMSE and R^2 per channel
    """
    residuals = z_true - z_pred
    ss_res = jnp.sum(residuals ** 2, axis=0)
    ss_tot = jnp.sum((z_true - jnp.mean(z_true, axis=0)) ** 2, axis=0)

    return {
        'rmse_per_channel': jnp.sqrt(jnp.mean(residuals ** 2, axis=0)),
        'r2_per_channel': 1 - ss_res / (ss_tot + 1e-10),
        'overall_rmse': float(jnp.sqrt(jnp.mean(residuals ** 2))),
        'overall_r2': float(1 - jnp.sum(ss_res) / (jnp.sum(ss_tot) + 1e-10)),
    }


def plot_results(
    z_true: jnp.ndarray,
    z_pred: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    channel_names: list[str],
    save_dir: Path,
    max_samples: int = 2000,
) -> None:
    """
    Generate diagnostic plots for state-space model fit.

    Args:
        z_true: (T, output_dim) true observations
        z_pred: (T, output_dim) predicted observations
        params: Fitted parameters {'A', 'B', 'C'}
        channel_names: Names for each output channel
        save_dir: Directory to save plots
        max_samples: Max samples to plot (for readability)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Limit samples for plotting
    n_plot = min(len(z_true), max_samples)
    z_true_plot = np.array(z_true[:n_plot])
    z_pred_plot = np.array(z_pred[:n_plot])

    # Plot 1: Time series comparison for each channel (4x3 grid)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('State-Space Model: Predicted vs True', fontsize=14)

    for i, (ax, name) in enumerate(zip(axes.flat, channel_names)):
        ax.plot(z_true_plot[:, i], 'b-', alpha=0.7, linewidth=0.8, label='True')
        ax.plot(z_pred_plot[:, i], 'r--', alpha=0.7, linewidth=0.8, label='Predicted')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'statespace_fit.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_dir / 'statespace_fit.png'}")
    plt.close()

    # Plot 2: Eigenvalue plot
    fig, ax = plt.subplots(figsize=(8, 8))
    eigvals = np.array(jnp.linalg.eigvals(params['A']))
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')
    ax.scatter(np.real(eigvals), np.imag(eigvals), s=100, c='red', zorder=5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Eigenvalues of A (Unit Circle = Stability Boundary)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'eigenvalues.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_dir / 'eigenvalues.png'}")
    plt.close()

    # Plot 3: R^2 per channel bar chart
    metrics = compute_fit_metrics(z_true, z_pred)
    r2_values = np.array(metrics['r2_per_channel'])

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(channel_names, r2_values)

    # Color bars by R^2 value
    for bar, r2 in zip(bars, r2_values):
        if r2 > 0.8:
            bar.set_color('green')
        elif r2 > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Good fit (R²=0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate fit (R²=0.5)')
    ax.set_ylabel('R² Score')
    ax.set_title('Fit Quality per Channel')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_dir / 'r2_per_channel.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_dir / 'r2_per_channel.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fit linear state-space model to Betaflight flight log"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV log file"
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=16,
        help="Hidden state dimension (default: 16, recommended range: 8-32)"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum optimization iterations (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization (default: 42)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation"
    )
    parser.add_argument(
        "--trim-data",
        action="store_true",
        help="Trim data to most consistent region using residual analysis"
    )

    args = parser.parse_args()

    # Load data
    u, z = load_flight_log(args.csv_file)

    # Normalize
    u_norm, u_mean, u_std = normalize_data(u)
    z_norm, z_mean, z_std = normalize_data(z)

    print(f"\nInput stats - mean: {np.array(u_mean)}, std: {np.array(u_std)}")
    print(f"Output stats - mean range: [{float(z_mean.min()):.2f}, {float(z_mean.max()):.2f}]")

    # Optional: Trim data to best region
    if args.trim_data:
        u_norm, z_norm, start_idx, end_idx = trim_data_by_residuals(
            u_norm, z_norm,
            state_dim=8,  # Use smaller state for initial fit
            maxiter_initial=20,
            seed=args.seed,
            verbose=True
        )

    # Fit model
    params_flat, opt_state = fit_statespace_model(
        u_norm, z_norm,
        state_dim=args.state_dim,
        maxiter=args.maxiter,
        seed=args.seed,
        verbose=True
    )

    # Extract parameters from flat array
    n_x = args.state_dim
    n_u = INPUT_DIM
    n_z = OUTPUT_DIM
    A = params_flat[:n_x*n_x].reshape((n_x, n_x))
    B = params_flat[n_x*n_x: n_x*n_x+n_x*n_u].reshape((n_x, n_u))
    C = params_flat[n_x*n_x+n_x*n_u:].reshape((n_z, n_x))
    params = {'A': A, 'B': B, 'C': C}

    # Compute predictions and metrics
    z_pred = simulate_statespace(A, B, C, u_norm, jnp.zeros(n_x))
    metrics = compute_fit_metrics(z_norm, z_pred)

    # Print results
    print("\n=== Fit Results ===")
    print(f"Overall R²: {metrics['overall_r2']:.4f}")
    print(f"Overall RMSE: {metrics['overall_rmse']:.4f}")
    print("\nPer-channel R²:")
    for name, r2 in zip(CHANNEL_NAMES, metrics['r2_per_channel']):
        print(f"  {name}: {float(r2):.4f}")

    # Check stability
    stability = check_stability(params['A'])
    print(f"\nStability:")
    print(f"  Spectral radius: {stability['spectral_radius']:.4f}")
    print(f"  Is stable: {stability['is_stable']}")

    # Generate plots
    if not args.no_plot:
        plot_results(z_norm, z_pred, params, CHANNEL_NAMES, Path('figures'))

    # Print matrix shapes
    print("\n=== Model Parameters ===")
    print(f"A shape: {params['A'].shape}")
    print(f"B shape: {params['B'].shape}")
    print(f"C shape: {params['C'].shape}")


if __name__ == "__main__":
    main()
