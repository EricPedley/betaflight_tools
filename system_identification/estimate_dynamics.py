#!/usr/bin/env python3
"""
Script to estimate system dynamics from flight controller CSV logs.
Loads data with pandas and creates numpy arrays for analysis.
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import scipy
    from scipy.signal import correlate
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install pandas numpy")
    sys.exit(1)


def load_flight_log(csv_path):
    """Load flight log CSV file and extract key columns as numpy arrays."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    print(f"Loading log file: {csv_path}")

    # Load the CSV file with pandas
    df = pd.read_csv(csv_path)

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Filter out data before first throttle activation
    throttle_col = 'rcCommand[3]'
    if throttle_col in df.columns:
        throttle_min = 1000
        # Find first index where throttle exceeds minimum
        first_active_idx = (df[throttle_col] > throttle_min).idxmax()
        if df[throttle_col].iloc[first_active_idx] > throttle_min:
            original_len = len(df)
            df = df.iloc[first_active_idx:].reset_index(drop=True)
            print(f"Filtered data: removed {original_len - len(df)} rows before throttle activation")
        else:
            print(f"Warning: Throttle never exceeds {throttle_min}, keeping all data")
    else:
        print(f"Warning: Throttle column '{throttle_col}' not found")

    # Extract motor output columns as numpy array
    # Average across all 4 motors for a single output value per row
    motor_cols = ['motor[0]', 'motor[1]', 'motor[2]', 'motor[3]']
    if all(col in df.columns for col in motor_cols):
        motor_output = df[motor_cols].mean(axis=1).values
        print(f"\nMotor output shape: {motor_output.shape}")
    else:
        print(f"Warning: Motor columns not found. Available columns: {list(df.columns)}")
        motor_output = None

    # Extract eRPM columns as numpy array (n, 4) shape
    erpm_cols = ['eRPM[0]', 'eRPM[1]', 'eRPM[2]', 'eRPM[3]']
    if all(col in df.columns for col in erpm_cols):
        erpm = df[erpm_cols].values
        print(f"eRPM shape: {erpm.shape}")
    else:
        print(f"Warning: eRPM columns not found. Available columns: {list(df.columns)}")
        erpm = None

    # Extract accelerometer Z column as numpy array
    acc_z_col = 'accSmooth[2]'
    if acc_z_col in df.columns:
        acc_z = df[acc_z_col].values
        print(f"Accelerometer Z shape: {acc_z.shape}")
    else:
        print(f"Warning: Accelerometer Z column '{acc_z_col}' not found. Available columns: {list(df.columns)}")
        acc_z = None

    return df, motor_output, erpm, acc_z

def detect_residual_cutoffs(residuals: np.ndarray):
    """
    Detect cutoff indices by finding the longest contiguous block
    where residuals are within ±threshold of zero (centered at 0).
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

def estimate_time_delay(erpm_mean: np.ndarray, acc_z: np.ndarray, sample_rate: float):
    """
    Estimate time delay between eRPM and accelerometer using cross-correlation.

    Args:
        erpm_mean: Mean eRPM across 4 motors (1D array)
        acc_z: Accelerometer Z readings (1D array)
        sample_rate: Sample rate in Hz

    Returns:
        delay_ms: Estimated delay in milliseconds
        max_corr: Maximum correlation coefficient
    """
    # Detrend and normalize signals
    erpm_normalized = erpm_mean - np.mean(erpm_mean)
    acc_z_normalized = acc_z - np.mean(acc_z)

    # Compute cross-correlation
    correlation = correlate(acc_z_normalized, erpm_normalized, mode='full')
    lags = np.arange(-len(acc_z) + 1, len(acc_z))

    # Find lag with maximum correlation
    max_corr_idx = np.argmax(np.abs(correlation))
    delay_samples = lags[max_corr_idx]
    max_corr = correlation[max_corr_idx] / (np.linalg.norm(erpm_normalized) * np.linalg.norm(acc_z_normalized))

    # Convert to milliseconds
    delay_ms = (delay_samples / sample_rate) * 1000

    return delay_ms, max_corr, correlation, lags

def validate_delay_compensation(erpm: np.ndarray, residuals_trimmed: np.ndarray, residuals_shifted: np.ndarray, sample_rate: float, delay_ms: float, plot: bool = True):
    """
    Validate delay compensation by computing cross-correlations between residuals and individual eRPM channels,
    and analyze autocorrelation of residuals.

    Args:
        erpm: eRPM array with shape (n, 4) for 4 motors
        residuals_trimmed: Residuals before delay compensation
        residuals_shifted: Residuals after delay compensation
        sample_rate: Sample rate in Hz
        delay_ms: Estimated delay in milliseconds
        plot: Whether to generate validation plots
    """
    # Compute cross-correlations for each motor
    motor_names = ['M1', 'M2', 'M3', 'M4']
    xcorr_results_trimmed = []
    xcorr_results_shifted = []

    for i in range(4):
        erpm_ch = erpm[:, i]
        erpm_normalized = erpm_ch - np.mean(erpm_ch)

        # Cross-correlation with original residuals
        corr_trimmed = correlate(residuals_trimmed - np.mean(residuals_trimmed), erpm_normalized, mode='full')
        xcorr_results_trimmed.append(corr_trimmed)

        # Cross-correlation with delay-compensated residuals
        corr_shifted = correlate(residuals_shifted - np.mean(residuals_shifted), erpm_normalized, mode='full')
        xcorr_results_shifted.append(corr_shifted)

        # Normalize and find peak
        peak_trimmed = np.max(np.abs(corr_trimmed)) / (np.linalg.norm(erpm_normalized) * np.linalg.norm(residuals_trimmed))
        peak_shifted = np.max(np.abs(corr_shifted)) / (np.linalg.norm(erpm_normalized) * np.linalg.norm(residuals_shifted))

        print(f"Motor {i}: Residual-eRPM correlation (original: {peak_trimmed:.4f}, delay-compensated: {peak_shifted:.4f})")

    # Compute autocorrelation of residuals
    residuals_trimmed_normalized = residuals_trimmed - np.mean(residuals_trimmed)
    residuals_shifted_normalized = residuals_shifted - np.mean(residuals_shifted)

    autocorr_trimmed = correlate(residuals_trimmed_normalized, residuals_trimmed_normalized, mode='full')
    autocorr_shifted = correlate(residuals_shifted_normalized, residuals_shifted_normalized, mode='full')

    # Normalize by maximum (at zero lag)
    autocorr_trimmed = autocorr_trimmed / np.max(autocorr_trimmed)
    autocorr_shifted = autocorr_shifted / np.max(autocorr_shifted)

    # Calculate statistics at key lags
    center_idx = len(autocorr_trimmed) // 2
    lag_1_idx = center_idx + 1
    lag_10_idx = center_idx + 10
    lag_100_idx = center_idx + 100

    print(f"\nResidual Autocorrelation Analysis:")
    print(f"  Original residuals: lag-1={autocorr_trimmed[lag_1_idx]:.4f}, lag-10={autocorr_trimmed[lag_10_idx]:.4f}, lag-100={autocorr_trimmed[lag_100_idx]:.4f}")
    print(f"  Delay-compensated:  lag-1={autocorr_shifted[lag_1_idx]:.4f}, lag-10={autocorr_shifted[lag_10_idx]:.4f}, lag-100={autocorr_shifted[lag_100_idx]:.4f}")

    if plot:
        figures_dir = Path('figures')
        figures_dir.mkdir(exist_ok=True)

        lags = np.arange(-len(residuals_trimmed) + 1, len(residuals_trimmed))
        lags_ms = (lags / sample_rate) * 1000

        # Plot 1: Cross-correlation with individual eRPM channels (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Correlation: Residuals vs Individual eRPM Channels', fontsize=14)

        for i, ax in enumerate(axes.flat):
            motor_name = motor_names[i]
            ax.plot(lags_ms, xcorr_results_trimmed[i], label='Original residuals', linewidth=1, alpha=0.7)
            ax.plot(lags_ms, xcorr_results_shifted[i], label='Delay-compensated residuals', linewidth=1, alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Zero lag')
            ax.axvline(x=delay_ms, color='orange', linestyle='--', alpha=0.5, label=f'Detected delay: {delay_ms:.3f} ms')
            ax.set_xlabel('Lag (ms)')
            ax.set_ylabel('Cross-Correlation')
            ax.set_title(f'{motor_name}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(figures_dir / 'xcorr_residuals_vs_erpm.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {figures_dir / 'xcorr_residuals_vs_erpm.png'}")
        plt.close(fig)

        # Plot 2: Autocorrelation of residuals (zoomed and full views)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Autocorrelation of Residuals', fontsize=14)

        # Full autocorrelation view (limited to ±500ms to reduce noise)
        lags_full = lags_ms
        max_lag_ms = 500
        max_lag_idx = np.searchsorted(lags_full, max_lag_ms, side='left')
        min_lag_idx = np.searchsorted(lags_full, -max_lag_ms, side='right')

        ax = axes[0]
        ax.plot(lags_full[min_lag_idx:max_lag_idx], autocorr_trimmed[min_lag_idx:max_lag_idx],
                label='Original residuals', linewidth=1, alpha=0.7)
        ax.plot(lags_full[min_lag_idx:max_lag_idx], autocorr_shifted[min_lag_idx:max_lag_idx],
                label='Delay-compensated residuals', linewidth=1, alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Zero lag')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Normalized Autocorrelation')
        ax.set_title('Full View (±500 ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Zoomed view (±50ms)
        max_lag_ms_zoom = 50
        max_lag_idx_zoom = np.searchsorted(lags_full, max_lag_ms_zoom, side='left')
        min_lag_idx_zoom = np.searchsorted(lags_full, -max_lag_ms_zoom, side='right')

        ax = axes[1]
        ax.plot(lags_full[min_lag_idx_zoom:max_lag_idx_zoom], autocorr_trimmed[min_lag_idx_zoom:max_lag_idx_zoom],
                label='Original residuals', linewidth=1.5, alpha=0.7)
        ax.plot(lags_full[min_lag_idx_zoom:max_lag_idx_zoom], autocorr_shifted[min_lag_idx_zoom:max_lag_idx_zoom],
                label='Delay-compensated residuals', linewidth=1.5, alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Zero lag')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Normalized Autocorrelation')
        ax.set_title('Zoomed View (±50 ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(figures_dir / 'autocorr_residuals.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {figures_dir / 'autocorr_residuals.png'}")
        plt.close(fig)

def estimate_thrust_curve(erpm: np.ndarray, acc_z: np.ndarray, plot: bool = True, sample_rate: float = None):
    # Calculate residuals from full dataset
    A_full = np.array([
        [1, np.sum(omega), np.sum(np.square(omega))]
        for omega in erpm
    ])
    b = acc_z
    i_start = len(erpm)//4
    i_end = 3*len(erpm)//4
    x_initial, _, _, _ = np.linalg.lstsq(A_full[i_start:i_end], b[i_start:i_end], rcond=None)
    estimated_thrust_full = A_full @ x_initial
    residuals_full = b - estimated_thrust_full

    # Detect cutoffs based on residuals
    start_idx, end_idx = detect_residual_cutoffs(residuals_full)
    print(f"Detected cutoffs: start={start_idx}, end={end_idx}")
    print(f"Using {end_idx - start_idx + 1} samples out of {len(erpm)}")

    # Re-fit model using only trimmed data
    A_trimmed = A_full[start_idx:end_idx+1]
    b_trimmed = b[start_idx:end_idx+1]
    x, residuals, rank, s = np.linalg.lstsq(A_trimmed, b_trimmed, rcond=None)

    # Estimate time delay on trimmed data using cross-correlation
    # Calculate mean eRPM across 4 motors
    erpm_trimmed = erpm[start_idx:end_idx+1]
    erpm_mean = np.mean(erpm_trimmed, axis=1)

    # Get sample rate from time column if available (assuming uniform sampling)
    # Default to 8 kHz if not provided (typical for flight controllers)
    if sample_rate is None:
        sample_rate = 8000.0  # Hz, typical for flight controllers

    delay_ms, max_corr, _, _ = estimate_time_delay(erpm_mean, b_trimmed, sample_rate)
    print(f"Estimated time delay: {delay_ms:.3f} ms (correlation: {max_corr:.4f})")

    # Convert delay to samples and shift eRPM to align with accelerometer
    delay_samples = int(np.round(delay_ms * sample_rate / 1000))
    print(f"Delay in samples: {delay_samples}")

    # Apply delay compensation by shifting eRPM
    A_trimmed_shifted = A_full[start_idx:end_idx+1].copy()
    erpm_trimmed_shifted = np.roll(erpm_trimmed, -delay_samples, axis=0)  # Negative to advance eRPM
    A_trimmed_shifted = np.array([
        [1, np.sum(omega), np.sum(np.square(omega))]
        for omega in erpm_trimmed_shifted
    ])

    # Re-fit model with shifted eRPM
    x_shifted, _, _, _ = np.linalg.lstsq(A_trimmed_shifted, b_trimmed, rcond=None)
    estimated_thrust_shifted = A_trimmed_shifted @ x_shifted
    residuals_shifted = b_trimmed - estimated_thrust_shifted

    print(f"Original model residual std: {np.std(residuals_full[start_idx:end_idx+1]):.4f}")
    print(f"Delay-compensated residual std: {np.std(residuals_shifted):.4f}")

    if plot:
        figures_dir = Path('figures')
        figures_dir.mkdir(exist_ok=True)

        # Calculate estimated thrust using trimmed-data coefficients
        estimated_thrust = A_full @ x
        residuals_trimmed = b_trimmed - (A_trimmed @ x)

        # Calculate residual statistics for threshold lines (centered at 0)
        std_residuals = np.std(residuals_full)

        # Plot 1: Residuals time series with cutoffs marked
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(residuals_full, linewidth=0.8, label='Initial residuals')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero error')
        ax.axhline(y=1 * std_residuals, color='orange', linestyle=':', alpha=0.7, label='±1σ')
        ax.axhline(y=-1 * std_residuals, color='orange', linestyle=':', alpha=0.7)
        ax.axhline(y=2 * std_residuals, color='purple', linestyle=':', alpha=0.7, label='±2σ')
        ax.axhline(y=-2 * std_residuals, color='purple', linestyle=':', alpha=0.7)
        ax.axvline(x=start_idx, color='g', linestyle='-', linewidth=2, label='Cutoff start')
        ax.axvline(x=end_idx, color='orange', linestyle='-', linewidth=2, label='Cutoff end')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Residual (Actual - Estimated)')
        ax.set_title('Residuals Time Series with Detected Cutoffs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(figures_dir / 'residuals_time_series.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {figures_dir / 'residuals_time_series.png'}")
        plt.close(fig)

        # Plot 2: Residuals after trimming
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(residuals_trimmed, linewidth=0.8, color='green')
        ax.axhline(y=0, color='r', linestyle='--', label='Zero error')
        ax.set_xlabel('Sample Index (trimmed data)')
        ax.set_ylabel('Residual (Actual - Estimated)')
        ax.set_title('Residuals Time Series (Trimmed Data with Final Model)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(figures_dir / 'residuals_time_series_trimmed.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {figures_dir / 'residuals_time_series_trimmed.png'}")
        plt.close(fig)

        # Plot 2b: Residuals with delay compensation
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(residuals_trimmed, linewidth=0.8, color='blue', label='Without delay compensation', alpha=0.6)
        ax.plot(residuals_shifted, linewidth=0.8, color='green', label='With delay compensation')
        ax.axhline(y=0, color='r', linestyle='--', label='Zero error')
        ax.set_xlabel('Sample Index (trimmed data)')
        ax.set_ylabel('Residual (Actual - Estimated)')
        ax.set_title(f'Residuals Comparison: Original vs Delay-Compensated (Delay: {delay_ms:.3f} ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(figures_dir / 'residuals_time_series_delay_compensated.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {figures_dir / 'residuals_time_series_delay_compensated.png'}")
        plt.close(fig)

        # Plot 3: Thrust estimation (using trimmed data coefficients)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(b, estimated_thrust, alpha=0.5, s=10, label='Estimated vs Actual')
        ax.plot([b.min(), b.max()], [b.min(), b.max()], 'r--', label='Perfect fit')
        ax.set_xlabel('Actual Thrust (acc_z)')
        ax.set_ylabel('Estimated Thrust')
        ax.set_title('Thrust Curve Estimation: Estimated vs Actual (Trimmed Data Model)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(figures_dir / 'thrust_curve_estimation.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {figures_dir / 'thrust_curve_estimation.png'}")
        plt.close(fig)

        # Validation: Cross-correlation between residuals and individual eRPM channels
        print("\n=== Residual-eRPM Cross-Correlation Validation ===")
        validate_delay_compensation(erpm_trimmed, residuals_trimmed, residuals_shifted, sample_rate, delay_ms, plot=True)

    return x


def main():
    parser = argparse.ArgumentParser(
        description="Estimate system dynamics from flight controller CSV logs"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV log file"
    )

    args = parser.parse_args()
    _, motor_output, erpm, acc_z = load_flight_log(args.csv_file)
    estimate_thrust_curve(erpm, acc_z, plot=True)

    # Example: Print first few values
    if motor_output is not None:
        print(f"\nFirst 10 motor output values: {motor_output[:10]}")
    if erpm is not None:
        print(f"First 10 eRPM values: {erpm[:10]}")
    if acc_z is not None:
        print(f"First 10 accelerometer Z values: {acc_z[:10]}")


if __name__ == "__main__":
    main()
