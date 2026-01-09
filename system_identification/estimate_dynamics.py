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

def estimate_thrust_curve(erpm: np.ndarray, acc_z: np.ndarray, plot: bool = True):
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
