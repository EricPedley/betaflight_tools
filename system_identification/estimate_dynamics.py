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

    # Extract motor output columns as numpy array
    # Average across all 4 motors for a single output value per row
    motor_cols = ['motor[0]', 'motor[1]', 'motor[2]', 'motor[3]']
    if all(col in df.columns for col in motor_cols):
        motor_output = df[motor_cols].mean(axis=1).values
        print(f"\nMotor output shape: {motor_output.shape}")
    else:
        print(f"Warning: Motor columns not found. Available columns: {list(df.columns)}")
        motor_output = None

    # Extract eRPM columns as numpy array
    # Average across all 4 motors for a single eRPM value per row
    erpm_cols = ['eRPM[0]', 'eRPM[1]', 'eRPM[2]', 'eRPM[3]']
    if all(col in df.columns for col in erpm_cols):
        erpm = df[erpm_cols].mean(axis=1).values
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

    # Example: Print first few values
    if motor_output is not None:
        print(f"\nFirst 10 motor output values: {motor_output[:10]}")
    if erpm is not None:
        print(f"First 10 eRPM values: {erpm[:10]}")
    if acc_z is not None:
        print(f"First 10 accelerometer Z values: {acc_z[:10]}")


if __name__ == "__main__":
    main()
