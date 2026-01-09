#!/usr/bin/env python3
"""
Script to visualize flight controller CSV logs using Rerun.

Usage:
    python log_to_rerun.py <path_to_csv_file>
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import rerun as rr
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install rerun-sdk numpy")
    sys.exit(1)


def parse_csv_header(header_row):
    """Parse the CSV header to extract column names."""
    # Clean up column names (remove spaces, handle special characters)
    columns = [col.strip() for col in header_row]
    return columns


def convert_value(value_str):
    """Convert a string value to appropriate numeric type or keep as string."""
    value_str = value_str.strip()

    # Handle empty values
    if not value_str:
        return None

    # Try to convert to number
    try:
        # Try integer first
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        # Keep as string (for flags, status values, etc.)
        return value_str


def log_to_rerun(csv_path):
    """Read CSV file and log all data to Rerun."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Initialize Rerun
    rr.init("flight_log_viewer", spawn=True)

    print(f"Loading log file: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        # Read header
        header = next(reader)
        columns = parse_csv_header(header)

        print(f"Found {len(columns)} columns")

        # Track numeric columns for time series
        numeric_columns = set()

        # Read and log data
        row_count = 0
        for row in reader:
            if len(row) != len(columns):
                continue  # Skip malformed rows

            # Create dictionary of column -> value
            data = {}
            for col_name, value_str in zip(columns, row):
                value = convert_value(value_str)
                data[col_name] = value

                # Track which columns have numeric data
                if isinstance(value, (int, float)):
                    numeric_columns.add(col_name)

            # Use time as the timeline
            time_us = data.get('time (us)')
            if time_us is None:
                continue

            # Set timeline
            rr.set_time("loop_iteration", sequence=data.get('loopIteration', row_count))
            rr.set_time("time", timestamp=time_us * 1e-6)  # Convert microseconds to seconds

            # Log PID values
            if all(key in data for key in ['axisP[0]', 'axisP[1]', 'axisP[2]']):
                rr.log("pid/P/roll", rr.Scalars(data['axisP[0]']))
                rr.log("pid/P/pitch", rr.Scalars(data['axisP[1]']))
                rr.log("pid/P/yaw", rr.Scalars(data['axisP[2]']))

            if all(key in data for key in ['axisI[0]', 'axisI[1]', 'axisI[2]']):
                rr.log("pid/I/roll", rr.Scalars(data['axisI[0]']))
                rr.log("pid/I/pitch", rr.Scalars(data['axisI[1]']))
                rr.log("pid/I/yaw", rr.Scalars(data['axisI[2]']))

            if all(key in data for key in ['axisD[0]', 'axisD[1]']):
                rr.log("pid/D/roll", rr.Scalars(data['axisD[0]']))
                rr.log("pid/D/pitch", rr.Scalars(data['axisD[1]']))

            if all(key in data for key in ['axisF[0]', 'axisF[1]', 'axisF[2]']):
                rr.log("pid/F/roll", rr.Scalars(data['axisF[0]']))
                rr.log("pid/F/pitch", rr.Scalars(data['axisF[1]']))
                rr.log("pid/F/yaw", rr.Scalars(data['axisF[2]']))

            # Log gyro data
            if all(key in data for key in ['gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]']):
                rr.log("sensors/gyro_adc/roll", rr.Scalars(data['gyroADC[0]']))
                rr.log("sensors/gyro_adc/pitch", rr.Scalars(data['gyroADC[1]']))
                rr.log("sensors/gyro_adc/yaw", rr.Scalars(data['gyroADC[2]']))

            if all(key in data for key in ['gyroUnfilt[0]', 'gyroUnfilt[1]', 'gyroUnfilt[2]']):
                rr.log("sensors/gyro_unfilt/roll", rr.Scalars(data['gyroUnfilt[0]']))
                rr.log("sensors/gyro_unfilt/pitch", rr.Scalars(data['gyroUnfilt[1]']))
                rr.log("sensors/gyro_unfilt/yaw", rr.Scalars(data['gyroUnfilt[2]']))

            # Log accelerometer data
            if all(key in data for key in ['accSmooth[0]', 'accSmooth[1]', 'accSmooth[2]']):
                rr.log("sensors/acc/x", rr.Scalars(data['accSmooth[0]']))
                rr.log("sensors/acc/y", rr.Scalars(data['accSmooth[1]']))
                rr.log("sensors/acc/z", rr.Scalars(data['accSmooth[2]']))

            # Log RC commands
            if all(key in data for key in ['rcCommand[0]', 'rcCommand[1]', 'rcCommand[2]', 'rcCommand[3]']):
                rr.log("rc/command/roll", rr.Scalars(data['rcCommand[0]']))
                rr.log("rc/command/pitch", rr.Scalars(data['rcCommand[1]']))
                rr.log("rc/command/yaw", rr.Scalars(data['rcCommand[2]']))
                rr.log("rc/command/throttle", rr.Scalars(data['rcCommand[3]']))

            # Log setpoints
            if all(key in data for key in ['setpoint[0]', 'setpoint[1]', 'setpoint[2]', 'setpoint[3]']):
                rr.log("setpoint/roll", rr.Scalars(data['setpoint[0]']))
                rr.log("setpoint/pitch", rr.Scalars(data['setpoint[1]']))
                rr.log("setpoint/yaw", rr.Scalars(data['setpoint[2]']))
                rr.log("setpoint/throttle", rr.Scalars(data['setpoint[3]']))

            # Log motor outputs
            if all(key in data for key in ['motor[0]', 'motor[1]', 'motor[2]', 'motor[3]']):
                rr.log("motors/output/m1", rr.Scalars(data['motor[0]']))
                rr.log("motors/output/m2", rr.Scalars(data['motor[1]']))
                rr.log("motors/output/m3", rr.Scalars(data['motor[2]']))
                rr.log("motors/output/m4", rr.Scalars(data['motor[3]']))

            # Log eRPM
            if all(key in data for key in ['eRPM[0]', 'eRPM[1]', 'eRPM[2]', 'eRPM[3]']):
                rr.log("motors/erpm/m1", rr.Scalars(data['eRPM[0]']))
                rr.log("motors/erpm/m2", rr.Scalars(data['eRPM[1]']))
                rr.log("motors/erpm/m3", rr.Scalars(data['eRPM[2]']))
                rr.log("motors/erpm/m4", rr.Scalars(data['eRPM[3]']))

            # Log battery voltage and current
            if 'vbatLatest (V)' in data and data['vbatLatest (V)'] is not None:
                rr.log("battery/voltage", rr.Scalars(data['vbatLatest (V)']))

            if 'amperageLatest (A)' in data and data['amperageLatest (A)'] is not None:
                rr.log("battery/current", rr.Scalars(data['amperageLatest (A)']))

            if 'energyCumulative (mAh)' in data and data['energyCumulative (mAh)'] is not None:
                rr.log("battery/energy", rr.Scalars(data['energyCumulative (mAh)']))

            # Log RSSI
            if 'rssi' in data and data['rssi'] is not None:
                rr.log("radio/rssi", rr.Scalars(data['rssi']))

            # Log debug values
            for i in range(8):
                key = f'debug[{i}]'
                if key in data and data[key] is not None:
                    rr.log(f"debug/{i}", rr.Scalars(data[key]))

            # Log status/flags (as text)
            if 'flightModeFlags (flags)' in data:
                mode = data['flightModeFlags (flags)']
                if mode is not None:
                    rr.log("status/flight_mode", rr.TextLog(str(mode)))

            if 'failsafePhase (flags)' in data:
                failsafe = data['failsafePhase (flags)']
                if failsafe is not None:
                    rr.log("status/failsafe", rr.TextLog(str(failsafe)))

            row_count += 1

            # Progress indicator
            if row_count % 1000 == 0:
                print(f"Processed {row_count} rows...")

    print(f"\nCompleted! Logged {row_count} rows to Rerun.")
    print(f"Numeric columns found: {len(numeric_columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize flight controller CSV logs using Rerun"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV log file"
    )

    args = parser.parse_args()
    log_to_rerun(args.csv_file)


if __name__ == "__main__":
    main()
