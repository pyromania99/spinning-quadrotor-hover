"""
Calculate yaw rate from existing telemetry CSV files.
This script reads CSV files and computes yaw rate by differentiating yaw with respect to time.
"""

import pandas as pd
import numpy as np
import os
import sys

def calculate_yaw_rate(csv_filepath):
    """
    Load a telemetry CSV and calculate yaw rate from ATTITUDE messages.
    
    Args:
        csv_filepath: Path to the CSV file
        
    Returns:
        DataFrame with attitude data including calculated yaw rate
    """
    print(f"Loading: {csv_filepath}")
    
    # Load the CSV
    df = pd.read_csv(csv_filepath)
    
    # Filter only ATTITUDE messages with valid yaw data
    attitude_df = df[df['Message Type'] == 'ATTITUDE'].copy()
    attitude_df = attitude_df.dropna(subset=['Yaw'])
    
    if len(attitude_df) < 2:
        print("Not enough ATTITUDE data points to calculate yaw rate")
        return None
    
    # Calculate time differences (in seconds)
    attitude_df['dt'] = attitude_df['Timestamp'].diff()
    
    # Calculate yaw differences (in radians)
    attitude_df['dyaw'] = attitude_df['Yaw'].diff()
    
    # Handle yaw wrap-around (crossing from +π to -π or vice versa)
    attitude_df['dyaw'] = attitude_df['dyaw'].apply(
        lambda x: x - 2*np.pi if x > np.pi else (x + 2*np.pi if x < -np.pi else x)
    )
    
    # Yaw rate in rad/s
    attitude_df['Yaw_Rate_Calculated'] = attitude_df['dyaw'] / attitude_df['dt']
    
    # Remove first row (NaN values from diff)
    attitude_df = attitude_df.iloc[1:]
    
    return attitude_df[['Timestamp', 'Roll', 'Pitch', 'Yaw', 'dt', 'dyaw', 'Yaw_Rate_Calculated']]


def save_results(result_df, original_filepath):
    """Save the calculated results to a new CSV file."""
    output_dir = os.path.dirname(original_filepath)
    base_name = os.path.splitext(os.path.basename(original_filepath))[0]
    output_filepath = os.path.join(output_dir, f"{base_name}_yaw_rate.csv")
    
    result_df.to_csv(output_filepath, index=False)
    print(f"Results saved to: {output_filepath}")
    
    return output_filepath


def print_statistics(result_df):
    """Print statistics about the yaw rate."""
    print("\n=== Yaw Rate Statistics ===")
    print(f"Mean yaw rate: {result_df['Yaw_Rate_Calculated'].mean():.4f} rad/s")
    print(f"Std deviation: {result_df['Yaw_Rate_Calculated'].std():.4f} rad/s")
    print(f"Min yaw rate:  {result_df['Yaw_Rate_Calculated'].min():.4f} rad/s")
    print(f"Max yaw rate:  {result_df['Yaw_Rate_Calculated'].max():.4f} rad/s")
    print(f"Median:        {result_df['Yaw_Rate_Calculated'].median():.4f} rad/s")
    print(f"\nTotal attitude samples: {len(result_df)}")
    print(f"Time span: {result_df['Timestamp'].max() - result_df['Timestamp'].min():.2f} seconds")


if __name__ == "__main__":
    # Check if a file path was provided
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to the most recent file in test directory
        test_dir = os.path.join(os.path.dirname(__file__), 'test')
        csv_files = [f for f in os.listdir(test_dir) if f.startswith('mavlink_data_') and f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found in test directory")
            sys.exit(1)
        
        # Get the most recent file
        csv_files.sort(reverse=True)
        csv_file = os.path.join(test_dir, csv_files[0])
    
    # Calculate yaw rate
    result = calculate_yaw_rate(csv_file)
    
    if result is not None:
        # Print statistics
        print_statistics(result)
        
        # Display first few rows
        print("\n=== First 10 samples ===")
        print(result.head(10).to_string(index=False))
        
        # Save results
        output_file = save_results(result, csv_file)
        
        print(f"\nDone! Check {output_file} for full results.")
