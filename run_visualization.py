#!/usr/bin/env python3
"""
Convenience script to run the AQI visualization from any directory.
Usage: python run_visualization.py
"""

import os
import subprocess
import sys

def run_visualization():
    """Run the visualization script with proper environment setup."""
    
    # Get the project root (parent directory of this script)
    project_root = os.path.dirname(os.path.abspath(__file__))
    gas_emission_dir = os.path.join(project_root, "gas-emission-prediction")
    src_dir = os.path.join(gas_emission_dir, "src")
    venv_path = os.path.join(project_root, ".venv", "bin", "python")
    
    # Verify paths exist
    if not os.path.isfile(venv_path):
        print(f"Error: Virtual environment not found at {venv_path}")
        print("Please ensure the .venv is set up in the project root.")
        sys.exit(1)
    
    if not os.path.isdir(src_dir):
        print(f"Error: src directory not found at {src_dir}")
        sys.exit(1)
    
    # Run the script
    script_path = os.path.join(src_dir, "visualize_predictions.py")
    
    try:
        result = subprocess.run(
            [venv_path, script_path],
            cwd=src_dir,
            check=True,
            capture_output=False
        )
        print("\n✓ Visualization complete!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running visualization: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_visualization())
