#!/bin/bash

# Script to run the AQI visualization
# Navigate to project root
cd "$(dirname "$0")/gas-emission-prediction"

# Activate virtual environment
source .venv/bin/activate

# Navigate to src directory
cd src

# Run the visualization script
python visualize_predictions.py

echo "✓ Visualization complete!"
