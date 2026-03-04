.PHONY: help visualize run-viz clean

help:
	@echo "AQI Gas Emission Prediction - Available Commands"
	@echo "================================================"
	@echo "make visualize      - Run the AQI prediction visualization"
	@echo "make run-viz        - Alias for visualize"
	@echo "make clean          - Remove generated plots"

visualize:
	@cd gas-emission-prediction/src && \
	. ../../.venv/bin/activate && \
	python visualize_predictions.py

run-viz: visualize

clean:
	rm -rf plots/*.png
	@echo "Cleaned up plots"
