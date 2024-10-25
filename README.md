# Exceptional Model Mining for Time-Series Data: Descriptive Space Feature Generation and Interval-Based Quality Measures

**Brief Description:**  
As part of the course Resarch Topics in Data Mining this project has been setup to study generation a novel quality measure and generating the descriptive space from the target space in an EMM setting for time-series data (specifically stock data).

## Project Structure

### Folder with Outdated Versions of Code and Datasets
- `OLD/`

### Notebooks with Experiments and Results
- `beam_search_experiment_top3.ipynb`  
- `beam_search_plots.ipynb`  
- `beam_search_window_experiment.ipynb`  

### Script for Beam Search Implementation
- `beam_search_module.py`

### Script for Generating plots
- `plot_sg_and_pop.py`  

### Zip Folder with Datasets
- `datasets.zip`

### Scripts for Data Processing and Feature Generation
- `feature_time_series_generator.py`  
- `get_data.py`  
- `process_data.py`

## Installation
### Prerequisites
Can be installed by:  
pip install -r requirements.txt

### Datasets folder
Make sure to unzip datasets.zip to get the datasets folder.

## Usage
Steps for running this project:
1. Run data gathering script get_data.py
2. Run data processing script process_data.py
3. Run notebooks for experiments/results:
    - beam_search_experiment_top3.ipnyb
    - beam_search_window_experiment.ipnyb
    - beam_search_plots.ipnyb