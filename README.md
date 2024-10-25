# Exceptional Model Mining for Time-Series Data: Descriptive Space Feature Generation and Interval-Based Quality Measures

**Brief Description:**  
Provide a short description of the project, the research problem being tackled, and the methodology used.


## Overview

Explain the motivation behind the project and give a brief overview of the research objectives.  
- **Purpose**: Why is the project important? What specific problem are you addressing?  
- **Research Focus**: Briefly explain the methodologies (data collection, modeling) and any specific machine learning algorithms or statistical methods used.

## Project Structure
Folder with outdated versions of code and datasets:  
OLD  
  
Notebooks with experiments and results:  
beam_search_experiment_top3.ipnyb  
beam_search_plots.ipnyb  
beam_search_window_experiment.ipnyb  

Script for beam search implementation:  
beam_search_module.py  

Zip folder with datasets:  
datasets.zip

Scripts for data processing and feature generation:  
feature_time_series_generator.py  
get_data.py   
plot_sg_and_pop.py  
process_data.py  

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