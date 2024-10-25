# Exceptional Model Mining for Time-Series Data: Descriptive Space Feature Generation and Interval-Based Quality Measures

**Brief Description:**  
Provide a short description of the project, the research problem being tackled, and the methodology used.


## Overview

Explain the motivation behind the project and give a brief overview of the research objectives.  
- **Purpose**: Why is the project important? What specific problem are you addressing?  
- **Research Focus**: Briefly explain the methodologies (data collection, modeling) and any specific machine learning algorithms or statistical methods used.

## Project Structure

Provide an outline of the folder structure and files in your repo:  
OLD                                 # folder with outdated versions of code and datasets  
beam_search_experiment_top3.ipnyb   # Notebook for producing top-3 results from beam search  
beam_search_module.py               # Script with beam search implementation  
beam_search_plots.ipnyb             # Notebook for producing results plots from beam search  
beam_search_window_experiment.ipnyb # Notebook for producing window experiment results from beam search  
datasets.zip                        # Zip folder containing the starting data required  
feature_time_series_generator.py    # Script with functions for generating features from time-series  
get_data.py                         # Script to get data from Yahoo Finance based on the original dataset  
plot_sg_and_pop.py                  # Script with functions for plotting results  
process_data.py                     # Script for processing data from get_data.py  
README.md                           # This file  


## Installation
### Prerequisites
Can be installed by:  
pip install -r requirements.txt

### Datasets folder
Make sure to unzip datasets.zip to achieve the following results:  
├── datasets  
│   ├── stock_data_for_emm.pkl  

## Usage
Steps for running this project:
1. Run data gathering script get_data.py
2. Run data processing script process_data.py
3. Run notebooks for experiments/results:
    - beam_search_experiment_top3.ipnyb
    - beam_search_window_experiment.ipnyb
    - beam_search_plots.ipnyb