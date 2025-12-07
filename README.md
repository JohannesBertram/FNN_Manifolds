# FNN Manifolds

Code for 'Manifold and Modules: How Function Develops in a Neural Foundation Model' @ Data on the Brain and Mind, NeurIPS 2025. See PDF for the paper. Arxiv link coming soon.

For detailed information on the encoding manifold pipeline refer to https://github.com/dyballa/NeuralEncodingManifolds.

## Installation

[`requirements.txt`](/requirements.txt) - Requirements to run with Python 3.11.2 

## Running experiments

### Building Manifolds

- run build-CNN-manifold.ipynb
- this will throw and error right after the factors are read. If the factor files are already present, choose the number of factors to use and run everything from there. Else run the tensor decomposition in matlab first.

### Sampling from models

- get the FNN checkpoints and save them in CNN_sampling/fnn 
- same for the minimodels
- run FNN_sampling/sampling....ipynb

### Running Tensor Decomposition

After sampling data and creating the matlab file with the first part of plotting/encoding_manifolds.ipynb do:
- Navigate to data/mat_data
- Add tensor toolbox to matlab path
- run run_permcp('matlab_data_filename', 'shift', 2, 30, 50, 8)

## Plots

Then, using the code in plotting/ you can generate encoding manifolds, decoding manifolds and trajectories, alongside additional paper figures.

## Note

Repository cleaning and usage guide still in progress, if you want to run experiments, please contact johannes.bertram[at]yale.edu






