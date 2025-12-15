# FNN Manifolds

TODO get retina tensor data and ajust in tubularity. 

Code for 'Manifold and Modules: How Function Develops in a Neural Foundation Model' @ Data on the Brain and Mind, NeurIPS 2025. See PDF for the paper. Arxiv link coming soon.

For detailed information on the encoding manifold pipeline refer to https://github.com/dyballa/NeuralEncodingManifolds.

## Repository Structure

- FNN_sampling: Code to sample activity from the FNN model
- plotting: Main analysis pipelines to produces the figures from the paper
- permuted-decomposition: Matlab code for the nonnegative tensor decomposition
- data: Data directory, containing some example data to run analysis
- fig: directory for all results figures

## Installation

[`requirements.txt`](/requirements.txt) - via `pip install -r requirements.txt`

Tested on Python 3.11 and 3.13.

## Experiments

Example data for running experiments is included in data/

### Sampling Activity from Models

Run FNN_sampling/sampling-fnn.ipynb to obtain intermediate layer FNN samples. Running this the first time will download FNN checkpoints.

additional_figures.py can be used for obtaining some additional analysis of FNN, focusing on activations of intermediate layers and decodability of stimuli.

### Encoding Manifolds

- run plotting/encoding_manifolds.ipynb
- this will throw and error right after the factors are read. If the factor files are already present, choose the number of factors to use and run everything from there. Else run the tensor decomposition in matlab first.

### Decoding Manifolds and Trajectories

- run plotting/decoding_analysis.ipynb

### Tubularity Analysis

- run plotting/tubularity.ipynb

### Running Tensor Decomposition

After sampling data and creating the matlab file in data/mat_data with the first part of plotting/encoding_manifolds.ipynb do:
- Add tensor toolbox to matlab path
- run run_permcp('matlab_data_filename', 'shift', 2, 30, 50, 8)

This repository contains example data of tensor decompositions to run the full pipeline, but the data is not sufficient to allow for robust factor selection.

## Note

Work in progress. If you have questions, or want to run experiments, feel free to reach out to me! johannes.bertram[at]yale.edu






