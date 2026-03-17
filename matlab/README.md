# MATLAB Tensor Decomposition

The permuted nonnegative CP decomposition is performed using the [Tensor Toolbox for MATLAB](https://www.tensortoolbox.org/):

> Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB.
> www.tensortoolbox.org / https://gitlab.com/tensors/tensor_toolbox

## Setup

1. Install the Tensor Toolbox into your MATLAB path.
2. Copy the following files into your Tensor Toolbox directory:
   - `perm_cp_opt.m`
   - `perm_tt_cp_fg.m`
   - `perm_tt_cp_fun.m`

## Usage

Call `run_permcp` from MATLAB with the tensor data file and decomposition parameters:

```matlab
run_permcp('matlab_data_filename', 'shift', F, max_iters, num_reps, num_workers)
```

| Argument | Description |
|----------|-------------|
| `matlab_data_filename` | Path to `.mat` file containing the tensor `X` |
| `shift_type` | Shift strategy — use `'shift'` for circular shifts |
| `F` | Number of CP components |
| `max_iters` | Maximum optimisation iterations per rep (default: 50) |
| `num_reps` | Number of random initialisations (default: 30) |
| `num_workers` | Parallel workers (default: 8) |

See documentation inside `run_permcp.m` for full argument descriptions.

## Output

Each choice of `F` produces a `.mat` file in `data/decompositions/` containing:
- `factors` — cell array of factor matrices for each rep
- `lams` — lambda (component weight) vectors for each rep
- `objs` — objective value for each rep
