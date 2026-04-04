# FGW Generalization Experiment

Tests whether FGW-aligned encoding topology **generalizes to held-out stimuli**.

## Motivation

The toy experiment in `notebooks/10_fgw_training_signal.ipynb` showed that FGW regularization brings a student network's encoding manifold 56% closer to a clustered teacher — but only evaluated on the *same stimuli used during training*. This experiment tests whether the alignment holds on held-out stimuli, providing evidence that FGW shapes network *weights* (and therefore generalizable tuning structure) rather than just the seen-stimulus activation pattern.

## Experiment design

Two stimulus splits of the synthetic grating dataset (8 ori × 4 SF × 8 noise reps):

| Split | Train | Test |
|-------|-------|------|
| **Noise** | reps 0–3 (128 stimuli) | reps 4–7 (128 stimuli, different noise) |
| **SF** | SF ∈ {1.0, 2.0} (128 stimuli) | SF ∈ {3.0, 4.0} (128 stimuli, held-out scales) |

Three models per run:
- **Teacher** — trained on ALL stimuli with selectivity + diversity regularizers → clustered manifold (reference)
- **Baseline** — CE only on train stimuli
- **FGW student** — CE + FGW on train stimuli, aligning h1 manifold to teacher's train manifold

A lambda sweep (λ ∈ {0.0, 0.02, 0.05, 0.1, 0.2, 0.5}) with 3 random seeds is run for both splits.

**Key metric**: `GW(student → teacher)` computed independently on train AND test stimuli.

## Usage

```bash
cd experiments/fgw_generalization
../../.venv/bin/python run.py       # train + save results (~10–20 min on CPU)
../../.venv/bin/python analyze.py   # load results, print table, save figures
```

## Outputs (`results/`)

| File | Contents |
|------|----------|
| `metrics.json` | All scalar results (one row per split × lambda × seed × model) |
| `learning_curves.json` | Epoch-level CE / FGW / total losses |
| `embeddings.npz` | PCA 2-D encoding manifold coords for all conditions |
| `teacher_Cs.npz` | Teacher structural cost matrices + tuning vectors |
| `figures/lambda_sweep_noise.png` | GW / accuracy / OSI vs λ (noise split) |
| `figures/lambda_sweep_sf.png` | Same for SF split |
| `figures/encoding_manifolds_noise.png` | 3×2 manifold scatter (mid λ) |
| `figures/gen_gap_noise.png` | Train vs test GW bar chart |

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters |
| `data.py` | Grating generation + noise/SF splits |
| `model.py` | MLP (64→128→64→8, identical to nb10) |
| `losses.py` | Teacher regularizers + linearized FGW loss |
| `train.py` | Training loops (teacher / baseline / FGW student) |
| `evaluate.py` | Encoding manifold metrics + GW distance |
| `run.py` | Entry point — runs all experiments, saves results |
| `analyze.py` | Loads results, prints table, generates figures |
