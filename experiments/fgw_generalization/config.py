"""All hyperparameters for the FGW generalization experiment."""

CONFIG = dict(
    # ── Data ──────────────────────────────────────────────────────────────────
    n_ori=8,
    n_reps_total=8,   # total noise repetitions per (ori, SF) pair
    n_reps_train=4,   # reps 0..n_reps_train-1  → train set
                      # reps n_reps_train..end   → test  set
    grid=8,           # grating image: grid×grid pixels
    noise=0.15,       # Gaussian noise std on gratings

    # Interleaved SF setup: even-indexed SFs → train, odd-indexed → test (interpolation)
    # 7 SFs: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # train: {1.0, 2.0, 3.0, 4.0}  test: {1.5, 2.5, 3.5}
    freqs=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    n_freq=7,

    # ── Model ─────────────────────────────────────────────────────────────────
    input_dim=64,
    h1_dim=128,
    h2_dim=64,
    output_dim=8,

    # ── Teacher training ──────────────────────────────────────────────────────
    n_epochs_teacher=500,
    lr_teacher=1e-3,
    lam_sel=0.3,    # selectivity regularizer weight
    lam_div=0.1,    # diversity   regularizer weight
    teacher_seed=0,

    # ── Student training ──────────────────────────────────────────────────────
    n_epochs=500,
    lr=1e-3,
    lambda_fgw_sweep=[0.0, 0.02, 0.05, 0.1, 0.2, 0.5],
    alpha_fgw=0.5,        # structural vs feature weight in FGW (0=feature only, 1=GW only)
    T_update_every=20,    # re-solve FGW transport plan every N epochs
    n_seeds=3,            # independent random init seeds per (lambda, split)

    # ── GW evaluation ─────────────────────────────────────────────────────────
    gw_epsilon=0.05,      # entropic regularization
    gw_max_iter=200,
)
