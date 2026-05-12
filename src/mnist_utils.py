"""MNIST manifold dissociation utilities.

Self-contained helpers for notebook: model, losses, training,
data loading, checkpoint I/O, and manifold analysis.

Only the cluster_pref_stim loss is supported.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA


# ── CNN model ─────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """2-conv CNN for MNIST.

    Architecture:
        Conv1(1→16, 3×3) → ReLU → MaxPool(2) → (B, 16, 13, 13)
        Conv2(16→32, 3×3) → ReLU → MaxPool(2) → (B, 32,  5,  5)
        Flatten → (B, 800)  ← "neural population"
        FC(800→10)

    Each of the 800 neurons corresponds to a (channel, spatial_position) pair
    in the final feature maps.  Use ``channel_map`` to colour by channel.
    """

    N_CHANNELS = 32
    SPATIAL     = 5   # feature map side length after two pools

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, self.N_CHANNELS, kernel_size=3)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(self.N_CHANNELS * self.SPATIAL ** 2, 10)
        self.relu  = nn.ReLU()

    def _h2(self, x):
        """Return flattened conv2 activations, shape (B, 800)."""
        x = self.pool(self.relu(self.conv1(x)))   # (B, 16, 13, 13)
        x = self.pool(self.relu(self.conv2(x)))   # (B, 32,  5,  5)
        return x.view(x.size(0), -1)               # (B, 800)

    def forward(self, x):
        return self.fc(self._h2(x))

    @property
    def channel_map(self):
        """Channel index for each neuron, shape (800,).

        Neurons 0..24 → channel 0, neurons 25..49 → channel 1, etc.
        """
        return np.repeat(np.arange(self.N_CHANNELS), self.SPATIAL ** 2)

    @property
    def spatial_map(self):
        """(row, col) in the 5×5 feature map for each neuron, shape (800, 2)."""
        rc = np.array([(r, c)
                       for r in range(self.SPATIAL)
                       for c in range(self.SPATIAL)])
        return np.tile(rc, (self.N_CHANNELS, 1))


# ── Loss functions ─────────────────────────────────────────────────────────────

def _batch_tuning_vecs(h2, y, n_classes=10):
    """Per-class mean of h2 in current batch → (N_neurons, n_present) or None."""
    vecs = []
    for c in range(n_classes):
        m = (y == c)
        if m.sum() > 0:
            vecs.append(h2[m].mean(0))
    return torch.stack(vecs, dim=1) if len(vecs) >= 2 else None   # (N, n_present)


def _batch_tuning_vecs_labeled(h2, y, n_classes=10):
    """Per-class mean of h2, returning both the matrix and class label indices.

    Returns
    -------
    tv     : (N_neurons, k_present) tensor, or None
    labels : (k_present,) LongTensor of actual class indices, or None
    """
    vecs, labels = [], []
    for c in range(n_classes):
        m = (y == c)
        if m.sum() > 0:
            vecs.append(h2[m].mean(0))
            labels.append(c)
    if len(vecs) < 2:
        return None, None
    return (torch.stack(vecs, dim=1),
            torch.tensor(labels, dtype=torch.long, device=h2.device))


def l_cluster_pref_stim(h2, y, n_classes=10, tau=0.07):
    """Cluster neurons by preferred stimulus on the encoding manifold.

    Operates on **raw activation profiles** across the batch rather than the
    compressed 10-D tuning vector, giving a much richer similarity signal.

    For each neuron, its response vector across the B samples in the batch
    serves as its high-dimensional "feature".  Neurons are labelled by their
    preferred stimulus (argmax of per-class mean activation).  A supervised
    contrastive loss then pulls same-preference neurons together and pushes
    different-preference neurons apart in this activation space.

    Parameters
    ----------
    tau : float — temperature (lower = tighter clusters, default 0.07)
    """
    B, N = h2.size()   # (batch, neurons)
    if B < 4:
        return h2.new_tensor(0.0)

    # h2 is (B, N) — transpose to (N, B) so each neuron has a B-dim profile
    profiles = h2.T                                         # (N, B)

    # Determine preferred stimulus per neuron (no grad through argmax)
    with torch.no_grad():
        tv = _batch_tuning_vecs(h2, y, n_classes)
        if tv is None:
            return h2.new_tensor(0.0)
        pref = tv.argmax(dim=1)                             # (N,)

    # Normalise profiles to unit length for cosine similarity
    z = F.normalize(profiles, dim=1)                        # (N, B)
    sim = z @ z.T / tau                                     # (N, N)

    # Numerical stability: subtract row max before exp
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim - sim_max)                      # (N, N)

    # Positive mask: same preferred stimulus, excluding self
    pos_mask = (pref.unsqueeze(0) == pref.unsqueeze(1))     # (N, N)
    pos_mask.fill_diagonal_(False)

    # Only include anchors with at least one positive
    has_pos = (pos_mask.sum(dim=1) > 0)
    if not has_pos.any():
        return h2.new_tensor(0.0)

    n_pos = pos_mask.sum(dim=1).float().clamp(min=1)        # (N,)

    # SupCon: log-sum-exp denominator minus log of positive sum
    log_den = torch.log(exp_sim.sum(dim=1) + 1e-8)
    log_num = torch.log((exp_sim * pos_mask).sum(dim=1) + 1e-8)
    per_anchor = (log_den - log_num) / n_pos

    return per_anchor[has_pos].mean()


def l_triplet(h2, y, n_classes=10, margin=1.0):
    """Hard-margin triplet loss on unit-norm neuron batch profiles."""
    B, N = h2.size()
    if B < 4:
        return h2.new_tensor(0.0)
    profiles = h2.T
    with torch.no_grad():
        tv = _batch_tuning_vecs(h2, y, n_classes)
        if tv is None:
            return h2.new_tensor(0.0)
        pref = tv.argmax(dim=1)
    z    = F.normalize(profiles, dim=1)
    dists = torch.cdist(z, z, p=2)
    same = (pref.unsqueeze(0) == pref.unsqueeze(1))
    diff = ~same
    same.fill_diagonal_(False)
    valid = (same.sum(1) > 0) & (diff.sum(1) > 0)
    if not valid.any():
        return h2.new_tensor(0.0)
    pos_d = dists.clone(); pos_d[~same] = 0.0
    neg_d = dists.clone(); neg_d[~diff] = 1e9
    loss = F.relu(pos_d.max(1).values - neg_d.min(1).values + margin)
    return loss[valid].mean()


def l_fisher(h2, y, n_classes=10):
    """Fisher discriminant ratio loss: minimise S_within / S_between."""
    B, N = h2.size()
    if B < 4:
        return h2.new_tensor(0.0)
    profiles = h2.T
    with torch.no_grad():
        tv = _batch_tuning_vecs(h2, y, n_classes)
        if tv is None:
            return h2.new_tensor(0.0)
        pref = tv.argmax(dim=1)
    classes = pref.unique()
    if len(classes) < 2:
        return h2.new_tensor(0.0)
    grand = profiles.mean(0, keepdim=True)
    S_w = h2.new_tensor(0.0); S_b = h2.new_tensor(0.0)
    for c in classes:
        mask = (pref == c); g = profiles[mask]
        if len(g) == 0: continue
        cent = g.mean(0, keepdim=True)
        S_w = S_w + ((g - cent) ** 2).sum()
        S_b = S_b + mask.sum() * ((cent - grand) ** 2).sum()
    return S_w / (S_b + 1e-8)


def l_centroid_pull(h2, y, n_classes=10):
    """Pull each neuron's batch profile toward its preferred-digit centroid."""
    B, N = h2.size()
    if B < 4:
        return h2.new_tensor(0.0)
    profiles = h2.T
    with torch.no_grad():
        tv = _batch_tuning_vecs(h2, y, n_classes)
        if tv is None:
            return h2.new_tensor(0.0)
        pref = tv.argmax(dim=1)
    classes = pref.unique()
    total, count = h2.new_tensor(0.0), 0
    for c in classes:
        mask = (pref == c)
        if mask.sum() < 2: continue
        g = profiles[mask]; cent = g.mean(0, keepdim=True).detach()
        total = total + ((g - cent) ** 2).mean(); count += 1
    return total / count if count > 0 else total


def l_ortho_centroids(h2, y, n_classes=10):
    """Encourage class response centroids to be orthogonal in neuron space."""
    tv = _batch_tuning_vecs(h2, y, n_classes)
    if tv is None:
        return h2.new_tensor(0.0)
    k = tv.shape[1]
    if k < 2:
        return h2.new_tensor(0.0)
    tv_norm = F.normalize(tv, dim=0)
    G = tv_norm.T @ tv_norm
    I = torch.eye(k, device=G.device)
    off = G - I
    return (off ** 2).sum() / (k * (k - 1))


def l_smooth(h2, y, n_classes=10, tau=0.07):
    """Anti-clustering: push same-preferred-stimulus neurons apart."""
    B, N = h2.size()
    if B < 4:
        return h2.new_tensor(0.0)
    profiles = h2.T
    with torch.no_grad():
        tv = _batch_tuning_vecs(h2, y, n_classes)
        if tv is None:
            return h2.new_tensor(0.0)
        pref = tv.argmax(dim=1)
    z = F.normalize(profiles, dim=1)
    sim = z @ z.T / tau
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim - sim_max)
    pos_mask = (pref.unsqueeze(0) != pref.unsqueeze(1))  # REVERSED
    pos_mask.fill_diagonal_(False)
    has_pos = (pos_mask.sum(dim=1) > 0)
    if not has_pos.any():
        return h2.new_tensor(0.0)
    n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
    log_den = torch.log(exp_sim.sum(dim=1) + 1e-8)
    log_num = torch.log((exp_sim * pos_mask).sum(dim=1) + 1e-8)
    per_anchor = (log_den - log_num) / n_pos
    return per_anchor[has_pos].mean()


def l_sparse(h2, y, n_classes=10):
    """Sparse coding: each neuron responds to exactly one class."""
    tv = _batch_tuning_vecs(h2, y, n_classes)
    if tv is None:
        return h2.new_tensor(0.0)
    tv_relu = F.relu(tv)
    with torch.no_grad():
        pref = tv_relu.argmax(dim=1)
    k = tv_relu.shape[1]
    pref_one_hot = F.one_hot(pref, k).float()
    non_pref_mask = 1.0 - pref_one_hot
    return (tv_relu * non_pref_mask).mean()


def l_mixed(h2, y, n_classes=10):
    """Mixed selectivity: maximise entropy of per-neuron class-response distribution."""
    tv = _batch_tuning_vecs(h2, y, n_classes)
    if tv is None:
        return h2.new_tensor(0.0)
    p = F.softmax(tv, dim=1)
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)
    return -entropy.mean()


def l_dim_collapse(h2, y, n_classes=10, keep_dims=2):
    """Collapse encoding manifold to a low-dimensional subspace."""
    tv = _batch_tuning_vecs(h2, y, n_classes)
    if tv is None:
        return h2.new_tensor(0.0)
    k = tv.shape[1]
    if k <= keep_dims:
        return h2.new_tensor(0.0)
    tv_c = tv - tv.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(tv_c, full_matrices=False)
    except RuntimeError:
        return h2.new_tensor(0.0)
    s_sq = S.pow(2)
    return s_sq[keep_dims:].sum() / (s_sq.sum() + 1e-8)


def l_dim_expand(h2, y, n_classes=10):
    """Expand encoding manifold to maximum dimensionality / isotropy."""
    tv = _batch_tuning_vecs(h2, y, n_classes)
    if tv is None:
        return h2.new_tensor(0.0)
    if tv.shape[1] < 2:
        return h2.new_tensor(0.0)
    tv_c = tv - tv.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(tv_c, full_matrices=False)
    except RuntimeError:
        return h2.new_tensor(0.0)
    s_sq = S.pow(2)
    s_prob = s_sq / (s_sq.sum() + 1e-8)
    entropy = -(s_prob * torch.log(s_prob + 1e-8)).sum()
    return -entropy


def l_ring(h2, y, n_classes=10, tau=0.07, ring_bandwidth=1):
    """Ring topology: neurons tuned to adjacent digit classes are similar."""
    B, N = h2.size()
    if B < 4:
        return h2.new_tensor(0.0)
    profiles = h2.T
    with torch.no_grad():
        tv, class_labels = _batch_tuning_vecs_labeled(h2, y, n_classes)
        if tv is None:
            return h2.new_tensor(0.0)
        pref_local  = tv.argmax(dim=1)
        pref_global = class_labels[pref_local]
    z = F.normalize(profiles, dim=1)
    sim = z @ z.T / tau
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim - sim_max)
    diff = (pref_global.unsqueeze(0) - pref_global.unsqueeze(1)).abs()
    circ_dist = torch.minimum(diff, n_classes - diff)
    pos_mask = (circ_dist <= ring_bandwidth)
    pos_mask.fill_diagonal_(False)
    has_pos = (pos_mask.sum(dim=1) > 0)
    if not has_pos.any():
        return h2.new_tensor(0.0)
    n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
    log_den = torch.log(exp_sim.sum(dim=1) + 1e-8)
    log_num = torch.log((exp_sim * pos_mask).sum(dim=1) + 1e-8)
    per_anchor = (log_den - log_num) / n_pos
    return per_anchor[has_pos].mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, loss_type='baseline', lam=0.0,
                n_epochs=10, lr=1e-3, device='cpu', verbose=True,
                aux_fn=None, val_loader=None, patience=15):
    """Train model with CE + optional auxiliary manifold loss.

    Parameters
    ----------
    loss_type  : 'baseline' | 'cluster_pref_stim'
    lam        : float — auxiliary loss weight (0 = CE only)
    verbose    : print one line per epoch
    aux_fn     : callable(h2, y) → scalar loss, overrides loss_type dispatch
    val_loader : DataLoader for validation (early stopping on val CE loss).
                 When provided, ``n_epochs`` acts as a ceiling and training
                 stops early when val CE loss fails to improve for ``patience``
                 epochs. The best checkpoint (by val loss) is restored.
    patience   : int — epochs without val improvement before stopping

    Returns
    -------
    log : dict with lists 'ce', 'aux', 'train_acc', 'val_ce', 'val_acc'
    """
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    log       = {'ce': [], 'aux': [], 'train_acc': [], 'val_ce': [], 'val_acc': []}

    best_val_ce   = float('inf')
    best_state    = None
    epochs_no_imp = 0

    for ep in range(n_epochs):
        model.train()
        tot_ce = tot_aux = correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            h2     = model._h2(xb)
            logits = model.fc3(h2) if hasattr(model, 'fc3') else model.fc(h2)
            ce     = criterion(logits, yb)
            aux    = aux_fn(h2, yb) if (aux_fn and lam > 0) else ce.new_tensor(0.0)
            (ce + lam * aux).backward()
            optimizer.step()
            n = len(yb)
            tot_ce  += ce.item()  * n
            tot_aux += aux.item() * n
            correct += (logits.argmax(1) == yb).sum().item()
            total   += n

        ep_ce, ep_aux, ep_acc = tot_ce/total, tot_aux/total, correct/total
        log['ce'].append(ep_ce)
        log['aux'].append(ep_aux)
        log['train_acc'].append(ep_acc)

        # ── Validation ───────────────────────────────────────────────────────
        if val_loader is not None:
            model.eval()
            v_ce = v_correct = v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    h2     = model._h2(xb)
                    logits = model.fc3(h2) if hasattr(model, 'fc3') else model.fc(h2)
                    v_ce     += criterion(logits, yb).item() * len(yb)
                    v_correct += (logits.argmax(1) == yb).sum().item()
                    v_total   += len(yb)
            val_ce  = v_ce / v_total
            val_acc = v_correct / v_total
            log['val_ce'].append(val_ce)
            log['val_acc'].append(val_acc)

            if val_ce < best_val_ce - 1e-5:
                best_val_ce   = val_ce
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1

            if verbose:
                aux_str = f'  L_aux={ep_aux:.4f}' if lam > 0 else ''
                print(f'  ep {ep+1:>3}/{n_epochs}  ce={ep_ce:.4f}  '
                      f'val_ce={val_ce:.4f}  val_acc={val_acc:.4f}'
                      f'{aux_str}  no_imp={epochs_no_imp}', flush=True)

            if epochs_no_imp >= patience:
                if verbose:
                    print(f'  Early stop at epoch {ep+1} (patience={patience})')
                break
        else:
            log['val_ce'].append(None)
            log['val_acc'].append(None)
            if verbose:
                aux_str = f'  L_{loss_type}={ep_aux:.4f}' if lam > 0 else ''
                print(f'  ep {ep+1:>2}/{n_epochs}  ce={ep_ce:.4f}  '
                      f'train_acc={ep_acc:.4f}{aux_str}', flush=True)

    # Restore best weights when using early stopping
    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)

    return log


def test_accuracy(model, test_loader, device='cpu'):
    model.eval().to(device)
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    return correct / total


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mnist_data(batch_size=256, data_dir=None):
    """Return (train_loader, test_loader).

    Downloads MNIST to data_dir if not already present.
    """
    from torchvision import datasets, transforms
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=512,        shuffle=False, num_workers=0)
    return train_loader, test_loader


def load_mnist_data_with_val(batch_size=256, data_dir=None, val_frac=0.1, seed=0):
    """Return (train_loader, val_loader, test_loader).

    Splits the 60 000 training samples into train/val using a fixed random
    seed so the split is identical across conditions and model seeds.
    """
    from torchvision import datasets, transforms
    import torch.utils.data as tud
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds    = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    n_val   = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    train_ds, val_ds = tud.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))

    train_loader = tud.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = tud.DataLoader(val_ds,   batch_size=512,         shuffle=False, num_workers=0)
    test_loader  = tud.DataLoader(test_ds,  batch_size=512,         shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_tuning_vecs(model, test_loader, n_classes=10, device='cpu'):
    """Mean hidden activation per digit class → (N_neurons, n_classes)."""
    model.eval().to(device)
    sums   = [None] * n_classes
    counts = np.zeros(n_classes, dtype=int)
    with torch.no_grad():
        for xb, yb in test_loader:
            h2 = model._h2(xb.to(device)).cpu().numpy()
            for c in range(n_classes):
                m = (yb.numpy() == c)
                if m.sum():
                    sums[c] = h2[m].sum(0) if sums[c] is None else sums[c] + h2[m].sum(0)
                    counts[c] += m.sum()
    cols = []
    for c in range(n_classes):
        if sums[c] is None:
            ref = sums[next(i for i in range(n_classes) if sums[i] is not None)]
            cols.append(np.full(ref.shape, np.nan, dtype=np.float32))
        else:
            cols.append((sums[c] / counts[c]).astype(np.float32))
    return np.stack(cols, axis=1)


def build_tensor4d(model, test_loader, n_per_class=50, device='cpu'):
    """Build (N_neurons, 10, n_per_class, 1) tensor for subpop_utils.

    Returns (tensor4d, stim_labels) where stim_labels[i] = digit class of point i
    when the S*D axis is flattened.
    """
    model.eval().to(device)
    all_h2, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_h2.append(model._h2(xb.to(device)).cpu().numpy())
            all_y.append(yb.numpy())
    all_h2 = np.concatenate(all_h2)   # (10000, N)
    all_y  = np.concatenate(all_y)

    N   = all_h2.shape[1]
    t4d = np.zeros((N, 10, n_per_class, 1), dtype=np.float32)
    cnt = np.zeros(10, dtype=int)
    for i in range(len(all_y)):
        c = int(all_y[i])
        if cnt[c] < n_per_class:
            t4d[:, c, cnt[c], 0] = all_h2[i]
            cnt[c] += 1
        if cnt.min() >= n_per_class:
            break

    stim_labels = np.repeat(np.arange(10), n_per_class)
    return t4d, stim_labels


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────

def cnn_checkpoint_path(ckpt_dir, condition, seed, n_epochs, lam=0.0,
                        early_stop=False, patience=15):
    """Canonical CNN checkpoint filename."""
    os.makedirs(ckpt_dir, exist_ok=True)
    if early_stop:
        tag = f'cnn_{condition}_lam{lam}_es{patience}_s{seed}.pt'
    else:
        tag = f'cnn_{condition}_lam{lam}_e{n_epochs}_s{seed}.pt'
    return os.path.join(ckpt_dir, tag)


def load_cnn_checkpoint(path):
    """Load CNN from checkpoint; returns model in eval mode."""
    model = CNN()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.eval()


def train_or_load_cnn(condition, seed, train_loader, ckpt_dir,
                      n_epochs=10, lr=1e-3, device='cpu', verbose=True,
                      lam=0.0, aux_fn=None, val_loader=None, patience=15):
    """Return a trained CNN, loading from checkpoint if it exists.

    Parameters
    ----------
    condition  : str  — used as checkpoint tag; built-in dispatch for
                 'baseline' and 'cluster_pref_stim'; any other name
                 requires ``aux_fn`` to be supplied explicitly.
    seed       : int  random seed
    lam        : float  auxiliary loss weight (0 = CE only)
    aux_fn     : callable(h2, y) → scalar, optional.
                 If provided, overrides the built-in condition dispatch.
                 Ignored when ``lam <= 0``.
    val_loader : DataLoader — when provided, enables early stopping on val CE
                 loss. ``n_epochs`` becomes a ceiling. Best weights restored.
    patience   : int — epochs without val CE improvement before stopping.

    Returns
    -------
    model : CNN (eval mode)
    log   : training log dict, or None if loaded from cache
    """
    early_stop = val_loader is not None
    path = cnn_checkpoint_path(ckpt_dir, condition, seed, n_epochs, lam,
                               early_stop=early_stop, patience=patience)
    if os.path.exists(path):
        print(f'  Loaded from cache: {os.path.basename(path)}')
        return load_cnn_checkpoint(path), None

    mode_str = f'early-stop (patience={patience}, max={n_epochs})' if early_stop else f'{n_epochs} epochs'
    print(f'  Training CNN {condition}  λ={lam}  seed={seed}  ({mode_str}) …')
    torch.manual_seed(seed)
    model = CNN().to(device)

    if lam <= 0 or condition == 'baseline':
        aux_fn_call = None
    elif aux_fn is not None:
        aux_fn_call = aux_fn
    elif condition == 'cluster_pref_stim':
        aux_fn_call = lambda h2, y: l_cluster_pref_stim(h2, y)
    else:
        raise ValueError(
            f"Unknown condition {condition!r} with no aux_fn supplied. "
            f"Built-in conditions: baseline, cluster_pref_stim. "
            f"For custom losses pass aux_fn explicitly."
        )

    # Pre-flight check: verify aux_fn fires and returns a non-zero scalar
    if aux_fn_call is not None:
        _xb, _yb = next(iter(train_loader))
        _xb, _yb = _xb.to(device), _yb.to(device)
        with torch.no_grad():
            _h2 = model._h2(_xb)
            _val = aux_fn_call(_h2, _yb)
        print(f'  Pre-flight aux check: {condition} loss = {_val.item():.6f}  '
              f'(should be non-zero)')

    log = train_model(model, train_loader, loss_type=condition, lam=lam,
                      n_epochs=n_epochs, lr=lr, device=device, verbose=verbose,
                      aux_fn=aux_fn_call, val_loader=val_loader, patience=patience)

    torch.save(model.state_dict(), path)
    print(f'  Saved → {path}')
    return model.eval(), log


# ── Manifold analysis ─────────────────────────────────────────────────────────

def encoding_pca(tuning_vecs, n_components=2):
    """PCA of neuron tuning vectors.

    Parameters
    ----------
    tuning_vecs : (N_neurons, n_classes)

    Returns
    -------
    coords      : (N_neurons, n_components)
    pref_class  : (N_neurons,) int — argmax class per neuron
    pca         : fitted PCA object
    """
    pca    = PCA(n_components=n_components)
    coords = pca.fit_transform(tuning_vecs)
    pref   = tuning_vecs.argmax(axis=1)
    return coords, pref, pca


def encoding_gw_distance(tv_a, tv_b, n_subsample=300, rng_seed=0):
    """Approx Gromov-Hausdorff distance between two tuning-vector point clouds."""
    from .metrics import compute_gromov_hausdorff_approx
    rng   = np.random.default_rng(rng_seed)
    n     = min(tv_a.shape[0], tv_b.shape[0], n_subsample)
    ia    = rng.choice(tv_a.shape[0], n, replace=False)
    ib    = rng.choice(tv_b.shape[0], n, replace=False)
    return float(compute_gromov_hausdorff_approx(tv_a[ia], tv_b[ib]))


def make_neighbor_pairs(channel_map, spatial_map, spatial=5):
    """Precompute spatially adjacent neuron index pairs (4-connectivity, same channel).

    Returns
    -------
    src_idx, dst_idx : np.ndarray of int, each length n_channels × 2×S×(S−1)
    """
    src, dst = [], []
    n_channels = int(channel_map.max()) + 1
    for ch in range(n_channels):
        neurons = np.where(channel_map == ch)[0]   # 25 indices, row-major
        for r in range(spatial):
            for c in range(spatial):
                i = r * spatial + c
                if c + 1 < spatial:                # right neighbor
                    src.append(neurons[i]); dst.append(neurons[i + 1])
                if r + 1 < spatial:                # bottom neighbor
                    src.append(neurons[i]); dst.append(neurons[i + spatial])
    return np.array(src), np.array(dst)


# ── Perturbation / stability helpers ─────────────────────────────────────────

def make_noisy_loader(test_loader, sigma, seed=42):
    """Return a list of (noisy_x, y) batch tuples with additive Gaussian noise.

    Noise is added in the normalised pixel space (post-ToTensor/Normalize).
    sigma=0 returns a clean copy.
    """
    torch.manual_seed(seed)
    batches = []
    for xb, yb in test_loader:
        if sigma > 0.0:
            xb = xb + sigma * torch.randn_like(xb)
        batches.append((xb, yb))
    return batches


def tuning_vec_cosine_stability(tv_clean, tv_noisy):
    """Mean per-neuron cosine similarity between clean and noisy tuning vectors.

    Parameters
    ----------
    tv_clean, tv_noisy : (N_neurons, n_classes) ndarray

    Returns
    -------
    float in [-1, 1]; higher = more stable tuning under perturbation
    """
    norm_c  = np.linalg.norm(tv_clean, axis=1, keepdims=True) + 1e-8
    norm_n  = np.linalg.norm(tv_noisy, axis=1, keepdims=True) + 1e-8
    cos_sim = np.sum((tv_clean / norm_c) * (tv_noisy / norm_n), axis=1)
    return float(np.mean(cos_sim))


# ── Rotation helpers ──────────────────────────────────────────────────────────

def rotate_batch(x, angle_deg):
    """Apply a fixed rotation to a float32 image batch (B, 1, H, W)."""
    import torchvision.transforms.functional as TF
    return TF.rotate(x, angle=float(angle_deg),
                     interpolation=TF.InterpolationMode.BILINEAR)


def make_rotated_eval_loader(dataset, angle_deg, digit_filter=None, batch_size=512):
    """Evaluate-time loader: all samples rotated by a single fixed angle.

    Returns a list of (Tensor, Tensor) — iterable like a DataLoader.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)
    batches = []
    for xb, yb in loader:
        if digit_filter is not None:
            mask = torch.zeros(len(yb), dtype=torch.bool)
            for d in digit_filter:
                mask |= (yb == d)
            if mask.sum() == 0:
                continue
            xb, yb = xb[mask], yb[mask]
        xb = rotate_batch(xb, angle_deg)
        batches.append((xb, yb))
    return batches


def build_tensor4d_rot(model, test_dataset, digits, angles_deg,
                       n_per_cell=20, device='cpu'):
    """Build (N_neurons, n_digits, n_angles, n_per_cell) tensor4d for rotation data.

    Returns (t4d, stim_labels) where stim_labels[i] = digit index for S*D condition i.
    """
    model.eval().to(device)
    S, D = len(digits), len(angles_deg)
    N    = 800
    t4d  = np.zeros((N, S, D, n_per_cell), dtype=np.float32)
    cnt  = np.zeros((S, D), dtype=int)
    for ai, angle in enumerate(angles_deg):
        loader = make_rotated_eval_loader(test_dataset, angle_deg=angle,
                                          digit_filter=digits)
        with torch.no_grad():
            for xb, yb in loader:
                h2 = model._h2(xb.to(device)).cpu().numpy()
                for b in range(len(yb)):
                    d_idx    = digits.index(int(yb[b]))
                    n_filled = cnt[d_idx, ai]
                    if n_filled < n_per_cell:
                        t4d[:, d_idx, ai, n_filled] = h2[b]
                        cnt[d_idx, ai] += 1
    stim_labels = np.repeat(np.arange(S), D)
    return t4d, stim_labels


def train_or_load_rotation_cnn(condition, seed, train_dataset, train_digits,
                                train_angles_deg, ckpt_dir, n_epochs=10,
                                lr=1e-3, device='cpu', lam=0.0):
    """Train CNN on rotation-augmented MNIST subset; load from cache if available.

    Each batch is rotated by a single angle randomly chosen from train_angles_deg.
    """
    import random
    n_ang = len(train_angles_deg)
    nd    = len(train_digits)
    tag   = (f'rot_cnn_{condition}_d{nd}_a{n_ang}'
             f'_lam{lam}_e{n_epochs}_s{seed}.pt')
    path  = os.path.join(ckpt_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.exists(path):
        print(f'  ✓ Loaded from cache: {os.path.basename(path)}')
        return load_cnn_checkpoint(path), None
    print(f'  Training rotation CNN {condition}  λ={lam}  seed={seed}  …')
    torch.manual_seed(seed)
    random.seed(seed)
    model = CNN().to(device)
    digit_mask = np.isin(np.array(train_dataset.targets), train_digits)
    indices    = np.where(digit_mask)[0]
    sub_ds     = torch.utils.data.Subset(train_dataset, indices)

    def collate_rot(batch):
        xs    = torch.stack([b[0] for b in batch])
        ys    = torch.tensor([b[1] for b in batch])
        angle = random.choice(train_angles_deg)
        return rotate_batch(xs, angle), ys

    train_loader = torch.utils.data.DataLoader(
        sub_ds, batch_size=256, shuffle=True, num_workers=0,
        collate_fn=collate_rot)
    if lam > 0 and condition == 'cluster_pref_stim':
        n_cls  = len(train_digits)
        aux_fn = lambda h2, y: l_cluster_pref_stim(h2, y, n_classes=n_cls)
    else:
        aux_fn = None
    log = train_model(model, train_loader, loss_type=condition, lam=lam,
                      n_epochs=n_epochs, lr=lr, device=device, verbose=True,
                      aux_fn=aux_fn)
    torch.save(model.state_dict(), path)
    print(f'  Saved → {path}')
    return model.eval(), log
