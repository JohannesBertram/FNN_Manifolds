import os
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.sparse import load_npz
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import tueplots
from tueplots import bundles
from tueplots.constants.color import rgb
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from tueplots import axes
import cv2
import torch
from fnn import microns
from CNN_sampling.utils import createFlowDataset

# Ensure figures output directory exists
os.makedirs("../figs", exist_ok=True)
os.makedirs("../fig", exist_ok=True)


def create_intensity_plot_hidden():
    PREFIX = "fnn07_act_i3_n2000_SCL0_7_TL37_hidden_maxFr_maxNr_seed1"

    basedir = '../data/sampled_data'
    tensor4d = np.load(f'{basedir}/tensor4d_{PREFIX}.npy')

    c = ["#fbc02c", "#cfd8dc", "#33691d", "#88144f"] 

    intensity = np.mean(tensor4d, axis=(1,2)).reshape(-1, 37)

    indices = [
        list(range(550, 600)), # yellow spike
        list(range(1850, 1900)), # gray spike
        list(range(1750, 1850)) + list(range(1900, 1950)), #greens + blue
        list(range(400, 450)) + list(range(1050, 1100)) # pinks
    ]

    filtered_intensities = []
    filtered_stds = []
    n_samples = [50, 50, 150, 100]

    for i in range(len(indices)):

        filtered_intensities.append(np.mean(intensity[np.array(indices[i])], axis=0))
        filtered_stds.append(np.std(intensity[np.array(indices[i])], axis=0))
    

    with plt.rc_context({**bundles.neurips2024(), **axes.lines()}):

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(len(filtered_intensities)):
            time_points = np.arange(len(filtered_intensities[i]))

            ax.plot(time_points, filtered_intensities[i], color=c[i], alpha=0.8, linewidth=5)
            
            ax.fill_between(time_points, 
                            filtered_intensities[i] - filtered_stds[i]/np.sqrt(n_samples[i]),
                            filtered_intensities[i] + filtered_stds[i]/np.sqrt(n_samples[i]),
                            color=c[i], alpha=0.3)

        ax.set_xlabel("Time", fontsize=40)
        ax.set_ylabel("Response Intensity", fontsize=40)
        ax.tick_params(axis="both", which="major", labelsize=30)

        plt.savefig("../figs/intensity_hidden.png")
        plt.savefig("../figs/intensity_hidden.pdf")


def create_intensity_plot_block2():
    PREFIX = "fnn07_act_i3_n2000_SCL0_7_TL37_blocks2_maxFr_maxNr_seed1"

    basedir = '../data/sampled_data'
    tensor4d = np.load(f'{basedir}/tensor4d_{PREFIX}.npy')

    c = ["#cfd8dc", "#FFFF00", "#03a9f4"]#, "#88144f"] 

    intensity = np.mean(tensor4d, axis=(1,2)).reshape(-1, 37)

    indices = [
        list(range(1900, 1950)), # grey cluster
        [133, 943, 1353, 128, 767, 112, 1709, 104, 107, 135, 761, 1726, 969, 589, 929, 105, 102, 556, 1606, 1608, 554, 1736, 407, 1729, 768, 1307, 1619, 927, 778, 1328, 1325, 1331, 118, 1334, 772, 701, 1368, 535, 575, 1320, 1314, 525, 562, 757, 920, 1607, 580, 246], # intensity spike
        list(range(1700, 1750)) + list(range(1800, 1850)), # blue/orange spike 
        #list(range(350, 400)) + list(range(1100, 1150)) + list(range(300, 350)) + list(range(650, 700)) # multi color spike
    ]

    filtered_intensities = []
    filtered_stds = []
    n_samples = []

    for i in range(len(indices)):

        filtered_intensities.append(np.mean(intensity[np.array(indices[i])], axis=0))
        filtered_stds.append(np.std(intensity[np.array(indices[i])], axis=0))
        n_samples.append(len(indices[i]))


    with plt.rc_context({**bundles.neurips2024(), **axes.lines()}):

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(len(filtered_intensities)):
            time_points = np.arange(len(filtered_intensities[i]))

            ax.plot(time_points, filtered_intensities[i], color=c[i], alpha=0.8, linewidth=5)
            
            ax.fill_between(time_points, 
                            filtered_intensities[i] - filtered_stds[i]/np.sqrt(n_samples[i]),
                            filtered_intensities[i] + filtered_stds[i]/np.sqrt(n_samples[i]),
                            color=c[i], alpha=0.3)

        ax.set_xlabel("Time", fontsize=40)
        ax.set_ylabel("Response Intensity", fontsize=40)
        ax.tick_params(axis="both", which="major", labelsize=30)
        
        plt.savefig("../figs/intensity_block2.png")
        plt.savefig("../figs/intensity_block2.pdf")


def intensity_comparison_plot():

    def get_activations(frames, model_session=8, model_scan=5):
        """Get max activations for given frames"""
        # Load model
        model, ids = microns.scan(session=model_session, scan_idx=model_scan)
        
        # Hook to capture activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        layers = ["core.feedforward.inputs.0", "core.feedforward.inputs.1", "core.feedforward.inputs.2", 
                 "core.feedforward.blocks.0.convs.1", "core.feedforward.blocks.1.convs.1", 
                 "core.feedforward.blocks.2.convs.2", "core.recurrent.conv", "core.recurrent.out", 
                 "readout", "unit"]

        # Register hooks
        for name, module in model.named_modules():
            if name in layers:
                module.register_forward_hook(hook_fn(name))

        # Forward pass
        model.predict(stimuli=frames)
        
        # Get max activations
        layer_names = list(activations.keys())
        max_activations = [activations[layer].max().item() for layer in layer_names]
        return max_activations, layer_names

    # 1. Load MP4 stimulus
    cap = cv2.VideoCapture("../data/stimulus_17797_7_3_v4_compressed.mp4")
    mp4_frames = []
    counter = 0
    while counter < 1000:
        counter += 1
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 144))
        mp4_frames.append(resized)
    cap.release()
    mp4_frames = np.array(mp4_frames, dtype="uint8")

    # 2. Load flow stimulus frames
    mydirs = list(map(str, range(0, 360, 45)))
    categories = ['grat_W12', 'grat_W1', 'grat_W2',
                  'neg1dotflow_D1_bg', 'neg3dotflow_D1_bg', 'neg1dotflow_D2_bg', 'neg3dotflow_D2_bg',
                  'pos1dotflow_D1_bg', 'pos3dotflow_D1_bg', 'pos1dotflow_D2_bg', 'pos3dotflow_D2_bg']
    
    # Assuming createFlowDataset exists and returns datasets
    # Replace this with your actual flow dataset loading code
    flow_datasets = createFlowDataset(categories, 'flowstims', mydirs, (800, 600), (144, 256),
                                    scl_factor, N_INSTANCES, trial_len, stride)
    
    # Convert first flow dataset to frames format
    flow_frames = np.array([frame.reshape(144, 256) for frame in flow_datasets[0][:1000]], dtype="uint8")

    # 3. Get activations for both stimuli
    mp4_activations, layer_names = get_activations(mp4_frames)
    flow_activations, _ = get_activations(flow_frames)

    # 4. Plot comparison
    plt.figure(figsize=(12, 6))
    x = range(len(layer_names))
    plt.plot(x, mp4_activations, 'o-', label='MP4 Stimulus', linewidth=2)
    plt.plot(x, flow_activations, 's-', label='Flow Stimulus', linewidth=2)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Maximum Activation')
    plt.title('Maximum Activation per Layer: Stimulus Comparison')
    plt.xticks(x, [f'L{i}' for i in range(len(layer_names))], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figs/intensity_comparison_microns.pdf', dpi=300)
    plt.show()

def create_knn_lr_plots():

    prefixes = [
        "fnn07_act_i3_n2000_SCL0_7_TL37_inputs0_maxFr_maxNr_seed1",
        "fnn07_act_i3_n2000_SCL0_7_TL37_inputs1_maxFr_maxNr_seed1",
        "fnn07_act_i3_n2000_SCL0_7_TL37_inputs2_maxFr_maxNr_seed1",
        "fnn07_act_i3_n2000_SCL0_7_TL37_blocks0_maxFr_maxNr_seed1",
        "fnn07_act_i3_n2000_SCL0_7_TL37_blocks1_maxFr_maxNr_seed1",
        "fnn07_act_i3_n2000_SCL0_7_TL37_blocks2_maxFr_maxNr_seed1",
        "fnn07_act_i3_n2000_SCL0_7_TL37_hidden_maxFr_maxNr_seed1",
        "fnn07_act_i3_n2000_SCL0_7_TL37_recurrentout_maxFr_maxNr_seed1",
        "fnn07_act_i3_n2000_SCL0_7_TL37_position_maxFr_maxNr_seed1",
        "fnn07_seed3"
    ]

    layers = [
        "Encoder L1",
        "Encoder L2",
        "Encoder L4",
        "Encoder L5",
        "Encoder L7",
        "Encoder L8",
        "Hidden", 
        "Hidden-Out",
        "Readout",
        "Output"
    ]

    layers_files = [
        "Encoder_L1",
        "Encoder_L2",
        "Encoder_L4",
        "Encoder_L5",
        "Encoder_L7",
        "Encoder_L8",
        "Recurrent",
        "RecurrentOut",
        "Readout",
        "Output"
    ]

    # desired new order
    new_order = [0, 3, 1, 4, 2, 5, 6, 7, 8, 9]


    basedir = '../data/sampled_data'
    cache_dir = '../data/lr_cache'
    os.makedirs(cache_dir, exist_ok=True)

    

    # ---------- Logistic Regression ----------
    n_runs = 5
    all_time_inc_acc_lin, all_time_dec_acc_lin = [], []
    all_time_inc_std_lin, all_time_dec_std_lin = [], []
    all_accuracies_lin = []

    for idx, prefix in enumerate(prefixes):
        tensor4d = np.load(f'{basedir}/tensor4d_{prefix}.npy')
        n_samples = 88

        cache_file_inc_mean_old = os.path.join(cache_dir, f'layer_{idx}_{layers_files[idx]}_inc_mean.npy')
        cache_file_dec_mean_old = os.path.join(cache_dir, f'layer_{idx}_{layers_files[idx]}_dec_mean.npy')
        cache_file_inc_std_old = os.path.join(cache_dir, f'layer_{idx}_{layers_files[idx]}_inc_std.npy')
        cache_file_dec_std_old = os.path.join(cache_dir, f'layer_{idx}_{layers_files[idx]}_dec_std.npy')
        print(os.path.exists(cache_file_inc_mean_old))
        print(cache_file_dec_mean_old)

        cache_file_inc_mean = os.path.join(cache_dir, f'layer_{idx}_inc_mean.npy')
        cache_file_dec_mean = os.path.join(cache_dir, f'layer_{idx}_dec_mean.npy')
        cache_file_inc_std = os.path.join(cache_dir, f'layer_{idx}_inc_std.npy')
        cache_file_dec_std = os.path.join(cache_dir, f'layer_{idx}_dec_std.npy')
        #print(cache_file_inc_mean)

        if (os.path.exists(cache_file_inc_mean_old) and os.path.exists(cache_file_dec_mean_old) and
            os.path.exists(cache_file_inc_std_old) and os.path.exists(cache_file_dec_std_old)):
            print("loading data")
            time_inc_acc_mean = np.load(cache_file_inc_mean_old)
            time_dec_acc_mean = np.load(cache_file_dec_mean_old)
            time_inc_acc_std = np.load(cache_file_inc_std_old)
            time_dec_acc_std = np.load(cache_file_dec_std_old)
        elif (os.path.exists(cache_file_inc_mean) and os.path.exists(cache_file_dec_mean) and
            os.path.exists(cache_file_inc_std) and os.path.exists(cache_file_dec_std)):
            print("loading data")
            time_inc_acc_mean = np.load(cache_file_inc_mean)
            time_dec_acc_mean = np.load(cache_file_dec_mean)
            time_inc_acc_std = np.load(cache_file_inc_std)
            time_dec_acc_std = np.load(cache_file_dec_std)
        else:
            print(prefix)
            runs_time_inc_acc, runs_time_dec_acc = [], []
            for run in range(n_runs):
                print(run)
                clf = LogisticRegression(max_iter=200, solver='lbfgs', random_state=42+run)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+run)
                time_inc_acc, time_dec_acc = [], []

                for i in range(37):
                    print(i)
                    X_inc = np.transpose(tensor4d, (2, 1, 0, 3))[:, :, :, :i+1]
                    X_inc = np.mean(X_inc, axis=3).reshape(n_samples, -1)
                    y = np.array(list(range(11)) * 8)[:n_samples]
                    acc_inc = cross_val_score(clf, X_inc, y, cv=cv, scoring='accuracy')
                    time_inc_acc.append(acc_inc.mean())

                    X_dec = np.transpose(tensor4d, (2, 1, 0, 3))[:, :, :, i:]
                    X_dec = np.mean(X_dec, axis=3).reshape(n_samples, -1)
                    y = np.array(list(range(11)) * 8)[:n_samples]
                    acc_dec = cross_val_score(clf, X_dec, y, cv=cv, scoring='accuracy')
                    time_dec_acc.append(acc_dec.mean())

                runs_time_inc_acc.append(time_inc_acc)
                runs_time_dec_acc.append(time_dec_acc)

            time_inc_acc_mean = np.mean(runs_time_inc_acc, axis=0)
            time_dec_acc_mean = np.mean(runs_time_dec_acc, axis=0)
            time_inc_acc_std = np.std(runs_time_inc_acc, axis=0)
            time_dec_acc_std = np.std(runs_time_dec_acc, axis=0)

            np.save(cache_file_inc_mean, time_inc_acc_mean)
            np.save(cache_file_dec_mean, time_dec_acc_mean)
            np.save(cache_file_inc_std, time_inc_acc_std)
            np.save(cache_file_dec_std, time_dec_acc_std)

        all_time_inc_acc_lin.append(time_inc_acc_mean)
        all_time_dec_acc_lin.append(time_dec_acc_mean)
        all_time_inc_std_lin.append(time_inc_acc_std)
        all_time_dec_std_lin.append(time_dec_acc_std)
        all_accuracies_lin.extend(time_inc_acc_mean)
        all_accuracies_lin.extend(time_dec_acc_mean)

    y_min_lin, y_max_lin = min(all_accuracies_lin), max(all_accuracies_lin)

    # ---------- KNN ----------
    k_neighbors = 3
    loocv = LeaveOneOut()
    all_time_inc_acc_knn, all_time_dec_acc_knn = [], []
    all_accuracies_knn = []

    for prefix in prefixes:
        tensor4d = np.load(f'{basedir}/tensor4d_{prefix}.npy')
        n = len(tensor4d)

        time_inc_acc, time_dec_acc = [], []

        for i in range(37):
            # Increasing window
            X = np.transpose(tensor4d, (2, 1, 0, 3))[:, :, :, :i+1]
            X = np.mean(X, axis=3).reshape(88, n)
            y = np.array(list(range(11)) * 8)

            preds, trues = [], []
            for train_idx, test_idx in loocv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                knn = KNeighborsClassifier(n_neighbors=k_neighbors)
                knn.fit(X_train, y_train)
                preds.append(knn.predict(X_test)[0])
                trues.append(y_test[0])
            time_inc_acc.append(accuracy_score(trues, preds))

            # Decreasing window
            X = np.transpose(tensor4d, (2, 1, 0, 3))[:, :, :, i:]
            X = np.mean(X, axis=3).reshape(88, n)
            y = np.array(list(range(11)) * 8)

            preds, trues = [], []
            for train_idx, test_idx in loocv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                knn = KNeighborsClassifier(n_neighbors=k_neighbors)
                knn.fit(X_train, y_train)
                preds.append(knn.predict(X_test)[0])
                trues.append(y_test[0])
            time_dec_acc.append(accuracy_score(trues, preds))

        all_time_inc_acc_knn.append(time_inc_acc)
        all_time_dec_acc_knn.append(time_dec_acc)
        all_accuracies_knn.extend(time_inc_acc)
        all_accuracies_knn.extend(time_dec_acc)

    y_min_knn, y_max_knn = min(all_accuracies_knn), max(all_accuracies_knn)

    # ---------- PLOT ----------
    with plt.rc_context({**bundles.neurips2024(), **axes.lines()}):
        fig = plt.figure(figsize=(15, 12))
        time_points = np.arange(1, 38)

        # ---- 1) Compute global y-limits ----
        all_vals = []

        # KNN values
        for arr in all_time_inc_acc_knn + all_time_dec_acc_knn:
            all_vals.append(np.min(arr))
            all_vals.append(np.max(arr))

        # LR values (mean ± std)
        for mean_arr, std_arr in zip(all_time_inc_acc_lin, all_time_inc_std_lin):
            all_vals.append(np.min(mean_arr - std_arr))
            all_vals.append(np.max(mean_arr + std_arr))
        for mean_arr, std_arr in zip(all_time_dec_acc_lin, all_time_dec_std_lin):
            all_vals.append(np.min(mean_arr - std_arr))
            all_vals.append(np.max(mean_arr + std_arr))

        global_ymin = min(all_vals)
        global_ymax = max(all_vals)

        # ---- Create 2x1 layout ----
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

        # ---- LR subplot grid (TOP) ----
        #gs_lr = gs[0].subgridspec(2, 5)
        gs_lr = gs[0].subgridspec(1, 3)
        #lr_axes = [fig.add_subplot(gs_lr[i, j]) for i in range(2) for j in range(5)]
        lr_axes = [fig.add_subplot(gs_lr[i, j]) for i in range(1) for j in range(3)]

        for j, idx in enumerate(new_order):
            ax = lr_axes[j]

            # Increasing window
            mean_inc = all_time_inc_acc_lin[idx]
            std_inc = all_time_inc_std_lin[idx]
            ax.plot(time_points, mean_inc, color='red', linewidth=2, marker='o', markersize=4,
                    label='Increasing Window (Start → t)', alpha=0.8)
            ax.fill_between(time_points, mean_inc - std_inc/np.sqrt(5), mean_inc + std_inc/np.sqrt(5),
                            color='red', alpha=0.2)

            # Decreasing window
            mean_dec = all_time_dec_acc_lin[idx]
            std_dec = all_time_dec_std_lin[idx]
            ax.plot(time_points, mean_dec, color='blue', linewidth=2, marker='s', markersize=4,
                    label='Decreasing Window (t → End)', alpha=0.8)
            ax.fill_between(time_points, mean_dec - std_dec/np.sqrt(5), mean_dec + std_dec/np.sqrt(5),
                            color='blue', alpha=0.2)

            ax.set_ylim(global_ymin, global_ymax)
            ax.set_title(layers[idx], fontsize=20)
            if j >= 0:
                ax.set_xlabel('Time Point', fontsize=20)
            if j % 1 == 0:
                ax.set_ylabel('Accuracy', fontsize=20)
            if j == 0:
                ax.legend(loc='best', fontsize=10)

        # ---- KNN subplot grid (BOTTOM) ----
        #gs_knn = gs[1].subgridspec(2, 5)
        gs_knn = gs[1].subgridspec(1, 3)
        #knn_axes = [fig.add_subplot(gs_knn[i, j]) for i in range(2) for j in range(5)]
        knn_axes = [fig.add_subplot(gs_knn[i, j]) for i in range(1) for j in range(3)]

        for j, idx in enumerate(new_order):
            ax = knn_axes[j]
            ax.plot(time_points, all_time_inc_acc_knn[idx],
                    color='red', linewidth=2, marker='o', markersize=4,
                    label='Increasing Window (Start → t)', alpha=0.8)
            ax.plot(time_points, all_time_dec_acc_knn[idx],
                    color='blue', linewidth=2, marker='s', markersize=4,
                    label='Decreasing Window (t → End)', alpha=0.8)

            ax.set_ylim(global_ymin, global_ymax)
            ax.set_title(layers[idx], fontsize=20)
            if j >= 0:
                ax.set_xlabel('Time Point', fontsize=20)
            if j % 1 == 0:
                ax.set_ylabel('Accuracy', fontsize=20)
            if j == 0:
                ax.legend(loc='best', fontsize=10)

        plt.tight_layout(h_pad=3)
        plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3)  # leave space for titles

        # --- Block titles positioned relative to grids ---
        # LR title: above the first row block
        lr_top = lr_axes[0].get_position().y1  # top of first LR subplot
        fig.text(0.5, lr_top + 0.03, "Logistic Regression (LR)",
                ha="center", va="bottom", fontsize=24, weight="bold")

        # KNN title: above the second row block
        knn_top = knn_axes[0].get_position().y1  # top of first KNN subplot
        fig.text(0.5, knn_top + 0.03, "K-Nearest Neighbors (KNN)",
                ha="center", va="bottom", fontsize=24, weight="bold")
        plt.savefig("../figs/knn_lr_combined.png", bbox_inches="tight")
        plt.savefig("../figs/knn_lr_combined.pdf", bbox_inches="tight")

# ==== OSI & DSI utilities (moved from CNN_sampling/osi_dsi.py) ====

def compute_osi(responses: np.ndarray, angles: np.ndarray) -> float:
    numerator = np.abs(np.sum(responses * np.exp(1j * 2 * angles)))
    denominator = np.sum(responses)
    return numerator / (denominator + 1e-8)

def compute_dsi(responses: np.ndarray, angles: np.ndarray) -> float:
    numerator = np.abs(np.sum(responses * np.exp(1j * angles)))
    denominator = np.sum(responses)
    return numerator / (denominator + 1e-8)

def save_pink_noise_video(frames: np.ndarray, output_path="pink_noise_input.mp4", fps=30):
    if len(frames.shape) != 3:
        raise ValueError("Expected input shape [T, H, W] for grayscale video.")
    T, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=False)
    for frame in frames:
        out.write(frame)
    out.release()

def generate_directional_pink_noise_frames(
    num_directions=16,
    frames_per_direction=37,
    frame_size=(144, 256),
    seed=42
):
    np.random.seed(seed)
    H, W = frame_size
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)

    def pink_noise(shape):
        white = np.random.randn(*shape)
        f = np.fft.fft2(white)
        fshift = np.fft.fftshift(f)
        Y, X = np.ogrid[:shape[0], :shape[1]]
        center = (shape[0] / 2, shape[1] / 2)
        dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
        dist[dist == 0] = 1
        fshift = fshift / dist
        f_ishift = np.fft.ifftshift(fshift)
        pink = np.fft.ifft2(f_ishift).real
        pink = (pink - pink.min()) / (pink.max() - pink.min()) * 255
        return pink.astype(np.uint8)

    frames = []
    direction_labels = []

    for i, angle in enumerate(angles):
        base = pink_noise(frame_size)
        flow_x = np.cos(angle)
        flow_y = np.sin(angle)
        for t in range(frames_per_direction):
            dx = int(flow_x * t * 1.5)
            dy = int(flow_y * t * 1.5)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            frames.append(shifted)
            direction_labels.append(i)

    save_pink_noise_video(np.array(frames, dtype=np.uint8))
    return np.array(frames, dtype=np.uint8), angles, np.array(direction_labels)

def load_flow_stimuli():
    scl_factor = 0.7
    N_INSTANCES = 3
    trial_len = 75 // 2
    stride = 1
    mydirs = list(map(str, range(0, 360, 45)))
    categories = ['grat_W12', 'grat_W1', 'grat_W2',
                  'neg1dotflow_D1_bg', 'neg3dotflow_D1_bg', 'neg1dotflow_D2_bg', 'neg3dotflow_D2_bg',
                  'pos1dotflow_D1_bg', 'pos3dotflow_D1_bg', 'pos1dotflow_D2_bg', 'pos3dotflow_D2_bg']
    flow_dataset = createFlowDataset(categories, 'flowstims', mydirs, (800, 600), (144, 256),
                             scl_factor, N_INSTANCES, trial_len, stride)[0]
    flow_frames = np.array([frame.reshape(144, 256) for frame in flow_dataset], dtype="uint8")
    return flow_frames

def compute_osi_dsi_per_neuron(output, angles, dir_labels):
    num_frames, num_neurons = output.shape
    num_dirs = len(angles)
    osi_list = []
    dsi_list = []
    for neuron_idx in range(num_neurons):
        responses_by_dir = {k: [] for k in range(num_dirs)}
        for t in range(num_frames):
            val = output[t, neuron_idx].item()
            direction_idx = dir_labels[t]
            responses_by_dir[direction_idx].append(val)
        mean_responses = np.array([np.mean(responses_by_dir[i]) for i in range(num_dirs)])
        osi = compute_osi(mean_responses, angles)
        dsi = compute_dsi(mean_responses, angles)
        osi_list.append(osi)
        dsi_list.append(dsi)
    return np.array(osi_list), np.array(dsi_list)

def compute_flow_osi_dsi_max_per_neuron(model, angles, device):
    flow_frames = load_flow_stimuli()
    T_total = flow_frames.shape[0]
    num_categories = 11
    num_directions = 8
    frames_per_trial = T_total // (num_categories * num_directions)
    assert T_total % (num_categories * num_directions) == 0, "Unexpected number of frames"
    all_osi = []
    all_dsi = []
    for cat_idx in range(num_categories):
        start_idx = cat_idx * num_directions * frames_per_trial
        end_idx = (cat_idx + 1) * num_directions * frames_per_trial
        cat_frames = flow_frames[start_idx:end_idx]
        dir_labels = []
        for d in range(num_directions):
            dir_labels.extend([d] * frames_per_trial)
        dir_labels = np.array(dir_labels)
        with torch.no_grad():
            output = model.predict(stimuli=cat_frames)
        osi, dsi = compute_osi_dsi_per_neuron(output, angles, dir_labels)
        all_osi.append(osi)
        all_dsi.append(dsi)
    all_osi = np.stack(all_osi)
    all_dsi = np.stack(all_dsi)
    max_osi = np.mean(all_osi, axis=0)
    max_dsi = np.mean(all_dsi, axis=0)
    return max_osi, max_dsi

def plot_comparison_hist_and_scatter(
    pink_osi, pink_dsi, flow_osi, flow_dsi, save_path="../fig/osi_dsi_comparison.pdf"
):
    mean_osi = np.mean(pink_osi)
    mean_dsi = np.mean(pink_dsi)
    mean_osi_flow = np.mean(flow_osi)
    mean_dsi_flow = np.mean(flow_dsi)
    with plt.rc_context({**bundles.neurips2024(), **axes.lines()}):
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 3, 1)
        plt.hist(pink_osi, bins=30, color="pink", edgecolor="k")
        plt.title(f"Pink Noise OSI\nMean = {mean_osi:.3f}", fontsize=20)
        plt.axvline(mean_osi, color='red', linestyle='--')
        plt.xlabel("OSI", fontsize=20)
        plt.ylabel("Neurons", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.subplot(2, 3, 2)
        plt.hist(pink_dsi, bins=30, color="pink", edgecolor="k")
        plt.title(f"Pink Noise DSI\nMean = {mean_dsi:.3f}", fontsize=20)
        plt.axvline(mean_dsi, color='red', linestyle='--')
        plt.xlabel("DSI", fontsize=20)
        plt.ylabel("Neurons", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.subplot(2, 3, 4)
        plt.hist(flow_osi, bins=30, color="blue", edgecolor="k")
        plt.title(f"Flow Stimuli OSI \nMean = {mean_osi_flow:.3f}", fontsize=20)
        plt.axvline(mean_osi_flow, color='red', linestyle='--')
        plt.xlabel("OSI", fontsize=20)
        plt.ylabel("Neurons", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.subplot(2, 3, 5)
        plt.hist(flow_dsi, bins=30, color="blue", edgecolor="k")
        plt.title(f"Flow Stimuli DSI \nMean = {mean_dsi_flow:.3f}", fontsize=20)
        plt.axvline(mean_dsi_flow, color='red', linestyle='--')
        plt.xlabel("DSI", fontsize=20)
        plt.ylabel("Neurons", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.subplot(2, 3, 3)
        plt.scatter(pink_osi, flow_osi, alpha=0.5, color="blue")
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("Pink OSI", fontsize=20)
        plt.ylabel("Flow OSI", fontsize=20)
        plt.title("OSI Comparison", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.subplot(2, 3, 6)
        plt.scatter(pink_dsi, flow_dsi, alpha=0.5, color="blue")
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("Pink DSI", fontsize=20)
        plt.ylabel("Flow DSI", fontsize=20)
        plt.title("DSI Comparison", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def create_osi_dsi_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ids = microns.scan(session=8, scan_idx=5)
    model.to(device)
    model.eval()
    pink_frames, pink_angles, pink_dir_labels = generate_directional_pink_noise_frames()
    with torch.no_grad():
        pink_output = model.predict(stimuli=pink_frames)
    pink_osi, pink_dsi = compute_osi_dsi_per_neuron(pink_output, pink_angles, pink_dir_labels)
    flow_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    flow_osi, flow_dsi = compute_flow_osi_dsi_max_per_neuron(model, flow_angles, device)
    plot_comparison_hist_and_scatter(pink_osi, pink_dsi, flow_osi, flow_dsi, save_path="../fig/osi_dsi_comparison.pdf")


create_intensity_plot_block2()
create_intensity_plot_hidden()
intensity_comparison_plot()
create_knn_lr_plots()
create_osi_dsi_plot()