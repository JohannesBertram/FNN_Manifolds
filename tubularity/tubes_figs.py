# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import analysis pipeline functions from tubesTest.py if needed (or copy them here)
from tubesTest import preprocess_curves, run_clustering, compute_cluster_metrics, aggregate_metrics, plot_scenario_overview, plot_per_cluster_bars, plot_radius_profiles, plot_scenario_overview_proper

data_paths = [
    os.path.abspath(os.path.join(os.pardir, 'data', 'retina_tensor_traces.npy')),
    os.path.abspath(os.path.join(os.pardir, 'data', 'V1_tensor_traces.npy')),
    os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_inputs0_maxFr_maxNr_seed1.npy')),
    #os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_blocks0_maxFr_maxNr_seed1.npy')),
    #os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_inputs1_maxFr_maxNr_seed1.npy')),
    #os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_blocks1_maxFr_maxNr_seed1.npy')),
    #os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_inputs2_maxFr_maxNr_seed1.npy')),
    os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_blocks2_maxFr_maxNr_seed2.npy')),
    os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_hidden_maxFr_maxNr_seed1.npy')),
    os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_recurrentout_maxFr_maxNr_seed1.npy')),
    os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_act_i3_n2000_SCL0_7_TL37_position_maxFr_maxNr_seed1.npy')),
    os.path.abspath(os.path.join(os.pardir, 'data', 'sampled_data', 'tensor4d_fnn07_seed3.npy'))
]

exp_names = [
    "Retina",
    "V1",
    "Enc1", 
    "Enc13", 
    "Rec",
    "RecOut",
    "Readout",
    "Output"
]

use_ground_truth_labels = True

result_dict = {}

bio_t = []
art_t = []
bio_c = []
art_c = []
enc_c = []
later_c = []

for i in range(len(data_paths)):
    data_path = data_paths[i]
    data = np.load(data_path)
    
    if "position" in data_path:
        data = data[:, :6]
        #data = np.concatenate((np.zeros((len(data), 6, 8, 1)), data), axis=3)
    elif "fnn" in data_path:
        data = data[:, np.array([0, 6, 7, 8, 9, 10])]
        #data = np.concatenate((np.zeros((len(data), 6, 8, 1)), data), axis=3)
    print(f"Loaded data shape: {data.shape}")

    n_neurons, stim, dirs, time = data.shape

    orig_data = np.transpose(data, (1, 2, 3, 0)).reshape(stim*dirs*time, -1)

    bootstrapped_t = []
    bootstrapped_c = []
    for j in range(30):
        rng = np.random.default_rng()

        data = rng.choice(orig_data, axis=1, size=orig_data.shape[1], replace=True) 
        print(f"before: {data.shape}")

        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=10))])
        #pipeline = Pipeline([('pca', PCA(n_components=10))])

        pca_traj_result = pipeline.fit_transform(data)
        data = pca_traj_result.reshape(stim*dirs, time, -1)

        print(data.shape)
        curves_raw = list(data)

        M = 150  # Number of points to resample each curve to
        curves = preprocess_curves(curves_raw, M=M)
        labels, _ = run_clustering(curves, metric="H1", alpha=0.5, min_cluster_size=4)

        if use_ground_truth_labels:
            labels = np.repeat(np.array(range(stim)), dirs)  # Set ground truth labels

        results = compute_cluster_metrics(curves, labels, smoothing=2.0, q=0.9, B=100)
        
        t_avg, c_avg, t_var, c_var = aggregate_metrics(results)
        #bootstrapped_results.append((t_avg, c_avg))
        bootstrapped_t.append(t_avg)
        bootstrapped_c.append(c_avg)

        if i < 2:
            bio_t.append(t_avg)
            bio_c.append(c_avg)
        else:
            art_t.append(t_avg)
            art_c.append(c_avg)
        if i >= 2 and i < 4:
            enc_c.append(c_avg)
        elif i >= 4:
            later_c.append(c_avg)
        print(f"mean S_tight={t_avg:.3f} (var={t_var:.3f}), mean S_cross={c_avg:.3f} (var={c_var:.3f})")

    if results:
        plot_scenario_overview_proper(curves_raw, curves, results, labels, 
                                title="curves + fitted centerlines", layer=exp_names[i])
        #plot_per_cluster_bars(results, scenario_name="Loaded Data")
        #plot_radius_profiles(results, scenario_name="Loaded Data")
        result_dict[exp_names[i]] = {}
        result_dict[exp_names[i]]["s_tight"] = np.mean(bootstrapped_t)
        result_dict[exp_names[i]]["s_tight_var"] = np.var(bootstrapped_t)
        result_dict[exp_names[i]]["s_cross"] = np.mean(bootstrapped_c)
        result_dict[exp_names[i]]["s_cross_var"] = np.var(bootstrapped_c)
        result_dict[exp_names[i]]["clusters"] = len(np.unique(labels))

        """if results:
        #plot_scenario_overview_proper(curves_raw, curves, results, labels, 
        #                        title="curves + fitted centerlines", layer=exp_names[i])
        #plot_per_cluster_bars(results, scenario_name="Loaded Data")
        #plot_radius_profiles(results, scenario_name="Loaded Data")
        result_dict[exp_names[i]] = {}
        result_dict[exp_names[i]]["s_tight"] = t_avg
        result_dict[exp_names[i]]["s_tight_var"] = t_var
        result_dict[exp_names[i]]["s_cross"] = c_avg
        result_dict[exp_names[i]]["s_cross_var"] = c_var
        result_dict[exp_names[i]]["clusters"] = len(np.unique(labels))"""

from scipy import stats
u_stat1, p_val_mw1 = stats.mannwhitneyu(bio_t, art_t, alternative='less')
print(u_stat1, p_val_mw1)
u_stat2, p_val_mw2 = stats.mannwhitneyu(bio_c, art_c, alternative='less')
print(u_stat2, p_val_mw2)
u_stat3, p_val_mw3 = stats.mannwhitneyu(later_c, enc_c, alternative='less')
print(u_stat3, p_val_mw3)

import matplotlib.pyplot as plt
import numpy as np

# Prepare data
exp_names = list(result_dict.keys())
s_tight = [result_dict[name]["s_tight"] for name in exp_names]
s_tight_var = [result_dict[name]["s_tight_var"] for name in exp_names]
s_cross = [result_dict[name]["s_cross"] for name in exp_names]
s_cross_var = [result_dict[name]["s_cross_var"] for name in exp_names]
clusters = [result_dict[name]["clusters"] for name in exp_names]

print(result_dict)

x = np.arange(len(exp_names))
width = 0.35


from tueplots import bundles, axes
from tueplots.constants.color import rgb

with plt.rc_context({**bundles.neurips2024(), **axes.lines()}):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'red'
    color2 = 'blue'

    # Left y-axis for s_tight
    rects1 = ax1.bar(x - width/2, s_tight, width, yerr=np.sqrt(s_tight_var) / np.sqrt(100), 
                    label='Tightness', color=color1, capsize=5)
    ax1.set_ylabel('Tightness', color=color1, fontsize=20)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=20)

    # Right y-axis for s_cross
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, s_cross, width, yerr=np.sqrt(s_cross_var) / np.sqrt(100), 
                    label='Crossing', color=color2, capsize=5)
    ax2.set_ylabel('Crossing', color=color2, fontsize=20)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=20)

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=20)
    ax1.set_xlim(-0.5, len(exp_names) - 0.5)
    #ax1.set_title('Tightness and Crossing Scores', fontsize=20)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=20)

    # Highlight biological and artificial regions
    ax1.axvspan(-0.5, 1.5, color='green', alpha=0.1)
    ax1.text(0.5, ax1.get_ylim()[1]*0.95, 'Biological', ha='center', va='top',
            fontsize=16, color='green', alpha=0.8)

    ax1.axvspan(1.5, 7.5, color='purple', alpha=0.1)
    ax1.text(3.5, ax1.get_ylim()[1]*0.95, 'FNN', ha='center', va='top',
            fontsize=16, color='purple', alpha=0.8)

    plt.tight_layout()
    plt.savefig("metrics_hdbscan.pdf")
