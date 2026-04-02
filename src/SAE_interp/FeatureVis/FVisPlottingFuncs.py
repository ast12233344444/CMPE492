import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm


def _plot_single_feature(args):
    """
    Standalone function to plot a single feature, designed to be run in a separate process.
    """
    layer, feature_no, top_k_max, top_k_avg, save_dir = args

    k = len(top_k_max)
    fig, axes = plt.subplots(4, k, figsize=(k * 3.5, 14.0))

    if k == 1:
        axes = axes.reshape(4, 1)

    def plot_subset(data, row_clean, row_heat, title_prefix):
        for i, (val, cls_act, spatial_acts, img_np) in enumerate(data):
            # Normalize image
            img_min, img_max = img_np.min(), img_np.max()
            img_np = (img_np - img_min) / (img_max - img_min + 1e-5)

            grid_size = int(math.sqrt(spatial_acts.shape[0]))
            spatial_heatmap = spatial_acts.reshape(grid_size, grid_size)
            spatial_heatmap[spatial_heatmap < 1e-6] = np.nan

            # 1. Plot Clean Image
            ax_clean = axes[row_clean, i]
            ax_clean.imshow(img_np)
            ax_clean.axis('off')
            ax_clean.set_title(f"{title_prefix} Top {i + 1}\nAct: {val:.2f} | [CLS]: {cls_act:.2f}", fontsize=11)

            # 2. Plot Heatmap Image
            ax_heat = axes[row_heat, i]
            ax_heat.imshow(img_np)
            im = ax_heat.imshow(
                spatial_heatmap,
                cmap='jet',
                alpha=0.5,
                extent=[0, img_np.shape[1], img_np.shape[0], 0]
            )
            ax_heat.axis('off')

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    plot_subset(top_k_max, row_clean=0, row_heat=1, title_prefix="Max")
    plot_subset(top_k_avg, row_clean=2, row_heat=3, title_prefix="Avg")

    row_labels = ['Max Clean', 'Max Heatmap', 'Avg Clean', 'Avg Heatmap']
    for ax, row_title in zip(axes[:, 0], row_labels):
        ax.annotate(row_title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    plt.suptitle(f"Layer {layer} | Feature {feature_no} Maximizers", fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"layer_{layer}_feature_{feature_no}.png")
    # Reduced DPI slightly from 150 to 120; it is usually indistinguishable and saves I/O time
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def save_top_k_visualizations(location_maximizers_max, location_maximisers_avg,
                              save_dir="/home/ahmet/PycharmProjects/CMPE492/alternative_SAE/feature_maximizers", max_workers=8):
    """
    Prepares data and dispatches plotting tasks to a ProcessPool for parallel execution.
    """
    os.makedirs(save_dir, exist_ok=True)

    tasks = []
    for layer, features in location_maximizers_max.items():
        for feature_no, heap_max in features.items():
            heap_avg = location_maximisers_avg[layer].get(feature_no, [])

            if not heap_max or not heap_avg:
                continue

            # Helper to detach tensors and move them to CPU numpy arrays before multiprocessing
            def preprocess_heap(heap):
                sorted_heap = sorted(heap, key=lambda x: x[0], reverse=True)
                processed = []
                for val, _, img_tensor, act_tensor in sorted_heap:
                    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    acts = act_tensor.squeeze().cpu().numpy()
                    cls_act = acts[0]
                    spatial_acts = acts[1:]
                    processed.append((val, cls_act, spatial_acts, img_np))
                return processed

            proc_max = preprocess_heap(heap_max)
            proc_avg = preprocess_heap(heap_avg)

            tasks.append((layer, feature_no, proc_max, proc_avg, save_dir))

    print(f"Submitting {len(tasks)} plotting tasks to ProcessPool with {max_workers} workers...")

    # Run the plotting in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show a progress bar for the parallel execution
        list(tqdm(executor.map(_plot_single_feature, tasks), total=len(tasks), desc="Saving Visualizations"))

    print("All visualizations saved successfully!")

def _plot_feature_pair(args):
    """
    Plots a pair of features side-by-side.
    Source feature on the left (3 cols), Destination feature on the right (3 cols).
    """
    layer, head, src_feat, dst_feat, src_max, src_avg, dst_max, dst_avg, save_dir = args

    k = len(src_max) # Expected to be 3 based on your n_samples
    fig, axes = plt.subplots(4, k * 2, figsize=(k * 2 * 3.5, 14.0))

    def plot_subset(data, row_clean, row_heat, col_offset, title_prefix):
        for i, (val, cls_act, spatial_acts, img_np) in enumerate(data):
            # Normalize image
            img_min, img_max = img_np.min(), img_np.max()
            img_np = (img_np - img_min) / (img_max - img_min + 1e-5)

            grid_size = int(math.sqrt(spatial_acts.shape[0]))
            spatial_heatmap = spatial_acts.reshape(grid_size, grid_size)
            spatial_heatmap[spatial_heatmap < 1e-6] = np.nan

            # 1. Plot Clean Image
            ax_clean = axes[row_clean, col_offset + i]
            ax_clean.imshow(img_np)
            ax_clean.axis('off')
            ax_clean.set_title(f"{title_prefix} Top {i + 1}\nAct: {val:.2f} | [CLS]: {cls_act:.2f}", fontsize=11)

            # 2. Plot Heatmap Image
            ax_heat = axes[row_heat, col_offset + i]
            ax_heat.imshow(img_np)
            im = ax_heat.imshow(
                spatial_heatmap,
                cmap='jet',
                alpha=0.5,
                extent=[0, img_np.shape[1], img_np.shape[0], 0]
            )
            ax_heat.axis('off')

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    # Plot Source Feature (Left side, col_offset = 0)
    plot_subset(src_max, row_clean=0, row_heat=1, col_offset=0, title_prefix="Src Max")
    plot_subset(src_avg, row_clean=2, row_heat=3, col_offset=0, title_prefix="Src Avg")

    # Plot Destination Feature (Right side, col_offset = k)
    plot_subset(dst_max, row_clean=0, row_heat=1, col_offset=k, title_prefix="Dst Max")
    plot_subset(dst_avg, row_clean=2, row_heat=3, col_offset=k, title_prefix="Dst Avg")

    row_labels = ['Max Clean', 'Max Heatmap', 'Avg Clean', 'Avg Heatmap']
    for ax, row_title in zip(axes[:, 0], row_labels):
        ax.annotate(row_title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    plt.suptitle(f"Layer {layer} | Head {head} | Source Feat {src_feat} (Left) -> Dest Feat {dst_feat} (Right)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Leave room for the suptitle

    save_path = os.path.join(save_dir, f"pair_src_{src_feat}_to_dst_{dst_feat}.png")
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def save_qk_pair_visualizations(layer, head, feature_pairs, location_maximizers_max, location_maximisers_avg, save_dir, max_workers=8):
    """
    Prepares paired data and dispatches plotting tasks to ProcessPool.
    """
    os.makedirs(save_dir, exist_ok=True)
    tasks = []

    def preprocess_heap(heap):
        sorted_heap = sorted(heap, key=lambda x: x[0], reverse=True)
        processed = []
        for val, _, img_tensor, act_tensor in sorted_heap:
            img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            acts = act_tensor.squeeze().cpu().numpy()
            cls_act = acts[0]
            spatial_acts = acts[1:]
            processed.append((val, cls_act, spatial_acts, img_np))
        return processed

    for pair in feature_pairs:
        src_feat, dst_feat = pair[0], pair[1]

        # Extract heaps for both features in the pair
        src_heap_max = location_maximizers_max[layer].get(src_feat, [])
        src_heap_avg = location_maximisers_avg[layer].get(src_feat, [])
        dst_heap_max = location_maximizers_max[layer].get(dst_feat, [])
        dst_heap_avg = location_maximisers_avg[layer].get(dst_feat, [])

        if not src_heap_max or not src_heap_avg or not dst_heap_max or not dst_heap_avg:
            continue

        proc_src_max = preprocess_heap(src_heap_max)
        proc_src_avg = preprocess_heap(src_heap_avg)
        proc_dst_max = preprocess_heap(dst_heap_max)
        proc_dst_avg = preprocess_heap(dst_heap_avg)

        tasks.append((layer, head, src_feat, dst_feat, proc_src_max, proc_src_avg, proc_dst_max, proc_dst_avg, save_dir))

    if tasks:
        print(f"Submitting {len(tasks)} pair plotting tasks for Layer {layer} Head {head}...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(_plot_feature_pair, tasks), total=len(tasks), desc=f"L{layer}H{head} Pairs"))
