import json
import math
import os
import random

import numpy as np
from PIL import Image
import torch
from PIL.ImageChops import offset
from matplotlib import pyplot as plt
from sympy.strategies.core import switch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import ViTCompGraph
from src.TracingAlgorithms import TracingAlgorithms
from src.data_setups import TargetCorruptedImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO select more carefully when feeding inputs into IG Algorithm
#TODO PLOT ACTIVATIONS (AND MAYBE GRADS) OF RESPECTIVE EDGES ON PCA

class Experiments:
    @staticmethod
    def analyse_topk_contribution(model, data_path, processor, classes,
                                  n_samples=90 * 10, topk_plots=[1, 2, 3, 5, 10],
                                  measure_at_logit="out", metric_factory=None, metric_target="inc"):
        n_samples_per_class = (n_samples + len(classes) * (len(classes) - 1) - 1) // (len(classes) * (len(classes) - 1))
        dataloaders =[[None for _ in range(len(classes))] for _ in range(len(classes))]

        topk = {k: [] for k in topk_plots}
        topk_inclusions = {k: {} for k in topk_plots}
        topk_inclusions["total_trials"] = n_samples
        sole_edge_improvements = {}
        for true_i in range(len(classes)):
            for corr_i in range(len(classes)):
                if true_i == corr_i:
                    continue
                begin_idx = (true_i * (len(classes) - 1) + corr_i) * n_samples_per_class
                end_idx = min(begin_idx + n_samples_per_class, n_samples)
                my_dataset = TargetCorruptedImageDataset(
                    data_path=data_path,
                    processor=processor,
                    true_label=classes[true_i],
                    distorted_label=classes[corr_i],
                    offset = begin_idx,
                )
                my_dataset.n_samples = end_idx - begin_idx

                dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=1)

                for clean, adversarial in tqdm(dataloader, "loading data..."):
                    metric = metric_factory(true_i, corr_i)

                    compGraph = ViTCompGraph.Graph.from_model(model)

                    TracingAlgorithms.EAP_direct(model, compGraph, adversarial, clean, metric_fn=metric, measure_at_logit=measure_at_logit, metric_target=metric_target)

                    edge_names = []
                    for edge in compGraph.edges:
                        edgeObj = compGraph.edges[edge]
                        if edgeObj.in_graph:
                            edge_names.append(edgeObj.name)
                    pli = []
                    full_loss = TracingAlgorithms.patch_activations_direct(model, [],
                                                                                     clean, adversarial,
                                                                                     metric, measure_at_logit=measure_at_logit)
                    full_clean = TracingAlgorithms.patch_activations_direct(model, [],
                                                                                       adversarial, clean,
                                                                                       metric, measure_at_logit=measure_at_logit)
                    full_attack = full_loss - full_clean
                    for i in range(len(edge_names)):
                        if edge_names[i] not in sole_edge_improvements:
                            sole_edge_improvements[edge_names[i]] = []
                        edges_to_patch = [edge_names[i]]

                        patched_loss = TracingAlgorithms.patch_activations_direct(model, edges_to_patch,
                                                                                    clean, adversarial,
                                                                                    metric, measure_at_logit=measure_at_logit)
                        imp = full_loss - patched_loss
                        pli.append((imp, edge_names[i]))
                        sole_edge_improvements[edge_names[i]].append(imp / full_attack)
                    pli = sorted(pli, reverse=True)
                    for k in topk_plots:
                        topk_edges = [edge_name for (edge_score, edge_name) in pli[:k]]
                        patched_loss = TracingAlgorithms.patch_activations_direct(model, topk_edges,
                                                                                    clean, adversarial,
                                                                                    metric, measure_at_logit=measure_at_logit)
                        topk[k].append((full_loss - patched_loss) / full_attack)
                        for edge in topk_edges:
                            if edge in topk_inclusions[k]:
                                topk_inclusions[k][edge] += 1
                            else:
                                topk_inclusions[k][edge] = 1

        ###---- PLOTTING OF THE SOLE EDGE IMPORTANCE HISTOGRAMS-----
        max_set = [(value, key) for (key, value) in topk_inclusions[topk_plots[-1]].items()]
        max_set = sorted(max_set, reverse=True)
        edge_impss = {}
        for i in range(min(10, len(max_set))):
            velue, key = max_set[i]
            edge_impss[key] = sole_edge_improvements[key]
        for (key, val) in edge_impss.items():
            while len(val) < n_samples:
                val.append(0)

        if len(edge_impss) > 0:
            n_edge_plots = len(edge_impss)
            n_edge_cols = 3
            n_edge_rows = math.ceil(n_edge_plots / n_edge_cols)

            # Create a new figure specifically for the edge histograms
            # We use a distinct figure object (fig_edges) to avoid conflict with the next plot
            fig_edges, axes_edges = plt.subplots(n_edge_rows, n_edge_cols, figsize=(5 * n_edge_cols, 4 * n_edge_rows))
            axes_edges_flat = axes_edges.flatten()

            for idx, (edge_name, values) in enumerate(edge_impss.items()):
                ax = axes_edges_flat[idx]

                # Sanitization: Ensure values are simple floats (handling Torch tensors)
                clean_values = []
                for v in values:
                    if hasattr(v, 'item'):
                        clean_values.append(v.item())
                    elif hasattr(v, 'detach'):
                        clean_values.append(v.detach().cpu().numpy())
                    else:
                        clean_values.append(v)

                # Plot histogram
                ax.hist(clean_values, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
                ax.set_title(f'{edge_name}', fontsize=10)
                ax.set_xlabel('Sole Edge Improvement')
                ax.set_ylabel('Frequency')
                ax.grid(axis='y', linestyle='--', alpha=0.5)

            # Remove empty subplots if grid is larger than number of edges
            for i in range(n_edge_plots, len(axes_edges_flat)):
                fig_edges.delaxes(axes_edges_flat[i])

            plt.tight_layout()

            # Save path for the edge improvements
            edge_imp_path = "/home/ahmet/PycharmProjects/CMPE492/results/topk_sole_edge_improvements.png"
            plt.savefig(edge_imp_path)
            print(f"Edge improvement histograms saved to {edge_imp_path}")
            plt.close(fig_edges)
        ###---- PLOTTING OF THE SOLE EDGE IMPORTANCE HISTOGRAMS-----

        ###---- PLOTTING THE TOPK EDGE IMPROVEMENT PLOTS--------
        n_plots = len(topk_plots)
        n_cols = 3
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_flat = axes.flatten()

        for idx, k in enumerate(topk_plots):
            ax = axes_flat[idx]
            values = topk[k]
            clean_values = []
            for v in values:
                if hasattr(v, 'item'):
                    clean_values.append(v.item())
                elif hasattr(v, 'detach'):
                    clean_values.append(v.detach().cpu().numpy())
                else:
                    clean_values.append(v)

            ax.hist(clean_values, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'Top-{k} Edges Contribution')
            ax.set_xlabel('Fraction of Attack Explained')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        for i in range(n_plots, len(axes_flat)):
            fig.delaxes(axes_flat[i])
        plt.tight_layout()

        json_save_path = "/home/ahmet/PycharmProjects/CMPE492/results/topk_inclusion_data.json"
        with open(json_save_path, 'w') as f:
            json.dump(topk_inclusions, f, indent=4)
        save_path = "/home/ahmet/PycharmProjects/CMPE492/results/topk_contribution_subplots.png"
        plt.savefig(save_path)
        print(f"Analysis complete. Plot saved to {save_path}")
        plt.close()
        ###---- PLOTTING THE TOPK EDGE IMPROVEMENT PLOTS--------

    @staticmethod
    def analyse_pairwise_corruption(model, dataloader, title, metric, measure_at_logit, metric_target="dec", switch_sides=False):

        print(f"analyzing {title}")

        compGraph = ViTCompGraph.Graph.from_model(model)
        if not os.path.exists(f"/home/ahmet/PycharmProjects/CMPE492/results/{title}/"):
            os.mkdir(f"/home/ahmet/PycharmProjects/CMPE492/results/{title}/")
        TracingAlgorithms.EAP(model, compGraph, dataloader,
                              img_path=f"/home/ahmet/PycharmProjects/CMPE492/results/{title}/compgraph.png",
                              metric_fn=metric, measure_at_logit=measure_at_logit, metric_target=metric_target,
                              switch_sides=switch_sides)

        edge_names = []
        for edge in compGraph.edges:
            edgeObj = compGraph.edges[edge]
            if edgeObj.in_graph:
                edge_names.append(edgeObj.name)

        patched_loss_improvements = {}
        pli = []

        full_corr_metric, full_corr_metric_cache = TracingAlgorithms.patch_activations(model, [],
                                                                         dataloader,
                                                                         metric, measure_at_logit=measure_at_logit)

        full_clean_metric, full_clean_metric_cache = TracingAlgorithms.patch_activations(model, [],
                                                                           dataloader,
                                                                           metric, measure_at_logit=measure_at_logit, flip_dl=True)

        if measure_at_logit == "in":
            deviation = (full_corr_metric + full_clean_metric) / 2
        else:
            deviation = (full_corr_metric - full_clean_metric)

        print(full_corr_metric, full_clean_metric, deviation)
        stdev = np.std(full_corr_metric_cache - full_clean_metric_cache)
        pli.append({"edge": "full_attack", "improvement": deviation, "stdev": stdev, "robustness": deviation / stdev})

        for i in range(len(edge_names)):
            edges_to_patch = [edge_names[i]]

            patched_loss, patched_loss_cache = TracingAlgorithms.patch_activations(model, edges_to_patch,
                                                                                   dataloader,
                                                                                   metric, measure_at_logit=measure_at_logit)
            deviation = full_corr_metric - patched_loss
            stdev = np.std(full_corr_metric_cache - patched_loss_cache)

            patched_loss_improvements[edge_names[i]] = {"improvement": deviation, "stdev": stdev,
                                                            "robustness": deviation / stdev}
            pli.append({"edge": edge_names[i], "improvement": deviation, "stdev": stdev, "robustness": deviation / stdev})

        pli = sorted(pli, key=lambda x: x["improvement"], reverse=True)

        json_path = f"/home/ahmet/PycharmProjects/CMPE492/results/{title}/impedgeinfo.json"
        with open(json_path, 'w') as f:
            json.dump(pli, f, indent=4)

        # We'll plot the top 50 edges for clarity
        plot_data = pli
        names = [x["edge"] for x in plot_data]
        robustness_vals = [x["robustness"] for x in plot_data]
        improvement_vals = [x["improvement"] for x in plot_data]

        fig, ax1 = plt.subplots(figsize=(12, 15))

        y_pos = np.arange(len(names))

        # Plot Robustness as the primary metric
        ax1.barh(y_pos, improvement_vals, align='center', color='skyblue', label='Robustness (Imp/Std)')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.invert_yaxis()  # Highest robustness at the top
        ax1.set_xlabel('Raw Loss Improvement')
        ax1.set_title(f'Edge Importance: {title}')

        #ax2 = ax1.twiny()
        #ax2.plot(improvement_vals, y_pos, 'ro', markersize=4, label='Raw Improvement')
        #ax2.set_xlabel('Raw Loss Improvement')

        fig.tight_layout()
        plt.savefig(f"/home/ahmet/PycharmProjects/CMPE492/results/{title}/impedgeinfo.png")
        plt.close()

    @staticmethod
    def plot_edge_info(model, processor, edge, classes, true_label_i, distorted_label_i, resolution = 100):
        true_label = classes[true_label_i]
        distorted_label = classes[distorted_label_i]

        def metric(logits, unused):
            return (logits[:, distorted_label_i] - logits[:, true_label_i]).mean()

        data_path = "/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset"
        path_clean = os.path.join(data_path, true_label, true_label)
        path_corrupted = os.path.join(data_path, true_label, distorted_label)
        files_clean = os.listdir(path_clean)
        file_name = random.sample(files_clean, 1)[0]

        dpoints_clean, dpoints_adversarial =[], []
        dpoint_clean = Image.open(os.path.join(path_clean, file_name)).convert("RGB")
        dpoints_clean.append(dpoint_clean)
        dpoint_adversarial = Image.open(os.path.join(path_corrupted, file_name)).convert("RGB")
        dpoints_adversarial.append(dpoint_adversarial)

        input_adversarial = processor(images=dpoints_adversarial, return_tensors="pt")["pixel_values"].to(device)
        input_clean = processor(images=dpoints_clean, return_tensors="pt")["pixel_values"].to(device)

        full_corrupt_metric, _ = TracingAlgorithms.patch_activations(model, [], input_clean, input_adversarial, [], metric)
        full_clean_metric, _= TracingAlgorithms.patch_activations(model, [], input_adversarial, input_clean, [], metric)

        TracingAlgorithms.trace_contrib_integrated_gradients(model, input_clean, input_adversarial, edge, resolution, metric, [])
        edge_patched_on_adversarial, _ = TracingAlgorithms.patch_activations(model, [edge], input_clean, input_adversarial, [], metric)
        print(f"corrupt to clean patch edge: {edge_patched_on_adversarial} - {full_corrupt_metric} = {edge_patched_on_adversarial - full_corrupt_metric}")

    @staticmethod
    def plot_activation_patterns(model, processor, edge, classes, class_a, class_b):
        true_label = classes[class_a]
        distorted_label = classes[class_b]

        data_path = "/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset"
        path_clean = os.path.join(data_path, true_label, true_label)
        path_corrupted = os.path.join(data_path, true_label, distorted_label)
        path_other = os.path.join(data_path, distorted_label, distorted_label)

        paths = [path_clean, path_corrupted, path_other]
        filesets = [os.listdir(path_clean), os.listdir(path_corrupted), os.listdir(path_other)]

        dpoints_clean_a = []
        dpoints_clean_b = []
        dpoints_corrupted = []
        for fileset, path in zip(filesets, paths):
            for file in fileset:
                dpoint = Image.open(os.path.join(path, file)).convert("RGB")
                if path == path_clean:
                    dpoints_clean_a.append(dpoint)
                if path == path_corrupted:
                    dpoints_corrupted.append(dpoint)
                if path == path_other:
                    dpoints_clean_b.append(dpoint)

        dpoints_clean_a = processor(images=dpoints_clean_a, return_tensors="pt")["pixel_values"].to(device)
        dpoints_clean_b = processor(images=dpoints_clean_b, return_tensors="pt")["pixel_values"].to(device)
        dpoints_corrupted = processor(images=dpoints_corrupted, return_tensors="pt")["pixel_values"].to(device)

        TracingAlgorithms.trace_activation_patterns(model, edge, dpoints_clean_a, dpoints_corrupted, dpoints_clean_b, reduce_mean=True)
        TracingAlgorithms.trace_activation_patterns(model, edge, dpoints_clean_a, dpoints_corrupted, dpoints_clean_b, reduce_mean=False)
