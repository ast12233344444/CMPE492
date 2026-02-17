import json
import os

import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt

from src import ViTCompGraph
from src.TracingAlgorithms import TracingAlgorithms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiments:
    @staticmethod
    def analyse_topk_contribution(model, processor, classes, n_samples=90 * 10, topk_plots=[1, 2, 3, 5, 10]):
        n_samples_per_class = (n_samples + len(classes) * (len(classes) - 1) - 1) // (len(classes) * (len(classes) - 1))
        samples_clean = [None for _ in range(n_samples)]
        samples_adversarial = [None for _ in range(n_samples)]
        labels = []
        for true_i in range(len(classes)):
            for corr_i in range(len(classes)):
                if true_i == corr_i:
                    continue
                begin_idx = (true_i * (len(classes) - 1) + corr_i) * n_samples_per_class
                end_idx = min(begin_idx + n_samples_per_class, len(samples_clean))

                path_clean = os.path.join("/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset", classes[true_i],
                                          classes[true_i])
                path_corrupted = os.path.join("/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset",
                                              classes[true_i], classes[corr_i])
                names = os.listdir(path_clean)
                names = random.sample(names, end_idx - begin_idx)
                for sample_idx in range(len(names)):
                    dpoint_clean = Image.open(os.path.join(path_clean, names[sample_idx])).convert("RGB")
                    dpoint_adversarial = Image.open(os.path.join(path_corrupted, names[sample_idx])).convert("RGB")
                    samples_clean[sample_idx + begin_idx] = dpoint_clean
                    samples_adversarial[sample_idx + begin_idx] = dpoint_adversarial
                    labels.append((true_i, corr_i))

        input_adversarial = processor(images=samples_adversarial, return_tensors="pt")["pixel_values"].to(device)
        input_clean = processor(images=samples_clean, return_tensors="pt")["pixel_values"].to(device)

        topk = {k: [] for k in topk_plots}
        for i in range(len(samples_clean)):
            true_i, corr_i = labels[i]
            clean = input_clean[i:(i + 1)]
            adversarial = input_adversarial[i:(i + 1)]

            def metric(logits, unused):
                return (logits[:, corr_i] - logits[:, true_i]).mean()

            compGraph = ViTCompGraph.Graph.from_model(model)

            TracingAlgorithms.EAP(model, compGraph, clean, adversarial, true_i,
                                  img_path=f"ignore_compgraph.png", metric_fn=metric)

            edge_names = []
            for edge in compGraph.edges:
                edgeObj = compGraph.edges[edge]
                if edgeObj.in_graph:
                    edge_names.append(edgeObj.name)
            pli = []
            full_loss, full_loss_cache = TracingAlgorithms.patch_activations(model, [],
                                                                             clean, adversarial,
                                                                             torch.tensor([true_i]),
                                                                             metric)
            full_clean, full_clean_cache = TracingAlgorithms.patch_activations(model, [],
                                                                               adversarial, clean,
                                                                               torch.tensor([true_i]),
                                                                               metric)
            full_attack = full_loss - full_clean
            for i in range(len(edge_names)):
                edges_to_patch = [edge_names[i]]

                patched_loss, patched_loss_cache = TracingAlgorithms.patch_activations(model, edges_to_patch,
                                                                                       clean, adversarial,
                                                                                       torch.tensor([true_i]), metric)
                imp = full_loss - patched_loss
                pli.append(imp)
            pli = sorted(pli, reverse=True)
            for k in topk_plots:
                topk[k].append(sum(pli[:k]) / full_attack)

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

            ax.hist(clean_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'Top-{k} Edges Contribution')
            ax.set_xlabel('Fraction of Attack Explained')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        for i in range(n_plots, len(axes_flat)):
            fig.delaxes(axes_flat[i])
        plt.tight_layout()

        save_path = "/home/ahmet/PycharmProjects/CMPE492/results/topk_contribution_subplots.png"
        plt.savefig(save_path)
        print(f"Analysis complete. Plot saved to {save_path}")
        plt.close()

    @staticmethod
    def analyse_pairwise_corruption(model, processor, true_label, distorted_label, true_label_i, distorted_label_i):

        def generate_pairwise_dataset(data_path, processor, true_label, distorted_label):
            path_clean = os.path.join(data_path, true_label, true_label)
            path_corrupted = os.path.join(data_path, true_label, distorted_label)
            files_clean = os.listdir(path_clean)
            files_corrupted = os.listdir(path_corrupted)
            dpoints_clean = []
            dpoints_adversarial = []
            for file in files_clean:
                dpoint_clean = Image.open(os.path.join(path_clean, file)).convert("RGB")
                dpoints_clean.append(dpoint_clean)
                dpoint_adversarial = Image.open(os.path.join(path_corrupted, file)).convert("RGB")
                dpoints_adversarial.append(dpoint_adversarial)

            input_adversarial = processor(images=dpoints_adversarial, return_tensors="pt")["pixel_values"].to(device)
            input_clean = processor(images=dpoints_clean, return_tensors="pt")["pixel_values"].to(device)

            return input_clean, input_adversarial

        print(f"analyzing {true_label} to  {distorted_label}")
        datas_clean, datas_adversarial = generate_pairwise_dataset(
            "/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset", processor,
            true_label, distorted_label)

        # datas_adversarial, datas_clean = datas_adversarial[:16], datas_clean[:16]

        def metric(logits, unused):
            return (logits[:, distorted_label_i] - logits[:, true_label_i]).mean()

        compGraph = ViTCompGraph.Graph.from_model(model)

        TracingAlgorithms.EAP(model, compGraph, datas_clean, datas_adversarial, true_label_i,
                              img_path=f"results/{true_label}->{distorted_label}_compgraph.png", metric_fn=metric)

        edge_names = []
        for edge in compGraph.edges:
            edgeObj = compGraph.edges[edge]
            if edgeObj.in_graph:
                edge_names.append(edgeObj.name)

        patched_loss_improvements = {}
        pli = []

        full_loss, full_loss_cache = TracingAlgorithms.patch_activations(model, [],
                                                                         datas_clean, datas_adversarial,
                                                                         torch.tensor([]),
                                                                         metric)

        full_clean, full_clean_cache = TracingAlgorithms.patch_activations(model, [],
                                                                           datas_adversarial, datas_clean,
                                                                           torch.tensor([]),
                                                                           metric)

        imp = full_loss - full_clean
        stdev = np.std(full_loss_cache - full_clean_cache)
        pli.append({"edge": "full_attack", "improvement": imp, "stdev": stdev, "robustness": imp / stdev})

        for i in range(len(edge_names)):
            edges_to_patch = [edge_names[i]]

            patched_loss, patched_loss_cache = TracingAlgorithms.patch_activations(model, edges_to_patch,
                                                                                   datas_clean, datas_adversarial,
                                                                                   torch.tensor([]),
                                                                                   metric)
            imp = full_loss - patched_loss
            stdev = np.std(full_loss_cache - patched_loss_cache)
            if stdev > 0:
                patched_loss_improvements[edge_names[i]] = {"improvement": imp, "stdev": stdev,
                                                            "robustness": imp / stdev}
                pli.append({"edge": edge_names[i], "improvement": imp, "stdev": stdev, "robustness": imp / stdev})
            else:
                patched_loss_improvements[edge_names[i]] = {"improvement": imp, "stdev": stdev, "robustness": 0}
                pli.append({"edge": edge_names[i], "improvement": imp, "stdev": stdev, "robustness": 0})

        pli = sorted(pli, key=lambda x: x["improvement"], reverse=True)

        json_path = f"results/{true_label}->{distorted_label}_impedgeinfo.json"
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
        bars = ax1.barh(y_pos, robustness_vals, align='center', color='skyblue', label='Robustness (Imp/Std)')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.invert_yaxis()  # Highest robustness at the top
        ax1.set_xlabel('Robustness Score')
        ax1.set_title(f'Edge Importance: {true_label} -> {distorted_label}')

        ax2 = ax1.twiny()
        ax2.plot(improvement_vals, y_pos, 'ro', markersize=4, label='Raw Improvement')
        ax2.set_xlabel('Raw Loss Improvement')

        fig.tight_layout()
        plt.savefig(f"results/{true_label}->{distorted_label}_impedgeinfo.png")
        plt.close()