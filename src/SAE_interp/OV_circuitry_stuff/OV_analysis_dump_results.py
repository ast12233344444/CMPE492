import os
from matplotlib import pyplot as plt
import numpy as np
import json
import seaborn as sns
import pandas as pd


def plt_graph(graph, details = None, layer = -1):
    data = []
    for node, edges in graph.items():
        for target_class, potence_val in edges:
            data.append({"Feature": node, "Class": target_class, "Potence": potence_val})

    df = pd.DataFrame(data)

    if df.empty:
        print("No features passed the cutoff effect to visualize.")
    else:
        pivot_df = df.pivot(index="Feature", columns="Class", values="Potence")

        pivot_df = pivot_df.fillna(0)

        pivot_df['abs_impact'] = pivot_df.abs().sum(axis=1)
        pivot_df = pivot_df.sort_values(by='abs_impact', ascending=False).drop(columns=['abs_impact'])

        fig_height = max(6, len(pivot_df) * 0.3)
        plt.figure(figsize=(14, fig_height))

        max_val = df["Potence"].abs().max()

        ax = sns.heatmap(pivot_df,
                         cmap="coolwarm",
                         center=0,
                         vmin=-max_val,
                         vmax=max_val,
                         annot=False,
                         linewidths=0.5,
                         linecolor='lightgray',
                         cbar_kws={'label': 'Feature Potence (Effect)'})
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        if details is None:
            plt.title(f"Feature-Class direct injection effect Heatmap (effect on loss)", fontsize=16, pad=20)
        else:
            plt.title(f"Feature-Class direct injection effect Heatmap (effect on loss) ({details})", fontsize=16, pad=20)
        plt.ylabel("Features", fontsize=12)
        plt.xlabel("Classes", fontsize=12)

        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"feature_potence_{layer}.png"))
        df.to_csv(os.path.join(out_path, f"feature_potence_{layer}.csv"))
        plt.show()


if __name__ == "__main__":
    feature_potence_path = "/home/ahmet/PycharmProjects/CMPE492/alternative_SAE/feature_potence_calc.json"
    average_attention_data_path = "/home/ahmet/PycharmProjects/CMPE492/alternative_SAE/avg_attention_scores.json"
    out_path = "/home/ahmet/PycharmProjects/CMPE492/alternative_SAE/OV_dump/"
    os.makedirs(out_path, exist_ok=True)

    feature_potence_data = json.load(open(feature_potence_path))
    average_attention_data = json.load(open(average_attention_data_path))
    feature_presence_data = average_attention_data["class_presence_probs"]
    average_attention_data = average_attention_data["class_avg_attentions"]


    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    features_per_layer = {}
    for layer_i in [10, 11]:
        cutoff_effect = 3e-4
        n_toks = 197
        n_heads = 12
        full_layer_effect_datas_by_class = []
        features_per_layer[f"l{layer_i}"] = []

        for head_i in range(n_heads):
            att_data = None
            for i in range(len(classes)):
                if att_data is None:
                    div_coeff = np.zeros(len(average_attention_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"]) + 1)
                    att_data = np.zeros(len(average_attention_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"]) + 1)
                pres = np.array(feature_presence_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"] + [1])
                att_data += pres * np.array(average_attention_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"] + [1 / n_toks])
                div_coeff += pres
            att_data /= (div_coeff + 1e-9)

            for i in range(len(classes)):
                pot_data = np.array(feature_potence_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"])
                if head_i == 0:
                    full_layer_effect_datas_by_class.append(att_data * pot_data)
                else:
                    full_layer_effect_datas_by_class[i] += att_data * pot_data

        plt.figure(figsize=(16, 9))
        for i in range(len(classes)):
            plt.hist(np.clip(full_layer_effect_datas_by_class[i], a_min = -1, a_max = 1), bins=1000, label = classes[i], alpha = 0.2)
        plt.title(f"Feature-Class average attention histogram (effect on loss) (Layer {layer_i})", fontsize=16, pad=20)
        plt.axvline(cutoff_effect, linestyle='--', color = 'red')
        plt.axvline(-cutoff_effect, linestyle='--', color = 'red')
        plt.legend()

        graph_layer = {}
        for clas in classes:
            graph_layer[clas] = []
        for i in range(len(classes)):
            for j in range(len(full_layer_effect_datas_by_class[i])):
                if abs(full_layer_effect_datas_by_class[i][j]) > cutoff_effect:
                    if f"feature_{j}" not in graph_layer:
                        graph_layer[f"feature_{j}"] = []
                        features_per_layer[f"l{layer_i}"].append(f"feature{j}")
                    graph_layer[f"feature_{j}"].append((classes[i], np.cbrt(full_layer_effect_datas_by_class[i][j])))

        plt_graph(graph_layer, details = f"layer {layer_i}", layer=layer_i)
