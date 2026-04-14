import os
import torch
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import ViTImageProcessor, ViTForImageClassification

# --- Import your existing modules! ---
from src.SAE.train_sae import SparseAutoencoder
from src.SAE_interp.FeatureVis.FeatureVisBoring import get_feature_maximisers
from src.SAE_interp.FeatureVis.FVisPlottingFuncs import save_top_k_visualizations

# (Assuming trace_feature_lineage is saved in graph_vis.py, import it here)
# If it's in a different file, adjust the import accordingly.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels


def plot_lineage_features(
    model,
    dataloader,
    SAEs,
    lineage_history,
    target_layer,
    target_feature,
    save_root_dir,
    n_samples=5
):
    """
    Parses the lineage history, extracts maximizers for all features in the graph,
    and saves the visualisations into a dedicated descendant folder.
    """
    # 1. Parse locations from lineage history
    locations = []
    for layer_node, features in lineage_history.items():
        for feat_idx, score in features:
            locations.append(f"{layer_node}-{feat_idx}")

    print(f"\nExtracting maximizers for {len(locations)} features across the lineage...")

    # 2. Get Maximizers using your existing function
    loc_max, loc_avg = get_feature_maximisers(model, dataloader, SAEs, locations, n_samples)

    # 3. Create the specific folder for this descendant
    descendant_dir = os.path.join(save_root_dir, f"descendant_l{target_layer}_f{target_feature}")
    os.makedirs(descendant_dir, exist_ok=True)

    print(f"\nSaving lineage visualizations to {descendant_dir}...")

    # 4. Use your existing plotting function
    save_top_k_visualizations(loc_max, loc_avg, save_dir=descendant_dir)
    print("Lineage visualization complete!")

def trace_feature_lineage(
        start_layer: int,
        start_feature: int,
        cache_dir: str,
        stop_layer: int = 1,
        top_k: int = 10
):
    """
    Traces the contributing features backward through the network layers
    and returns a structured dictionary of the top K contributors per layer.
    """
    print(f"--- Tracing Lineage for Feature {start_feature} in Layer lnb{start_layer} ---")

    current_contributions = None

    # This dictionary will store our graph data.
    # We initialize it with the target feature (giving it a nominal score of 1.0 to represent the root).
    lineage_history = {
        f"lnb{start_layer}": [(start_feature, 1.0)]
    }

    # Step backward from the start_layer down to stop_layer
    for l in range(start_layer, stop_layer - 1, -1):
        l_node = f"lnb{l}"
        l_minus_1_node = f"lnb{l - 1}"

        # Load the interaction matrix W between l-1 and l
        matrix_path = os.path.join(cache_dir, f"interaction_scores_{l_minus_1_node}_to_{l_node}.pt")

        if not os.path.exists(matrix_path):
            print(f"Missing matrix file: {matrix_path}. Stopping trace here.")
            break

        print(f"Loading {l_minus_1_node} -> {l_node}...")
        W = torch.load(matrix_path, map_location="cpu")

        if current_contributions is None:
            # First step: Extract the k'th row for the target feature
            current_contributions = W[start_feature, :].clone()
        else:
            # Subsequent steps: Vector-Matrix multiplication (b @ W)
            current_contributions = torch.matmul(current_contributions, W)

        # Extract the highest contributing features at layer l-1
        top_vals, top_indices = torch.topk(current_contributions, k=top_k)

        # Save them to our history tracker
        layer_top_k = []
        print(f"Top {top_k} contributors in {l_minus_1_node}:")
        for rank, (val, idx) in enumerate(zip(top_vals, top_indices)):
            feat_idx = idx.item()
            score = val.item()

            layer_top_k.append((feat_idx, score))
            print(f"  #{rank + 1} | Feature {feat_idx:<5} | Score: {score:.4f}")

        # Add this layer's top K list to the dictionary
        lineage_history[l_minus_1_node] = layer_top_k

    return lineage_history


if __name__ == "__main__":
    # --- 1. Configuration & Paths ---
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    CACHE_DIR = "C:\\Users\\ast12\\PycharmProjects\\CMPE492\\results\\attn_caches"
    SAE_DIR = "C:\\Users\\ast12\\PycharmProjects\\CMPE492\\saved_models"
    VIS_OUTPUT_DIR = "C:\\Users\\ast12\\PycharmProjects\\CMPE492\\results\\lineage_visualizations"

    TARGET_LAYER = 11
    TARGET_FEATURE = 11034
    STOP_LAYER = 6
    TOP_K_CONTRIBUTORS = 5  # How many ancestors per layer to trace/plot
    N_IMG_SAMPLES = 5  # How many images to plot per feature

    EXPANSION_FACTOR = 16
    L1_COEFF = "0.0001"

    # --- 2. Run the Trace to get the Feature Graph ---
    graph_data = trace_feature_lineage(
        start_layer=TARGET_LAYER,
        start_feature=TARGET_FEATURE,
        cache_dir=CACHE_DIR,
        stop_layer=STOP_LAYER,
        top_k=TOP_K_CONTRIBUTORS
    )

    # Find out which unique layers are actually in our graph so we only load necessary SAEs
    layers_in_graph = set()
    for layer_node in graph_data.keys():
        layers_in_graph.add(layer_node)

    # --- 3. Setup Model and Dataloader ---
    print("\nSetting up Model and Dataloader...")
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager").to(device)
    model.eval()

    batch_size = 32
    dataset = datasets.CIFAR10(root='..\\data', train=True, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # --- 4. Load only the Required SAEs ---
    print("\nLoading Required SAEs...")
    SAEs = {}
    for layer_node in layers_in_graph:
        sae_path = os.path.join(SAE_DIR, f"sae_{layer_node}_ef{EXPANSION_FACTOR}_l1{L1_COEFF}.pt")

        SAE_metadata = torch.load(sae_path, map_location=device)
        SAE_model = SparseAutoencoder(input_dim=768, expansion_factor=EXPANSION_FACTOR).to(device)
        SAE_model.load_state_dict(SAE_metadata["model_state_dict"])

        SAEs[layer_node] = SAE_model
        print(f"  Loaded SAE for {layer_node}")

    # --- 5. Run the Extraction and Visualisation Pipeline ---
    plot_lineage_features(
        model=model,
        dataloader=dataloader,
        SAEs=SAEs,
        lineage_history=graph_data,
        target_layer=TARGET_LAYER,
        target_feature=TARGET_FEATURE,
        save_root_dir=VIS_OUTPUT_DIR,
        n_samples=N_IMG_SAMPLES
    )