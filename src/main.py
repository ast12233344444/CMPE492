import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision import datasets
import ViTCompGraph
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from torchviz import make_dot
from nnsight import NNsight
import torch.nn.functional as F

def get_target_layer_index(edge_str):
    """Parses edge string to find the target layer index for sorting."""
    _, v = edge_str.split("->")
    if v == "logits": return 999
    # Format usually 'a10.h2' or 'm5'
    if v.startswith("a"): return 2*int(v.split(".")[0][1:])
    if v.startswith("m"): return 2*int(v[1:])+1
    return -1

def patch_activations(model, edges_to_patch, clean_input, corrupted_input, truth_label):
    model_nnsight = NNsight(model)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    device = clean_input.device
    edges_sorted = sorted(edges_to_patch, key=get_target_layer_index)

    # 1. PRE-COMPUTE CLEAN VALUES (Static Targets)
    clean_values = {}
    with model_nnsight.trace(clean_input, trace=False):
        for layer_idx in range(n_layers):
            layer_module = model_nnsight.vit.encoder.layer[layer_idx]

            # MLP Output
            clean_values[f"m{layer_idx}"] = layer_module.output.dense.output.save()

            # Attention Heads (Projected)
            # We need the output of the SelfAttention (before the dense Output layer)
            # AND the weights of the Dense Output layer to manually project later.
            concat_heads = layer_module.attention.attention.output[0]
            W_O = layer_module.attention.output.dense.weight

            for h in range(n_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                # Project specific head
                head_z = concat_heads[:, :, start:end]
                W_O_head = W_O[:, start:end]
                clean_values[f"a{layer_idx}.h{h}"] = (head_z @ W_O_head.T).save()

    # 2. ITERATIVE PATCHING (Dynamic Steering)
    patched_logits = None
    patched_loss = None

    with model_nnsight.trace(corrupted_input) as tracer:

        # We iterate through the edges.
        # Since nnsight builds a graph, the order of definition here matters.
        # Upstream patches defined first will affect downstream nodes naturally.
        for edge in edges_sorted:
            u, v = edge.split("->")

            # --- A. GET LIVE VALUE OF SOURCE (u) ---
            # This is the key fix: We don't use a saved dictionary.
            # We compute the value of 'u' right now in the graph.

            current_u_value = None

            if u.startswith("m"):
                l_src = int(u[1:])
                # Live output of the MLP
                current_u_value = model_nnsight.vit.encoder.layer[l_src].output.dense.output

            elif u.startswith("a"):
                l_src = int(u.split(".")[0][1:])
                h_src = int(u.split(".")[1][1:])

                # Re-construct the specific head's output from the live stream
                layer_module = model_nnsight.vit.encoder.layer[l_src]
                concat_heads = layer_module.attention.attention.output[0]
                W_O = layer_module.attention.output.dense.weight

                start, end = h_src * head_dim, (h_src + 1) * head_dim

                head_z = concat_heads[:, :, start:end]
                W_O_head = W_O[:, start:end]
                current_u_value = head_z @ W_O_head.T

            # --- B. CALCULATE STEERING ---
            # Steering = Target (Clean) - Current (Potentially modified by upstream patches)
            steering = clean_values[u] - current_u_value

            # --- C. APPLY PATCH TO DESTINATION (v) ---
            if v == "logits":
                model_nnsight.classifier.input += steering

            elif v.startswith("a"):
                l_tgt = int(v.split(".")[0][1:])
                h_tgt = int(v.split(".")[1][1:])

                layer_module = model_nnsight.vit.encoder.layer[l_tgt]
                ln_module = layer_module.layernorm_before

                # Through-LN Logic
                current_resid_input = ln_module.input
                corrected_post_ln = ln_module(current_resid_input + steering)
                current_post_ln = ln_module(current_resid_input)  # OR ln_module.output if available

                delta_post_ln = corrected_post_ln - current_post_ln

                # Project and inject
                W_Q = layer_module.attention.attention.query.weight
                W_K = layer_module.attention.attention.key.weight
                W_V = layer_module.attention.attention.value.weight

                delta_q = delta_post_ln @ W_Q.T
                delta_k = delta_post_ln @ W_K.T
                delta_v = delta_post_ln @ W_V.T

                start = h_tgt * head_dim
                end = (h_tgt + 1) * head_dim

                layer_module.attention.attention.query.output[:, :, start:end] += delta_q[:, :, start:end]
                layer_module.attention.attention.key.output[:, :, start:end] += delta_k[:, :, start:end]
                layer_module.attention.attention.value.output[:, :, start:end] += delta_v[:, :, start:end]

            elif v.startswith("m"):
                l_tgt = int(v[1:])
                target_input = model_nnsight.vit.encoder.layer[l_tgt].layernorm_after.input
                target_input += steering

        # --- D. METRICS ---
        patched_logits = model_nnsight.classifier.output.save()

        batch_size = corrupted_input.shape[0]
        target_tensor = torch.full((batch_size,), truth_label, device=device, dtype=torch.long)
        patched_loss = F.cross_entropy(patched_logits, target_tensor).save()

    return patched_loss.value.item(), patched_logits.value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the model and processor
model_id = 'nateraw/vit-base-patch16-224-cifar10'
processor = ViTImageProcessor.from_pretrained(model_id)
model = ViTForImageClassification.from_pretrained(model_id).to(device)
model.eval()


def visualize_transformers_arch(model):
    print(f"=== {model_id} Architecture Map ===\n")

    # 1. Embeddings (Patches + Position + CLS)
    # This combines everything we discussed: 196 patches + 1 CLS token
    embeds = model.vit.embeddings
    print(f"1. Embeddings: {type(embeds)}")
    print(f"   - Path: model.vit.embeddings")
    print(f"   - Projection: {embeds.patch_embeddings.projection}\n")

    # 2. The Encoder Blocks
    layers = model.vit.encoder.layer
    print(f"2. Transformer Layers: {len(layers)} blocks")
    print(f"   - Path to one block: model.vit.encoder.layer[i]")

    # Peek inside the first block
    sample_layer = layers[0]
    print(f"   - Attention Path: layer.attention.attention")
    print(f"   - MLP (Intermediate): {sample_layer.intermediate.dense}")
    print(f"   - MLP (Output): {sample_layer.output.dense}\n")

    # 3. The Pooler (Specific to HF)
    # This extracts the CLS token state before the head
    print(f"3. Pooler: {model.vit.pooler}")
    print(f"   - Path: model.vit.pooler\n")

    # 4. Classification Head
    print(f"4. Head: {model.classifier}")
    print(f"   - Path: model.classifier")
    print(f"   - Classes: {model.config.num_labels} (CIFAR-10)")

visualize_transformers_arch(model)
print("\n=== Detailed Layer-by-Layer Flow ===")
summary(
    model,
    input_size=(1, 3, 224, 224), # Note: torchinfo expects (Batch, C, H, W)
    depth=1000,                    # How deep into nested layers you want to see
    device=device.type          # Match your current device
)

# 2. Define the transform using the Processor
# Transformers processors handle resizing and ImageNet normalization automatically
def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels


# 3. Load CIFAR-10 Test Set
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 4. Evaluation Loop
correct = 0
total = 0

print(f"Benchmarking {model_id}...")
with torch.no_grad():
    for pixel_values, labels in tqdm(test_loader):
        pixel_values, labels = pixel_values.to(device), labels.to(device)

        outputs = model(pixel_values)
        predictions = outputs.logits.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

print(f"\nFinal Accuracy: {100 * correct / total:.2f}%")


# 1. Create a dummy input
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 2. Run a forward pass
output = model(dummy_input)

# 3. Create the visualization from the logits
# This traces backward from the output to the input
dot = make_dot(output.logits, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("vit_computational_graph")

# 1. Extract config from your actual model
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

graph = ViTCompGraph.Graph.from_model(model, False)
graph.apply_topn(100, level="edge")

in_graph_nodes = 0
in_graph_edges = 0
for node_name, node in graph.nodes.items():
    if node.in_graph:
        in_graph_nodes += 1
for edge_name, edge in graph.edges.items():
    if edge.in_graph:
        in_graph_edges += 1
print("in graph nodes: {}, in graph edges: {}".format(in_graph_nodes, in_graph_edges))



print("="*25, "GENERATED GRAPH", "="*25)
print("NODES:")
print(len(graph.nodes))
print("EDGES:")
print(len(graph.edges))


graph.to_image("ViTGraph.png")