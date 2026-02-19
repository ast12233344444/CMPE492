import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision import datasets
import ViTCompGraph
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from torchviz import make_dot

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