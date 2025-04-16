# %%
import torch
import torch.nn as nn
import time

# === Load checkpoint ===
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
state_dict = checkpoint["model_state_dict"]

print("📦 Checkpoint loaded. Inspecting layers:\n")

# === Analyze state_dict and infer architecture ===
layer_shapes = []
layer_order = []

for k, v in state_dict.items():
    print(f"{k:40} {tuple(v.shape)}")
    if "weight" in k:
        layer_order.append((k, v.shape))

# === Define custom Actor-Critic model ===
class ActorCriticModel(nn.Module):
    def __init__(self, actor_layers, critic_layers):
        super(ActorCriticModel, self).__init__()
        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x):
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output

# === Reconstruct actor and critic layers ===
actor_layers = []
critic_layers = []

for name, shape in layer_order:
    out_features, in_features = shape
    if name.startswith("actor"):
        actor_layers.append(nn.Linear(in_features, out_features))
        if "weight" in name and not name.endswith("bias"):
            actor_layers.append(nn.ReLU())  # Assuming ReLU activations
    elif name.startswith("critic"):
        critic_layers.append(nn.Linear(in_features, out_features))
        if "weight" in name and not name.endswith("bias"):
            critic_layers.append(nn.ReLU())  # Assuming ReLU activations

# === Create model ===
model = ActorCriticModel(actor_layers, critic_layers)
model.eval()

print("\n✅ Model reconstructed and weights loaded.")
print(model)

# === Filter unexpected keys ===
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(filtered_state_dict)

# === Inference benchmarking ===
# You may need to change input_dim if your model expects something else
input_dim = actor_layers[0].in_features  # Assuming actor's first layer input dim
dummy_input = torch.randn(1, input_dim)

# Warm-up
for _ in range(10):
    _ = model(dummy_input)

# Time it
start = time.perf_counter()
for _ in range(num_runs := 10000):
    _ = model(dummy_input)
end = time.perf_counter()

avg_time = (end - start) / num_runs
print(f"\n🚀 Average inference time over {num_runs} runs: {avg_time * 1e3:.3f} ms")
# %%