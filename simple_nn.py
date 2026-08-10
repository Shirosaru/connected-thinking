import random

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


BaseModule = nn.Module if nn is not None else object


class IdeaBridgeNet(BaseModule):
    """A dual-branch network where two idea streams are fused through a bridge layer."""

    def __init__(self, input_dim=7, branch_hidden=(8, 6), bridge_dim=4, output_dim=3):
        if nn is None:
            raise RuntimeError("PyTorch is required to build IdeaBridgeNet.")
        super().__init__()
        if len(branch_hidden) < 1:
            raise ValueError("branch_hidden must contain at least one size.")
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1.")
        if bridge_dim < 1:
            raise ValueError("bridge_dim must be >= 1.")

        self.input_dim = input_dim
        self.branch_hidden = tuple(branch_hidden)
        self.branch_dims = (input_dim,) + self.branch_hidden
        self.bridge_dim = bridge_dim
        self.output_dim = output_dim

        branch_dims = self.branch_dims
        self.left_layers = nn.ModuleList(
            [nn.Linear(branch_dims[i], branch_dims[i + 1]) for i in range(len(branch_dims) - 1)]
        )
        self.right_layers = nn.ModuleList(
            [nn.Linear(branch_dims[i], branch_dims[i + 1]) for i in range(len(branch_dims) - 1)]
        )

        self.bridge_proj = nn.Linear(branch_dims[-1] * 4, bridge_dim)
        self.bridge_gate = nn.Linear(bridge_dim, bridge_dim)
        self.out = nn.Linear(bridge_dim, output_dim)

    def _run_branch(self, x, layers, prefix, trace):
        h = x
        for idx, layer in enumerate(layers, start=1):
            h = torch.relu(layer(h))
            trace[f"{prefix}_L{idx}"] = h
        return h

    def forward(self, left_x, right_x):
        trace = {}
        left_final = self._run_branch(left_x, self.left_layers, "left", trace)
        right_final = self._run_branch(right_x, self.right_layers, "right", trace)

        interaction = torch.cat(
            [left_final, right_final, torch.abs(left_final - right_final), left_final * right_final],
            dim=1,
        )

        bridge = torch.tanh(self.bridge_proj(interaction))
        fused = torch.tanh(self.bridge_gate(bridge)) + bridge
        output = torch.sigmoid(self.out(fused))

        trace["left"] = left_final
        trace["right"] = right_final
        trace["bridge"] = bridge
        trace["fused"] = fused

        return output, trace


class IdeaTriBridgeNet(BaseModule):
    """A three-branch network where three concept streams converge through a shared bridge."""

    def __init__(self, input_dim=7, branch_hidden=(8, 6), bridge_dim=4, output_dim=3):
        if nn is None:
            raise RuntimeError("PyTorch is required to build IdeaTriBridgeNet.")
        super().__init__()
        if len(branch_hidden) < 1:
            raise ValueError("branch_hidden must contain at least one size.")
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1.")
        if bridge_dim < 1:
            raise ValueError("bridge_dim must be >= 1.")

        self.input_dim = input_dim
        self.branch_hidden = tuple(branch_hidden)
        self.branch_dims = (input_dim,) + self.branch_hidden
        self.bridge_dim = bridge_dim
        self.output_dim = output_dim

        self.left_layers = nn.ModuleList(
            [nn.Linear(self.branch_dims[i], self.branch_dims[i + 1]) for i in range(len(self.branch_dims) - 1)]
        )
        self.center_layers = nn.ModuleList(
            [nn.Linear(self.branch_dims[i], self.branch_dims[i + 1]) for i in range(len(self.branch_dims) - 1)]
        )
        self.right_layers = nn.ModuleList(
            [nn.Linear(self.branch_dims[i], self.branch_dims[i + 1]) for i in range(len(self.branch_dims) - 1)]
        )

        self.bridge_proj = nn.Linear(self.branch_dims[-1] * 6, bridge_dim)
        self.bridge_gate = nn.Linear(bridge_dim, bridge_dim)
        self.out = nn.Linear(bridge_dim, output_dim)

    def _run_branch(self, x, layers, prefix, trace):
        h = x
        for idx, layer in enumerate(layers, start=1):
            h = torch.relu(layer(h))
            trace[f"{prefix}_L{idx}"] = h
        return h

    def forward(self, left_x, center_x, right_x):
        trace = {}
        left_final = self._run_branch(left_x, self.left_layers, "left", trace)
        center_final = self._run_branch(center_x, self.center_layers, "center", trace)
        right_final = self._run_branch(right_x, self.right_layers, "right", trace)

        interaction = torch.cat(
            [
                left_final,
                center_final,
                right_final,
                torch.abs(left_final - center_final),
                torch.abs(center_final - right_final),
                left_final * center_final * right_final,
            ],
            dim=1,
        )

        bridge = torch.tanh(self.bridge_proj(interaction))
        fused = torch.tanh(self.bridge_gate(bridge)) + bridge
        output = torch.sigmoid(self.out(fused))

        trace["left"] = left_final
        trace["center"] = center_final
        trace["right"] = right_final
        trace["bridge"] = bridge
        trace["fused"] = fused

        return output, trace


class DeepDualFocalNet(BaseModule):
    """Mirrored dual-sided deep branches converging to one focal node."""

    def __init__(self, branch_dims=(7, 10, 8, 6, 4, 3, 2), focal_dim=1, output_dim=3):
        if nn is None:
            raise RuntimeError("PyTorch is required to build DeepDualFocalNet.")
        super().__init__()
        if len(branch_dims) < 2:
            raise ValueError("branch_dims must contain at least two sizes.")
        if branch_dims[0] != 7:
            raise ValueError("branch_dims must start with input size 7.")
        if branch_dims[-1] < 1:
            raise ValueError("branch_dims final size must be >= 1.")

        self.branch_dims = tuple(branch_dims)
        self.focal_dim = focal_dim
        self.output_dim = output_dim

        self.left_layers = nn.ModuleList(
            [
                nn.Linear(self.branch_dims[i], self.branch_dims[i + 1])
                for i in range(len(self.branch_dims) - 1)
            ]
        )
        self.right_layers = nn.ModuleList(
            [
                nn.Linear(self.branch_dims[i], self.branch_dims[i + 1])
                for i in range(len(self.branch_dims) - 1)
            ]
        )

        self.focal_in_dim = self.branch_dims[-1] * 2
        self.focal = nn.Linear(self.focal_in_dim, self.focal_dim)
        self.out = nn.Linear(self.focal_dim, self.output_dim)

    def _run_branch(self, x, layers, prefix, trace):
        h = x
        for idx, layer in enumerate(layers, start=1):
            h = torch.sigmoid(layer(h))
            trace[f"{prefix}_L{idx}"] = h
        return h

    def forward(self, left_x, right_x):
        trace = {}
        left_final = self._run_branch(left_x, self.left_layers, "left", trace)
        right_final = self._run_branch(right_x, self.right_layers, "right", trace)

        focal_in = torch.cat([left_final, right_final], dim=1)
        focal = torch.tanh(self.focal(focal_in))
        output = torch.sigmoid(self.out(focal))

        trace["left_final"] = left_final
        trace["right_final"] = right_final
        trace["focal"] = focal

        return output, trace


def make_pair_targets(left, right):
    left_sum = sum(left)
    right_sum = sum(right)
    both_dense = 1 if (left_sum >= 4 and right_sum >= 4) else 0
    left_right_balance = 1 if abs(left_sum - right_sum) <= 1 else 0
    edge_sync = 1 if (left[0] == right[0] and left[-1] == right[-1]) else 0
    return [both_dense, left_right_balance, edge_sync]


def make_training_data(n=64, seed=3):
    rng = random.Random(seed)
    left_rows = []
    right_rows = []
    targets = []

    for _ in range(n):
        left = [rng.randint(0, 1) for _ in range(7)]
        right = [rng.randint(0, 1) for _ in range(7)]
        left_rows.append(left)
        right_rows.append(right)
        targets.append(make_pair_targets(left, right))

    return left_rows, right_rows, targets


def make_cat_car_demo_data():
    """Create a tiny intentionally structured dataset around cat/car idea bridging."""
    left_rows = [
        [1, 0, 1, 0, 0, 1, 0],  # cat-like
        [1, 1, 0, 0, 1, 0, 0],  # cat-like + motion
        [0, 1, 0, 1, 0, 1, 0],  # mixed
        [1, 0, 0, 1, 1, 0, 0],  # cat-like + vehicle-like
        [0, 0, 1, 0, 1, 1, 0],  # vehicle-like
        [0, 1, 0, 0, 0, 0, 1],  # bridge-ish
    ]
    right_rows = [
        [0, 0, 0, 1, 1, 0, 1],  # car-like
        [0, 1, 0, 1, 0, 0, 1],  # car-like + motion
        [1, 0, 1, 0, 0, 1, 0],  # matching cat-like
        [1, 1, 0, 0, 1, 0, 0],  # mixed bridge
        [0, 0, 1, 1, 0, 0, 1],  # vehicle-like
        [1, 0, 1, 0, 1, 0, 0],  # bridge-ish
    ]
    targets = [
        [0, 0, 1],
        [1, 1, 1],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
    ]
    return left_rows, right_rows, targets


def make_triple_bridge_demo_data():
    """Create a small three-way example around cat/car/road ideas."""
    left_streams = [
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 0],
    ]
    center_streams = [
        [0, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 0],
    ]
    right_streams = [
        [0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0],
    ]
    targets = [
        [0, 0, 1],
        [1, 1, 1],
        [0, 1, 0],
        [1, 1, 1],
    ]
    return left_streams, center_streams, right_streams, targets


def to_tensor(data, device):
    return torch.tensor(data, dtype=torch.float32, device=device)


def train_model(model, left_data, right_data, targets, device, epochs=3000, lr=0.02):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    left_t = to_tensor(left_data, device)
    right_t = to_tensor(right_data, device)
    target_t = to_tensor(targets, device)

    best_loss = 1e9
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred, _ = model(left_t, right_t)
        loss = criterion(pred, target_t)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value

        if epoch % 500 == 0:
            print(f"epoch {epoch}: loss={loss_value:.5f} best={best_loss:.5f}")

    print("Training complete")


def train_model_triple(model, left_data, center_data, right_data, targets, device, epochs=3000, lr=0.02):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    left_t = to_tensor(left_data, device)
    center_t = to_tensor(center_data, device)
    right_t = to_tensor(right_data, device)
    target_t = to_tensor(targets, device)

    best_loss = 1e9
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred, _ = model(left_t, center_t, right_t)
        loss = criterion(pred, target_t)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value

        if epoch % 500 == 0:
            print(f"epoch {epoch}: loss={loss_value:.5f} best={best_loss:.5f}")

    print("Training complete")


def _node_positions(branch_dims, focal_dim, output_dim):
    pos = {}

    z_step = 0.7
    for layer_idx, width in enumerate(branch_dims):
        y_center = (width - 1) / 2.0
        x_dist = 8.0 - (layer_idx * 1.0)
        z = layer_idx * z_step
        for i in range(width):
            pos[f"L{layer_idx}_{i}"] = (-x_dist, i - y_center, z)
            pos[f"R{layer_idx}_{i}"] = (x_dist, i - y_center, z)

    focal_z = len(branch_dims) * z_step
    focal_center = (focal_dim - 1) / 2.0
    for i in range(focal_dim):
        pos[f"F{i}"] = (0.0, i - focal_center, focal_z)

    out_z = focal_z + z_step
    out_center = (output_dim - 1) / 2.0
    for i in range(output_dim):
        pos[f"O{i}"] = (0.0, i - out_center, out_z)

    return pos


def export_3d_plot(model, output_path="network_3d.png"):
    if plt is None:
        print("matplotlib not installed; skipping 3D plot export")
        return

    if hasattr(model, "branch_dims"):
        branch_dims = model.branch_dims
    else:
        branch_dims = (7, 10, 8, 6, 4, 3, 2)

    focal_dim = getattr(model, "focal_dim", getattr(model, "bridge_dim", 1))
    pos = _node_positions(branch_dims, focal_dim, model.output_dim)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    def draw_nodes(names, color, size):
        xs = [pos[n][0] for n in names]
        ys = [pos[n][1] for n in names]
        zs = [pos[n][2] for n in names]
        ax.scatter(xs, ys, zs, color=color, s=size)

    def draw_full_connections(src_names, dst_names, linear_weight):
        w = linear_weight.detach().cpu().numpy()
        for dst_idx, dst in enumerate(dst_names):
            for src_idx, src in enumerate(src_names):
                weight = float(w[dst_idx][src_idx])
                s = pos[src]
                d = pos[dst]
                color = "#0a9396" if weight >= 0 else "#ae2012"
                alpha = min(0.9, 0.25 + abs(weight) * 0.55)
                ax.plot(
                    [s[0], d[0]],
                    [s[1], d[1]],
                    [s[2], d[2]],
                    color=color,
                    alpha=alpha,
                    linewidth=1.1,
                )

    left_layers = []
    right_layers = []
    for layer_idx, width in enumerate(branch_dims):
        left_layers.append([f"L{layer_idx}_{i}" for i in range(width)])
        right_layers.append([f"R{layer_idx}_{i}" for i in range(width)])

    out = [f"O{i}" for i in range(model.output_dim)]

    for idx, layer in enumerate(model.left_layers):
        draw_full_connections(left_layers[idx], left_layers[idx + 1], layer.weight)
    for idx, layer in enumerate(model.right_layers):
        draw_full_connections(right_layers[idx], right_layers[idx + 1], layer.weight)

    if hasattr(model, "bridge_proj"):
        bridge = [f"F{i}" for i in range(getattr(model, "bridge_dim", focal_dim))]
        src_names = left_layers[-1] + right_layers[-1]
        bridge_weight = torch.ones(len(bridge), len(src_names), dtype=torch.float32)
        draw_full_connections(src_names, bridge, bridge_weight)
        out_weight = torch.ones(model.output_dim, len(bridge), dtype=torch.float32)
        draw_full_connections(bridge, out, out_weight)

        draw_nodes(left_layers[0] + right_layers[0], "#94d2bd", 45)
        for idx in range(1, len(left_layers)):
            size = 65 + (idx * 6)
            draw_nodes(left_layers[idx] + right_layers[idx], "#ee9b00", size)
        draw_nodes(bridge, "#bb3e03", 140)
        draw_nodes(out, "#9b2226", 95)
        dims_text = "->".join(str(x) for x in branch_dims)
        ax.set_title(f"Bridge {dims_text} -> Bridge -> {model.output_dim}")
    else:
        focal = [f"F{i}" for i in range(getattr(model, "focal_dim", focal_dim))]
        draw_full_connections(left_layers[-1] + right_layers[-1], focal, model.focal.weight)
        draw_full_connections(focal, out, model.out.weight)

        draw_nodes(left_layers[0] + right_layers[0], "#94d2bd", 45)
        for idx in range(1, len(left_layers)):
            size = 65 + (idx * 6)
            draw_nodes(left_layers[idx] + right_layers[idx], "#ee9b00", size)
        draw_nodes(focal, "#bb3e03", 145)
        draw_nodes(out, "#9b2226", 95)
        dims_text = "->".join(str(x) for x in model.branch_dims)
        ax.set_title(f"Dual-Sided {dims_text} -> Focal -> {model.output_dim}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"saved 3D network plot to {output_path}")


def run_cat_car_demo():
    if torch is None:
        print("PyTorch is not installed. Install with: python3 -m pip install --user torch")
        return

    random.seed(2)
    torch.manual_seed(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    print("Running cat/car bridge demo")

    left_data, right_data, targets = make_cat_car_demo_data()
    model = IdeaBridgeNet(input_dim=7, branch_hidden=(8, 6), bridge_dim=4, output_dim=3).to(device)

    train_model(model, left_data, right_data, targets, device, epochs=1200, lr=0.03)

    with torch.no_grad():
        left_t = to_tensor(left_data, device)
        right_t = to_tensor(right_data, device)
        pred, trace = model(left_t, right_t)
        pred_cpu = pred.detach().cpu().numpy()
        bridge_cpu = trace["bridge"].detach().cpu().numpy()

    print("\nCat/car bridge predictions:")
    for i, (left, right) in enumerate(zip(left_data, right_data)):
        print(
            f"  cat-like: {left} | car-like: {right} | target={targets[i]} | pred={ [round(float(x), 4) for x in pred_cpu[i]] } | bridge={ [round(float(x), 4) for x in bridge_cpu[i]] }"
        )


def run_triple_bridge_demo():
    if torch is None:
        print("PyTorch is not installed. Install with: python3 -m pip install --user torch")
        return

    random.seed(3)
    torch.manual_seed(3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    print("Running three-way bridge demo")

    left_data, center_data, right_data, targets = make_triple_bridge_demo_data()
    model = IdeaTriBridgeNet(input_dim=7, branch_hidden=(10, 8), bridge_dim=4, output_dim=3).to(device)

    train_model_triple(model, left_data, center_data, right_data, targets, device, epochs=1400, lr=0.025)

    with torch.no_grad():
        left_t = to_tensor(left_data, device)
        center_t = to_tensor(center_data, device)
        right_t = to_tensor(right_data, device)
        pred, trace = model(left_t, center_t, right_t)
        pred_cpu = pred.detach().cpu().numpy()
        bridge_cpu = trace["bridge"].detach().cpu().numpy()

    print("\nThree-way bridge predictions:")
    for i, (left, center, right) in enumerate(zip(left_data, center_data, right_data)):
        print(
            f"  left={left} | center={center} | right={right} | target={targets[i]} | pred={ [round(float(x), 4) for x in pred_cpu[i]] } | bridge={ [round(float(x), 4) for x in bridge_cpu[i]] }"
        )


def main():
    if torch is None:
        print("PyTorch is not installed. Install with: python3 -m pip install --user torch")
        return

    random.seed(2)
    torch.manual_seed(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    left_data, right_data, targets = make_training_data(n=256, seed=5)

    model = IdeaBridgeNet(input_dim=7, branch_hidden=(14, 10, 8), bridge_dim=4, output_dim=3).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Bridge branch dims:", (7,) + model.branch_hidden)
    print("Trainable parameters:", total_params)

    train_model(model, left_data, right_data, targets, device, epochs=3500, lr=0.015)

    with torch.no_grad():
        left_t = to_tensor(left_data[:8], device)
        right_t = to_tensor(right_data[:8], device)
        pred, trace = model(left_t, right_t)
        pred_cpu = pred.detach().cpu().numpy()
        bridge_cpu = trace["bridge"].detach().cpu().numpy()

    print("\nSample predictions (first 8 rows):")
    for i in range(8):
        print(
            left_data[i],
            right_data[i],
            "target",
            targets[i],
            "pred",
            [round(float(x), 4) for x in pred_cpu[i]],
            "bridge",
            [round(float(x), 4) for x in bridge_cpu[i]],
        )

    export_3d_plot(model, "network_3d.png")


if __name__ == "__main__":
    main()
