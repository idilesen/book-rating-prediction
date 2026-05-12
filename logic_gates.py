"""
logic_gates.py — Logic Gate Perceptron + Decision Boundary Visualization
Bonus module for Book Rating Predictor project.

Implements AND, OR, NAND gates from scratch using the same ScratchMLP,
then plots decision boundaries with Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlp_scratch import ScratchMLP


# ── Gate definitions ──────────────────────────────────────────────────────────
GATES = {
    "AND":  np.array([[0], [0], [0], [1]], dtype=float),
    "OR":   np.array([[0], [1], [1], [1]], dtype=float),
    "NAND": np.array([[1], [1], [1], [0]], dtype=float),
    "XOR":  np.array([[0], [1], [1], [0]], dtype=float),  # non-linear — needs hidden layer
}

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def train_gate(gate_name: str, epochs: int = 5000, lr: float = 0.1) -> tuple:
    y = GATES[gate_name]

    # XOR needs a hidden layer; others can be solved linearly but we use same arch
    model = ScratchMLP(layers=[2, 8, 1], lr=lr)

    losses = []
    for epoch in range(epochs):
        out   = model.forward(X)
        model.backward(X, y, out)
        if epoch % 50 == 0:
            loss = float(np.mean((y - out) ** 2))
            losses.append((epoch, loss))

    final_out = model.forward(X)
    final_loss = float(np.mean((y - final_out) ** 2))
    print(f"{gate_name:5s}  |  Final MSE: {final_loss:.5f}  |  "
          f"Predictions: {final_out.flatten().round(2)}")
    return model, losses


def plot_decision_boundary(ax, model, title: str, y_true: np.ndarray):
    """Shade the decision boundary on a 0-1 grid."""
    h = 0.005
    xx, yy = np.meshgrid(np.arange(-0.1, 1.15, h),
                         np.arange(-0.1, 1.15, h))
    grid   = np.c_[xx.ravel(), yy.ravel()]
    Z      = model.forward(grid).reshape(xx.shape)

    # Background shading
    ax.contourf(xx, yy, Z, levels=50, cmap="RdYlGn", alpha=0.85, vmin=0, vmax=1)
    ax.contour( xx, yy, Z, levels=[0.5], colors="white", linewidths=1.5, linestyles="--")

    # Data points
    colors = ["#FF4D6D" if v == 0 else "#00D68F" for v in y_true.flatten()]
    for i, (xi, yi_) in enumerate(X):
        ax.scatter(xi, yi_, s=220, color=colors[i],
                   edgecolors="white", linewidths=1.5, zorder=5)
        label = int(y_true[i][0])
        ax.text(xi + 0.04, yi_ + 0.04, str(label),
                fontsize=11, fontweight="bold", color="white", zorder=6)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=10)
    ax.set_xlabel("Input A", color="#A0A8B8", fontsize=9)
    ax.set_ylabel("Input B", color="#A0A8B8", fontsize=9)
    ax.tick_params(colors="#A0A8B8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2A5A")


def plot_loss_curves(ax, loss_data: dict):
    """Plot training loss curves for all gates."""
    palette = {"AND": "#7C6FFF", "OR": "#00D68F", "NAND": "#FFB830", "XOR": "#FF4D6D"}
    for gate, losses in loss_data.items():
        epochs_lst = [l[0] for l in losses]
        mse_lst    = [l[1] for l in losses]
        ax.plot(epochs_lst, mse_lst, label=gate, color=palette[gate], linewidth=2)

    ax.set_facecolor("#13132A")
    ax.set_title("Training Loss (MSE) per Gate", fontsize=13,
                 fontweight="bold", color="white", pad=10)
    ax.set_xlabel("Epoch", color="#A0A8B8", fontsize=9)
    ax.set_ylabel("MSE", color="#A0A8B8", fontsize=9)
    ax.tick_params(colors="#A0A8B8")
    ax.legend(facecolor="#1C1C3A", labelcolor="white", edgecolor="#2A2A5A")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2A5A")


def run():
    print("=" * 50)
    print("  Logic Gate Training — ScratchMLP")
    print("=" * 50)

    models    = {}
    loss_data = {}
    for gate in GATES:
        model, losses = train_gate(gate)
        models[gate]    = model
        loss_data[gate] = losses

    print("\nPlotting decision boundaries...")

    # Layout: 2×2 decision boundaries + 1 loss curve spanning bottom
    fig = plt.figure(figsize=(14, 10), facecolor="#0D0D1A")
    fig.suptitle("Logic Gate Decision Boundaries — MLP from Scratch",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    gate_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]

    for ax, gate in zip(gate_axes, GATES):
        ax.set_facecolor("#13132A")
        final_out = models[gate].forward(X)
        acc = np.mean((final_out > 0.5).astype(int) == GATES[gate].astype(int)) * 100
        plot_decision_boundary(ax, models[gate], f"{gate} Gate  ({acc:.0f}% acc)", GATES[gate])

    plt.tight_layout()
    #plt.savefig("/mnt/user-data/outputs/logic_gates_plot.png",
    #           dpi=150, bbox_inches="tight", facecolor="#0D0D1A")
    #print("Saved: logic_gates_plot.png")
    plt.show()


if __name__ == "__main__":
    run()