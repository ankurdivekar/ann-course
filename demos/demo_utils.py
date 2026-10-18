"""
demo_utils.py
Shared, reusable functions for the ANN course Streamlit demos (Lessons 1-3).

Nothing in here imports streamlit -- these are plain, testable functions that
the Streamlit apps (01_GradientDescent.py, 02_Perceptrons.py,
03_ActivationFunctions.py) build their UIs around.
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1-D practice functions:  y = f(x)  and their slopes  dy/dx
# ----------------------------------------------------------------------------
# Each entry gives everything the demo needs to draw and descend a curve.

FUNCTIONS = {
    "y = x^2  (one simple valley)": {
        "f": lambda x: x**2,
        "grad": lambda x: 2 * x,
        "x_range": (-5.0, 5.0),
        "default_start": 4.0,
        "true_min_x": 0.0,
        "note": "The classic bowl. One minimum, at x = 0. Great for a first look.",
    },
    "y = (x - 3)^2 + 5  (shifted valley)": {
        "f": lambda x: (x - 3) ** 2 + 5,
        "grad": lambda x: 2 * (x - 3),
        "x_range": (-4.0, 10.0),
        "default_start": -3.0,
        "true_min_x": 3.0,
        "note": "Same shape, moved. The minimum is at x = 3, y = 5.",
    },
    "y = x^4 - 3x^3 + 2  (two valleys)": {
        "f": lambda x: x**4 - 3 * x**3 + 2,
        "grad": lambda x: 4 * x**3 - 9 * x**2,
        "x_range": (-1.5, 3.3),
        "default_start": -1.0,
        "true_min_x": 2.25,
        "note": (
            "A shallow dip near x = 0 and the true deepest point near x = 2.25. "
            "Where you start decides which one you fall into."
        ),
    },
    "y = x^2 + 3*sin(3x)  (bumpy)": {
        "f": lambda x: x**2 + 3 * np.sin(3 * x),
        "grad": lambda x: 2 * x + 9 * np.cos(3 * x),
        "x_range": (-4.0, 4.0),
        "default_start": 3.5,
        "true_min_x": -0.44,
        "note": "A bowl with ripples. Lots of little traps for gradient descent.",
    },
}


def gradient_descent(grad_fn, start, learning_rate, n_steps):
    """Run gradient descent on a 1-D function.

    Args:
        grad_fn: function returning the slope dy/dx at a point x.
        start: starting x position.
        learning_rate: step-size multiplier.
        n_steps: how many update steps to take.

    Returns:
        1-D numpy array of length ``n_steps + 1`` with every x visited,
        starting with ``start``.
    """
    x = float(start)
    history = [x]
    for _ in range(n_steps):
        x = x - learning_rate * grad_fn(x)
        # Stop recording once the numbers blow up so plots stay drawable.
        if not np.isfinite(x) or abs(x) > 1e6:
            history.append(np.sign(x) * 1e6 if np.isfinite(x) else np.nan)
            break
        history.append(x)
    return np.array(history)


def step_table(f, xs):
    """Build a list of per-step dicts (step, x, y, slope-was-not-computed-here).

    Handy for showing a table of what happened at each iteration.
    """
    rows = []
    for i, x in enumerate(xs):
        rows.append({"step": i, "x": float(x), "y  (height / loss)": float(f(x))})
    return rows


def diverged(xs):
    """True if the descent ran away to infinity / NaN."""
    return not np.all(np.isfinite(xs)) or np.nanmax(np.abs(xs)) > 1e5


# ----------------------------------------------------------------------------
# Fitting a straight line  y = w*x + b  by gradient descent on MSE loss
# ----------------------------------------------------------------------------


def make_line_data(true_w=2.3, true_b=4.0, n=40, noise=3.0, seed=0):
    """Generate noisy (x, y) points that roughly follow a straight line."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = true_w * x + true_b + rng.normal(0, noise, size=n)
    return x, y


def mse_loss(w, b, x, y):
    """Mean squared error of the line y = w*x + b against the data."""
    predictions = w * x + b
    return float(np.mean((predictions - y) ** 2))


def mse_gradients(w, b, x, y):
    """Slopes of the MSE loss with respect to w and b."""
    error = (w * x + b) - y
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)
    return float(dw), float(db)


def fit_line_gradient_descent(
    x, y, learning_rate=0.01, n_steps=400, start_w=0.0, start_b=0.0
):
    """Fit a line to data with gradient descent.

    Returns:
        dict with arrays ``w``, ``b``, ``loss`` (each length ``n_steps + 1``).
    """
    w, b = float(start_w), float(start_b)
    ws, bs, losses = [w], [b], [mse_loss(w, b, x, y)]
    for _ in range(n_steps):
        dw, db = mse_gradients(w, b, x, y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ws.append(w)
        bs.append(b)
        losses.append(mse_loss(w, b, x, y))
        if not np.isfinite(losses[-1]) or losses[-1] > 1e12:
            break
    return {"w": np.array(ws), "b": np.array(bs), "loss": np.array(losses)}


# ----------------------------------------------------------------------------
# Plotting helpers (matplotlib, Agg backend -- see the app for backend setup)
# ----------------------------------------------------------------------------


def plot_descent_on_curve(f, x_range, xs, current_step):
    """Figure: the curve y = f(x) with the descent path drawn up to
    ``current_step``, and the current position highlighted."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    # Widen the view so the curve is drawn wherever the descent actually goes
    # (e.g. when the student picks a starting point outside the default range).
    finite_xs = xs[np.isfinite(xs)]
    lo = min(x_range[0], finite_xs.min()) if finite_xs.size else x_range[0]
    hi = max(x_range[1], finite_xs.max()) if finite_xs.size else x_range[1]
    margin = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - margin, hi + margin, 500)
    ax.plot(grid, f(grid), color="steelblue", linewidth=2, label="y = f(x)")

    shown = xs[: current_step + 1]
    ax.plot(
        shown,
        f(shown),
        "o-",
        color="crimson",
        markersize=5,
        linewidth=1.5,
        alpha=0.7,
        label="steps taken",
    )
    ax.scatter([xs[0]], [f(xs[0])], color="green", s=90, zorder=6, label="start")
    ax.scatter(
        [shown[-1]],
        [f(shown[-1])],
        color="black",
        s=110,
        zorder=7,
        label=f"now (step {current_step})",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Walking downhill on the curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Keep the y-axis sane even if a step overshoots hugely.
    # y-limits: show the valley (default range) and the visited points, but
    # don't let one huge overshoot squash everything.
    base_y = f(np.linspace(x_range[0], x_range[1], 200))
    path_y = f(shown[np.isfinite(shown)]) if np.any(np.isfinite(shown)) else base_y
    y_lo = min(base_y.min(), np.min(path_y))
    y_hi = max(base_y.max(), np.percentile(path_y, 95))
    pad = 0.1 * (y_hi - y_lo + 1e-9)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    fig.tight_layout()
    return fig


def plot_loss_curve(values, current_step=None, ylabel="y  (height / loss)"):
    """Figure: value at each step, optionally marking the current step."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    steps = np.arange(len(values))
    ax.plot(steps, values, "-", color="purple", linewidth=1.5)
    ax.plot(steps, values, "o", color="purple", markersize=3, alpha=0.5)
    if current_step is not None and current_step < len(values):
        ax.scatter(
            [current_step], [values[current_step]], color="black", s=90, zorder=6
        )
    ax.set_xlabel("step number")
    ax.set_ylabel(ylabel)
    ax.set_title("Loss over time -- we want this to go down")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_line_fit(x, y, w, b, step, loss):
    """Figure: the data scatter with the current fitted line."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="steelblue", label="data", zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(
        xs, w * xs + b, color="crimson", linewidth=2.5, label=f"y = {w:.2f}x + {b:.2f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Step {step}   |   loss = {loss:,.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Perceptrons (02_Perceptrons.ipynb): step activation, logic gates, and
# Rosenblatt's mistake-driven learning rule.
# ----------------------------------------------------------------------------

# All four possible combinations of two binary inputs, and the three classic
# logic-gate truth tables built on top of them (same data as the notebook).
GATE_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

GATES = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


def step(z):
    """Step activation: outputs 1 if z >= 0, otherwise 0."""
    return np.where(np.asarray(z) >= 0, 1, 0)


def perceptron_output(x1, x2, w1, w2, b):
    """Weighted sum and step output for a single (x1, x2) pair."""
    z = w1 * x1 + w2 * x2 + b
    return float(z), int(step(z))


def perceptron_predict(inputs, w, b):
    """Step-activated predictions for a batch of (x1, x2) rows."""
    z = inputs @ w + b
    return step(z)


def count_mistakes(inputs, labels, w, b):
    """How many of the 4 gate rows the current (w, b) gets wrong."""
    return int(np.sum(perceptron_predict(inputs, w, b) != labels))


def train_perceptron_trace(
    inputs, labels, learning_rate=0.1, n_epochs=10, w_init=(0.0, 0.0), b_init=0.0
):
    """Run Rosenblatt's perceptron rule, recording state after every example.

    Unlike the notebook's per-epoch history, this records one entry per
    individual weight update (plus one entry for the untrained starting
    point), so a UI can scrub through training example by example.

    Returns:
        A list of dicts, each with: epoch, example (-1 for the starting
        point), w, b, x, target, prediction, mistakes_after.
    """
    w = np.array(w_init, dtype=float)
    b = float(b_init)
    trace = [
        {
            "epoch": 0,
            "example": -1,
            "w": w.copy(),
            "b": b,
            "x": None,
            "target": None,
            "prediction": None,
            "mistakes_after": count_mistakes(inputs, labels, w, b),
        }
    ]
    for epoch in range(1, n_epochs + 1):
        for i, (x, target) in enumerate(zip(inputs, labels)):
            prediction = int(step(np.dot(x, w) + b))
            error = int(target) - prediction
            w = w + learning_rate * error * x
            b = b + learning_rate * error
            trace.append(
                {
                    "epoch": epoch,
                    "example": i,
                    "w": w.copy(),
                    "b": b,
                    "x": x,
                    "target": int(target),
                    "prediction": prediction,
                    "mistakes_after": count_mistakes(inputs, labels, w, b),
                }
            )
    return trace


def plot_perceptron_boundary(inputs, labels, w, b, title="Decision boundary"):
    """Scatter the 4 gate points (colored by label) and the current boundary."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = np.where(labels == 1, "tab:red", "tab:blue")
    ax.scatter(inputs[:, 0], inputs[:, 1], c=colors, s=200, zorder=3, edgecolor="black")
    for (x1, x2), label in zip(inputs, labels):
        ax.annotate(
            str(int(label)),
            (x1, x2),
            textcoords="offset points",
            xytext=(10, 8),
            fontsize=11,
        )

    if abs(w[1]) > 1e-9:
        x1_line = np.linspace(-0.6, 1.6, 100)
        x2_line = -(w[0] * x1_line + b) / w[1]
        ax.plot(x1_line, x2_line, "k--", linewidth=2, label="decision boundary")
        ax.legend(loc="upper right", fontsize=9)
    elif abs(w[0]) > 1e-9:
        ax.axvline(
            -b / w[0],
            color="black",
            linestyle="--",
            linewidth=2,
            label="decision boundary",
        )
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mistakes_curve(mistake_counts, current_epoch=None):
    """Line plot of mistakes per epoch, echoing plot_loss_curve's style."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    epochs = np.arange(1, len(mistake_counts) + 1)
    ax.plot(epochs, mistake_counts, "o-", color="purple", linewidth=1.5)
    if current_epoch is not None and 1 <= current_epoch <= len(mistake_counts):
        ax.scatter(
            [current_epoch],
            [mistake_counts[current_epoch - 1]],
            color="black",
            s=90,
            zorder=6,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("mistakes")
    ax.set_title("Mistakes per epoch -- we want this to hit zero")
    ax.set_ylim(bottom=-0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Activation functions (03_ActivationFunctions.ipynb): differentiability,
# sigmoid, ReLU, and training a single neuron with gradient descent.
# ----------------------------------------------------------------------------


def numerical_slope(f, x, h=1e-4):
    """Estimate the slope of f at x from a tiny step forward and backward."""
    return (f(x + h) - f(x - h)) / (2 * h)


# Three functions used to build intuition for differentiability: one smooth,
# one with a corner, one with a jump (same trio as the notebook).
DIFFERENTIABILITY_FUNCTIONS = {
    "Smooth curve:  y = x^2": {
        "f": lambda x: x**2,
        "x_range": (-3.0, 3.0),
        "note": "A textbook smooth curve -- one well-defined slope everywhere.",
    },
    "Corner:  y = |x|": {
        "f": lambda x: np.abs(x),
        "x_range": (-3.0, 3.0),
        "note": "Two straight edges meet at x = 0 with different slopes (-1 and +1).",
    },
    "Jump: the step function": {
        "f": lambda x: step(x).astype(float),
        "x_range": (-3.0, 3.0),
        "note": "The height leaps from 0 to 1 at z = 0 -- the slope there is unbounded",
    },
}


def sigmoid(z):
    """Smoothly squashes any real number into the range (0, 1)."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    """The slope of the sigmoid function at z."""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """Outputs z if positive, otherwise 0."""
    return np.maximum(0, z)


def relu_grad(z):
    """The slope of ReLU: 1 where z > 0, 0 where z <= 0 (by convention)."""
    return np.where(np.asarray(z) > 0, 1.0, 0.0)


# A registry of activations, mirroring FUNCTIONS above -- everything a UI
# needs to plot an activation and (if it has one) its derivative.
ACTIVATIONS = {
    "Step (Lesson 2)": {"f": lambda z: step(z).astype(float), "grad": None},
    "Sigmoid": {"f": sigmoid, "grad": sigmoid_grad},
    "ReLU": {"f": relu, "grad": relu_grad},
}


def train_neuron_history(
    inputs,
    labels,
    activation,
    activation_grad,
    learning_rate=0.5,
    n_epochs=2000,
    seed=0,
    zero_init=False,
):
    """Train a single neuron with batch gradient descent on the MSE loss.

    Unlike the notebook's `train_neuron`, this also keeps the weights at
    every epoch (not just the loss), so a UI can scrub through training and
    redraw the decision surface at any point along the way.

    `zero_init=True` starts from `w = 0, b = 0` instead of small random
    numbers -- useful for demonstrating a "dead" ReLU neuron, whose slope at
    z = 0 is 0, so gradient descent never takes a single step.

    Returns:
        dict with arrays "w" (n_epochs+1, n_features), "b" (n_epochs+1,) and
        "loss" (n_epochs+1,), all aligned by epoch index.
    """
    if zero_init:
        w = np.zeros(inputs.shape[1])
        b = 0.0
    else:
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 0.5, size=inputs.shape[1])
        b = 0.1
    w_hist, b_hist, loss_hist = [w.copy()], [b], []

    for _ in range(n_epochs):
        z = inputs @ w + b
        pred = activation(z)
        error = pred - labels
        loss_hist.append(float(np.mean(error**2)))

        dz = (2 * error * activation_grad(z)) / len(labels)  # chain rule
        dw = inputs.T @ dz
        db = np.sum(dz)

        w = w - learning_rate * dw
        b = b - learning_rate * db
        w_hist.append(w.copy())
        b_hist.append(b)

    z = inputs @ w + b
    loss_hist.append(float(np.mean((activation(z) - labels) ** 2)))

    return {"w": np.array(w_hist), "b": np.array(b_hist), "loss": np.array(loss_hist)}


def decision_surface_grid(w, b, activation, grid_range=(-0.6, 1.6), resolution=150):
    """Grid of a neuron's output over 2-D input space, for a heatmap plot."""
    grid = np.linspace(grid_range[0], grid_range[1], resolution)
    x1_grid, x2_grid = np.meshgrid(grid, grid)
    z = activation(w[0] * x1_grid + w[1] * x2_grid + b)
    return x1_grid, x2_grid, z


def plot_decision_surface(w, b, activation, inputs, labels, title="Decision surface"):
    """Shade input space by the neuron's output and scatter the 4 gate points."""
    import matplotlib.pyplot as plt

    x1_grid, x2_grid, z = decision_surface_grid(w, b, activation)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    mesh = ax.contourf(x1_grid, x2_grid, z, levels=20, cmap="RdBu_r")
    fig.colorbar(mesh, ax=ax, label="neuron output")

    colors = np.where(labels == 1, "black", "white")
    ax.scatter(inputs[:, 0], inputs[:, 1], c=colors, s=200, zorder=3, edgecolor="black")
    for (x1, x2), label in zip(inputs, labels):
        ax.annotate(
            str(int(label)),
            (x1, x2),
            textcoords="offset points",
            xytext=(10, 8),
            color="black",
        )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_activation_curve(name, z_range=(-8, 8), n_points=400):
    """Side-by-side plot of an activation function and its derivative."""
    import matplotlib.pyplot as plt

    spec = ACTIVATIONS[name]
    z = np.linspace(z_range[0], z_range[1], n_points)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(z, spec["f"](z), color="steelblue", linewidth=2)
    ax1.set_title(name)
    ax1.set_xlabel("z")
    ax1.set_ylabel("output")
    ax1.grid(True, alpha=0.3)

    if spec["grad"] is not None:
        ax2.plot(z, spec["grad"](z), color="darkorange", linewidth=2)
        ax2.set_title(f"{name} -- slope")
    else:
        ax2.plot(z, np.zeros_like(z), color="darkorange", linewidth=2)
        ax2.axvline(0, color="crimson", linestyle="--", label="slope is unbounded here")
        ax2.legend(fontsize=8)
        ax2.set_title(f"{name} -- slope (flat, except an infinite jump at z=0)")
    ax2.set_xlabel("z")
    ax2.set_ylabel("slope")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_differentiability_probe(spec_name, x0, h):
    """Zoom in on a function at x0 with step size h, showing the left and
    right secant lines used to estimate the slope from each side."""
    import matplotlib.pyplot as plt

    spec = DIFFERENTIABILITY_FUNCTIONS[spec_name]
    f = spec["f"]
    lo, hi = spec["x_range"]

    x = np.linspace(lo, hi, 400)
    y = f(x)

    left_slope = float((f(x0) - f(x0 - h)) / h)
    right_slope = float((f(x0 + h) - f(x0)) / h)
    central_slope = float((f(x0 + h) - f(x0 - h)) / (2 * h))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color="steelblue", linewidth=2, zorder=2)

    ax.plot(
        [x0 - h, x0],
        [f(x0 - h), f(x0)],
        "o-",
        color="crimson",
        linewidth=2,
        label=f"left slope = {left_slope:.2f}",
    )
    ax.plot(
        [x0, x0 + h],
        [f(x0), f(x0 + h)],
        "o-",
        color="darkorange",
        linewidth=2,
        label=f"right slope = {right_slope:.2f}",
    )
    ax.scatter([x0], [f(x0)], color="black", zorder=5, s=80)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{spec_name}\ncentral-difference estimate = {central_slope:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig
