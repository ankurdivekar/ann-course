"""
03_ActivationFunctions.py
Interactive Activation Function demos for a high-school Artificial Neural
Networks class.

Run it with:
    cd demos
    streamlit run 03_ActivationFunctions.py

Three demos:
  1. "Is it differentiable?" -- zoom in on a smooth curve, a corner, and a
     jump, and watch the left/right slope estimates agree or disagree.
  2. "Meet sigmoid and ReLU" -- compare the step, sigmoid, and ReLU
     activations side by side with their derivatives.
  3. "Train a neuron (and watch XOR fail again)" -- run real gradient
     descent (not the perceptron rule) on AND, OR, and XOR, with either
     activation, and see the same linear-separability wall from Lesson 2.

If Streamlit segfaults on macOS, see ../references/Demos/SEGFAULT_FIX.md
The matplotlib backend fix from that document is applied below.
"""

import os

import numpy as np
import streamlit as st
from demo_utils import (
    ACTIVATIONS,
    DIFFERENTIABILITY_FUNCTIONS,
    GATE_INPUTS,
    GATES,
    numerical_slope,
    plot_activation_curve,
    plot_decision_surface,
    plot_differentiability_probe,
    relu,
    relu_grad,
    sigmoid,
    sigmoid_grad,
    train_neuron_history,
)

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg", force=True)  # non-GUI backend: required under Streamlit
import matplotlib.pyplot as plt

plt.ioff()


st.set_page_config(
    page_title="Activation Functions Demo", page_icon="📈", layout="wide"
)

st.title("📈 Activation Functions -- giving a neuron a gradient to follow")
st.markdown(
    "Lesson 2's step-activated perceptron couldn't be trained by gradient "
    "descent -- its slope is 0 almost everywhere, and undefined right at "
    "the jump. These demos build up why that matters, meet two smoother "
    "activations, and then show that smoothness alone still can't solve "
    "XOR."
)

demo = st.sidebar.radio(
    "Choose a demo",
    [
        "1 · Is it differentiable?",
        "2 · Meet sigmoid and ReLU",
        "3 · Train a neuron (and watch XOR fail again)",
    ],
)
st.sidebar.markdown("---")


# ===========================================================================
# DEMO 1 -- Differentiability, up close
# ===========================================================================
if demo.startswith("1"):
    st.header("Demo 1 · Is it differentiable?")
    st.markdown(
        "A function is **differentiable** at a point if it has one "
        "well-defined slope there. Pick a point `x0` and a tiny step size "
        "`h`, and watch whether the slope estimated from the left agrees "
        "with the slope estimated from the right."
    )

    with st.sidebar:
        st.subheader("⚙️ Settings")
        fn_name = st.selectbox("Function", list(DIFFERENTIABILITY_FUNCTIONS.keys()))
        spec = DIFFERENTIABILITY_FUNCTIONS[fn_name]
        st.caption(spec["note"])

        lo, hi = spec["x_range"]
        x0 = st.slider("Point to probe, x0", float(lo), float(hi), 0.0, 0.1)
        h = st.select_slider(
            "Step size, h", options=[1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001], value=0.1
        )

    f = spec["f"]
    left_slope = float((f(x0) - f(x0 - h)) / h)
    right_slope = float((f(x0 + h) - f(x0)) / h)
    central_slope = float((f(x0 + h) - f(x0 - h)) / (2 * h))

    col_plot, col_info = st.columns([3, 2])

    with col_plot:
        fig = plot_differentiability_probe(fn_name, x0, h)
        st.pyplot(fig, clear_figure=True)

    with col_info:
        st.metric("left slope", f"{left_slope:.3f}")
        st.metric("right slope", f"{right_slope:.3f}")
        st.metric("central-difference estimate", f"{central_slope:.3f}")

        st.markdown("---")
        if abs(left_slope - right_slope) < 0.05:
            st.success(
                "✅ Left and right slopes agree -- looks differentiable "
                "here. Try shrinking `h` further: the estimate should stay "
                "put."
            )
        else:
            st.error(
                "❌ Left and right slopes disagree -- **not differentiable** "
                "at this point. Notice shrinking `h` doesn't fix it; the "
                "disagreement (or, for the jump, the sheer size of the "
                "slope) persists no matter how closely you zoom in."
            )

    with st.expander("🧪 Things to try"):
        st.markdown(
            "- On `y = x^2`, probe any point and shrink `h` all the way to "
            "`0.001`. The left and right slopes should keep agreeing.\n"
            "- On `y = |x|`, set `x0 = 0`. No matter how small `h` gets, "
            "left stays near `-1` and right stays near `+1`.\n"
            "- On the step function, set `x0 = 0` and watch the left slope "
            "explode as `h` shrinks (`1/h` grows without bound) while the "
            "right slope stays `0`.\n"
            "- Move `x0` away from `0` on the corner or the jump -- the "
            "function is perfectly differentiable everywhere *except* that "
            "one point."
        )


# ===========================================================================
# DEMO 2 -- Meet the activation functions
# ===========================================================================
elif demo.startswith("2"):
    st.header("Demo 2 · Meet sigmoid and ReLU")
    st.markdown(
        "Same activation, two views: the function itself (left) and its "
        "slope (right). Gradient descent only cares about the right-hand "
        "plot."
    )

    with st.sidebar:
        st.subheader("⚙️ Settings")
        activation_name = st.selectbox("Activation", list(ACTIVATIONS.keys()))

    fig = plot_activation_curve(activation_name)
    st.pyplot(fig, clear_figure=True)

    if activation_name.startswith("Step"):
        st.error(
            "The step function's slope is **0** everywhere except `z = 0`, "
            "where it's unbounded. Gradient descent has nothing usable to "
            "work with -- this is exactly why Lesson 2 needed a separate "
            "training rule."
        )
    elif activation_name == "Sigmoid":
        st.success(
            "Sigmoid's slope is small but never exactly zero, everywhere. "
            "It peaks at `0.25` when `z = 0` and fades toward `0` far from "
            "the middle (a real practical issue called the *vanishing "
            "gradient problem*, for a later lesson)."
        )
    else:
        st.success(
            "ReLU's slope is exactly the step function from Lesson 2 -- "
            "but now it's the *derivative*, not the activation, so it only "
            "fails to exist at one point (`z = 0`) instead of being flat "
            "everywhere."
        )

    with st.expander("🧪 Things to try"):
        st.markdown(
            "- Compare Sigmoid's and ReLU's slope plots. Which one stays "
            "at a constant, useful value further from `z = 0`?\n"
            "- Switch back to Step and look at where its slope plot puts "
            "the dashed red line -- that's the same jump you probed in "
            "Demo 1.\n"
            "- Predict what ReLU's slope plot would look like *before* "
            "clicking it. Were you right?"
        )


# ===========================================================================
# DEMO 3 -- Training a neuron with real gradient descent
# ===========================================================================
else:
    st.header("Demo 3 · Train a neuron (and watch XOR fail again)")
    st.markdown(
        "Same loss as Lesson 1's line-fitting demo -- mean squared error -- "
        "but now wrapped around a neuron's activation output. Because the "
        "activation is differentiable, we can train with gradient descent "
        "directly, no perceptron rule required."
    )

    with st.sidebar:
        st.subheader("⚙️ Settings")
        activation_name = st.selectbox(
            "Activation", ["Sigmoid", "ReLU", "Step (Lesson 2)"], index=0
        )
        zero_init = False
        if not activation_name.startswith("Step"):
            gate_name = st.selectbox("Gate to learn", list(GATES.keys()), index=0)
            learning_rate = st.select_slider(
                "Learning rate",
                options=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
                value=0.3,
            )
            n_epochs = st.slider("Epochs to run", 100, 5000, 3000, step=100)
            if activation_name == "ReLU":
                zero_init = st.checkbox(
                    "Start from w = 0, b = 0 (demonstrate a dead ReLU neuron)"
                )

    if activation_name.startswith("Step"):
        st.error(
            "The step function has no usable slope (see Demos 1 and 2), so "
            "gradient descent cannot train it -- there is nothing to "
            "compute here. This is exactly why Lesson 2 trained its "
            "perceptron with a separate mistake-counting rule instead. "
            "Pick Sigmoid or ReLU to actually train a neuron."
        )
    else:
        activation, activation_grad = (
            (sigmoid, sigmoid_grad)
            if activation_name == "Sigmoid"
            else (relu, relu_grad)
        )
        labels = GATES[gate_name]
        history = train_neuron_history(
            GATE_INPUTS,
            labels,
            activation,
            activation_grad,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            zero_init=zero_init,
        )

        st.markdown("### ▶️ Step through training")
        epoch_idx = st.slider("Drag to watch the neuron train", 0, n_epochs, n_epochs)
        w_now = history["w"][epoch_idx]
        b_now = history["b"][epoch_idx]
        loss_now = history["loss"][epoch_idx]

        col_plot, col_info = st.columns([3, 2])
        with col_plot:
            title = f"{activation_name} neuron on {gate_name} -- epoch {epoch_idx}"
            fig = plot_decision_surface(
                w_now, b_now, activation, GATE_INPUTS, labels, title=title
            )
            st.pyplot(fig, clear_figure=True)
        with col_info:
            st.metric("weights", f"w=[{w_now[0]:.2f}, {w_now[1]:.2f}], b={b_now:.2f}")
            st.metric("loss (mean squared error)", f"{loss_now:.5f}")
            st.caption(
                "Shading shows the neuron's raw output; black dots are "
                "target 1, white dots are target 0."
            )

        st.markdown("### 📉 Loss curve")
        fig_loss, ax = plt.subplots(figsize=(8, 3))
        ax.plot(history["loss"], color="purple")
        ax.axvline(epoch_idx, color="black", linestyle="--", alpha=0.6)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean squared error")
        ax.set_title("Loss over training")
        ax.grid(True, alpha=0.3)
        fig_loss.tight_layout()
        st.pyplot(fig_loss, clear_figure=True)

        final_loss = history["loss"][-1]
        if zero_init and final_loss > 0.15:
            st.error(
                "💀 **Dead neuron.** Starting at `w = 0, b = 0` makes "
                "`z = 0` for every input, and ReLU's slope right there is "
                "0 by convention -- gradient descent never takes a single "
                "real step. Uncheck the box to start from small random "
                "weights instead."
            )
        elif gate_name == "XOR":
            st.error(
                f"💥 **Plateaus at {final_loss:.3f}, not 0.** Sigmoid and "
                'ReLU are both *monotonic*, so "the neuron fires" always '
                'means "z is above some fixed number" -- still a '
                "straight-line boundary. Smoothing the activation fixed "
                "*trainability*, not *linear separability*: XOR needs more "
                "than one neuron (next lesson)."
            )
        elif final_loss < 0.02:
            st.success(
                f"✅ Converged! Loss reached {final_loss:.5f} -- gradient "
                f"descent, not the perceptron rule, learned **{gate_name}**."
            )
        else:
            st.warning("Getting there -- try more epochs or a different learning rate.")

    with st.expander("🧪 Things to try"):
        st.markdown(
            "- Train **AND** with Sigmoid, then with ReLU. Both should "
            "reach a very low loss, just via different decision-surface "
            "shapes.\n"
            "- Switch to **XOR** with either activation and let it run the "
            "full 5000 epochs. The loss curve flattens out well above "
            "zero -- gradient descent is working, the *problem* isn't "
            "trainable by one neuron.\n"
            "- Pick **ReLU**, check the dead-neuron box, and train **AND**. "
            "Watch the loss curve go completely flat from epoch 0.\n"
            "- Crank the learning rate to **1.5** on ReLU's AND. Does it "
            "still converge, the way Lesson 1 warned a too-large learning "
            "rate might not?"
        )

st.sidebar.markdown("---")
st.sidebar.caption("ANN course · Lesson 03 · Activation Functions")
