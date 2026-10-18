"""
02_Perceptrons.py
Interactive Perceptron demos for a high-school Artificial Neural Networks class.

Run it with:
    cd demos
    streamlit run 02_Perceptrons.py

Two demos:
  1. "Build a perceptron by hand" -- move weights and bias with sliders and
     watch the step-activated output and decision boundary respond live.
  2. "Train it (and watch XOR fail)" -- run Rosenblatt's perceptron learning
     rule step by step on AND, OR, or XOR, and see why a single perceptron
     can solve two of the three but never the third.

If Streamlit segfaults on macOS, see ../references/Demos/SEGFAULT_FIX.md
The matplotlib backend fix from that document is applied below.
"""

import os

import numpy as np
import streamlit as st
from demo_utils import (
    GATE_INPUTS,
    GATES,
    perceptron_output,
    perceptron_predict,
    plot_mistakes_curve,
    plot_perceptron_boundary,
    train_perceptron_trace,
)

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg", force=True)  # non-GUI backend: required under Streamlit
import matplotlib.pyplot as plt

plt.ioff()


st.set_page_config(page_title="Perceptron Demo", page_icon="🧠", layout="wide")

st.title("🧠 Perceptrons -- a machine that draws one straight line")
st.markdown(
    "A perceptron combines weighted inputs plus a bias into one number `z`, "
    "then fires (**1**) or stays silent (**0**) depending on whether `z` "
    "crosses zero -- the same weighted-sum-plus-bias idea as the line "
    "`y = mx + c` from Lesson 0. These demos let you build one by hand, then "
    "train one, and see exactly where a single perceptron runs out of road."
)

demo = st.sidebar.radio(
    "Choose a demo",
    ["1 · Build a perceptron by hand", "2 · Train it (and watch XOR fail)"],
)
st.sidebar.markdown("---")


# ===========================================================================
# DEMO 1 -- Hand-picking weights
# ===========================================================================
if demo.startswith("1"):
    st.header("Demo 1 · Build a perceptron by hand")
    st.markdown(
        r"$$z = w_1 x_1 + w_2 x_2 + b \qquad "
        r"\text{output} = \begin{cases}1 & z \ge 0\\0 & z < 0\end{cases}$$"
        "\n\nTry to find weights that reproduce a logic gate below -- exactly "
        "like hand-picking `(m, c)` for a line in Lesson 0, just with an "
        "extra input and a hard yes/no decision on top."
    )

    with st.sidebar:
        st.subheader("⚙️ Weights")
        w1 = st.slider("w₁", -3.0, 3.0, 1.0, 0.1)
        w2 = st.slider("w₂", -3.0, 3.0, 1.0, 0.1)
        b = st.slider("b (bias)", -3.0, 3.0, -0.5, 0.1)

        st.markdown("---")
        gate_name = st.selectbox("Compare against gate", list(GATES.keys()))

    w = np.array([w1, w2])
    labels = GATES[gate_name]
    predictions = perceptron_predict(GATE_INPUTS, w, b)
    mistakes = int(np.sum(predictions != labels))

    col_plot, col_info = st.columns([3, 2])

    with col_plot:
        fig = plot_perceptron_boundary(
            GATE_INPUTS, labels, w, b, title=f"Target: {gate_name}"
        )
        st.pyplot(fig, clear_figure=True)

    with col_info:
        st.markdown("### 🔢 Truth table")
        for (x1, x2), target in zip(GATE_INPUTS, labels):
            z, pred = perceptron_output(x1, x2, w1, w2, b)
            mark = "✅" if pred == target else "❌"
            st.write(
                f"{mark}  x=({x1},{x2})  z={z:+.2f}  ->  out={pred}  (target={target})"
            )

        st.markdown("---")
        if mistakes == 0:
            st.success(f"🎉 These weights implement **{gate_name}** perfectly!")
        else:
            st.warning(f"{mistakes} of 4 rows wrong. Keep adjusting the sliders.")
            if gate_name == "XOR":
                st.info(
                    "Spoiler: no `(w1, w2, b)` will ever get this to 0 "
                    "mistakes -- see Demo 2 for why."
                )

    with st.expander("🧪 Things to try"):
        st.markdown(
            "- Reproduce **AND**: try `w1 = 1, w2 = 1, b = -1.5`.\n"
            "- Reproduce **OR**: try `w1 = 1, w2 = 1, b = -0.5`.\n"
            "- Flip a sign (`w1 = -1`) -- what logic gate do you get now?\n"
            "- Switch the target to **XOR** and try every combination you "
            "can think of. How close can you get?"
        )


# ===========================================================================
# DEMO 2 -- Training with the perceptron rule
# ===========================================================================
else:
    st.header("Demo 2 · Train it (and watch XOR fail)")
    st.markdown(
        "Rosenblatt's perceptron rule needs no calculus, unlike the "
        "gradient descent from Lesson 1: compare the prediction to the "
        "target, and nudge the weights toward it.\n\n"
        r"$$w_i \leftarrow w_i + \text{lr} \times "
        r"(\text{target} - \text{prediction}) \times x_i$$"
    )

    with st.sidebar:
        st.subheader("⚙️ Settings")
        gate_name = st.selectbox("Gate to learn", list(GATES.keys()), index=0)
        learning_rate = st.select_slider(
            "Learning rate", options=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1
        )
        n_epochs = st.slider("Epochs to run", 1, 40, 15)

    labels = GATES[gate_name]
    trace = train_perceptron_trace(
        GATE_INPUTS, labels, learning_rate=learning_rate, n_epochs=n_epochs
    )

    st.markdown("### ▶️ Step through training, one example at a time")
    step_idx = st.slider(
        "Drag to scrub through updates", 0, len(trace) - 1, len(trace) - 1
    )
    state = trace[step_idx]
    w_now = state["w"]
    b_now = state["b"]

    col_plot, col_info = st.columns([3, 2])

    with col_plot:
        title = f"{gate_name} -- epoch {state['epoch']}"
        fig = plot_perceptron_boundary(GATE_INPUTS, labels, w_now, b_now, title=title)
        st.pyplot(fig, clear_figure=True)

    with col_info:
        st.metric("weights", f"w=[{w_now[0]:.2f}, {w_now[1]:.2f}], b={b_now:.2f}")
        st.metric("mistakes right now", state["mistakes_after"])
        if state["x"] is not None:
            correct = (
                "✅ correct" if state["prediction"] == state["target"] else "❌ mistake"
            )
            st.write(
                f"Last example: x = ({state['x'][0]}, {state['x'][1]}), "
                f"target = {state['target']}, prediction = {state['prediction']} "
                f"-- {correct}"
            )
        else:
            st.write("Starting weights, before any training example is seen.")

    epoch_mistakes = [
        row["mistakes_after"] for row in trace if row["example"] == len(GATE_INPUTS) - 1
    ]
    st.markdown("### 📉 Mistakes per epoch")
    st.pyplot(
        plot_mistakes_curve(epoch_mistakes, current_epoch=state["epoch"] or None),
        clear_figure=True,
    )

    final_mistakes = epoch_mistakes[-1] if epoch_mistakes else state["mistakes_after"]
    if final_mistakes == 0:
        st.success(
            f"✅ Converged! The perceptron learned **{gate_name}** with zero mistakes."
        )
    elif gate_name == "XOR":
        st.error(
            "💥 **Never converges.** XOR isn't linearly separable, so no "
            "straight-line boundary -- and therefore no single perceptron "
            "-- can get this to zero mistakes, no matter how long you train "
            "or which learning rate you pick. Lesson 3 shows that swapping "
            "in a smoother activation function doesn't fix this either."
        )
    else:
        st.warning(
            "Still making mistakes -- try more epochs or a different learning rate."
        )

    with st.expander("🧪 Things to try"):
        st.markdown(
            "- Train **AND** or **OR** to convergence, then note how few "
            "epochs it takes.\n"
            "- Switch to **XOR** and run the full 40 epochs. Does the "
            "mistake count ever reach 0?\n"
            "- Drag the step slider slowly through the first epoch on AND "
            "-- watch the boundary line swing after each individual "
            "mistake.\n"
            "- Try a very small learning rate (`0.01`) on AND. Does it "
            "still converge? How many more epochs does it need?"
        )

st.sidebar.markdown("---")
st.sidebar.caption("ANN course · Lesson 02 · Perceptrons")
