"""
01_GradientDescent.py
Interactive Gradient Descent demos for a high-school Artificial Neural Networks class.

Run it with:
    cd demos
    streamlit run 01_GradientDescent.py

Two demos:
  1. "Downhill on a curve" -- pick a function, a learning rate and a starting
     point, then step through gradient descent one iteration at a time.
  2. "Fit a line to data"  -- watch gradient descent tune the slope and
     intercept of a line to match noisy data (a tiny neural network).

If Streamlit segfaults on macOS, see ../references/Demos/SEGFAULT_FIX.md
The matplotlib backend fix from that document is applied below.
"""

import os

import numpy as np
import streamlit as st
from demo_utils import (
    FUNCTIONS,
    diverged,
    fit_line_gradient_descent,
    gradient_descent,
    make_line_data,
    plot_descent_on_curve,
    plot_line_fit,
    plot_loss_curve,
    step_table,
)

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg", force=True)  # non-GUI backend: required under Streamlit
import matplotlib.pyplot as plt

plt.ioff()


st.set_page_config(page_title="Gradient Descent Demo", page_icon="⛰️", layout="wide")

st.title("⛰️ Gradient Descent -- watch a machine learn")
st.markdown(
    "Gradient descent is the rule almost every neural network uses to learn: "
    "**feel which way is downhill, take a small step, repeat.** "
    "These two demos let you run it by hand and see what the *learning rate* does."
)

demo = st.sidebar.radio(
    "Choose a demo",
    ["1 · Downhill on a curve", "2 · Fit a line to data"],
)
st.sidebar.markdown("---")


# ===========================================================================
# DEMO 1 -- Gradient descent on a 1-D curve
# ===========================================================================
if demo.startswith("1"):
    st.header("Demo 1 · Downhill on a curve")
    st.markdown(
        "We want to find the lowest point of a curve $y = f(x)$. "
        "Gradient descent uses the **slope** at the current point:\n\n"
        r"$$x_{\text{new}} = x_{\text{old}} - \text{learning rate} \times "
        r"\text{slope at } x_{\text{old}}$$"
    )

    with st.sidebar:
        st.subheader("⚙️ Settings")
        fn_name = st.selectbox("Function to minimize", list(FUNCTIONS.keys()))
        spec = FUNCTIONS[fn_name]
        f, grad = spec["f"], spec["grad"]
        lo, hi = spec["x_range"]

        st.caption(spec["note"])

        start_slider = st.slider(
            "Starting point x₀",
            min_value=float(lo) - 3.0,
            max_value=float(hi) + 3.0,
            value=float(spec["default_start"]),
            step=0.1,
        )
        start_x = st.number_input(
            "…or type an exact starting point x₀",
            min_value=float(lo) - 20.0,
            max_value=float(hi) + 20.0,
            value=float(start_slider),
            step=0.1,
            format="%.3f",
        )

        lr_preset = st.select_slider(
            "Learning rate",
            options=[0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0, 1.05, 1.1, 1.5, 2.0],
            value=0.1,
        )
        learning_rate = st.number_input(
            "…or type an exact learning rate",
            min_value=0.0,
            max_value=5.0,
            value=float(lr_preset),
            step=0.01,
            format="%.4f",
        )

        n_steps = st.slider("How many steps to run", 1, 100, 30)

    # Run the whole descent, then let the student scrub through it.
    xs = gradient_descent(grad, start_x, learning_rate, n_steps)
    last_step = len(xs) - 1

    st.markdown("### ▶️ Step through the descent")
    current = st.slider(
        "Drag to move through the iterations",
        min_value=0,
        max_value=last_step,
        value=last_step,
        step=1,
    )

    col_plot, col_info = st.columns([3, 2])

    with col_plot:
        fig = plot_descent_on_curve(f, (lo, hi), xs, current)
        st.pyplot(fig, clear_figure=True)

    with col_info:
        x_now = xs[current]
        slope_now = grad(x_now)
        st.metric("current x", f"{x_now:.4f}")
        st.metric("current height  y = f(x)", f"{f(x_now):.4f}")
        st.metric("slope at current x", f"{slope_now:.4f}")

        if current < last_step:
            nxt = x_now - learning_rate * slope_now
            st.markdown(
                f"**Next step:**\n\nx − lr·slope = {x_now:.3f} − {learning_rate:g}·({slope_now:.3f}) = **{nxt:.3f}**"
            )
        direction = "left ⬅️" if slope_now > 0 else ("right ➡️" if slope_now < 0 else "nowhere — flat!")
        st.info(
            f"Slope is {'positive' if slope_now > 0 else ('negative' if slope_now < 0 else 'zero')}, "
            f"so downhill is **{direction}**"
        )

    # Verdict on how the run went. "Exploding" = ran to infinity, OR the height
    # ended up higher than where we started (steps growing instead of shrinking).
    exploding = diverged(xs) or (np.isfinite(f(xs[-1])) and f(xs[-1]) > f(xs[0]) + 1e-9)
    if exploding:
        st.error(
            "💥 **It's diverging.** The learning rate is too big: each step "
            "overshoots the bottom by *more* than the last one, so the point "
            "climbs the walls instead of settling. Try a smaller learning rate."
        )
    else:
        final = xs[-1]
        gap = abs(final - spec["true_min_x"])
        if gap < 0.05:
            st.success(
                f"✅ Reached x = {final:.4f} after {last_step} steps — "
                f"essentially the true minimum (x = {spec['true_min_x']})."
            )
        elif gap < 0.5:
            st.warning(
                f"🟡 Ended at x = {final:.4f}, close to the minimum "
                f"(x = {spec['true_min_x']}) but not there yet — try more steps "
                f"or a slightly bigger learning rate."
            )
        else:
            st.warning(
                f"🟠 Ended at x = {final:.4f}, still far from x = "
                f"{spec['true_min_x']}. Either the learning rate is tiny (slow "
                f"crawl) or the descent settled into a *different* valley — "
                f"try another starting point."
            )

    st.markdown("### 📉 Loss curve")
    st.pyplot(plot_loss_curve(f(xs), current), clear_figure=True)

    with st.expander("📋 See every step as a table"):
        st.dataframe(step_table(f, xs), width="stretch", height=300)

    with st.expander("🧪 Things to try"):
        st.markdown(
            "- Set the learning rate to **1.0** on `y = x^2`. What happens? Now **1.05**.\n"
            "- Set it very small (**0.003**). How many steps to reach the bottom?\n"
            "- On `y = x^4 - 3x^3 + 2`, start at **x = -1**, then at **x = 3**. "
            "Do you land in the same valley?\n"
            "- On the bumpy function, try several starting points — the result "
            "changes a lot."
        )


# ===========================================================================
# DEMO 2 -- Fitting a line to data
# ===========================================================================
else:
    st.header("Demo 2 · Fit a line to data")
    st.markdown(
        "Now the 'curve' is a **loss function**. We have data points and want the "
        "straight line $\\hat{y} = w x + b$ that fits best. 'Best' = smallest "
        "**mean squared error**. Gradient descent tunes two numbers at once, "
        "$w$ (slope) and $b$ (intercept) — this is a neural network with one "
        "neuron."
    )

    with st.sidebar:
        st.subheader("⚙️ Settings")
        true_w = st.slider("True slope of the data", 0.0, 5.0, 2.3, 0.1)
        true_b = st.slider("True intercept of the data", -5.0, 10.0, 4.0, 0.5)
        noise = st.slider("Noise in the data", 0.0, 8.0, 3.0, 0.5)
        st.markdown("---")
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05],
            value=0.01,
        )
        n_steps = st.slider("How many steps to run", 10, 2000, 400, step=10)

    x_data, y_data = make_line_data(true_w=true_w, true_b=true_b, noise=noise)
    history = fit_line_gradient_descent(x_data, y_data, learning_rate=learning_rate, n_steps=n_steps)
    last_step = len(history["w"]) - 1

    st.markdown("### ▶️ Step through the training")
    current = st.slider(
        "Drag to watch the line improve",
        min_value=0,
        max_value=last_step,
        value=last_step,
        step=1,
    )

    w_now = history["w"][current]
    b_now = history["b"][current]
    loss_now = history["loss"][current]

    col_plot, col_info = st.columns([3, 2])
    with col_plot:
        st.pyplot(plot_line_fit(x_data, y_data, w_now, b_now, current, loss_now), clear_figure=True)
    with col_info:
        st.metric("slope  w", f"{w_now:.3f}", help=f"data was built with slope {true_w}")
        st.metric("intercept  b", f"{b_now:.3f}", help=f"data was built with intercept {true_b}")
        st.metric("loss (mean squared error)", f"{loss_now:,.3f}")
        st.caption("Watch w and b crawl toward the true values while the loss drops.")

    if not np.isfinite(history["loss"][-1]) or history["loss"][-1] > 1e6:
        st.error("💥 The loss blew up — this learning rate is too large for this data. Lower it (try 0.005 or 0.002).")
    elif history["loss"][-1] > history["loss"][0] * 0.5:
        st.warning(
            "🟡 Training barely moved. The learning rate is small or there aren't enough steps — increase one of them."
        )
    else:
        st.success(
            f"✅ Final line: y = {history['w'][-1]:.3f}·x + {history['b'][-1]:.3f} "
            f"(data came from y = {true_w}·x + {true_b} + noise). "
            f"Loss fell from {history['loss'][0]:,.1f} to {history['loss'][-1]:,.2f}."
        )

    st.markdown("### 📉 Loss curve")
    st.pyplot(
        plot_loss_curve(history["loss"], current, ylabel="mean squared error"),
        clear_figure=True,
    )

    with st.expander("🧪 Things to try"):
        st.markdown(
            "- Use **400 steps** at learning rate **0.01** — good fit. Now drop to "
            "**50 steps**: the line is still improving when training stops.\n"
            "- Push the learning rate to **0.05**. Does the loss curve wobble or blow up?\n"
            "- Crank the **noise** up. The line still finds the trend, but the "
            "final loss can't reach zero — the noise sets a floor.\n"
            "- Watch **b** (intercept): it often lags behind **w** because the "
            "x-values are large, so w's slope of the loss is steeper."
        )

st.sidebar.markdown("---")
st.sidebar.caption("ANN course · Lesson 01 · Gradient Descent")
