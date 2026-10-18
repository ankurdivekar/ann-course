# Intro to Artificial Neural Networks

Course notebooks for **Intro to Artificial Neural Networks**.

## Getting started

Clone the repository:

```bash
git clone https://github.com/ankurdivekar/ann-course.git
cd ann-course
```

Set up the environment (requires [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run jupyter lab
```

## Interactive demos

The `demos/` folder has interactive Streamlit apps to play with alongside the notebooks:

```bash
cd demos
uv run streamlit run 01_GradientDescent.py
```

Swap in `02_Perceptrons.py` or `03_ActivationFunctions.py` for the other demos.

## License

Code is MIT-licensed and course text is CC BY-SA 4.0. See [LICENSE.md](LICENSE.md).

## Updates

This repository is updated automatically as new material is released. Run `git pull` regularly to get the latest notebooks.
