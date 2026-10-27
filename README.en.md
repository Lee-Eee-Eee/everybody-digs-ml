# Everybody Digs ML

<div align="center">

<img src="docs/assets/course-banner.svg" alt="Everybody Digs ML: learn machine learning by building and exploring" width="100%">

**Start with Python. Build models. See how they learn.**

[中文](README.md) · [Try it](#try-it-first) · [Learning paths](#learning-paths) · [Course map](#course-map) · [Get started](#get-started) · [Acknowledgments](#acknowledgments)

`24 notebooks`　 `139 exercises`　 `18 interactive lessons`　 `Python 3.12 · CPU`

</div>

---

How can a model recognize a handwritten digit? What happens to a weight during training? How is the next character in a sequence chosen?

This course brings **equations, code, and experiments** together. Work through a small calculation, implement it, then change a condition and explain the result. The path runs from Python and NumPy through classical machine learning, neural networks, Transformers, a tiny character GPT, graph networks, and diffusion.

> **Predict → run → change one thing → explain what happened.**

Notebook explanations and the interactive lab are primarily in Chinese. This guide provides English navigation; the [project briefs](docs/PROJECTS.md) and [reference sheet](docs/REFERENCE.md) are bilingual.

## Try it first

<p align="center">
  <a href="https://lee-eee-eee.github.io/everybody-digs-ml/interactive/#network"><img src="docs/assets/training-demo.gif" alt="A small network learns XOR: predictions and loss change during training" width="760"></a>
</p>

<p align="center"><sub>The animation shows an independent NumPy network training on synthetic XOR data.<br><a href="https://lee-eee-eee.github.io/everybody-digs-ml/interactive/">Open the interactive lab →</a></sub></p>

**Does increasing a weight always change the output?**

1. [Open the network experiment](https://lee-eee-eee.github.io/everybody-digs-ml/interactive/#network). Set input `x₁` to `0.6`, move `w₁₁`, and watch the hidden node and output.
2. Set `x₁` to `0` and move `w₁₁` again, keeping the other settings fixed. What changes now?
3. Explain the result using `a₁ = x₁w₁₁ + x₂w₂₁`, then follow how weights affect loss in [lesson 04](lessons/neural_network_basics/04_derivatives_numpy_mlp.ipynb).

<details>
<summary><strong>See the explanation</strong></summary>

When `x₁ = 0`, the contribution `x₁w₁₁` stays zero. With other settings fixed, moving `w₁₁` does not change this node's weighted sum or its downstream outputs. With nonzero input, the product changes with the weight.

</details>

## What you will build and investigate

| Question | Practice | Start here |
| :--- | :--- | :--- |
| How does a model recognize a digit? | Compare kNN, SVM, MLP, CNN and a small ResNet; inspect prediction errors. | [Classical ML](lessons/classical_ml/20_classical_classification.ipynb) · [CNN](lessons/computer_vision/09_cnn_training_and_residuals.ipynb) · [ResNet](lessons/computer_vision/23_resnet_from_blocks.ipynb) |
| How do parameters learn from mistakes? | Apply the chain rule, write a NumPy backward pass, and compare with autograd. | [Backpropagation](lessons/neural_network_basics/04_derivatives_numpy_mlp.ipynb) · [autograd](lessons/neural_network_basics/05_torch_tensors_autograd.ipynb) |
| How is text generated? | Work through Q/K/V and masking, train a tiny character GPT, then read minGPT. | [Attention](lessons/llm/12_attention_qkv.ipynb) · [GPT](lessons/llm/14_small_gpt_training.ipynb) · [minGPT](lessons/llm/15_mingpt_reading_and_generation.ipynb) |
| What can one experiment tell us? | Compare baselines, select on validation data, inspect errors, and record multiple seeds. | [Experiments](lessons/experiments/18_research_experiments.ipynb) · [Evaluation](lessons/experiments/19_evaluation_and_decisions.ipynb) |

The course also explores RNNs/LSTMs, GNNs, liquid neural networks, and diffusion. Worked calculations reveal individual operations; scikit-learn and PyTorch experiments connect them into complete workflows.

## Learning paths

![Course roadmap: Python, mathematics and training lead to classical ML, vision, sequences, language models and experiments](docs/assets/course-roadmap.svg)

**Python and mathematics → neural-network foundations → choose a model branch → experiments and projects.**

- **New to Python?** Start with [00](lessons/python_math/00_start_and_environment.ipynb), then follow 01–07.
- **Classical algorithms?** After 03 and 06–07, continue with [20–21](lessons/classical_ml/20_classical_classification.ipynb).
- **Images or language models?** Explore [08–09](lessons/computer_vision/08_convolution_from_windows.ipynb), [23](lessons/computer_vision/23_resnet_from_blocks.ipynb), or [12–15](lessons/llm/12_attention_qkv.ipynb).
- **Ready to investigate?** Use [18–19](lessons/experiments/18_research_experiments.ipynb), then choose a [project brief](docs/PROJECTS.md).

Lesson numbers are stable; the newer lessons 20–23 can be taken once their prerequisites are complete.

## Course map

Open a module to see its lessons and what you will practice. Lesson numbers remain stable across modules.

<details>
<summary><strong>Python and mathematics · python_math/</strong></summary>

| Lesson | Topic | Practice |
| :---: | :--- | :--- |
| [00](lessons/python_math/00_start_and_environment.ipynb) | Getting started | Run a first program and use exercise checks |
| [01](lessons/python_math/01_python_values_control.ipynb) | Python values and control flow | Process measurements with variables, lists and loops |
| [02](lessons/python_math/02_python_functions_files_classes.ipynb) | Functions, files and classes | Organize reusable computations and data input/output |
| [03](lessons/python_math/03_numpy_shapes_linear_algebra.ipynb) | NumPy and linear algebra | Use shapes, broadcasting and matrix multiplication |

</details>

<details>
<summary><strong>Neural-network foundations · neural_network_basics/</strong></summary>

| Lesson | Topic | Practice |
| :---: | :--- | :--- |
| [04](lessons/neural_network_basics/04_derivatives_numpy_mlp.ipynb) | Derivatives and a NumPy MLP | Derive backpropagation and train XOR |
| [05](lessons/neural_network_basics/05_torch_tensors_autograd.ipynb) | Tensors and autograd | Trace computation graphs and check gradients |
| [06](lessons/neural_network_basics/06_modules_data_training.ipynb) | Modules, data and training | Connect batches, loss, gradients and updates |
| [07](lessons/neural_network_basics/07_mlp_regression_classification.ipynb) | Regression and classification | Compare baselines and an MLP using validation |

</details>

<details>
<summary><strong>Classical machine learning · classical_ml/</strong></summary>

| Lesson | Topic | Practice |
| :---: | :--- | :--- |
| [20](lessons/classical_ml/20_classical_classification.ipynb) | Classical classification and clustering | Compare kNN, logistic regression, linear SVM and k-means |
| [21](lessons/classical_ml/21_regression_regularization.ipynb) | MSE, Ridge and LASSO | Study regression and regularization with continuous targets |

</details>

<details>
<summary><strong>Computer vision · computer_vision/</strong></summary>

| Lesson | Topic | Practice |
| :---: | :--- | :--- |
| [08](lessons/computer_vision/08_convolution_from_windows.ipynb) | Convolution from windows | Calculate convolution and explain shared weights |
| [09](lessons/computer_vision/09_cnn_training_and_residuals.ipynb) | CNNs and residuals | Train digit classifiers and track residual gradients |
| [23](lessons/computer_vision/23_resnet_from_blocks.ipynb) | A small ResNet | Train a residual classifier and compare it with a plain CNN |

</details>

<details>
<summary><strong>Sequence models · sequence_models/</strong></summary>

| Lesson | Topic | Practice |
| :---: | :--- | :--- |
| [10](lessons/sequence_models/10_rnn_time_and_bptt.ipynb) | RNNs and backpropagation through time | Unroll state updates and shared-parameter gradients |
| [11](lessons/sequence_models/11_lstm_gates_and_memory.ipynb) | LSTM gates and memory | Implement gates and compare with PyTorch |

</details>

<details>
<summary><strong>Language models · llm/</strong></summary>

| Lesson | Topic | Practice |
| :---: | :--- | :--- |
| [12](lessons/llm/12_attention_qkv.ipynb) | Q, K, V and causal attention | Calculate attention and test the causal mask |
| [13](lessons/llm/13_transformer_block.ipynb) | Transformer block | Assemble attention, normalization and residuals |
| [14](lessons/llm/14_small_gpt_training.ipynb) | Tiny character GPT | Train, generate and save a decoder |
| [15](lessons/llm/15_mingpt_reading_and_generation.ipynb) | Reading minGPT | Trace tensors and training through source code |

</details>

<details>
<summary><strong>Graphs, continuous time and generation · other_networks/</strong></summary>

| Lesson | Topic | Practice |
| :---: | :--- | :--- |
| [16](lessons/other_networks/16_gnn_message_passing.ipynb) | Graph neural networks | Aggregate neighbors and train a GCN |
| [17](lessons/other_networks/17_liquid_neural_networks.ipynb) | Liquid neural networks | Build a trainable LTC from continuous dynamics |
| [22](lessons/other_networks/22_diffusion_from_noise.ipynb) | Introduction to diffusion | Visualize image noise; train and sample a 1D model |

</details>

<details>
<summary><strong>Experiments and evaluation · experiments/</strong></summary>

| Lesson | Topic | Practice |
| :---: | :--- | :--- |
| [18](lessons/experiments/18_research_experiments.ipynb) | Research experiments | Compare controls and seeds; analyze extrapolation |
| [19](lessons/experiments/19_evaluation_and_decisions.ipynb) | Classification evaluation | Select a threshold on validation data |

</details>

## Get started

The examples run on CPU. Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/Lee-Eee-Eee/everybody-digs-ml.git
cd everybody-digs-ml
uv sync --locked
uv run jupyter lab
```

Open [lesson 00](lessons/python_math/00_start_and_environment.ipynb) in JupyterLab and run the cells in order. Fill each exercise cell and run the following check; expand the hint or reference answer when needed.

The project uses Python 3.12. Initial setup downloads dependencies. Select the Python kernel in the project's `.venv`: `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux.

<details>
<summary><strong>Data, hardware and troubleshooting</strong></summary>

- Default experiments use synthetic data, scikit-learn's bundled 8×8 `digits` dataset, and a short text corpus. No GPU is required. The separate 28×28 MNIST dataset is an optional download in lesson 20.
- The models are teaching examples: lesson 14 trains a character GPT; lesson 22 trains a one-dimensional diffusion denoiser and visualizes image noising.
- For `NameError` or unexpected state, restart the kernel and run cells from the beginning; make sure the exercise cell ran before its check.
- For missing dependencies, run `uv sync --locked` from the project directory and launch Jupyter with `uv run jupyter lab`. Use `uv run python -c "import sys; print(sys.executable)"` to find the project interpreter.

</details>

## Explore further

- **Interactive lab:** follow [backpropagation](interactive/index.html#backprop), [training](interactive/index.html#training), [convolution](interactive/index.html#convolution), and [residual connections](interactive/index.html#residual). For offline use, open [interactive/index.html](interactive/index.html) or the [standalone file](interactive/neural-network-lab.html). The lab is primarily in Chinese.
- **Projects:** pick a [digit, rare-event, character, or graph task](docs/PROJECTS.md), then record the data, baseline, result, and interpretation.
- **Reference:** [bilingual reference sheet](docs/REFERENCE.md) · [course design and sources](docs/COURSE_DESIGN.md) · [source map](references/SOURCES.md).
- **Feedback:** report a lesson, explanation, or exercise issue on [GitHub Issues](https://github.com/Lee-Eee-Eee/everybody-digs-ml/issues), including the lesson, environment, and steps to reproduce it.

<details>
<summary><strong>Repository and maintenance</strong></summary>

`lessons/` contains the notebooks by topic; `interactive/` contains the browser lab; `models/` contains the tiny GPT; `tools/chapters/` contains the sources used to generate lessons.

```bash
uv run python tools/validate_course.py
node tools/check_interactive.cjs
# Regenerate the XOR animation from its NumPy training script
uv run python tools/build_training_demo.py
uv run python -B tools/execute_course.py --answers --pattern '01*.ipynb' --jobs 1
```

`uv run python tools/build_course.py` rebuilds notebooks from their sources. Filled exercise answers are preserved; prose and other code cells are rebuilt. Edit the relevant file in `tools/chapters/` when maintaining a lesson.

</details>

## Acknowledgments

Thank you to the authors and maintainers whose resources inform the course's structure and further reading:

- [Build Your Own X: Machine Learning](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning) for its collection of hands-on projects.
- [micrograd](https://github.com/karpathy/micrograd) and [minGPT](https://github.com/karpathy/minGPT) for approachable computation graphs and GPT source code. Lesson 15 uses a local minGPT snapshot with its license retained.
- [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en), [Machine Learning with PyTorch and Scikit-Learn](https://github.com/rasbt/machine-learning-book), and [Happy-LLM](https://github.com/datawhalechina/happy-llm) for connecting explanation, code, and experiments.
- [ML for Beginners](https://github.com/microsoft/ML-For-Beginners) and [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) for course navigation, setup, and prerequisite descriptions.
- The authors of [PyTorch tutorials](https://docs.pytorch.org/tutorials/), [scikit-learn documentation](https://scikit-learn.org/stable/), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [ResNet](https://arxiv.org/abs/1512.03385), [GCN](https://arxiv.org/abs/1609.02907), [LTC](https://arxiv.org/abs/2006.04439), and [DDPM](https://arxiv.org/abs/2006.11239).
- The developers of [MathJax](https://www.mathjax.org/) for the lab's mathematics rendering.

See [course design and references](docs/COURSE_DESIGN.md) for our adaptations and [third-party notices](THIRD_PARTY.md) for included materials and their licenses.
