<div align="center">

<img src="docs/assets/course-banner.svg" alt="From Python to Neural Networks — Learn by building, understand by experimenting" width="100%">

**Learn by calculating, building, and experimenting.**<br>
从 Python 到神经网络：从一行代码出发，亲手理解模型如何学习。

[中文](README.md) · **English** · [Interactive lab](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/) · [Project briefs](docs/PROJECTS.md) · [Reference](docs/REFERENCE.md)

**20 notebooks** · **117 exercises** · **18 interactive lessons** · **CPU-friendly examples**

</div>

---

A neural network takes numbers as inputs, combines them through a sequence of operations, and adjusts its parameters using feedback. This course follows that journey: you will start with Python and arrays, calculate a gradient by hand, train models, and investigate their behavior with controlled experiments.

Each notebook connects a concrete problem to **worked calculations, executable code, exercises, and a question to take further**. The accompanying browser lab lets you change values and see the computation respond. Notebook explanations are primarily in Chinese; this guide, the [project briefs](docs/PROJECTS.md), the [reference sheet](docs/REFERENCE.md), and lesson 19 offer English navigation or bilingual material.

## Explore the course

- [What you will learn](#what-you-will-learn) · [Learning paths](#learning-paths) · [Course map](#course-map)
- [Interactive lab](#interactive-lab) · [Get started](#get-started) · [How to work through a lesson](#how-to-work-through-a-lesson)
- [Project work](#project-work) · [Troubleshooting](#troubleshooting) · [Repository and maintenance](#repository-and-maintenance) · [Acknowledgments](#acknowledgments)

## What you will learn

| Practice | What you will do |
| --- | --- |
| **Read and write the computation** | Use Python, NumPy, and PyTorch; label tensor axes; trace individual values through a model. |
| **Understand learning** | Apply the chain rule, write a small network's backward pass, and compare gradients with numerical checks and autograd. |
| **Build models** | Work from sliding windows to CNNs, from recurrent states to LSTMs, and from Q/K/V to a small character GPT; explore GCNs and liquid time-constant networks. |
| **Evaluate and investigate** | Compare against baselines, separate training, validation, and test data, inspect errors, and repeat experiments with different seeds. |
| **Read original work** | Connect an implementation to selected papers, official documentation, and a local minGPT source snapshot. |

The opening lessons introduce Python as they go. Curiosity and a willingness to run and change code are enough to start; algebra and derivatives become useful when you reach the neural-network lessons. Experienced Python users can skim 01–03 and begin with 04, returning to the earlier lessons whenever a shape or operation needs review.

## Learning paths

```text
00–03  Python, functions, arrays and linear algebra
   │
04–07  Derivatives → backpropagation → PyTorch training
   ├── 08–09  Images, convolution and residual connections
   ├── 10–11  Sequences, RNNs and LSTMs
   ├── 12–15  Attention, Transformers and character GPT
   └── 16–17  Graphs and continuous-time networks

18  Research experiments     19  Classification evaluation
```

- **New to Python?** Start at [00](lessons/00_start_and_environment.ipynb) and follow 01–07 in order.
- **Interested in language models?** Complete the foundations, then follow 12–15.
- **Interested in images?** Continue with 08–09 and the digit-recognition project.
- **Ready to investigate?** Use 18 for experiment design and 19 for classification metrics; each can be revisited alongside a model project.

## Course map

### Foundations · 00–07

| Lesson | Topic | You will be able to |
| --- | --- | --- |
| [00 · Start here](lessons/00_start_and_environment.ipynb) | Running a notebook | Run and modify a first program, find the project interpreter, and use an exercise check. |
| [01 · Python values and control flow](lessons/01_python_values_control.ipynb) | Variables, collections, conditions, loops | Process a small table of measurements. |
| [02 · Functions, files and classes](lessons/02_python_functions_files_classes.ipynb) | Reusable code and data | Read and write data, build functions, and define a small model class. |
| [03 · NumPy and linear algebra](lessons/03_numpy_shapes_linear_algebra.ipynb) | Shapes, broadcasting, matrix multiplication | Move from element-by-element loops to batch calculations. |
| [04 · Derivatives and a NumPy MLP](lessons/04_derivatives_numpy_mlp.ipynb) | Chain rule and backpropagation | Calculate gradients, check them numerically, and train a small XOR network. |
| [05 · Tensors and autograd](lessons/05_torch_tensors_autograd.ipynb) | Differentiable computation graphs | Trace gradients through branches and compare against manual calculations. |
| [06 · Modules, data and training](lessons/06_modules_data_training.ipynb) | Model and training loop | Connect batches, loss, gradient clearing, backpropagation, and updates. |
| [07 · Regression and classification](lessons/07_mlp_regression_classification.ipynb) | MLPs and baselines | Compare models with train/validation/test splits and training-set scaling. |

### Model branches · 08–17

| Lesson | Topic | You will be able to |
| --- | --- | --- |
| [08 · Convolution from windows](lessons/08_convolution_from_windows.ipynb) | Sliding windows and shared weights | Compute a convolution and follow the contributions to its gradients. |
| [09 · CNNs and residuals](lessons/09_cnn_training_and_residuals.ipynb) | Images and residual paths | Train digit classifiers and trace layer shapes and residual gradients. |
| [10 · RNNs and time](lessons/10_rnn_time_and_bptt.ipynb) | Recurrence and backpropagation through time | Unroll a state update and accumulate gradients for shared parameters. |
| [11 · LSTM gates and memory](lessons/11_lstm_gates_and_memory.ipynb) | Gated recurrent cells | Implement gate and state updates and compare with PyTorch. |
| [12 · Attention and Q/K/V](lessons/12_attention_qkv.ipynb) | Weighted reading and causal masking | Calculate attention weights and test the effect of changing future inputs. |
| [13 · Transformer block](lessons/13_transformer_block.ipynb) | Multi-head attention, normalization and residuals | Assemble the components of a Transformer block. |
| [14 · Train a tiny GPT](lessons/14_small_gpt_training.ipynb) | Character-level generation | Construct next-character targets, train a decoder, generate, and save it. |
| [15 · Read minGPT](lessons/15_mingpt_reading_and_generation.ipynb) | Reading an existing implementation | Follow tensors, model components, and training through source code. |
| [16 · Graph neural networks](lessons/16_gnn_message_passing.ipynb) | Message passing and GCNs | Aggregate neighbors, test permutation equivariance, and train a GCN. |
| [17 · Liquid neural networks](lessons/17_liquid_neural_networks.ipynb) | Continuous-time dynamics | Connect time steps and time constants to a trainable LTC cell. |

### Experiments and evaluation · 18–19

| Lesson | Topic | You will be able to |
| --- | --- | --- |
| [18 · Research experiments](lessons/18_research_experiments.ipynb) | Controls, seeds and extrapolation | Compare configurations, select using validation data, and record results. |
| [19 · Classification evaluation](lessons/19_evaluation_and_decisions.ipynb) | Imbalance, metrics and decisions | Read a confusion matrix, choose a threshold on validation data, and recognize leakage. |

## Interactive lab

The [online lab](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/) contains 18 browser lessons. Change a value, follow the computation, then reproduce the relevant operation in a notebook. For offline use, open [interactive/index.html](interactive/index.html) or the [standalone HTML file](interactive/neural-network-lab.html).

| When studying | Open | Look for |
| --- | --- | --- |
| 04 | [Network computation](interactive/index.html#network) | How inputs and weights combine at a node. |
| 04–05 | [Backpropagation](interactive/index.html#backprop) | How local derivatives carry a gradient along each edge. |
| 06 | [Shared weights](interactive/index.html#matrix) | How contributions from multiple examples combine. |
| 07 | [Training](interactive/index.html#training) | How the loss and classification boundary change together. |
| 08 | [Convolution](interactive/index.html#convolution) | Which image pixels contribute to a particular output. |
| 09 and 13 | [Residual connections](interactive/index.html#residual) | How a gradient splits at addition and merges at a shared input. |

## Get started

### 1. Get the course

Install [Git](https://git-scm.com/install/) and [uv](https://docs.astral.sh/uv/getting-started/installation/), then open a terminal:

```bash
git clone https://github.com/Lee-Eee-Eee/neural-networks-from-zero.git
cd neural-networks-from-zero
```

If you already have the folder, open a terminal in its root instead.

### 2. Set up the environment and launch Jupyter

```bash
uv sync --locked
uv run jupyter lab
```

The project specifies **Python 3.12** and CPU PyTorch. `uv sync --locked` installs the dependencies recorded in `uv.lock` into the project environment; `uv run` runs commands using that environment. The first setup may take time while packages are downloaded.

Open `lessons/00_start_and_environment.ipynb` in JupyterLab. If prompted for a kernel, select the Python environment associated with this project. To check the interpreter from a terminal:

```bash
uv run python -c "import sys; print(sys.executable)"
```

The path should point into the project's `.venv`: typically `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux. The introductory experiments use generated data, scikit-learn's bundled digits dataset, and a small course corpus.

## How to work through a lesson

1. **Read the opening question.** Before running a cell, predict the result or draw the relevant shapes.
2. **Run the notebook in order.** Earlier cells define names that later cells use; read both the code and its output.
3. **Fill in each exercise.** Replace the placeholder in the exercise cell, then run the following check cell. Expand the hint and reference answer when you need them.
4. **Change one condition.** Modify a value or setting, rerun the relevant cells, and explain the observed change using the computation.
5. **Keep an experiment record.** Note your question, data split, settings, seed, results, and an error or observation worth investigating.

The notebooks include saved example outputs. Your own changes are saved in your local notebook. Restart the kernel and run from the beginning when earlier state may have become inconsistent.

## Project work

After the foundation lessons, choose a [bilingual project brief](docs/PROJECTS.md). Each brief specifies prerequisites, a question, a first experiment, and an extension:

| Project | Central question | Suggested preparation |
| --- | --- | --- |
| **Digit recognition** | How do MLP and CNN errors differ on the same data? | 03, 06–09 |
| **Rare-event screening** | Does high accuracy mean rare positive cases are found? | 07, 18–19 |
| **Character prediction** | How does context length affect prediction and generation? | 12–14 |
| **Graph neighbors** | What happens when nodes are relabeled or edges change? | 03, 16, 18 |

For each project, compare against a baseline where appropriate, use validation data for choices, and report the held-out test result after those choices. The [bilingual reference sheet](docs/REFERENCE.md) collects notation, common shapes, the training loop, and debugging prompts.

## Troubleshooting

<details>
<summary><strong>The notebook uses a different Python or cannot import a dependency</strong></summary>

Run `uv sync --locked` from the repository root and launch Jupyter with `uv run jupyter lab`. Compare the notebook's Python with `uv run python -c "import sys; print(sys.executable)"`, then select the project environment as the kernel and restart it.

</details>

<details>
<summary><strong>A cell reports <code>NameError</code> or an unexpected value</strong></summary>

Notebook cells share state. Restart the kernel and run cells from the top in order; check that the exercise cell was filled and run before its check cell.

</details>

<details>
<summary><strong>A tensor operation reports a shape mismatch</strong></summary>

Print the `.shape` of each operand, label the sample, sequence, channel, or feature axes, and compare them with the operation's requirements. The [reference sheet](docs/REFERENCE.md) has common shape conventions.

</details>

## Repository and maintenance

```text
lessons/             Course notebooks and exercises
interactive/         Browser lab, standalone page and local MathJax assets
models/              Reusable tiny GPT model
```

| Path | Purpose |
| --- | --- |
| [`docs/`](docs/) | Bilingual project briefs and reference sheet. |
| [`tools/chapters/`](tools/chapters/) | Source used to generate notebook content. |
| [`tools/`](tools/) | Course generation, execution, and validation scripts. |
| [`references/`](references/) | Source map, teaching notes, and local minGPT snapshot. |
| [`support.py`](support.py) | Exercise feedback and figure helpers. |
| [`course_manifest.json`](course_manifest.json) | Lesson names, paths, cell counts, and exercise counts. |

Run the course checks from the repository root:

```bash
uv run python tools/validate_course.py
node tools/check_interactive.cjs
```

To execute one lesson and save its outputs, run:

```bash
uv run python -B tools/execute_course.py --pattern '01*.ipynb' --jobs 1
```

This writes back to the selected notebook. Add `--answers` to execute reference solutions into `.build/` instead. `uv run python tools/build_course.py` rebuilds notebooks from their sources: filled exercise answers are preserved, while notebook prose and other code cells are rebuilt. Edit the corresponding file in `tools/chapters/` when maintaining lesson content.

## Acknowledgments

Thank you to the authors and maintainers of these open resources for the ideas, explanations, and implementations that support this course:

- [Build Your Own X: Machine Learning](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning) for its collection of hands-on machine-learning projects and tutorials.
- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and [minGPT](https://github.com/karpathy/minGPT) for approachable computation-graph and language-model implementations; lesson 15 reads a local minGPT snapshot.
- [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en), [Machine Learning with PyTorch and Scikit-Learn](https://github.com/rasbt/machine-learning-book), and [ML From Scratch](https://github.com/eriklindernoren/ML-From-Scratch) for accessible ways to connect explanation, code, and experiments.
- The developers of [MathJax](https://www.mathjax.org/) for the mathematics display in the interactive lab.
- The authors of [PyTorch's tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [GCN](https://arxiv.org/abs/1609.02907), and [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439), whose work provides further reading throughout the course.

See the [source map](references/SOURCES.md) and [teaching notes](references/TEACHING_NOTES.md) for reading questions and teaching references, and [third-party notices](THIRD_PARTY.md) for included materials and their licenses.
