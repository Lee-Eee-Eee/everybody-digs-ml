# From Python to Neural Networks

[中文](README.md) · [Interactive lab](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/) · [Project briefs](docs/PROJECTS.md) · [Bilingual reference](docs/REFERENCE.md) · [Source map](references/SOURCES.md)

Start with a line of Python and build models that recognize handwritten digits, process sequences, generate characters, and aggregate information on graphs. The course connects **numerical examples, mathematical derivations, implementations, and experiments**: predict an outcome, run the code, inspect intermediate values, and explain what changes when you vary a condition.

The course contains 20 notebooks, 117 fill-in exercises, and an interactive browser lab. Notebook explanations are primarily in Chinese; this guide provides an English course map, and the [project briefs](docs/PROJECTS.md) are bilingual.

## Get started

1. **Set up the environment.** Install [uv](https://docs.astral.sh/uv/getting-started/installation/), open a terminal in the project directory, and run:

   ```bash
   uv sync --locked
   uv run jupyter lab
   ```

   The project uses Python 3.12 and CPU PyTorch. Select the Python interpreter in this project's environment as your notebook kernel: `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux. The introductory experiments use locally generated data, scikit-learn's bundled digits dataset, and a small course corpus.

2. **Open [00: Getting started](lessons/00_start_and_environment.ipynb).** Run the cells in order. Predict each result before running it, then inspect the output.
3. **Complete the exercises.** Fill in each exercise code cell and run the check immediately after it. Expand the hint or reference solution when needed. Notebook state depends on earlier cells; if a name is undefined, restart the kernel and run from the beginning in order.
4. **Run a small experiment.** Keep the data and training settings fixed, change one factor, and record your prediction, result, and explanation. After the foundations, continue with a [project brief](docs/PROJECTS.md).

To get a local copy first:

```bash
git clone https://github.com/Lee-Eee-Eee/neural-networks-from-zero.git
cd neural-networks-from-zero
```

## Learning path

```text
00–03  Python and NumPy
   ↓
04–07  Derivatives, backpropagation and training
   ├── 08–09  Images and convolutional networks
   ├── 10–11  Time series, RNNs and LSTMs
   ├── 12–15  Attention, Transformers and character GPT
   └── 16–17  Graph networks and continuous-time models
                ↓
          18 Research experiments   19 Classification evaluation
```

For a first pass, follow 00–07 in order, then choose a branch. Lesson 18 brings experiment design together; lesson 19 can be taken after 07 and is useful before a classification project.

| Lesson | Topic | What you will practice |
| --- | --- | --- |
| [00](lessons/00_start_and_environment.ipynb) | Orientation | Run, modify, and explain a first program |
| [01](lessons/01_python_values_control.ipynb) | Python values and control flow | Process measurements with containers, conditions, and loops |
| [02](lessons/02_python_functions_files_classes.ipynb) | Functions, files, and classes | Organize reusable computations and data input/output |
| [03](lessons/03_numpy_shapes_linear_algebra.ipynb) | NumPy and linear algebra | Express batch operations with shapes, broadcasting, and matrix multiplication |
| [04](lessons/04_derivatives_numpy_mlp.ipynb) | Derivatives and a NumPy MLP | Implement backpropagation with the chain rule and train XOR |
| [05](lessons/05_torch_tensors_autograd.ipynb) | Tensors and autograd | Trace computation graphs, branch accumulation, and input gradients |
| [06](lessons/06_modules_data_training.ipynb) | Models, data, and training loops | Connect batches, gradient clearing, backpropagation, and updates |
| [07](lessons/07_mlp_regression_classification.ipynb) | Regression and classification | Compare an MLP, scaling, validation, and baselines |
| [08](lessons/08_convolution_from_windows.ipynb) | Sliding windows and convolution | Calculate local windows, shared weights, and gradients |
| [09](lessons/09_cnn_training_and_residuals.ipynb) | CNNs and residual connections | Train a digit classifier and track shapes and gradients |
| [10](lessons/10_rnn_time_and_bptt.ipynb) | RNNs and backpropagation through time | Unroll state updates and accumulate shared-parameter gradients |
| [11](lessons/11_lstm_gates_and_memory.ipynb) | LSTMs | Implement a gated cell and compare it with PyTorch |
| [12](lessons/12_attention_qkv.ipynb) | Q, K, V and causal attention | Calculate attention and test the future-information mask |
| [13](lessons/13_transformer_block.ipynb) | Transformer components | Assemble multi-head attention, positions, normalization, and residuals |
| [14](lessons/14_small_gpt_training.ipynb) | Tiny character GPT | Construct prediction targets, train, generate, and save a model |
| [15](lessons/15_mingpt_reading_and_generation.ipynb) | Reading minGPT | Trace tensors, training, and intermediate representations in a real project |
| [16](lessons/16_gnn_message_passing.ipynb) | Graph neural networks | Calculate neighbor aggregation, test permutation equivariance, and train a GCN |
| [17](lessons/17_liquid_neural_networks.ipynb) | Liquid neural networks | Build a trainable LTC from continuous-time equations and time steps |
| [18](lessons/18_research_experiments.ipynb) | Research experiments | Design controls, multi-seed experiments, and validation selection |
| [19](lessons/19_evaluation_and_decisions.ipynb) | Classification evaluation | Compare precision and recall, choose a threshold on validation data, and identify leakage |

## Interactive lab

[Open online](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/), or open [interactive/index.html](interactive/index.html) offline. A [single-file version](interactive/neural-network-lab.html) is also available. Computation runs in the browser; use it to explore before working through the corresponding notebook.

| After or during | Experiment | Look for |
| --- | --- | --- |
| 04 | [Network computation](interactive/index.html#network) | Inputs, weights, and node outputs |
| 04–05 | [Backpropagation](interactive/index.html#backprop) | Local derivatives for inputs and weights |
| 06 | [Shared weights](interactive/index.html#matrix) | How gradients from multiple examples combine |
| 07 | [Training](interactive/index.html#training) | Changes in loss and the decision boundary |
| 08 | [Convolution](interactive/index.html#convolution) | The pixels read by each output |
| 09, 13 | [Residual connections](interactive/index.html#residual) | Gradient flow and merging at addition |

## Repository and maintenance

| Path | Contents |
| --- | --- |
| `lessons/` | Learning notebooks and fill-in exercises |
| `interactive/` | Browser lab and standalone version |
| `models/` | Reusable tiny GPT model |
| `docs/` | Bilingual project briefs and learning materials |
| `tools/chapters/` | Notebook generation source |
| `support.py` | Plotting and exercise-feedback helpers |
| `references/` | Sources, teaching notes, and a local minGPT copy |

To check the course from the repository root:

```bash
uv run python tools/validate_course.py
node tools/check_interactive.cjs
```

To execute one lesson and save its outputs, run `uv run python -B tools/execute_course.py --pattern '01*.ipynb' --jobs 1`. This writes back to the selected notebook. Add `--answers` to check reference solutions and write the results under `.build/`. Generate notebooks with `uv run python tools/build_course.py`; the generator preserves filled exercise answers but rebuilds notebook prose and other code cells. Edit the sources in `tools/chapters/` when maintaining course content.

The [source map](references/SOURCES.md) connects lessons to original papers and official documentation. [Teaching notes](references/TEACHING_NOTES.md) record organizational influences; [third-party notices](THIRD_PARTY.md) list included code and licenses.
