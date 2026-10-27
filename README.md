# 从 Python 到神经网络

<div align="center">

<img src="docs/assets/course-banner.svg" alt="From Python to Neural Networks — Learn by building, understand by experimenting" width="100%">

**从一行 Python 出发，亲手搭建、训练并理解机器学习模型。**

[English](README.en.md) &nbsp; · &nbsp; [开始学习](#开始学习) &nbsp; · &nbsp; [学习路线](#学习路线) &nbsp; · &nbsp; [课程目录](#课程目录) &nbsp; · &nbsp; [交互学习室](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/)

`24 个 Notebook` &nbsp; `139 道练习` &nbsp; `18 节交互实验` &nbsp; `Python 3.12 · CPU`

</div>

---

这是一套以**计算、代码和实验**为主线的学习教程：从 Python 和 NumPy 出发，理解经典机器学习、神经网络和语言模型，再用对照实验解释结果。每节课从一个具体问题开始，将公式推导、可运行代码、图示和练习放在一起。

**先预测，再计算；亲手实现，再比较结果。** 课程既有手写的核心计算，也使用 scikit-learn 和 PyTorch 构建完整实验。Notebook 正文以中文为主；[English guide](README.en.md)、[项目指南](docs/PROJECTS.md)和[术语速查](docs/REFERENCE.md)提供英文导航或双语内容。

### 先来一个小挑战

**把一条权重调大，输出一定会变吗？**

1. [打开交互学习室](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/#network)，保持其他设置不变，把输入 `x₁` 设为 `0.6`，拖动 `w₁`，观察节点输出。
2. 再把 `x₁` 设为 `0`，重新拖动 `w₁`。这次输出发生了什么？为什么？
3. 试着用加权和 `x₁w₁ + x₂w₂ + b` 解释，再去 [04 手写反向传播](lessons/neural_network_basics/04_derivatives_numpy_mlp.ipynb)追踪权重如何影响损失。

<details>
<summary>查看解释</summary>

当 `x₁ = 0`，这一条输入对加权和的贡献始终是 `x₁w₁ = 0`；其他量保持不变时，调整 `w₁` 不会改变当前输出。输入不为零时，该乘积会随 `w₁` 改变。

</details>

### 选择你的第一步

- **刚接触 Python 和 AI：** 从 [00 开始与环境](lessons/python_math/00_start_and_environment.ipynb) 出发，依次完成 01–07。
- **想从手写数字认识算法：** 在完成基础后，学习 [20 经典分类与聚类](lessons/classical_ml/20_classical_classification.ipynb)、[21 回归与正则化](lessons/classical_ml/21_regression_regularization.ipynb)，再比较 [CNN](lessons/computer_vision/09_cnn_training_and_residuals.ipynb) 与[小型 ResNet](lessons/computer_vision/23_resnet_from_blocks.ipynb)。
- **想理解语言模型：** 完成基础后，从 [12 注意力](lessons/llm/12_attention_qkv.ipynb)走到 [14 字符 GPT](lessons/llm/14_small_gpt_training.ipynb) 和 [15 minGPT 源码阅读](lessons/llm/15_mingpt_reading_and_generation.ipynb)。
- **想先直观看看：** 打开[交互学习室](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/)，改变网络权重，观察输出与梯度。

## 开始学习

### 1. 获取课程

安装 [Git](https://git-scm.com/downloads) 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在终端运行：

```bash
git clone https://github.com/Lee-Eee-Eee/neural-networks-from-zero.git
cd neural-networks-from-zero
```

如果已经有本项目的本地副本，直接在终端进入项目文件夹即可。

### 2. 创建环境并启动 Jupyter

```bash
uv sync --locked
uv run jupyter lab
```

`uv` 会根据项目配置创建隔离环境并安装依赖。课程使用 **Python 3.12** 与 CPU 版 PyTorch；首次安装需要下载依赖。进入 JupyterLab 后，打开 `lessons/python_math/00_start_and_environment.ipynb`，选择项目环境对应的 Python 内核。

| 系统 | 项目环境中的 Python |
| :--- | :--- |
| Windows | `.venv\Scripts\python.exe` |
| macOS / Linux | `.venv/bin/python` |

可用下面的命令确认当前项目使用的解释器：

```bash
uv run python -c "import sys; print(sys.executable)"
```

课程使用本地合成数据、scikit-learn 自带的手写数字数据和课程小语料，基础练习不需要下载大型数据集或配置 GPU。第 09 课使用 1,797 张 8×8 数字图像；第 14 课用 27 条合成短句进行字符级训练、验证和测试。第 20 课提供可选的 MNIST 下载实验，默认关闭；digits 与 28×28 MNIST 是不同的数据集。

### 3. 完成第一节课

1. 打开 [00 从这里开始](lessons/python_math/00_start_and_environment.ipynb)，按顺序运行代码格。
2. 在输出出现之前先预测结果，再修改一个数值，观察结果如何变化。
3. 在练习格中填写答案，运行紧随其后的检查格；遇到困难时展开提示和参考答案。
4. 打开[交互学习室](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/)，改变一条连接的权重，再回到 Notebook 解释变化。

## 学习路线

![课程路线：从 Python、数学和训练基础通往经典机器学习、视觉与序列、语言模型、其他网络及实验](docs/assets/course-roadmap.svg)

<sub>课程路线示意：先建立共同基础，再进入模型分支；第 20–23 课虽然编号较后，也可在相应基础完成后学习。</sub>

**文字路线：** Python 与数学 → 神经网络基础 → 经典算法 / 视觉 / 序列 / LLM / 其他网络 → 实验与项目。

**建议顺序：** 首次学习先完成 00–07，再根据兴趣选择分支；经典机器学习的 20–21 课建议在 03、06–07 后学习；第 23 课可在第 09 课后学习；第 18 课可在完成任一模型分支后学习，第 19 课在完成第 07 课后即可开始。完成每个阶段时，试着独立回答下列问题：

1. **基础：** 我能否解释一个数组的各个轴，并手算一个权重对损失的影响？
2. **模型：** 我能否追踪一次前向计算，说明参数怎样被更新？
3. **实验：** 我能否说明基线、数据划分、评价指标和一次失败案例？

## 课程目录

按兴趣展开模块；课号保持稳定，不必严格按数字顺序学习。

<details>
<summary><strong>Python 与数学 · python_math/</strong></summary>

| 课 | 主题 | 本课动手完成 |
| :---: | :--- | :--- |
| [00](lessons/python_math/00_start_and_environment.ipynb) | 开始与环境 | 运行第一段程序，认识课程与练习 |
| [01](lessons/python_math/01_python_values_control.ipynb) | Python 值与控制流 | 用变量、容器和循环处理测量值 |
| [02](lessons/python_math/02_python_functions_files_classes.ipynb) | 函数、文件与类 | 组织可复用的计算和数据读写 |
| [03](lessons/python_math/03_numpy_shapes_linear_algebra.ipynb) | NumPy 与线性代数 | 用形状、广播和矩阵乘法表达批量计算 |

</details>

<details>
<summary><strong>神经网络基础 · neural_network_basics/</strong></summary>

| 课 | 主题 | 本课动手完成 |
| :---: | :--- | :--- |
| [04](lessons/neural_network_basics/04_derivatives_numpy_mlp.ipynb) | 导数与手写 MLP | 推导反向传播，用 NumPy 训练 XOR |
| [05](lessons/neural_network_basics/05_torch_tensors_autograd.ipynb) | Tensor 与 autograd | 追踪计算图并核对梯度 |
| [06](lessons/neural_network_basics/06_modules_data_training.ipynb) | 模型、数据与训练循环 | 连接 batch、损失、反传和参数更新 |
| [07](lessons/neural_network_basics/07_mlp_regression_classification.ipynb) | 回归与分类 | 比较基线和 MLP，练习标准化与验证 |

</details>

<details>
<summary><strong>经典机器学习 · classical_ml/</strong></summary>

| 课 | 主题 | 本课动手完成 |
| :---: | :--- | :--- |
| [20](lessons/classical_ml/20_classical_classification.ipynb) | 手写数字与经典分类 | 比较 kNN、逻辑回归、线性 SVM 与 k-means |
| [21](lessons/classical_ml/21_regression_regularization.ipynb) | MSE、Ridge 与 LASSO | 用连续亮度预测理解回归损失和正则化 |

</details>

<details>
<summary><strong>视觉与残差 · computer_vision/</strong></summary>

| 课 | 主题 | 本课动手完成 |
| :---: | :--- | :--- |
| [08](lessons/computer_vision/08_convolution_from_windows.ipynb) | 卷积从滑动窗口开始 | 手算卷积输出、理解权重共享 |
| [09](lessons/computer_vision/09_cnn_training_and_residuals.ipynb) | CNN 与残差 | 训练数字分类器，追踪形状和残差梯度 |
| [23](lessons/computer_vision/23_resnet_from_blocks.ipynb) | 小型 ResNet | 组装并训练残差网络，比较普通 CNN 与投影捷径 |

</details>

<details>
<summary><strong>时序模型 · sequence_models/</strong></summary>

| 课 | 主题 | 本课动手完成 |
| :---: | :--- | :--- |
| [10](lessons/sequence_models/10_rnn_time_and_bptt.ipynb) | RNN 与时间反传 | 展开状态递推并累加共享参数梯度 |
| [11](lessons/sequence_models/11_lstm_gates_and_memory.ipynb) | LSTM | 手写门控单元，对照 PyTorch 并训练记忆任务 |

</details>

<details>
<summary><strong>语言模型 · llm/</strong></summary>

| 课 | 主题 | 本课动手完成 |
| :---: | :--- | :--- |
| [12](lessons/llm/12_attention_qkv.ipynb) | Q、K、V 与因果注意力 | 手算注意力，检验未来信息遮罩 |
| [13](lessons/llm/13_transformer_block.ipynb) | Transformer 组件 | 组装多头注意力、归一化与残差 |
| [14](lessons/llm/14_small_gpt_training.ipynb) | 微型字符 GPT | 构造下一字符目标、训练、生成与保存 |
| [15](lessons/llm/15_mingpt_reading_and_generation.ipynb) | 阅读 minGPT | 追踪真实源码中的张量和训练过程 |

</details>

<details>
<summary><strong>图、连续时间与生成 · other_networks/</strong></summary>

| 课 | 主题 | 本课动手完成 |
| :---: | :--- | :--- |
| [16](lessons/other_networks/16_gnn_message_passing.ipynb) | 图神经网络 | 手算邻居聚合，验证置换等变并训练 GCN |
| [17](lessons/other_networks/17_liquid_neural_networks.ipynb) | 液体神经网络 | 从连续时间方程到可训练 LTC |
| [22](lessons/other_networks/22_diffusion_from_noise.ipynb) | 扩散模型入门 | 观察图像加噪，训练一维去噪模型并采样 |

</details>

<details>
<summary><strong>实验与评价 · experiments/</strong></summary>

| 课 | 主题 | 本课动手完成 |
| :---: | :--- | :--- |
| [18](lessons/experiments/18_research_experiments.ipynb) | 研究实验 | 设置对照与多随机种子，分析外推 |
| [19](lessons/experiments/19_evaluation_and_decisions.ipynb) | 分类评价与决策 | 比较精确率与召回率，在验证集选择阈值 |

</details>

## 课程预览

下面的图来自学习本保存的实际输出：从两条计算路径理解残差块，再回到代码中追踪张量形状与梯度。

![小型残差块：卷积分支和投影或恒等捷径在加法节点汇合，随后经过 ReLU](docs/assets/preview-resnet.png)

<sub>图：[23 小型 ResNet](lessons/computer_vision/23_resnet_from_blocks.ipynb) · 主分支与捷径相加后再激活。</sub>

<details>
<summary><strong>再看两个例子：kNN 邻居投票与图像加噪</strong></summary>

![二维 kNN 示例：查询点及三个最近邻](docs/assets/preview-knn.png)

[20 经典分类](lessons/classical_ml/20_classical_classification.ipynb)：用二维样本观察距离与最近邻投票。

![手写数字在不同扩散步数下逐渐加入噪声](docs/assets/preview-diffusion.png)

[22 扩散模型](lessons/other_networks/22_diffusion_from_noise.ipynb)：固定同一图像与噪声，比较不同加噪时刻。训练与反向采样使用一维玩具数据。

</details>

## 交互学习室

<div align="center">

**[在线打开交互学习室 →](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/)**

</div>

实验在浏览器中计算。点选节点或修改参数，即可观察当前数值、梯度或模型行为；学习完一个概念，再回到对应 Notebook 自己实现。

| 学习到这里 | 打开实验 | 观察什么 |
| :--- | :--- | :--- |
| 04：网络前向 | [读图与计算](interactive/index.html#network) | 节点如何组合输入、权重与偏置 |
| 04–05：反向传播 | [沿边回传](interactive/index.html#backprop) | 梯度如何沿计算路径传播和相加 |
| 06：批量训练 | [共用权重](interactive/index.html#matrix) | 多个样本的梯度如何汇合 |
| 07：分类 | [真实训练](interactive/index.html#training) | 训练过程中的损失与决策边界 |
| 08：卷积 | [窗口与共享核](interactive/index.html#convolution) | 一个输出对应哪些输入像素 |
| 09、13：残差 | [两条支路](interactive/index.html#residual) | 加法节点处的前向和反向路径 |

离线时可打开 [interactive/index.html](interactive/index.html)；也提供[单文件版](interactive/neural-network-lab.html)。

## 把知识连成项目

![训练、验证与测试的职责](docs/assets/evaluation-workflow.svg)

<sub>实验流程示意：训练集拟合参数，验证集选择方案，锁定后使用测试集评价。</sub>

完成基础课程后，从[双语项目指南](docs/PROJECTS.md)中选择一条路线，记录问题、配置、基线、结果和解释。

| 项目 | 核心问题 | 对应课程 |
| :--- | :--- | :--- |
| **识别数字** | MLP 和 CNN 会在哪些样本上出错？ | 03、06–09 |
| **筛选稀有事件** | 高准确率能否说明模型找到了少见正例？ | 07、18–19 |
| **预测字符** | 上下文长度如何影响预测和生成？ | 12–14 |
| **图上的邻居** | 节点重编号或改变边，会怎样影响 GCN 输出？ | 03、16、18 |

阅读公式或排查代码时，可随手查阅[双语术语与形状速查](docs/REFERENCE.md)。

## 常见问题

<details>
<summary><strong>找不到课程的 Python 内核怎么办？</strong></summary>

先在项目根目录运行 `uv sync --locked` 和 `uv run jupyter lab`，再检查 Notebook 右上角选择的内核。运行 `uv run python -c "import sys; print(sys.executable)"` 可查看项目环境使用的解释器；若编辑器支持按路径选择解释器，请选择该路径。

</details>

<details>
<summary><strong>运行代码时出现 NameError 或结果与预期不同？</strong></summary>

Notebook 的后续代码格可能依赖前面建立的变量。重启内核后从第一格开始依次运行；完成练习时，先运行填写答案的代码格，再运行紧随其后的检查格。

</details>

<details>
<summary><strong>练习的提示和答案在哪里？</strong></summary>

练习与检查格之后有可展开的提示和参考答案。建议先独立尝试，再使用提示；检查结果也会指出哪些值还需要修改。

</details>

<details>
<summary><strong>需要 GPU 或大型数据集吗？</strong></summary>

课程默认使用 CPU 版 PyTorch，基础实验使用合成数据、scikit-learn 自带的 digits 数据和课程小语料。首次同步环境需要网络，训练速度会因设备不同而变化。

</details>

## 仓库结构与维护

```text
neural_networks_from_zero/
├── lessons/
│   ├── python_math/            # Python、数组与数学
│   ├── neural_network_basics/  # 导数、MLP、训练
│   ├── classical_ml/           # 经典算法与正则化
│   ├── computer_vision/        # CNN、残差与小型 ResNet
│   ├── sequence_models/        # RNN / LSTM
│   ├── llm/                    # 注意力、Transformer、GPT
│   ├── other_networks/         # GNN、LTC、扩散
│   └── experiments/            # 实验设计与评价
├── interactive/        # 浏览器交互实验与单文件版本
├── models/             # 课程复用的微型 GPT
├── docs/               # 双语项目指南、速查与文档资源
├── references/         # 阅读资料、来源说明与 minGPT 快照
├── tools/chapters/     # Notebook 内容的生成源
├── support.py          # 练习反馈与绘图辅助
├── course_manifest.json
├── pyproject.toml
└── uv.lock
```

<details>
<summary><strong>运行与维护命令</strong></summary>

在项目根目录执行：

```bash
# 检查 Notebook 结构、练习和本地链接
uv run python tools/validate_course.py

# 从已保存的 Notebook 输出更新课程预览图
uv run python tools/export_readme_previews.py

# 检查浏览器实验中的数值计算（需要 Node.js）
node tools/check_interactive.cjs

# 运行指定课程并写回该 Notebook 的输出
uv run python -B tools/execute_course.py --pattern '01*.ipynb' --jobs 1

# 验证指定课程的参考答案，结果保存在 .build/
uv run python -B tools/execute_course.py --answers --pattern '01*.ipynb' --jobs 1
```

运行 `uv run python tools/build_course.py` 可从 `tools/chapters/` 重建 Notebook。生成器会保留已填写的练习代码，但会重建课程文字和其他代码格；修改课程内容时请编辑生成源。

</details>

课程如何从参考资料发展出自己的例子与实验，见[课程设计与参考](docs/COURSE_DESIGN.md)。更多原论文、官方文档与阅读任务见[参考地图](references/SOURCES.md)和[教学参考](references/TEACHING_NOTES.md)；仓库内第三方材料及其许可证见[第三方说明](THIRD_PARTY.md)。

## 致谢

感谢以下项目、教程与研究工作为本课程的组织、实现和延伸阅读提供的启发：

- [Build Your Own X: Machine Learning](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning)——从零实现的项目索引，为课程项目练习的组织提供参考。
- [micrograd](https://github.com/karpathy/micrograd) 与 [minGPT](https://github.com/karpathy/minGPT)——分别以紧凑的自动求导和 GPT 实现，帮助读者理解计算图与生成模型；第 15 课使用了仓库内保留许可证的 minGPT 源码快照。
- [动手学深度学习](https://github.com/d2l-ai/d2l-zh)与 [Machine Learning with PyTorch and Scikit-Learn](https://github.com/rasbt/machine-learning-book)——为公式、代码、数据处理和模型评价相结合的学习方式提供参考。
- [Happy-LLM](https://github.com/datawhalechina/happy-llm) 与 [DDPM](https://arxiv.org/abs/2006.11239)——为语言模型的延伸阅读和扩散模型的推导提供参考。
- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/)与 [scikit-learn 文档](https://scikit-learn.org/stable/)——为框架接口、训练流程和评价方法提供可靠资料。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[GCN](https://arxiv.org/abs/1609.02907)、[ResNet](https://arxiv.org/abs/1512.03385)、[Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439) 等论文的作者，以及 [MathJax](https://www.mathjax.org/) 的开发者——让课程中的模型阅读和数学展示成为可能。

也感谢每一位分享知识、开源代码和教学材料的作者，以及使用这套课程、提出建议的学习者。
