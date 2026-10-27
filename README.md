# 从 Python 到神经网络

[English](README.en.md) · [交互学习室](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/) · [项目练习](docs/PROJECTS.md) · [双语速查](docs/REFERENCE.md) · [参考地图](references/SOURCES.md)

从一行 Python 计算出发，逐步写出能识别手写数字、处理序列、生成字符和聚合图信息的模型。课程把**数值计算、数学推导、代码实现和实验比较**放在一起：先预测结果，再运行代码、检查中间量，最后改变一个条件解释发生了什么。

课程包含 20 个循序渐进的 Notebook、117 个填写练习和一个浏览器交互实验室。Notebook 正文以中文为主；本页面提供中文入口，[English guide](README.en.md) 提供英文课程导航，[项目练习](docs/PROJECTS.md)提供双语任务。

## 开始学习

1. **准备环境。** 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在终端进入项目目录：

   ```powershell
   uv sync --locked
   uv run jupyter lab
   ```

   课程使用 Python 3.12 和 CPU 版 PyTorch。Notebook 运行内核应选择本项目环境中的 Python：Windows 为 `.venv\Scripts\python.exe`，macOS/Linux 为 `.venv/bin/python`。基础实验使用本地合成数据、scikit-learn 自带的 digits 数据和课程小语料。

2. **打开 [00 开始与环境](lessons/00_start_and_environment.ipynb)。** 按顺序运行代码格；在运行前先预测，再检查实际输出。
3. **完成练习。** 在标为“练习”的代码格填写答案，运行紧随其后的检查格；需要时展开提示与参考答案。代码格的状态依赖之前运行过的内容，遇到名称未定义时可重启内核并从头顺序运行。
4. **做一次小实验。** 固定数据与训练设置，改变一个因素，记录预测、结果和解释。完成基础线后，可以选择[项目练习](docs/PROJECTS.md)继续。

还没有本地副本时，先克隆：

```bash
git clone https://github.com/Lee-Eee-Eee/neural-networks-from-zero.git
cd neural-networks-from-zero
```

## 学习路线

```text
00–03  Python 与 NumPy
   ↓
04–07  导数、反向传播与训练
   ├── 08–09  图像与卷积网络
   ├── 10–11  时间序列、RNN 与 LSTM
   ├── 12–15  注意力、Transformer 与字符 GPT
   └── 16–17  图网络与连续时间模型
                ↓
          18 研究实验   19 分类评估
```

第一次学习建议按 00–07 顺序完成，再选感兴趣的分支。第 18 课串联实验设计；第 19 课在完成第 07 课后即可学习，适合在开展分类项目之前练习。

| 课程 | 主题 | 完成后可以做什么 |
| --- | --- | --- |
| [00](lessons/00_start_and_environment.ipynb) | 入门与环境 | 运行、修改并解释第一个程序 |
| [01](lessons/01_python_values_control.ipynb) | Python 值与控制流 | 用容器、条件和循环处理测量数据 |
| [02](lessons/02_python_functions_files_classes.ipynb) | 函数、文件与类 | 组织可复用的计算和数据读写 |
| [03](lessons/03_numpy_shapes_linear_algebra.ipynb) | NumPy 与线性代数 | 用形状、广播和矩阵乘法表达批量计算 |
| [04](lessons/04_derivatives_numpy_mlp.ipynb) | 导数与 NumPy MLP | 用链式法则手写反向传播并训练 XOR |
| [05](lessons/05_torch_tensors_autograd.ipynb) | Tensor 与自动求导 | 追踪计算图、分支梯度与输入梯度 |
| [06](lessons/06_modules_data_training.ipynb) | 模型、数据与训练循环 | 连接 batch、清梯度、反传和参数更新 |
| [07](lessons/07_mlp_regression_classification.ipynb) | 回归与分类 | 比较 MLP、标准化、验证集与基线 |
| [08](lessons/08_convolution_from_windows.ipynb) | 滑动窗口与卷积 | 计算局部窗口、共享权重及梯度 |
| [09](lessons/09_cnn_training_and_residuals.ipynb) | CNN 与残差 | 训练数字分类器并追踪形状和梯度 |
| [10](lessons/10_rnn_time_and_bptt.ipynb) | RNN 与时间反传 | 展开状态递推并累加共享参数的梯度 |
| [11](lessons/11_lstm_gates_and_memory.ipynb) | LSTM | 手写门控单元并与 PyTorch 对照 |
| [12](lessons/12_attention_qkv.ipynb) | Q、K、V 与因果注意力 | 手算注意力并检验未来信息遮罩 |
| [13](lessons/13_transformer_block.ipynb) | Transformer 组件 | 组装多头注意力、位置、归一化和残差 |
| [14](lessons/14_small_gpt_training.ipynb) | 微型字符 GPT | 构造预测目标、训练、生成和保存模型 |
| [15](lessons/15_mingpt_reading_and_generation.ipynb) | minGPT 源码阅读 | 追踪真实模型的张量、训练与中间表示 |
| [16](lessons/16_gnn_message_passing.ipynb) | 图网络 | 手算邻居聚合、验证置换等变并训练 GCN |
| [17](lessons/17_liquid_neural_networks.ipynb) | 液体神经网络 | 从连续时间方程和步长构建可训练 LTC |
| [18](lessons/18_research_experiments.ipynb) | 研究实验 | 设计对照、多种子实验与验证选择 |
| [19](lessons/19_evaluation_and_decisions.ipynb) | 分类评估与决策 | 比较精确率和召回率，在验证集选阈值，识别数据泄漏 |

## 交互学习室

[在线打开](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/)；离线可打开 [interactive/index.html](interactive/index.html)，也可使用[单文件版](interactive/neural-network-lab.html)。实验室在浏览器中计算，适合在 Notebook 推导前先动手观察。

| 学习位置 | 实验 | 观察重点 |
| --- | --- | --- |
| 04 | [读图与计算](interactive/index.html#network) | 节点的输入、权重与输出 |
| 04–05 | [沿边回传](interactive/index.html#backprop) | 输入和权重对应的局部导数 |
| 06 | [共用权重](interactive/index.html#matrix) | 多个样本的梯度如何汇合 |
| 07 | [真实训练](interactive/index.html#training) | 损失与分类边界的变化 |
| 08 | [窗口与共享核](interactive/index.html#convolution) | 每个输出读取的像素 |
| 09、13 | [两条支路](interactive/index.html#residual) | 加法处的梯度传播与汇合 |

## 文件与维护

| 路径 | 内容 |
| --- | --- |
| `lessons/` | 学习 Notebook，可直接填写练习 |
| `interactive/` | 浏览器实验室及单文件版 |
| `models/` | 可复用的微型 GPT 模型 |
| `docs/` | 双语项目练习等学习资料 |
| `tools/chapters/` | Notebook 生成源代码 |
| `support.py` | 绘图与练习反馈辅助函数 |
| `references/` | 来源、教学参考与 minGPT 本地副本 |

维护课程时，在项目根目录运行：

```powershell
uv run python tools/validate_course.py
node tools/check_interactive.cjs
```

从终端执行一课并保存输出，可运行 `uv run python -B tools/execute_course.py --pattern '01*.ipynb' --jobs 1`。该命令会写回所选 Notebook；检验参考答案可加 `--answers`，结果写入 `.build/`。生成 Notebook 使用 `uv run python tools/build_course.py`，生成器会保留已填写的练习答案，但会重建 Notebook 的文字和其他代码格；维护课程内容时应修改 `tools/chapters/` 中的生成源。

课程关联的原论文、官方文档和阅读问题见[参考地图](references/SOURCES.md)；课程组织上的借鉴见[教学参考](references/TEACHING_NOTES.md)。第三方代码及许可证见[第三方说明](THIRD_PARTY.md)。
