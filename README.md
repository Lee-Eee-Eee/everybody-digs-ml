# 从 Python 到神经网络

让电脑认出一张手写数字，或者接着 `the ` 写一句话，需要怎样的程序？这套课从一行计算、几条测量记录开始，逐渐写出能够学习的模型。每个新概念都落到一个小例子：算出数值、画出变化，再写成自己的函数。

可以先打开 [交互学习室](https://lee-eee-eee.github.io/neural-networks-from-zero/interactive/)，点一条连线、改一个权重，看它怎样影响输出与损失。来到反向传播时，沿着同一张图逐步回传；回到学习本，再将这条路径写成 NumPy 与 PyTorch。

## 从哪里开始

[00 开始与环境](lessons/00_start_and_environment.ipynb) 会运行第一个小程序。01–07 连成一条基础线；之后按图像、时间序列、注意力、图或连续时间网络继续。学习本含113个填写练习，演示保留实际运行结果，提示与答案可折叠查看。

## 交互实验怎样穿插

| 学到这里 | 打开实验 | 留意什么 |
| --- | --- | --- |
| 04 小网络前向 | [读图与计算](interactive/index.html#network) | 一个节点怎样由入边上的乘积算出来 |
| 04–05 反向传播 | [沿边回传](interactive/index.html#backprop) | 对输入与对权重使用不同的局部导数 |
| 06 批处理 | [共用权重](interactive/index.html#matrix) | 多份梯度怎样汇合成矩阵里的一个格 |
| 07 分类训练 | [真实训练](interactive/index.html#training) | 损失与边界如何一起改变 |
| 08 卷积 | [窗口与共享核](interactive/index.html#convolution) | 当前输出读取了哪些像素 |
| 09、13 残差 | [两条支路](interactive/index.html#residual) | 梯度在加法处分开，在共同输入处相加 |

实验室含18节，计算在浏览器现场完成。离线使用时打开 `interactive/index.html`；单文件版为 `interactive/neural-network-lab.html`。

## 课程

| 课 | 内容 | 亲手完成的核心任务 |
| --- | --- | --- |
| [00](lessons/00_start_and_environment.ipynb) | 运行与学习方法 | 选择本项目解释器，修改代码并检查结果 |
| [01](lessons/01_python_values_control.ipynb) | Python 值、容器、条件与循环 | 处理一份测量数字表 |
| [02](lessons/02_python_functions_files_classes.ipynb) | 函数、文件与类 | 读写小数据，写函数与最小可继承类 |
| [03](lessons/03_numpy_shapes_linear_algebra.ipynb) | NumPy 与线性代数 | 从逐项循环写到广播与矩阵乘法 |
| [04](lessons/04_derivatives_numpy_mlp.ipynb) | 导数、链式法则与手写 MLP | 核对每个权重、输入的导数，训练 XOR |
| [05](lessons/05_torch_tensors_autograd.ipynb) | Tensor 与 autograd | 追踪反传、分支累加和输入梯度 |
| [06](lessons/06_modules_data_training.ipynb) | Module、数据与训练循环 | 独立写出取 batch、清梯度、反传和更新 |
| [07](lessons/07_mlp_regression_classification.ipynb) | MLP 回归与分类 | 连接损失、模型、标准化、验证与基线 |
| [08](lessons/08_convolution_from_windows.ipynb) | 卷积从滑动窗口开始 | 手算输出、权重共享及梯度 |
| [09](lessons/09_cnn_training_and_residuals.ipynb) | CNN 与残差 | 训练数字分类器，核对逐层形状与残差梯度 |
| [10](lessons/10_rnn_time_and_bptt.ipynb) | RNN 与时间反传 | 展开状态递推，累加共享权重的梯度 |
| [11](lessons/11_lstm_gates_and_memory.ipynb) | LSTM 门与记忆 | 手写 Cell，对齐 PyTorch 并训练记忆任务 |
| [12](lessons/12_attention_qkv.ipynb) | Q、K、V 与因果注意力 | 手算权重、加权和，检验未来扰动 |
| [13](lessons/13_transformer_block.ipynb) | Transformer 组件 | 组装多头、位置、归一化与残差块 |
| [14](lessons/14_small_gpt_training.ipynb) | 微型 GPT 训练 | 构造下一字符目标、训练、生成和保存 |
| [15](lessons/15_mingpt_reading_and_generation.ipynb) | minGPT 源码阅读 | 沿真实项目追踪张量、训练与中间表示 |
| [16](lessons/16_gnn_message_passing.ipynb) | GNN 与消息传递 | 手算邻居聚合，验证置换等变，训练 GCN |
| [17](lessons/17_liquid_neural_networks.ipynb) | 液体神经网络 | 从 ODE、步长与时间常数写到可训练 LTC |
| [18](lessons/18_research_experiments.ipynb) | 从模型到研究 | 设对照、多种子实验、验证选择、解释外推误差 |

## 三条学习线

**实现。** 每个新操作先单独运行，再接入模型。数组形状、样本/位置/通道编号、参数与状态都在使用处解释。模型核心代码写在学习本里；第 14 课复用的 `models/tiny_gpt.py` 在第 13 课逐块搭建。

**推导。** 从具体数值走到公式，再用差分或框架核对。重点包括输入和权重的反传、共享参数跨时间的贡献、卷积与图聚合、注意力的归一化，以及连续时间状态的离散更新。

**研究。** 用同一划分、可解释基线和单变量对照回答小问题。各课末尾给原论文或官方源码的阅读问题；第 18 课把假设、配置选择、测试结果与记录保存接成完整流程。

## 模型之间怎样联系

| 结构 | 写进模型的关系 | 适合追问的实验 |
| --- | --- | --- |
| MLP | 特征间的全连接组合 | 非线性与容量改变了什么 |
| CNN | 局部连接、权重共享 | 平移、局部模式与感受野 |
| RNN / LSTM | 按顺序更新的状态 | 记忆间隔、状态与早期梯度 |
| Transformer | 依输入决定的信息读取 | mask、位置与上下文干预 |
| GNN | 沿边聚合、节点重编号对称性 | 图结构、打乱边与过平滑 |
| LTC | 输入/状态相关的连续时间动力学 | Δt、有效时间常数与不规则观测 |

## 独立 uv 环境

在 PowerShell 中运行：

```powershell
git clone https://github.com/Lee-Eee-Eee/neural-networks-from-zero.git
cd neural-networks-from-zero
uv sync --locked
uv run jupyter lab
```

已配置 Python 3.12 和官方 CPU 版 PyTorch。Notebook 内核选择本目录 `.venv\Scripts\python.exe`。基础实验使用本地合成数据、离线 digits 数据与自带小语料。

也可先激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

从终端执行一课并保存输出：

```powershell
uv run python -B tools/execute_course.py --pattern '01*.ipynb' --jobs 1
```

## 参考材料怎样使用

- 本地 NTU GenAI/ML：第 07、09 课连接 HW5 回归和 HW6 图像分类；第 15 课衔接 token、Logit Lens 与干预；后续可继续进入 RAG、微调、遗忘、扩散和语音作业。
- [minGPT](https://github.com/karpathy/minGPT)：第 12–15 课从逐头计算连接到原项目的合并投影、模型、训练与生成。本地参考副本在 `references/minGPT/`，保留原许可证与固定 commit。
- PyTorch 官方教程与原论文：在相应课末尾按具体问题阅读。详细映射与来源记录见 [参考地图](references/SOURCES.md)。
- [dive-into-llms](https://github.com/Lordog/dive-into-llms)：参考其中用一个任务串联输入、实现与输出的教学组织，具体阅读记录见 [教学参考](references/TEACHING_NOTES.md)。

## 文件

`lessons/` 是学习入口；`interactive/` 是交互室；`models/` 保存课程中搭出的可复用模型；`tools/chapters/` 保存课程生成源代码；`support.py` 提供绘图和练习反馈。填写练习时直接编辑 notebook。

交互计算可运行 `node tools/check_interactive.cjs` 核对。上游 minGPT 与 MathJax 保留各自的许可证，见 [第三方说明](THIRD_PARTY.md)。
