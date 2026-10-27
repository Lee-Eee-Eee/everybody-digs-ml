# 神经网络速查与词汇｜Neural-network reference

[课程首页](../README.md) · [English README](../README.en.md) · [项目练习 / Projects](PROJECTS.md)

阅读公式时先确定每个轴和单位，再对照一个具体样本。When reading a formula, identify each axis and its units, then trace one example.

## 常用记号 / Common notation

| 符号 / Symbol | 含义 / Meaning | 例子 / Example |
| --- | --- | --- |
| $B$ | 批大小 / Batch size | 一个 batch 中有多少样本 / Number of examples per batch |
| $T$ | 序列长度 / Sequence length | 每个样本有多少个位置 / Number of positions per example |
| $C$ | 通道或特征数 / Channels or features | 每个位置的向量宽度 / Vector width at each position |
| $x, y, \hat y$ | 输入、目标、预测 / Input, target, prediction | 回归中三者可都是数值 / All may be numeric in regression |
| $\theta$ | 模型的可训练参数 / Trainable model parameters | 权重与偏置 / Weights and biases |
| $L$ | 损失函数 / Loss | 根据预测与目标计算的标量 / A scalar computed from predictions and targets |
| $\nabla_\theta L$ | 损失对参数的梯度 / Gradient of loss with respect to parameters | 参数变化对损失的局部影响 / Local sensitivity of loss to parameters |

## 张量形状 / Tensor shapes

| 场景 / Context | 常见形状 / Typical shape | 读法 / Interpretation |
| --- | --- | --- |
| 批量特征 / Batched features | $(B, C)$ | $B$ 个样本，每个 $C$ 个特征 / $B$ examples with $C$ features each |
| 图像 / Images | $(B, C, H, W)$ | PyTorch 卷积常用的批、通道、高、宽 / Batch, channels, height, width for PyTorch convolutions |
| 序列表示 / Sequence representation | $(B, T, C)$ | 样本、位置、每个位置的表示 / Example, position, representation at that position |
| 注意力分数 / Attention scores | $(B, h, T_q, T_k)$ | 每头、每个查询对各键的分数 / Scores for each query and key in each head |
| 分类 logits / Classification logits | $(B, K)$ | 每个样本的 $K$ 个类别分数 / $K$ class scores for each example |

以上是常见约定；每个函数实际要求以代码为准。例如 PyTorch 中 `nn.Linear(in_features, out_features)` 的权重形状为 `(out_features, in_features)`。These are common conventions; check each function's actual interface. For example, `nn.Linear` stores weights as `(out_features, in_features)`.

## 训练流程 / Training loop

```python
prediction = model(inputs)       # 前向 / forward pass
loss = loss_fn(prediction, target)
optimizer.zero_grad()            # 清除旧梯度 / clear old gradients
loss.backward()                  # 计算梯度 / compute gradients
optimizer.step()                 # 更新参数 / update parameters
```

PyTorch 默认累加梯度；每次更新前通常需要清零。评价时可使用 `model.eval()` 和 `torch.no_grad()`：前者切换 dropout 等层的行为，后者关闭梯度记录。Gradients accumulate by default. For evaluation, `model.eval()` changes layers such as dropout, while `torch.no_grad()` disables gradient recording.

## 概念对照 / Concept map

| 中文 / English | 核心问题 / Key question | 课程 / Lesson |
| --- | --- | --- |
| 链式法则 / Chain rule | 输出变化如何沿依赖关系传回？How does sensitivity flow through dependencies? | [04](../lessons/04_derivatives_numpy_mlp.ipynb), [05](../lessons/05_torch_tensors_autograd.ipynb) |
| 过拟合 / Overfitting | 训练表现好时，未见样本表现如何？How well does the model perform on unseen data? | [07](../lessons/07_mlp_regression_classification.ipynb), [18](../lessons/18_research_experiments.ipynb) |
| 权重共享 / Weight sharing | 同一个参数在多个位置或时间使用时，梯度如何累加？How do gradients accumulate when a parameter is reused? | [08](../lessons/08_convolution_from_windows.ipynb), [10](../lessons/10_rnn_time_and_bptt.ipynb) |
| 因果遮罩 / Causal mask | 每个位置能读取哪些过去和当前位置？Which past and current positions can each position read? | [12](../lessons/12_attention_qkv.ipynb) |
| 残差连接 / Residual connection | 加法两条路径如何影响前向与反向？How do two addition paths affect forward and backward passes? | [09](../lessons/09_cnn_training_and_residuals.ipynb), [13](../lessons/13_transformer_block.ipynb) |
| 数据泄漏 / Data leakage | 哪些统计量或选择利用了不该使用的数据？Which statistics or choices used information they should not have used? | [19](../lessons/19_evaluation_and_decisions.ipynb) |

## 排查 / Troubleshooting

| 现象 / Symptom | 检查 / Check |
| --- | --- |
| `NameError` | 按顺序运行前面的代码格；必要时重启内核并从头运行。Run earlier cells in order; restart and run from the beginning if needed. |
| 形状不匹配 / Shape mismatch | 打印参与运算的 `.shape`，逐轴写出样本、位置、特征的含义。Print each `.shape` and label sample, position, and feature axes. |
| 损失或梯度不变 / Loss or gradients do not change | 检查参数是否属于优化器、前向是否使用这些参数，以及更新步骤是否执行。Check that parameters are in the optimizer, used in the forward pass, and updated. |
| 训练和评价行为不同 / Training and evaluation differ | 检查 `train()` / `eval()`、数据预处理与数据划分。Check `train()` / `eval()`, preprocessing, and data splits. |
| 结果无法复现 / Results differ across runs | 记录随机种子、划分、配置和依赖版本；比较多次运行。Record seeds, splits, configuration and dependency versions; compare repeated runs. |
