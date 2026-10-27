# 动手项目｜Build projects

[课程首页](../README.md) · [English README](../README.en.md)

这些项目把多节课连成一次可以复现的实验。先选一条路线，运行相关 Notebook，再记录自己的结果。每个项目均可使用课程现有数据和 CPU 环境；先完成基础版本，再尝试扩展。

These projects connect several lessons into reproducible experiments. Pick a track, run the linked notebooks, and record your own results. Each starts with the course's existing data and CPU environment.

## 共用交付物 / Shared deliverables

1. **问题与预期 / Question and hypothesis:** 用一两句话说明要比较什么、预期观察到什么。State the comparison and a testable expectation.
2. **实验记录 / Experiment record:** 记录数据与划分、预处理、模型、指标、随机种子和运行命令。Record data and splits, preprocessing, models, metrics, seeds, and commands.
3. **结果 / Results:** 展示基线与模型的验证结果、最终测试结果和至少一个有解释的错例。Show baseline and validation results, final test results, and at least one explained error.
4. **解释 / Interpretation:** 将结果与预期对照，指出支持结论的证据，并提出一个下一步对照。Compare evidence with the hypothesis and propose a follow-up control.

先在验证集做选择；完成模型与规则的选择后再评价测试集。跨随机种子重复时，记录每次结果，而不只保留最好的一次。Choose models and rules on validation data; evaluate the held-out test set after selection. For repeated seeds, retain every run.

## 项目 A：识别数字 / Recognize digits

**先修 / Prerequisites:** [03](../lessons/python_math/03_numpy_shapes_linear_algebra.ipynb), [06](../lessons/neural_network_basics/06_modules_data_training.ipynb), [07](../lessons/neural_network_basics/07_mlp_regression_classification.ipynb), [08–09](../lessons/computer_vision/09_cnn_training_and_residuals.ipynb)。

**问题 / Question:** 在相同训练与验证划分、相同评价指标下，MLP 和 CNN 的错例有何不同？How do MLP and CNN errors differ on the same split?

- 运行第 09 课，记录两种模型的参数量、验证表现与预测类别。Run lesson 09 and record parameter counts, validation performance, and predictions.
- 固定评价流程，对比两种模型的混淆矩阵；各选一个错例，观察它们的像素。Compare confusion matrices and inspect an error from each model.
- **扩展 / Extension:** 在验证集上研究平移后表现；锁定比较方案后再报告测试结果。Study a one-pixel shift on validation data, then report held-out results after locking the comparison.

## 项目 B：稀有事件筛选 / Screen rare events

**先修 / Prerequisites:** [07](../lessons/neural_network_basics/07_mlp_regression_classification.ipynb), [18](../lessons/experiments/18_research_experiments.ipynb), [19](../lessons/experiments/19_evaluation_and_decisions.ipynb)。

**问题 / Question:** 高准确率是否意味着模型能找到少见正例？Does high accuracy imply that rare positives are found?

- 比较多数类基线和逻辑回归；报告混淆矩阵、精确率、召回率和正例数量。Compare the majority baseline and logistic regression using confusion matrices, precision, recall, and positive counts.
- 在验证集上分别使用两个预先指定的召回率要求选择阈值，观察误报和漏报的变化。Use two predefined recall targets to select thresholds on validation data; compare false positives and false negatives.
- **扩展 / Extension:** 改变数据划分的随机种子，记录每次阈值和测试指标。Repeat with several split seeds and record thresholds and test metrics.

## 项目 C：字符预测 / Predict characters

**先修 / Prerequisites:** [12](../lessons/llm/12_attention_qkv.ipynb), [13](../lessons/llm/13_transformer_block.ipynb), [14](../lessons/llm/14_small_gpt_training.ipynb)。

**问题 / Question:** 上下文长度改变时，字符预测与生成会有什么变化？How does context length affect next-character prediction and generation?

- 沿用第 14 课的语料和划分，选择两种上下文长度，保持模型和训练流程中的其他设置一致。Use lesson 14's corpus and splits; compare two context lengths while holding other settings fixed.
- 记录验证损失、训练步数与固定提示下的生成样本；完成选择后报告测试损失。Record validation loss, training steps, and generations from the same prompt; report test loss after selection.
- **扩展 / Extension:** 多次初始化，检查结果是否稳定；说明不同上下文长度对计算量的影响。Repeat with multiple initializations and discuss the computational cost of different context lengths.

## 项目 D：图上的邻居 / Neighbors on a graph

**先修 / Prerequisites:** [03](../lessons/python_math/03_numpy_shapes_linear_algebra.ipynb), [16](../lessons/other_networks/16_gnn_message_passing.ipynb), [18](../lessons/experiments/18_research_experiments.ipynb)。

**问题 / Question:** 当节点重编号或边发生变化时，GCN 的输出会怎样变化？What changes when nodes are relabeled or edges are altered?

- 对同一图和特征做节点重编号，检验输出是否按相同方式重排。Relabel nodes and verify that outputs follow the same permutation.
- 固定特征与模型参数，改变边；比较节点表示与原图结果。Hold features and parameters fixed, change edges, and compare node representations.
- **扩展 / Extension:** 改变聚合次数，记录节点表示的差异并解释过平滑。Vary the number of aggregation steps and investigate oversmoothing.

## 自查 / Review your work

| 检查 / Check | 完成标准 / Done when |
| --- | --- |
| 可复现 / Reproducible | 给出运行命令、数据来源、划分和种子 / Commands, data, split and seeds are recorded |
| 可比较 / Comparable | 基线和模型使用同一评价数据与指标 / Baseline and model share evaluation data and metrics |
| 无泄漏 / Leakage-aware | 预处理只在训练集拟合，验证集用于选择，测试集用于最终评价 / Preprocessing fits training data; validation selects; test evaluates |
| 可解释 / Interpretable | 给出计数、曲线或错例，并将证据连接到问题 / Counts, curves or errors support the answer |
